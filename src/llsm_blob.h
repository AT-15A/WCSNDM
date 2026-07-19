/* llsm_blob.h
   ============================================================================
   紧凑二进制 .llsm 序列化（float32）+ 无损压缩，用于 wav→.llsm 缓存。

   设计：保存 "llsm_chunk_tolayer1() 之后、phasepropagate 之前" 的 L1 状态。
   引擎(llsm_utau)加载后执行与非缓存流程一致的：
       phasepropagate(-1) → 编辑/伸缩 → tolayer0 → phasepropagate(+1) → synth

   存储管线（仿 moresampler 的 FastLZ 思路，做无损压缩）：
     1. 序列化到内存缓冲(membuf)，全部字段均为 4 字节字(int/float)，布局见下。
     2. 字节平面 shuffle(stride 4)：把每个字的同一字节位聚到一起——dB 频谱/相位平滑，
        高字节(符号+指数)近乎常量 → 长游程，极利于压缩。
     3. LZSS 无损压缩 shuffle 后缓冲。
     4. 写入时自校验(解压回来 memcmp 一致)；不一致或压不动则退回"stored(原样)"，
        保证永不写出损坏/更大的缓存。

   文件格式：
     [u32 outer_magic]  'BLZ2'=shuffle+lzss / 'BLZ0'=stored(原样)
     [u32 raw_size]     内层序列化字节数(4 的倍数)
     [u32 payload_size] 其后载荷字节数
     [payload...]       BLZ2: lzss(shuffle(raw)) ; BLZ0: raw

   内层序列化(raw)布局（与压缩前的旧版完全一致）：
     header: magic('BUL1'), ver, nfrm, thop, fnyq, nspec, npsd, nchannel,
             maxnhar, maxnhar_e, hm_method
     每帧:   f0; voiced?{ rd, nhar, vsphse[nhar], nsp, vtmagn[nsp] };
             has_nm?{ npsd, nch, psd[npsd], edc[nch], 每通道 eenv(nhar,ampl,phse) };
             psdres_len, psdres[]

   conf 在加载时用 llsm_aoptions_toconf 重建，再补 NFRM/NSPEC。
   向后兼容：旧 'BUL1' 裸文件不再支持(读不出→返回 NULL→引擎自动重分析覆盖)。
   ============================================================================ */
#ifndef LLSM_BLOB_H
#define LLSM_BLOB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llsm.h"

#define LLSM_BLOB_MAGIC   0x314C5542   /* 'BUL1' 内层流 magic */
#define LLSM_BLOB_VERSION 2
#define LLSM_BLOB_OUT_C   0x325A4C42   /* 'BLZ2' 外层: shuffle+lzss */
#define LLSM_BLOB_OUT_S   0x305A4C42   /* 'BLZ0' 外层: stored(原样) */

/* conf 取值（被 llsm_utau.c 使用，保留） */
static int     blob_cgi(llsm_container* c, int k, int d)     { int* p = (int*)llsm_container_get(c,k); return p?*p:d; }
static FP_TYPE blob_cgf(llsm_container* c, int k, FP_TYPE d) { FP_TYPE* p = (FP_TYPE*)llsm_container_get(c,k); return p?*p:d; }

/* ===========================================================================
   可增长字节缓冲(写) / 游标(读)。所有 blob 字段都是 4 字节字。
   =========================================================================== */
typedef struct { unsigned char* p; size_t len, cap, pos; int err; } membuf;
static void mb_init(membuf* m){ m->p=NULL; m->len=0; m->cap=0; m->pos=0; m->err=0; }
static void mb_free(membuf* m){ free(m->p); mb_init(m); }
static void mb_ensure(membuf* m, size_t add){
    size_t nc; unsigned char* np;
    if (m->len + add <= m->cap) return;
    nc = m->cap ? m->cap*2 : 4096;
    while (nc < m->len + add) nc *= 2;
    np = (unsigned char*)realloc(m->p, nc);
    if (!np) { m->err=1; return; }
    m->p=np; m->cap=nc;
}
static void mb_w (membuf* m, const void* d, size_t n){ mb_ensure(m,n); if(m->err) return; memcpy(m->p+m->len,d,n); m->len+=n; }
static int  mb_r (membuf* m, void* d, size_t n){ if(m->pos+n>m->len){ memset(d,0,n); m->err=1; return 0; } memcpy(d,m->p+m->pos,n); m->pos+=n; return 1; }
static void mb_wi(membuf* m, int v)   { mb_w(m,&v,4); }
static void mb_wf(membuf* m, float v) { mb_w(m,&v,4); }
static void mb_wa(membuf* m, FP_TYPE* a, int n){ int i; float v; for(i=0;i<n;++i){ v=(float)a[i]; mb_w(m,&v,4); } }
static int  mb_ri(membuf* m, int* v)  { return mb_r(m,v,4); }
static int  mb_rf(membuf* m, float* v){ return mb_r(m,v,4); }
static int  mb_ra(membuf* m, FP_TYPE* a, int n){ int i; float v; for(i=0;i<n;++i){ if(!mb_r(m,&v,4)) return 0; a[i]=(FP_TYPE)v; } return 1; }
static void mb_skip(membuf* m, size_t n){ m->pos += n; if(m->pos>m->len) m->pos=m->len; }

/* ===========================================================================
   字节平面 shuffle (stride 4)。n 为 4 的倍数(全部 4 字节字)；尾部不足 4 原样拷。
   =========================================================================== */
static void blob_shuffle4(const unsigned char* in, unsigned char* out, size_t n){
    size_t w=n/4, i;
    for(i=0;i<w;++i){ out[i]=in[4*i]; out[w+i]=in[4*i+1]; out[2*w+i]=in[4*i+2]; out[3*w+i]=in[4*i+3]; }
    for(i=w*4;i<n;++i) out[i]=in[i];
}
static void blob_unshuffle4(const unsigned char* in, unsigned char* out, size_t n){
    size_t w=n/4, i;
    for(i=0;i<w;++i){ out[4*i]=in[i]; out[4*i+1]=in[w+i]; out[4*i+2]=in[2*w+i]; out[4*i+3]=in[3*w+i]; }
    for(i=w*4;i<n;++i) out[i]=in[i];
}

/* ===========================================================================
   LZSS 无损压缩 (12bit 窗口 4095 / 匹配 3..18 / hash-head 匹配查找)。
   压缩器与解压器为匹配对；losslessness 由写入时 round-trip 自校验兜底。
   =========================================================================== */
#define LZ_WIN   4095
#define LZ_MINM  3
#define LZ_MAXM  18
#define LZ_HSIZE 8192
static int lz_hash(const unsigned char* s){ return ((s[0]<<10) ^ (s[1]<<5) ^ s[2]) & (LZ_HSIZE-1); }

static unsigned char* lz_compress(const unsigned char* src, size_t n, size_t* outn){
    unsigned char* out; int* head; size_t ip=0, op=0, i;
    out = (unsigned char*)malloc(n + n/8 + 64);
    if (!out) return NULL;
    head = (int*)malloc(sizeof(int)*LZ_HSIZE);
    if (!head) { free(out); return NULL; }
    for (i=0;i<LZ_HSIZE;++i) head[i]=-1;
    while (ip < n) {
        size_t cpos=op; unsigned char ctrl=0; int b;
        out[op++]=0;
        for (b=0; b<8 && ip<n; ++b) {
            int matched=0;
            if (ip+2 < n) {
                int h=lz_hash(src+ip), cand=head[h];
                head[h]=(int)ip;
                if (cand>=0 && (int)ip-cand<=LZ_WIN) {
                    size_t off=(size_t)(ip-cand), k=0, maxk=n-ip;
                    if (maxk>LZ_MAXM) maxk=LZ_MAXM;
                    while (k<maxk && src[ip+k]==src[ip+k-off]) k++;
                    if (k>=LZ_MINM) {
                        size_t q;
                        ctrl |= (unsigned char)(1<<b);
                        out[op++]=(unsigned char)(((off>>8)<<4) | (k-LZ_MINM));
                        out[op++]=(unsigned char)(off & 0xFF);
                        for (q=1;q<k;++q) if (ip+q+2<n) { int hh=lz_hash(src+ip+q); head[hh]=(int)(ip+q); }
                        ip+=k; matched=1;
                    }
                }
            }
            if (!matched) out[op++]=src[ip++];
        }
        out[cpos]=ctrl;
    }
    free(head);
    *outn=op;
    return out;
}

static int lz_decompress(const unsigned char* src, size_t n, unsigned char* out, size_t outn){
    size_t ip=0, op=0;
    while (op < outn) {
        unsigned char ctrl; int b;
        if (ip>=n) return 0;
        ctrl=src[ip++];
        for (b=0; b<8 && op<outn; ++b) {
            if (ctrl & (1<<b)) {
                unsigned char c0,c1; size_t off,len,k;
                if (ip+1>=n) return 0;
                c0=src[ip++]; c1=src[ip++];
                off=((size_t)(c0>>4)<<8) | c1; len=(size_t)(c0&0xF)+LZ_MINM;
                if (off==0 || op<off) return 0;
                for (k=0;k<len && op<outn;++k){ out[op]=out[op-off]; op++; }
            } else {
                if (ip>=n) return 0;
                out[op++]=src[ip++];
            }
        }
    }
    return 1;
}

/* ===========================================================================
   保存：chunk 应处于 tolayer1 之后的状态。
   =========================================================================== */
static int llsm_blob_save(const char* path, llsm_chunk* chunk,
                          FP_TYPE thop, int maxnhar, int maxnhar_e, int hm_method) {
    membuf mb; FILE* f; int nfrm, nspec, npsd, nchannel, i, c;
    FP_TYPE fnyq;
    unsigned char* sh=NULL; unsigned char* cz=NULL; size_t cn=0;
    int use_comp=0;
    unsigned int omagic, oraw, opay;

    if (!path || !chunk || !chunk->conf) return -1;
    mb_init(&mb);

    nfrm     = blob_cgi(chunk->conf, LLSM_CONF_NFRM, 0);
    nspec    = blob_cgi(chunk->conf, LLSM_CONF_NSPEC, 0);
    npsd     = blob_cgi(chunk->conf, LLSM_CONF_NPSD, 0);
    nchannel = blob_cgi(chunk->conf, LLSM_CONF_NCHANNEL, 1);
    fnyq     = blob_cgf(chunk->conf, LLSM_CONF_FNYQ, 0);

    mb_wi(&mb, LLSM_BLOB_MAGIC); mb_wi(&mb, LLSM_BLOB_VERSION);
    mb_wi(&mb, nfrm); mb_wf(&mb, (float)thop); mb_wf(&mb, (float)fnyq);
    mb_wi(&mb, nspec); mb_wi(&mb, npsd); mb_wi(&mb, nchannel);
    mb_wi(&mb, maxnhar); mb_wi(&mb, maxnhar_e); mb_wi(&mb, hm_method);

    for (i = 0; i < nfrm; ++i) {
        llsm_container* fr = chunk->frames[i];
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_F0);
        float f0 = pf0 ? (float)*pf0 : 0.0f;
        int voiced = f0 > 0.0f;
        llsm_nmframe* nm; FP_TYPE* res; int rn;

        mb_wf(&mb, f0);
        mb_wi(&mb, voiced);
        if (voiced) {
            FP_TYPE* prd = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_RD);
            FP_TYPE* vs  = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_VSPHSE);
            FP_TYPE* vt  = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_VTMAGN);
            int nhar = vs ? llsm_fparray_length(vs) : 0;
            int nsp  = vt ? llsm_fparray_length(vt) : 0;
            mb_wf(&mb, prd ? (float)*prd : 1.0f);
            mb_wi(&mb, nhar); if (nhar) mb_wa(&mb, vs, nhar);
            mb_wi(&mb, nsp);  if (nsp)  mb_wa(&mb, vt, nsp);
        }
        nm = (llsm_nmframe*)llsm_container_get(fr, LLSM_FRAME_NM);
        mb_wi(&mb, nm ? 1 : 0);
        if (nm) {
            mb_wi(&mb, nm->npsd); mb_wi(&mb, nm->nchannel);
            mb_wa(&mb, nm->psd, nm->npsd);
            mb_wa(&mb, nm->edc, nm->nchannel);
            for (c = 0; c < nm->nchannel; ++c) {
                llsm_hmframe* e = nm->eenv[c];
                int en = e ? e->nhar : 0;
                mb_wi(&mb, en);
                if (en) { mb_wa(&mb, e->ampl, en); mb_wa(&mb, e->phse, en); }
            }
        }
        res = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_PSDRES);
        rn = res ? llsm_fparray_length(res) : 0;
        mb_wi(&mb, rn); if (rn) mb_wa(&mb, res, rn);
    }
    if (mb.err || mb.len == 0) { mb_free(&mb); return -3; }

    /* shuffle + lzss + 自校验 */
    sh = (unsigned char*)malloc(mb.len);
    if (sh) {
        blob_shuffle4(mb.p, sh, mb.len);
        cz = lz_compress(sh, mb.len, &cn);
        if (cz && cn < mb.len) {                      /* 压得动才用 */
            unsigned char* chk = (unsigned char*)malloc(mb.len);
            if (chk) {
                if (lz_decompress(cz, cn, chk, mb.len) && memcmp(chk, sh, mb.len)==0)
                    use_comp = 1;                      /* round-trip 通过 */
                free(chk);
            }
        }
    }

    f = fopen(path, "wb");
    if (!f) { free(sh); if(cz) free(cz); mb_free(&mb); return -2; }
    if (use_comp) {
        omagic=LLSM_BLOB_OUT_C; oraw=(unsigned int)mb.len; opay=(unsigned int)cn;
        fwrite(&omagic,4,1,f); fwrite(&oraw,4,1,f); fwrite(&opay,4,1,f);
        fwrite(cz,1,cn,f);
    } else {
        omagic=LLSM_BLOB_OUT_S; oraw=(unsigned int)mb.len; opay=(unsigned int)mb.len;
        fwrite(&omagic,4,1,f); fwrite(&oraw,4,1,f); fwrite(&opay,4,1,f);
        fwrite(mb.p,1,mb.len,f);
    }
    fclose(f);
    free(sh); if(cz) free(cz); mb_free(&mb);
    return 0;
}

/* ===========================================================================
   加载：重建一个处于 tolayer1 后状态的 chunk。
   =========================================================================== */
static llsm_chunk* llsm_blob_load(const char* path) {
    FILE* f; unsigned int omagic, oraw, opay;
    unsigned char* payload; membuf mb;
    int magic, ver, nfrm, nspec, npsd, nchannel, maxnhar, maxnhar_e, hm_method, i, c;
    float thop, fnyq;
    llsm_aoptions* aopt; llsm_container* conf; llsm_chunk* chunk;

    f = fopen(path, "rb");
    if (!f) return NULL;
    if (fread(&omagic,4,1,f)!=1) { fclose(f); return NULL; }
    if (omagic != LLSM_BLOB_OUT_C && omagic != LLSM_BLOB_OUT_S) { fclose(f); return NULL; }
    if (fread(&oraw,4,1,f)!=1 || fread(&opay,4,1,f)!=1) { fclose(f); return NULL; }
    payload = (unsigned char*)malloc(opay ? opay : 1);
    if (!payload) { fclose(f); return NULL; }
    if (fread(payload,1,opay,f)!=opay) { free(payload); fclose(f); return NULL; }
    fclose(f);

    mb_init(&mb);
    mb.p = (unsigned char*)malloc(oraw ? oraw : 1);
    if (!mb.p) { free(payload); return NULL; }
    mb.cap = oraw; mb.len = oraw; mb.pos = 0;
    if (omagic == LLSM_BLOB_OUT_C) {
        unsigned char* sh = (unsigned char*)malloc(oraw ? oraw : 1);
        if (!sh) { free(payload); mb_free(&mb); return NULL; }
        if (!lz_decompress(payload, opay, sh, oraw)) { free(sh); free(payload); mb_free(&mb); return NULL; }
        blob_unshuffle4(sh, mb.p, oraw);
        free(sh);
    } else {
        memcpy(mb.p, payload, oraw);
    }
    free(payload);

    if (!mb_ri(&mb,&magic) || magic != LLSM_BLOB_MAGIC) { mb_free(&mb); return NULL; }
    mb_ri(&mb,&ver);
    mb_ri(&mb,&nfrm); mb_rf(&mb,&thop); mb_rf(&mb,&fnyq);
    mb_ri(&mb,&nspec); mb_ri(&mb,&npsd); mb_ri(&mb,&nchannel);
    mb_ri(&mb,&maxnhar); mb_ri(&mb,&maxnhar_e); mb_ri(&mb,&hm_method);

    aopt = llsm_create_aoptions();
    aopt->thop = (FP_TYPE)thop; aopt->npsd = npsd;
    aopt->maxnhar = maxnhar; aopt->maxnhar_e = maxnhar_e; aopt->hm_method = hm_method;
    conf = llsm_aoptions_toconf(aopt, (FP_TYPE)fnyq);
    llsm_container_attach(conf, LLSM_CONF_NFRM,  llsm_create_int(nfrm),  llsm_delete_int, llsm_copy_int);
    llsm_container_attach(conf, LLSM_CONF_NSPEC, llsm_create_int(nspec), llsm_delete_int, llsm_copy_int);
    chunk = llsm_create_chunk(conf, 1);
    llsm_delete_container(conf);
    llsm_delete_aoptions(aopt);
    if (!chunk) { mb_free(&mb); return NULL; }

    for (i = 0; i < nfrm; ++i) {
        float f0; int voiced, has_nm, rn;
        llsm_container* fr = chunk->frames[i];
        mb_rf(&mb,&f0); mb_ri(&mb,&voiced);
        llsm_container_attach(fr, LLSM_FRAME_F0, llsm_create_fp((FP_TYPE)f0), llsm_delete_fp, llsm_copy_fp);
        if (voiced) {
            float rd; int nhar, nsp;
            mb_rf(&mb,&rd);
            llsm_container_attach(fr, LLSM_FRAME_RD, llsm_create_fp((FP_TYPE)rd), llsm_delete_fp, llsm_copy_fp);
            mb_ri(&mb,&nhar);
            if (nhar) { FP_TYPE* vs = llsm_create_fparray(nhar); mb_ra(&mb, vs, nhar);
                llsm_container_attach(fr, LLSM_FRAME_VSPHSE, vs, llsm_delete_fparray, llsm_copy_fparray); }
            mb_ri(&mb,&nsp);
            if (nsp)  { FP_TYPE* vt = llsm_create_fparray(nsp); mb_ra(&mb, vt, nsp);
                llsm_container_attach(fr, LLSM_FRAME_VTMAGN, vt, llsm_delete_fparray, llsm_copy_fparray); }
        }
        mb_ri(&mb,&has_nm);
        if (has_nm) {
            int p, ch; llsm_nmframe* nm;
            mb_ri(&mb,&p); mb_ri(&mb,&ch);
            nm = (llsm_nmframe*)llsm_container_get(fr, LLSM_FRAME_NM);
            if (nm && nm->npsd == p && nm->nchannel == ch) {
                mb_ra(&mb, nm->psd, p);
                mb_ra(&mb, nm->edc, ch);
                for (c = 0; c < ch; ++c) {
                    int en; llsm_hmframe* e;
                    mb_ri(&mb,&en);
                    e = nm->eenv[c];
                    if (e && en) {
                        if (e->nhar != en) {
                            e->ampl = (FP_TYPE*)realloc(e->ampl, sizeof(FP_TYPE)*en);
                            e->phse = (FP_TYPE*)realloc(e->phse, sizeof(FP_TYPE)*en);
                            e->nhar = en;
                        }
                        mb_ra(&mb, e->ampl, en); mb_ra(&mb, e->phse, en);
                    } else if (en) {
                        mb_skip(&mb, (size_t)en*2*4);   /* 跳过无法落位的数据 */
                    }
                }
            } else {
                mb_skip(&mb, (size_t)(p+ch)*4);
                for (c=0;c<ch;++c){ int en; mb_ri(&mb,&en); mb_skip(&mb,(size_t)en*2*4); }
            }
        }
        mb_ri(&mb,&rn);
        if (rn) { FP_TYPE* res = llsm_create_fparray(rn); mb_ra(&mb, res, rn);
            llsm_container_attach(fr, LLSM_FRAME_PSDRES, res, llsm_delete_fparray, llsm_copy_fparray); }
    }
    mb_free(&mb);
    return chunk;
}

#endif /* LLSM_BLOB_H */

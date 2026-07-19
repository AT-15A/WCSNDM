/* l1_path.c —— LLSM1 合成后端(被主引擎的 L flag 调用)。
   只 include LLSM1 头(经 l1_prefix.h 前缀隔离)，对外仅暴露 llsm1_render()。
   管线：wavread -> pyin -> LLSM1 analyze -> RPS -> layer1 分解
        -> 逐【输出帧】：帧映射(辅音保留伸缩)+插值 layer1 参数 + 变调到【该帧目标 f0】(跟随音高曲线)
        -> 重积分载波(逐帧 f0=pitchbend/颤音) -> synth -> 峰值归一+音量 -> wavwrite。
   f0_at 回调由主引擎提供(包装 pitchcurve_f0)，输入输出时间(秒)，返回该时刻目标 f0(Hz)。 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "external/libpyin/pyin.h"
#include "math-funcs.h"
#include "envelope.h"
#include "llsm.h"
#include "l1_edits.h"

typedef double (*l1_f0_fn)(void* ctx, double t_sec);

static double l1clamp(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

/* 在 L1 的声道对数幅度 vt[n](faxis 为各 bin 频率)上应用 g/Mo/ME/Mr —— 与 L2 apply_flags_to_frame
   同一套数学(g=等比频率平移；Mo=F1区幂律warp；ME=unsharp共振峰强调；Mr=3kHz歌手共振峰高斯bump)。
   注意 L1 的 vt 是自然对数幅度，Mr 的 dB 增益需 /8.6859 转 ln 域。 */
static void l1_apply_vt_edits(FP_TYPE* vt, int n, FP_TYPE* faxis, const L1Edits* ed, double tgt_f0, double src_f0) {
  if(ed && ed->has_Mf && ed->Mf>0){
    double dst_ = (src_f0>0&&tgt_f0>0)? 12.0*log(tgt_f0/src_f0)/log(2.0) : 0.0;
    mf_formant_tune(vt, n, faxis[n-1], tgt_f0, l1clamp(ed->Mf,0,100)/100.0, 1.0/8.6859, dst_);
  }
  int i, w;
  if(!ed || n<3) return;
  if(ed->has_g && ed->g!=0.0){                     /* g：等比频率平移。常规刻度:clamp±50,每单位9cents(与L2一致) */
    double R = pow(2.0, -l1clamp(ed->g,-50,50)*9.0/1200.0);
    FP_TYPE* tmp=(FP_TYPE*)malloc(n*sizeof(FP_TYPE));
    for(i=0;i<n;i++){ double si=i/(R>1e-6?R:1e-6); int i0=(int)floor(si); double a=si-i0;
      tmp[i]= i0<0?vt[0] : (i0>=n-1?vt[n-1] : (FP_TYPE)((1-a)*vt[i0]+a*vt[i0+1])); }
    memcpy(vt,tmp,n*sizeof(FP_TYPE)); free(tmp);
  }
  if(ed->has_Mo && ed->Mo!=0.0){                    /* Mo：开口度(F1区<1800Hz 幂律warp)。[CALIB 2026-07-09b] 0.5→1.0→2.0 原始4倍(与 L2 同步) */
    double gamma=pow(2.0,(l1clamp(ed->Mo,-100,100)/100.0)*2.0);
    int ip=0; while(ip<n-1 && faxis[ip]<1800.0) ip++;
    if(ip>=2){ FP_TYPE* tmp=(FP_TYPE*)malloc(n*sizeof(FP_TYPE)); for(i=0;i<n;i++)tmp[i]=vt[i];
      for(i=1;i<ip;i++){ double si=pow((double)i/ip,gamma)*ip; int i0=(int)floor(si); double a=si-i0;
        tmp[i]= i0<0?vt[0]:(i0>=n-1?vt[n-1]:(FP_TYPE)((1-a)*vt[i0]+a*vt[i0+1])); }
      memcpy(vt,tmp,n*sizeof(FP_TYPE)); free(tmp); }
  }
  if(ed->has_ME && ed->ME!=0.0){                    /* ME：共振峰强调(unsharp mask)。[CALIB 2026-07-12c] 正向2.0+单点钳±12dB(ln域/8.6859),负向1.0(与L2同步) */
    double v01=l1clamp(ed->ME,-100,100)/100.0;
    double emph=v01*(v01>0?2.0:1.0);
    double cap=12.0/8.6859;   /* ±12dB 换算 ln 域 */
    int half=0; while(half<n-1 && faxis[half]<1200.0) half++; if(half<1)half=1;
    FP_TYPE* sm=(FP_TYPE*)malloc(n*sizeof(FP_TYPE));
    for(i=0;i<n;i++){ double s=0;int c=0; for(w=-half;w<=half;w++){int k=i+w;if(k<0||k>=n)continue;s+=vt[k];c++;} sm[i]=(FP_TYPE)(c?s/c:vt[i]); }
    for(i=0;i<n;i++){ double d5=emph*(vt[i]-sm[i]); if(d5>cap)d5=cap; if(d5<-cap)d5=-cap; vt[i]+=(FP_TYPE)d5; }
    free(sm);
  }
  if(ed->has_RG && ed->RG!=0.0 && tgt_f0>50){   /* 自动混声(L1侧=倾斜部分,与L2同门控) */
    double st=69.0+12.0*log(tgt_f0/440.0)/log(2.0);
    double r=1.0/(1.0+exp(-(st-65.0)/2.0));
    double eff=r*l1clamp(ed->RG,-100,100)/100.0;
    if(eff>1e-4||eff<-1e-4){ int i9;
      for(i9=0;i9<n;i9++) if(faxis[i9]>1000.0){
        double db=-5.0*eff*log(faxis[i9]/1000.0)/log(2.0);
        if(db>8)db=8; if(db<-8)db=-8;
        vt[i9]+=(FP_TYPE)(db/8.6859); } }
  }
  if(ed->has_NA && ed->NA!=0.0)
    na_nasal(vt, n, faxis[n-1], l1clamp(ed->NA,-100,100)/100.0, 1.0/8.6859);
  if(ed->has_Mr && ed->Mr!=0.0){                    /* Mr：歌手共振峰(3kHz 高斯 bump，dB->ln) */
    double gln=((l1clamp(ed->Mr,-100,100)/100.0)*12.0)/8.6859, fc=3000.0, sig=1200.0;
    for(i=0;i<n;i++){ double d=(faxis[i]-fc)/sig; vt[i]+=(FP_TYPE)(gln*exp(-0.5*d*d)); }
  }
  if(ed->has_Mt && ed->Mt!=0.0){                    /* Mt：张力(mo语义,tilt部分)：>1kHz 谱倾斜 ±6dB/oct 封顶±10dB */
    double k=(l1clamp(ed->Mt,-100,100)/100.0)*6.0;
    for(i=0;i<n;i++) if(faxis[i]>1000.0){
      double db=k*log(faxis[i]/1000.0)/log(2.0);
      if(db>10)db=10; if(db<-10)db=-10;
      vt[i]+=(FP_TYPE)(db/8.6859); }
  }
  if(ed->has_MH && ed->MH!=0.0){                    /* MH：高频倾斜(>3kHz,MH>0=更压暗) */
    double k=-(l1clamp(ed->MH,-100,100)/100.0)*6.0;
    for(i=0;i<n;i++) if(faxis[i]>3000.0){
      double db=k*log(faxis[i]/3000.0)/log(2.0);
      if(db>12)db=12; if(db<-12)db=-12;
      vt[i]+=(FP_TYPE)(db/8.6859); }
  }
  if(ed->tm_n>1){                                   /* TM：音色对齐校正 EQ(主引擎按 bank 档案算好,dB->ln) */
    for(i=0;i<n;i++)
      vt[i]+=(FP_TYPE)(dn_interp_db_(ed->tm_corr_db, ed->tm_n, ed->tm_fnyq, faxis[i])/8.6859);
  }
}

/* sin_ret/nos_ret 任一非 NULL 时：不写文件、不归一，改为返回【原始 L1 分量】(malloc 拷贝)+长度+fs，
   供主引擎做混合：V1=L2谐波+L1噪声(取 nos)、V2=L1谐波+L2噪声(取 sin，并用 nos 作电平参考)。
   否则正常归一化并写 outwav。 */
int llsm1_render(const char* inwav, const char* outwav,
                 double out_len_ms, double consonant_ms, double velocity, double volume,
                 double offset_ms, double cutoff_ms,
                 const L1Edits* ed, l1_f0_fn f0_at, void* f0ctx,
                 FP_TYPE** sin_ret, FP_TYPE** nos_ret, int* comp_n, int* comp_fs,
                 int mj_cap) {
  int fs=0,nbit=0,nx=0,nfrm=0,nhop=256;
  FP_TYPE* x = wavread((char*)inwav, &fs,&nbit,&nx);
  if(!x||nx<=0){ fprintf(stderr,"[L1] read fail %s\n", inwav); return 1; }
  wcs_to_44100(&x, &nx, &fs, "L1");
  double thop_ms = 1000.0*nhop/fs;   /* 源帧粒度(分析 hop),源索引 p=t_src/thop_ms 恒用它 */

  /* Mj 转音平滑(与 L2 同机理):谐波合成同为"每帧恒定频率阶梯"OLA(llsm_layer0_synthesize),
     快速 pitchbend 下帧重叠失相位相干、中高谐波断裂。加密合成帧(nhop/F)→误差降 1/F²。
     自动因子:扫 f0_at 取峰值 半音/hop,F=ceil(st/0.4) 钳 [1,mj_cap];平缓 F=1 不掉速。 */
  int mjF = 1;
  if(mj_cap >= 1 && f0_at){
    double dt = thop_ms/1000.0, maxst = 0.0;
    for(double t=0; t < out_len_ms/1000.0; t += dt){
      double a=f0_at(f0ctx,t), b=f0_at(f0ctx,t+dt);
      if(a>50 && b>50){ double st=fabs(12.0*log(b/a)/log(2.0)); if(st>maxst)maxst=st; }
    }
    mjF=(int)ceil(maxst/0.4); if(mjF<1)mjF=1; if(mjF>mj_cap)mjF=mj_cap;
    if(mjF>1) fprintf(stderr,"[Mj/L1] transition smoothing: %dx frames (peak %.2f st/hop)\n", mjF, maxst);
  }
  int nhop_out = nhop;
  if(mjF>1){ nhop_out = nhop/mjF; if(nhop_out<16) nhop_out=16; }
  double thop_out_ms = 1000.0*nhop_out/fs;   /* 输出帧步进(加密时=细 hop) */

  llsm_parameters lp = llsm_init(4);
  lp.a_nosbandf[0]=2000; lp.a_nosbandf[1]=5000; lp.a_nosbandf[2]=9000;
  lp.a_mvf=12000; lp.a_nnos=192; lp.a_nosf=fs/2; lp.a_nhop=nhop; lp.a_nhar=400; lp.a_nhare=5;

  FP_TYPE* f0 = l1f0_cache_get(inwav, fs, nhop, &nfrm);
  if(!f0){
    pyin_config pp = pyin_init(nhop);
    pp.fmin=50; pp.fmax=900; pp.trange=24; pp.nf=(int)ceil(fs*0.025); pp.w=pp.nf/4; pp.bias=10;
    f0 = pyin_analyze(pp, x, nx, fs, &nfrm);
    if(f0 && nfrm>0) l1f0_cache_put(inwav, fs, nhop, f0, nfrm);
  }
  if(ed && ed->has_DN && ed->DN>0 && f0 && nfrm>4){ int fx=f0_sanitize(f0,nfrm); if(fx>0) fprintf(stderr,"[L1] f0fix: %d corrections\n",fx); }
  if(!f0||nfrm<=0){ fprintf(stderr,"[L1] pyin fail\n"); free(x); return 1; }

  llsm_layer0* model = llsm_layer0_analyze(lp, x, nx, fs, f0, nfrm, NULL);

  /* RPS 去载波 -> 相对相位域(不改 f0) */
  FP_TYPE* p0 = calloc(nfrm,sizeof(FP_TYPE));
  for(int i=0;i<nfrm;i++) p0[i] = model->frames[i]->f0>0 ? -model->frames[i]->sinu->phse[0]:0;
  llsm_layer0_phaseshift(model, p0); free(p0);

  /* DN 层②·谐波侧(L1)：V2 的谐波走这里——常驻嘶声会被谐波分析拟合成高次"谐波"幅度
     (贴着一条不随发声起伏的底线),噪声侧清了 y_sin 还带着。用 L2 静音印记(经 L1Edits
     传入)做绝对底噪参考,轨迹=谐波序号(f=(j+1)*f0中值)。layer1 分解前做:vt/vs 都吃到
     干净数据。纯 L1 模式无印记(dn_print_n=0)自动跳过。 */
  if(ed && ed->has_DN && ed->DN>0 && ed->dn_print_n>1){
    int maxnh=0, nv=0;
    for(int i=0;i<nfrm;i++){ llsm_frame* fr=model->frames[i];
      if(fr && fr->f0>0 && fr->sinu){ nv++; if(fr->sinu->nhar>maxnh) maxnh=fr->sinu->nhar; } }
    if(nv>=8 && maxnh>=8 && maxnh<=2048){
      double* pw=(double*)malloc(sizeof(double)*(size_t)nfrm*maxnh);
      double* ftrk=(double*)malloc(sizeof(double)*maxnh);
      double* gn=(double*)malloc(sizeof(double)*maxnh);
      double* f0v=(double*)malloc(sizeof(double)*nfrm);
      double f0m=0.0; int c2=0;
      if(pw&&ftrk&&gn&&f0v){
        for(int i=0;i<nfrm;i++) if(model->frames[i]->f0>0) f0v[c2++]=(double)model->frames[i]->f0;
        qsort(f0v,c2,sizeof(double),dn_cmp_d_); f0m=f0v[c2/2];
        for(int t=0;t<maxnh;t++) ftrk[t]=f0m*(t+1);
        for(int i=0;i<nfrm;i++){ llsm_frame* fr=model->frames[i];
          for(int t=0;t<maxnh;t++){
            double v=-1.0;
            if(fr && fr->f0>0 && fr->sinu && t<fr->sinu->nhar){ double a2=(double)fr->sinu->ampl[t]; v=a2*a2; }
            pw[(size_t)i*maxnh+t]=v; } }
        if(dn_harm_gain(pw,nfrm,maxnh,ftrk,ed->dn_print_db,ed->dn_print_n,ed->dn_print_fnyq,
                        ed->dn_nsnr_db, l1clamp(ed->DN,0,100)/100.0, gn)){
          for(int i=0;i<nfrm;i++){ llsm_frame* fr=model->frames[i];
            if(!(fr && fr->f0>0 && fr->sinu)) continue;
            for(int t=0;t<fr->sinu->nhar && t<maxnh;t++)
              fr->sinu->ampl[t]=(FP_TYPE)(fr->sinu->ampl[t]*sqrt(gn[t])); }
          fprintf(stderr,"[L1] DN harm-floor: on (snr=%.0f dB)\n", ed->dn_nsnr_db);
        }
      }
      free(pw);free(ftrk);free(gn);free(f0v);
    }
  }

  /* TM 音色对齐·噪声侧(L1)：192 点噪声谱(ln 域,warp 频轴)加同一条校正 EQ。
     谐波侧在 l1_apply_vt_edits 应用;此处覆盖纯 L1 与 V1 的 L1 噪声。源帧上做一次,
     后续输出帧经 llsm_copy_frame 自动带上。 */
  if(ed && ed->tm_n>1){
    FP_TYPE* fw = llsm_warp_freq(0, lp.a_nosf, lp.a_nnos, lp.a_noswarp);
    for(int i=0;i<nfrm;i++){ llsm_frame* fr=model->frames[i];
      if(!fr || !fr->noise || !fr->noise->spec) continue;
      for(int k=0;k<fr->noise->nnos && k<lp.a_nnos;k++){
        double fc=0.5*((double)fw[k]+(double)fw[k+1]);
        fr->noise->spec[k]+=(FP_TYPE)(dn_interp_db_(ed->tm_corr_db, ed->tm_n, ed->tm_fnyq, fc)/8.6859);
      }
    }
    free(fw);
  }

  llsm_layer1* m1 = llsm_layer1_from_layer0(lp, model, 2048, fs);
  const int nfft=2048, ns=nfft/2+1;
  FP_TYPE* faxis = llsm_uniform_faxis(2048, lp.a_nosf*2);
  FP_TYPE* vtmag = (FP_TYPE*)malloc(ns*sizeof(FP_TYPE));

  /* 辅音保留式映射参数(区段 = [offset, end_ms]，与 LLSM2 build_remap 一致，正确使用 oto 的 offset/cutoff) */
  double total_src = nfrm * thop_ms;
  double vel = pow(2.0, 1.0 - velocity/100.0);
  double end_ms = (cutoff_ms>=0) ? (total_src-cutoff_ms) : (offset_ms-cutoff_ms);
  if(end_ms < offset_ms+thop_ms) end_ms = offset_ms+thop_ms;
  if(end_ms > total_src) end_ms = total_src;
  double con_end = offset_ms + consonant_ms; if(con_end>end_ms) con_end=end_ms;
  double con_out = consonant_ms*vel; if(con_out>out_len_ms) con_out=out_len_ms;
  double vow_src = end_ms-con_end; if(vow_src<1e-6)vow_src=1e-6;
  double vow_out = out_len_ms-con_out; if(vow_out<0)vow_out=0;
  double ratio = vow_out/vow_src;

  int N = (int)(out_len_ms/thop_out_ms + 0.5); if(N<2)N=2;
  llsm_conf c2 = model->conf; c2.nfrm=N;
  c2.nhop=nhop_out; c2.thop=(FP_TYPE)nhop_out/fs;   /* 合成器 OLA hop 同步细化(否则密帧按粗 hop 错叠) */
  llsm_layer0* sm = llsm_create_empty_layer0(c2);

  /* A(颤音幅度联动)基线：预取逐帧目标 f0，±HW 帧平滑作基线(时间跨度~174ms,随加密放大帧数)，devst=偏离半音数 */
  int aHW = 15*mjF;
  FP_TYPE* tgtarr=(FP_TYPE*)calloc(N,sizeof(FP_TYPE));
  FP_TYPE* devst =(FP_TYPE*)calloc(N,sizeof(FP_TYPE));
  for(int k=0;k<N;k++) tgtarr[k]=(FP_TYPE)(f0_at?f0_at(f0ctx,k*thop_out_ms/1000.0):0.0);
  if(ed && ed->has_A){
    for(int k=0;k<N;k++){ double s=0;int c=0;
      for(int w=-aHW;w<=aHW;w++){int q=k+w; if(q<0||q>=N||tgtarr[q]<=0)continue; s+=log((double)tgtarr[q]);c++;}
      if(c>0 && tgtarr[k]>0) devst[k]=(FP_TYPE)(12.0*(log((double)tgtarr[k])-s/c)/log(2.0)); }
  }

  for(int k=0;k<N;k++){
    double t_out = k*thop_out_ms;
    double t_src = (t_out<con_out) ? (offset_ms + t_out/(vel>1e-6?vel:1e-6))
                                   : (con_end + (t_out-con_out)/(ratio>1e-9?ratio:1e-9));
    if(t_src<offset_ms) t_src=offset_ms;
    if(t_src>end_ms) t_src=end_ms;
    double p = t_src/thop_ms;
    int s0=(int)p; if(s0<0)s0=0; if(s0>nfrm-1)s0=nfrm-1;
    int s1=s0+1; if(s1>nfrm-1)s1=nfrm-1;
    double frac=p-s0; if(frac<0)frac=0; if(frac>1)frac=1;
    llsm_frame* a=model->frames[s0]; llsm_frame* b=model->frames[s1];
    llsm_frame* base=(a->f0>0)?a:b;
    double tgt = tgtarr[k];
    int voiced = (a->f0>0 && b->f0>0 && tgt>0);

    sm->frames[k] = llsm_create_frame(base->sinu->nhar, c2.nhare, c2.nnos, base->noise->nchannel);
    llsm_copy_frame(sm->frames[k], base);   /* 噪声 + 兜底 sinu */

    if(voiced){
      llsm_sinframe* d = sm->frames[k]->sinu;
      for(int bn=0;bn<ns;bn++) vtmag[bn]=m1->vt_resp_magn[s0][bn]*(1-frac)+m1->vt_resp_magn[s1][bn]*frac;
      l1_apply_vt_edits(vtmag, ns, faxis, ed, tgt, a->f0*(1-frac)+b->f0*frac);   /* g/Mo/ME/Mr 共通编辑(与 L2 同数学) */
      double origf0 = a->f0*(1-frac)+b->f0*frac;
      int nh0 = a->sinu->nhar<b->sinu->nhar?a->sinu->nhar:b->sinu->nhar;
      if(nh0>d->nhar) nh0=d->nhar;
      FP_TYPE freq[512]; int nh=nh0<512?nh0:512;
      for(int j=0;j<nh;j++){ freq[j]=(FP_TYPE)(tgt*(j+1.0)); if(freq[j]>lp.a_mvf){nh=j;break;} }
      if(nh<1){ sm->frames[k]->f0=0; continue; }
      FP_TYPE* na=interp1(faxis,vtmag,ns,freq,nh);
      for(int j=0;j<nh;j++) na[j]=exp(na[j]);
      FP_TYPE* np=llsm_harmonic_minphase(na,nh);
      FP_TYPE* la=interp1(faxis,m1->lip_resp_magn,ns,freq,nh);
      FP_TYPE* lpp=interp1(faxis,m1->lip_resp_phse,ns,freq,nh);
      /* 帧级增益：A(颤音幅度联动,偏离1半音*A/100*3dB,钳±12dB) + bh(辅音区谐波,gain=1+0.05bh) */
      double fgain=1.0;
      if(ed && ed->has_A && ed->A!=0.0){
        double gdb=l1clamp((l1clamp(ed->A,-100,100)/100.0)*3.0*devst[k],-12.0,12.0);
        fgain*=pow(10.0,gdb/20.0); }
      if(ed && ed->has_bh && ed->bh!=0.0 && t_out<con_out){
        double gb=1.0+0.05*l1clamp(ed->bh,-20,100); if(gb<1e-4)gb=1e-4; fgain*=gb; }
      for(int j=0;j<nh;j++){
        double vsa=m1->vs_har_ampl[s0][j]*(1-frac)+m1->vs_har_ampl[s1][j]*frac;
        double re=cos(m1->vs_har_phse[s0][j])*(1-frac)+cos(m1->vs_har_phse[s1][j])*frac;
        double im=sin(m1->vs_har_phse[s0][j])*(1-frac)+sin(m1->vs_har_phse[s1][j])*frac;
        d->ampl[j]=(FP_TYPE)(vsa*na[j]*la[j]*origf0/tgt*fgain);
        d->phse[j]=(FP_TYPE)(atan2(im,re)+np[j]+lpp[j]);
      }
      d->nhar=nh;
      sm->frames[k]->f0=(FP_TYPE)tgt;
      free(na);free(np);free(la);free(lpp);
    }
  }
  free(vtmag); free(faxis); llsm_delete_layer1(m1);

  /* 逐帧 f0 重积分载波 = pitchbend/颤音(步进用细 hop,与合成器 OLA 一致) */
  FP_TYPE* ph=calloc(N,sizeof(FP_TYPE));
  for(int k=1;k<N;k++) if(sm->frames[k]->f0>0) ph[k]=ph[k-1]+sm->frames[k]->f0*nhop_out/fs*2*M_PI;
  llsm_layer0_phaseshift(sm, ph); free(ph);

  lp.s_fs=fs;
  llsm_output* out = llsm_layer0_synthesize(lp, sm);
  int ny=out->ny;

  /* Mb(气声±12dB,全体噪声) + b(清辅音,仅无声帧,gain=1+0.05b)：作用 y_nos 后重混 y */
  if(ed && (ed->has_Mb||ed->has_Ab||ed->has_b) && out->y_nos && out->y_sin){
    double gmb = ed->has_Mb ? pow(10.0,(l1clamp(ed->Mb,-100,100)/100.0)*12.0/20.0) : 1.0;   /* 仅有声帧 */
    double gab = ed->has_Ab ? pow(10.0,(l1clamp(ed->Ab,-100,100)/100.0)*12.0/20.0) : 1.0;   /* 全帧 */
    double gb  = ed->has_b  ? (1.0+0.05*l1clamp(ed->b,-20,100)) : 1.0; if(gb<0)gb=0;
    for(int i=0;i<ny;i++){
      int k=i/nhop_out; if(k>N-1)k=N-1;   /* 样本→输出帧:用细 hop */
      double g = gab * ((sm->frames[k]->f0<=0) ? gb : gmb);
      out->y_nos[i]=(FP_TYPE)(out->y_nos[i]*g);
      out->y[i]=out->y_sin[i]+out->y_nos[i];
    }
  }
  free(tgtarr); free(devst);

  if(sin_ret || nos_ret){              /* 取原始分量,交给主引擎混合(不写不归一) */
    if(comp_n) *comp_n = ny; if(comp_fs) *comp_fs = fs;
    if(sin_ret){ *sin_ret=(FP_TYPE*)malloc((ny>0?ny:1)*sizeof(FP_TYPE));
      if(*sin_ret && out->y_sin) memcpy(*sin_ret, out->y_sin, ny*sizeof(FP_TYPE));
      else if(*sin_ret) memset(*sin_ret, 0, ny*sizeof(FP_TYPE)); }
    if(nos_ret){ *nos_ret=(FP_TYPE*)malloc((ny>0?ny:1)*sizeof(FP_TYPE));
      if(*nos_ret && out->y_nos) memcpy(*nos_ret, out->y_nos, ny*sizeof(FP_TYPE));
      else if(*nos_ret) memset(*nos_ret, 0, ny*sizeof(FP_TYPE)); }
  } else {
    FP_TYPE* y=out->y;
    double peak=1e-9; for(int i=0;i<ny;i++){ double av=fabs(y[i]); if(av>peak)peak=av; }
    double vscale = pow(0.5/peak, 1.0) * (volume>0?volume:100)/100.0;
    for(int i=0;i<ny;i++){ double v=y[i]*vscale; if(v>0.99)v=0.99; if(v<-0.99)v=-0.99; y[i]=(FP_TYPE)v; }
    wavwrite(y, ny, fs, 16, (char*)outwav);
  }

  llsm_deinit(lp); llsm_delete_layer0(model); free(f0); free(x); llsm_delete_output(out);
  return 0;
}

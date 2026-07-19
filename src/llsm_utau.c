/* llsm_utau.c
   ============================================================================
   基于 libllsm2 的 UTAU resampler 引擎（非 hifigan / 原生合成路径）。

   设计蓝本：
     - reference/pitch_shift.c   : LLSM2 端到端变调管线（analyze→L1→phase→F0→L0→synth）
     - reference/demo-stretch.c  : LLSM2 原生时间伸缩（interp_llsm_frame + phasepropagate）
     - timestretch.py            : 已验证的 UTAU 时间映射 build_remap()（本文件移植为 C）

   核心管线（一次循环完成 音长伸缩 + 变调 + pitchbend + flag 编辑）：
     1. 解析 UTAU 13 参数
     2. 解码 pitchbend(Base64) + note→f0；解析 flags
     3. wavread → pyin → llsm_analyze            （TODO: 接 .llsm 缓存）
     4. tolayer1(2048) → phasepropagate(-1)      进入可安全插值的 L1 相对相位域
     5. 按 build_remap() 建新 chunk(nfrm_new)，逐输出帧：
          - copy 最近源帧 + interp_llsm_frame(下一源帧, ratio)   ← 抄 demo-stretch
          - 覆盖 F0 = UTAU 音高曲线(该输出帧时刻)                  ← 抄 pitch_shift
          - 施加 flag 编辑（VTMAGN / RD / NM …）                  ← 本引擎扩展
          - 复制 PSDRES（带抖动，防循环机械感）
     6. tolayer0 → phasepropagate(+1)            按新 F0 重排谐波，共振峰保持
     7. llsm_synthesize → volume/归一化 → wavwrite

   Visual Studio 编译（与 pitch_shift.c 同一项目）：
     把 analyze_and_resynthesize.c / csv_to_wav.c / pitch_shift.c 都设为"从生成中排除"，
     仅 llsm_utau.c 设为活动源，生成后产物即 llsm_runner.exe，复制重命名为 llsm_utau.exe。

   进度标记：
     [OK]   已对照蓝本，逻辑应正确
     [TODO] 待实现 / 待接库
     [CHK]  需与参考实现交叉核对（尤其 pitchbend / flag 数值）
   ============================================================================ */

#ifdef _MSC_VER
#pragma warning(disable:4244)
#pragma warning(disable:4305)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include <sys/stat.h>

#include "llsm.h"
#include "llsmutils.h"
#include "pyin.h"
#include "wavfile.h"
#include <ciglet/ciglet.h>   /* fft/ifft（LD 倒谱低阶化用） */
#include <winsock2.h>         /* HF daemon 直连(C客户端,免python客户端开销) */
#pragma comment(lib, "ws2_32.lib")
#include "l1_edits.h"         /* L1/L2 共通编辑量 */

/* LLSM1 合成后端(llsm1_src/l1_path.c，符号经 l1ns_ 前缀隔离)。L flag 切到此路径。
   f0_at 回调逐输出帧取目标 f0(跟随音高曲线=pitchbend/颤音)；ed=共通声道编辑(g/Mo/ME/Mr)。 */
extern int llsm1_render(const char* inwav, const char* outwav,
                        double out_len_ms, double consonant_ms, double velocity, double volume,
                        double offset_ms, double cutoff_ms,
                        const L1Edits* ed, double (*f0_at)(void* ctx, double t_sec), void* f0ctx,
                        FP_TYPE** sin_ret, FP_TYPE** nos_ret, int* comp_n, int* comp_fs,
                        int mj_cap);
#include "llsm_blob.h"

/* 文件修改时间（秒）；不存在返回 -1。用于 .llsm 缓存新鲜度判断（仿 moresampler）。 */
static long file_mtime(const char* p) {
    struct _stat st;
    if (_stat(p, &st) == 0) return (long)st.st_mtime;
    return -1;
}

#ifndef DEFAULT_HOP
#define DEFAULT_HOP 512
#endif
#ifndef HF_TILT_COEF
#define HF_TILT_COEF  0.07   /* 自适应高频滚降基础系数：每半音上变调的 dB/oct 下倾(默认≈mo) */
#endif
#ifndef HF_TILT_PIVOT
#define HF_TILT_PIVOT 1500.0 /* 枢轴频率(Hz)：以下不动，以上按 log2(f/pivot) 递减 */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FP_TYPE* wavread(const char* path, int* fs, int* nbit, int* nx);
void     wavwrite(const FP_TYPE* y, int ny, int fs, int nbit, const char* path);

/* ===========================================================================
   [OK] 第 0 部分：UTAU 参数容器
   --------------------------------------------------------------------------
   UTAU resampler 调用：
     engine in.wav out.wav pitch velocity flags offset length consonant
            cutoff volume modulation tempo pitchbend
   =========================================================================== */
typedef struct {
    const char* in_file;
    const char* out_file;
    const char* pitch_note;   /* 目标音名，如 "A4"（实际曲线由 pitchbend 决定） */
    double      velocity;     /* 子音速度 0..200，默认 100 */
    const char* flags;        /* flag 串，如 "g-5Mb30" */
    double      offset;       /* ms：采样起点 */
    double      length;       /* ms：目标输出长度 */
    double      consonant;    /* ms：辅音固定区长度 */
    double      cutoff;       /* ms：采样终点（<0 为相对） */
    double      volume;       /* %：音量，默认 100 */
    double      modulation;   /* %：调制，默认 0 */
    double      tempo;        /* BPM（来自 "!120"） */
    const char* pitchbend;    /* Base64 音高弯曲串 */
} UtauArgs;

/* ===========================================================================
   [OK] 第 1 部分：音名 / 频率换算
   =========================================================================== */
static int note_to_midi(const char* s) {
    /* 形如 C4 / A#3 / F#5 */
    static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    char letter[3] = {0};
    int  oct, i, semitone = -1, k = 0;
    if (!s || !s[0]) return -1;
    letter[k++] = (char)toupper((unsigned char)s[0]);
    if (s[1] == '#') letter[k++] = '#';
    for (i = 0; i < 12; ++i)
        if (!strcmp(letter, names[i])) { semitone = i; break; }
    if (semitone < 0) return -1;
    oct = atoi(s + k);
    return 12 * (oct + 1) + semitone;
}
static double midi_to_hz(double midi) { return 440.0 * pow(2.0, (midi - 69.0) / 12.0); }

/* ===========================================================================
   [OK] 第 2 部分：UTAU pitchbend（Base64 + RLE）解码 → cents 数组
   --------------------------------------------------------------------------
   逐 2 字符 → 12bit 有符号整数(cents)；'#N#' 表示把上一个值再重复 N 次；末尾补 1 个 0。
   已与 hifisampler/util/parse_utau.py::pitch_string_to_cents 做数值对拍：
   tests/pitchbend_parity.py 8 个用例（含多段/多位 RLE、字母表边界）全部逐值一致。
   =========================================================================== */
static int b64_uint6(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return 0;
}
/* 返回 cents 数组（调用方 free），长度写入 *out_n。无 pitchbend 时返回 NULL,*out_n=0 */
static double* decode_pitchbend(const char* s, int* out_n) {
    int cap = 256, n = 0;
    double* arr = (double*)malloc(sizeof(double) * cap);
    int i = 0, len = s ? (int)strlen(s) : 0;
    *out_n = 0;
    if (!s || len == 0) { free(arr); return NULL; }
    while (i < len) {
        if (s[i] == '#') {            /* RLE：#N# 重复上一个值 N 次 */
            int rep = 0; ++i;
            while (i < len && s[i] != '#') { rep = rep * 10 + (s[i] - '0'); ++i; }
            if (i < len && s[i] == '#') ++i;
            double last = n > 0 ? arr[n - 1] : 0.0;
            int r;
            for (r = 0; r < rep; ++r) {
                if (n >= cap) { cap *= 2; arr = (double*)realloc(arr, sizeof(double)*cap); }
                arr[n++] = last;
            }
        } else if (i + 1 < len) {     /* 普通 2 字符对 → 12bit 有符号 */
            int v = b64_uint6(s[i]) * 64 + b64_uint6(s[i + 1]);
            if (v >= 2048) v -= 4096;
            if (n >= cap) { cap *= 2; arr = (double*)realloc(arr, sizeof(double)*cap); }
            arr[n++] = (double)v;     /* 单位：cents */
            i += 2;
        } else { ++i; }
    }
    /* 末尾补 1 个 0，对齐 hifisampler pitch_string_to_cents */
    if (n >= cap) { cap *= 2; arr = (double*)realloc(arr, sizeof(double)*cap); }
    arr[n++] = 0.0;
    *out_n = n;
    return arr;
}

/* ===========================================================================
   [TODO] 第 3 部分：UTAU 音高曲线求值
   --------------------------------------------------------------------------
   给定输出帧时刻 t_out_sec，返回目标 F0(Hz)。
     base_midi = note_to_midi(pitch_note)
     pitchbend 点间隔 = 60 / (tempo * 96) 秒（与 hifisampler 一致）
     cents(t) = 线性插值 pitchbend 数组
     f0 = midi_to_hz(base_midi + cents/100)
   [CHK] pitchbend 与输出时间轴的对齐（preutterance/overlap）属 UTAU 细节，先按
         "输出起点=pitchbend起点"实现，后续校准。
   =========================================================================== */
typedef struct {
    double  base_midi;
    double* cents;
    int     n;
    double  dt;       /* pitchbend 点间隔（秒） */
} PitchCurve;

static double pitchcurve_f0(const PitchCurve* pc, double t_out_sec) {
    if (!pc->cents || pc->n == 0) return midi_to_hz(pc->base_midi);
    double fidx = t_out_sec / pc->dt;
    int i0 = (int)floor(fidx);
    double a = fidx - i0;
    double c;
    if (i0 < 0)            c = pc->cents[0];
    else if (i0 >= pc->n-1) c = pc->cents[pc->n - 1];
    else                   c = (1.0 - a) * pc->cents[i0] + a * pc->cents[i0 + 1];
    return midi_to_hz(pc->base_midi + c / 100.0);
}

/* LLSM1 后端的逐帧 f0 回调(ctx=PitchCurve*)：把音高曲线(pitchbend/颤音)喂给 l1_path。 */
static double l1_f0_cb(void* ctx, double t_sec) {
    return pitchcurve_f0((const PitchCurve*)ctx, t_sec);
}


/* ===========================================================================
   [TODO] 第 4 部分：flags 解析
   --------------------------------------------------------------------------
   解析形如 "g-5Mb30Mt-20" 的串为键值表。键可为 1~2 字母（g, t, Mb, Mt, ME...）。
   先实现解析框架，具体作用在 apply_flags_to_frame 内逐个填。
   =========================================================================== */
typedef struct {
    double g;    /* 性别/共振峰 [-600,600] */
    double t;    /* 音高微调 cents [-1200,1200] */
    double Mt;   /* 张力(mo语义) [-100,100]：Rd 变化 + >1kHz 谱倾斜(正=紧/亮,负=松/柔) */
    double Rd;   /* 声门Rd缩放(原Mt改名) [-100,100]：纯 Rd ×2^(-(v/100)*2)，正=紧 */
    double Mw;   /* 噪声频率constant-Q平滑(原Md改名)：值=等效带数,建议16~24配Mg */
    double Mm;   /* 模型插值(mo语义) [0,100]：0=纯L1 ↔ 100=纯L2,分量级能量匹配混合(仅L2模式下有效) */
    double Ms;   /* 稳定化(mo语义) [0,10]整数：映射为源相位正则(Mv=Ms*10)+参数平滑(MS=Ms*10) */
    double Me;   /* 强制循环(mo语义,同 l flag) */
    double MC;   /* 粗糙感 [0,100]：逐周期随机幅度调制(roar) */
    double MG;   /* 咆哮growl [0,100]：f0/2 次谐波幅度调制 */
    double MD;   /* 失真(mo语义:更快的growl) [0,100]：~75Hz 快速幅度颤动 */
    double Mp;   /* 音高随机扰动 [0,100]：带限随机抖动加入音高曲线(全模式生效) */
    double u;    /* 直出(mo语义)：跳过一切修改,原区段直接输出 */
    double M;    /* M1=麦乐迪式调音内核(默认smooth:大窗+周期渐变) */
    double HF;   /* HF(toggle)：神经声码器路线——导出编辑后参数CSV+调用外挂渲染钩子(logmel+NSF-HiFiGAN) */
    double Mf;   /* 共振峰调谐 [0,100]：逐帧估F1→向最近谐波k*f0搬移(高音自动有芯,soprano resonance tuning) */
    double CR;   /* CR(toggle)：麦乐迪内核切 crisp 模式(小窗+吸附,瞬态型) */
    double Mb;   /* 元音气声(仅有声帧) */
    double Ab;   /* 全帧气声(原Mb) */
    double NA;   /* 鼻音度 ±100:500Hz反共振凹陷+250Hz鼻峰,正=鼻音 */
    double DN;   /* 智能降噪栈 [0,100]:≥1 启用。第一层=f0轨迹人声先验修复(八度/野点/漏检/误检) */
    double RG;   /* 自动混声 ±100:按音高相对换声点(F4)渐变声区。正=高音自动转头声,负=高音保持胸声(belt) */   /* 气声 [-100,100] → 调 NM/谐波比 */
    double b;    /* 清辅音增减 [-20,100]：仅无声帧噪声能量 gain=1+0.05*b, mo-compatible */
    double bh;   /* 辅音区谐波响度 [-20,100]：仅辅音固定区有声帧，缩放 VTMAGN, gain=1+0.05*bh */
    double ME;   /* 共振峰强调 [-100,100]：VTMAGN unsharp mask，正=峰更锐(共振峰强调)、负=压平 */
    double Mr;   /* 歌手共振峰 ~3kHz [-100,100]：高斯 dB bump，正=加亮穿透、负=削 */
    double Mo;   /* 开口度 [-100,100]：F1 区局部频率 warp，正=更"开"(F1上移)、负=更"闭"(F1下移) */
    double A;    /* 振幅-音高联动 [-100,100]：有声帧幅度随音高偏离基线(颤音)而起伏，增强颤音 */
    double Mn;   /* 噪声平滑/去颗粒 [-100,100]，默认0=适度平滑(k=0.5)；负=更颗粒,正=更平滑 */
    double MH;   /* 高频滚降强度缩放 [-100,100]，默认0；负=更亮(关滚降),正=更暖 */
    double MS;   /* 参数时间平滑/硬度 [0,100]，默认0=不平滑；越大越平滑越"硬"(对标mo规整化) */
    double Mz;   /* 降噪(noise reduction) [0,100]，默认0=关；谱减扣背景底噪 */
    double Mk;   /* 子谐波抑制(suppress-subharmonics) [0,100]，默认0=关；仅对有声帧压噪声谱→去痰 */
    double Ma;   /* 抗失真(anti-distortion) [0,100]，默认0=关；VTMAGN 频域平滑(低轻高重)抹高频伪峰 */
    double Mv;   /* 源相位正则化/硬度 [0,100]，默认0=关；VSPHSE 向最小相位衰减→去糊变硬变稳 */
    double Mx;   /* 实验/诊断：Mx1=只合成谐波(静音噪声)、Mx2=只合成噪声(静音谐波)，判断糊源 */
    double Mc;   /* 噪声相干化 [0,100]：对噪声分量梳状滤波(周期=基音)→逐周期重复=mo电流音相干噪声 */
    double Mg;   /* 噪声去颗粒 [0,100]：源侧对噪声模型(psd/PSDRES/edc/eenv)沿时间平滑→去颗粒/沙哑 */
    double Md;   /* 干燥度(mo语义) [-100,100]：高频(>5kHz)气声调制,正=更多高频气纹理,负=更干 */
    double Mq;   /* 始终生效的高次谐波滚降 [0,100]：>3kHz 按 -(Mq/100)*10 dB/oct 压，去高频沙/刺(对标mo顶部滚降) */
    double SK;   /* 极性反相(toggle)：整段输出 ×-1，对齐 mo 的声门脉冲极性约定 */
    double LD;   /* 降维(low-dimensional) [0,100]：VTMAGN 倒谱低阶化，退回类 LLSM1 低阶包络。值=平滑度 */
    double PB;   /* PbP 合成(toggle)：逐脉冲合成，脉冲由 LF 声门模型(Rd)⊛声道滤波直接生成(类WORLD)，非min-phase */
    double L;    /* LLSM1 后端(toggle，L1 启用)：整段绕过 LLSM2，走 libllsm v1 分析+合成(mo 同底座) */
    double V;    /* 混合：V1=L2谐波+L1干净噪声、V2=L1谐波+L2噪声。跨库拼接谐波/噪声分量,按能量匹配 */
    double P;    /* 峰值归一化 [0,100]，默认100(完全归一,2026-07用户定)；把峰值往0.5拉，P0=关 */
    double p;    /* 最终峰值归一 [-1,6]dB，可选；越大越安静 */
    double BX;   /* 频宽扩展 [0,100]:自动检测编解码截止悬崖(如金坷垃12.7kHz),噪声psd沿真实谱斜率外推补空气带。值=补带电平,无悬崖自动不动作 */
    double TM;   /* 音色对齐 [0,100]:同文件夹好样本建"参考音色档案",把当前音的染色(发闷/编码染色)拉回本人正常音色。值=校正强度 */
    double Mj;   /* 转音平滑 [1,6]:快速pitchbend下加密合成帧(thop/F),修中高谐波断点/沙哑。值=加密上限F,自动按音高斜率取1~F。≥1启用 */
    double Mi;   /* 神经蒸馏预设 [0,100]:把 Seed-VC 修复烘成逐帧64带参数差量(<wav>.nfx),渲染时按强度叠加到VTMAGN+psd。值=强度,无预设自动不动作 */
    /* …按需扩充 */
    int has_g, has_t, has_Mt, has_Mb, has_b, has_bh, has_ME, has_Mr, has_Mo, has_A, has_Mn, has_MH, has_MS, has_Mz, has_Mk, has_Ma, has_Mv, has_Mx, has_Mc, has_Mq, has_Mg, has_Md, has_SK, has_LD, has_PB, has_L, has_V, has_P, has_p;
    int has_Rd, has_Mw, has_Mm, has_Ms, has_Me, has_MC, has_MG, has_MD, has_Mp, has_u, has_M, has_CR, has_Mf, has_HF, has_Ab, has_NA, has_RG, has_DN, has_BX, has_TM, has_Mj, has_Mi;
    int has_e;   /* e flag：强制拉伸(force stretch)，优先级最高；长音也用拉伸不循环 */
    int has_l;   /* l flag：循环(loop)模式，延长时重复真实帧（ping-pong） */
} Flags;

static void parse_flags(const char* s, Flags* f) {
    memset(f, 0, sizeof(*f));
    if (!s) return;
    int i = 0, len = (int)strlen(s);
    while (i < len) {
        if (s[i] == '/') { ++i; continue; }
        /* 取 1~2 字母键 */
        char key[3] = {0}; int k = 0;
        if (isalpha((unsigned char)s[i])) {
            key[k++] = s[i++];
            /* 两字母 flag：M 前缀(全部 M*)，或 b 后接 h(=bh)，或 S 后接 K(=SK) */
            if (k == 1 && i < len && isalpha((unsigned char)s[i]) &&
                (key[0]=='M' || (key[0]=='b' && s[i]=='h') || (key[0]=='S' && s[i]=='K') || (key[0]=='L' && s[i]=='D') || (key[0]=='P' && s[i]=='B') || (key[0]=='R' && s[i]=='d') || (key[0]=='A' && s[i]=='b') || (key[0]=='N' && s[i]=='A') || (key[0]=='R' && s[i]=='G') || (key[0]=='D' && s[i]=='N') || (key[0]=='J' && s[i]=='Z') || (key[0]=='B' && s[i]=='X') || (key[0]=='T' && s[i]=='M') || (key[0]=='Q' && s[i]=='X') || (key[0]=='C' && s[i]=='R') || (key[0]=='H' && s[i]=='F')))
                key[k++] = s[i++];
        } else { ++i; continue; }
        /* 取可选数值（可带符号） */
        int j = i, neg = 0;
        if (j < len && (s[j]=='+'||s[j]=='-')) ++j;
        int has_num = 0;
        while (j < len && isdigit((unsigned char)s[j])) { ++j; has_num = 1; }
        double val = has_num ? atof(s + i) : 0.0;
        (void)neg;
        i = j;
        /* 分派(first-wins)：同名 flag 先出现者优先。UTAU 拼接顺序=音符 flag 在前、
           工程全局在后 → 音符设定覆盖全局(别名共享存储,QX/ME、JZ/DN 等同样先到先得)。 */
        if (!strcmp(key,"g"))  { if (!f->has_g) { f->g = val; f->has_g = 1; } }
        else if (!strcmp(key,"t"))  { if (!f->has_t) { f->t = val; f->has_t = 1; } }
        else if (!strcmp(key,"Mt")) { if (!f->has_Mt) { f->Mt = val; f->has_Mt = 1; } }
        else if (!strcmp(key,"Mb")) { if (!f->has_Mb) { f->Mb = val; f->has_Mb = 1; } }
        else if (!strcmp(key,"Ab")) { if (!f->has_Ab) { f->Ab = val; f->has_Ab = 1; } }   /* 全帧气声(原Mb) */
        else if (!strcmp(key,"NA")) { if (!f->has_NA) { f->NA = val; f->has_NA = 1; } }   /* 鼻音度 */
        else if (!strcmp(key,"RG")) { if (!f->has_RG) { f->RG = val; f->has_RG = 1; } }   /* 自动混声 */
        else if (!strcmp(key,"JZ")) { if (!f->has_DN) { f->DN = val; f->has_DN = 1; } }   /* 智能降噪栈(2026-07-06 改名:原 DN。原版编辑器把 N 当无值 flag 回写会吃掉数字) */
        else if (!strcmp(key,"DN")) { if (!f->has_DN) { f->DN = val; f->has_DN = 1; } }   /* 旧名兼容:=JZ */
        else if (!strcmp(key,"Mu")) { if (!f->has_NA) { f->NA = val; f->has_NA = 1; } }   /* 鼻音度(2026-07-06 改名:原 NA,同 N 冲突;M 前缀安全区) */
        else if (!strcmp(key,"BX")) { if (!f->has_BX) { f->BX = val; f->has_BX = 1; } }   /* 频宽扩展(补编解码截止) */
        else if (!strcmp(key,"TM")) { if (!f->has_TM) { f->TM = val; f->has_TM = 1; } }   /* 音色对齐(bank 档案修复) */
        else if (!strcmp(key,"Mj")) { if (!f->has_Mj) { f->Mj = val; f->has_Mj = 1; } }   /* 转音平滑(合成帧加密) */
        else if (!strcmp(key,"Mi")) { if (!f->has_Mi) { f->Mi = val; f->has_Mi = 1; } }   /* 神经蒸馏预设 */
        else if (!strcmp(key,"bh")) { if (!f->has_bh) { f->bh = val; f->has_bh = 1; } } /* 辅音区谐波响度 */
        else if (!strcmp(key,"b"))  { if (!f->has_b) { f->b = val; f->has_b = 1; } }   /* 清辅音增减 */
        else if (!strcmp(key,"QX")) { if (!f->has_ME) { f->ME = val; f->has_ME = 1; } }   /* 共振峰锐化(2026-07-12 改名:原 ME。原版编辑器把 E 当已知 flag 拆成 M/E 丢编辑) */
        else if (!strcmp(key,"ME")) { if (!f->has_ME) { f->ME = val; f->has_ME = 1; } }   /* 旧名兼容:=QX(仅在不吃字符的环境可用,如 CLI/OpenUtau) */
        else if (!strcmp(key,"Mr")) { if (!f->has_Mr) { f->Mr = val; f->has_Mr = 1; } }
        else if (!strcmp(key,"Mo")) { if (!f->has_Mo) { f->Mo = val; f->has_Mo = 1; } }
        else if (!strcmp(key,"Mn")) { if (!f->has_Mn) { f->Mn = val; f->has_Mn = 1; } }
        else if (!strcmp(key,"MH")) { if (!f->has_MH) { f->MH = val; f->has_MH = 1; } }
        else if (!strcmp(key,"MS")) { if (!f->has_MS) { f->MS = val; f->has_MS = 1; } }
        else if (!strcmp(key,"Mz")) { if (!f->has_Mz) { f->Mz = val; f->has_Mz = 1; } }
        else if (!strcmp(key,"Mk")) { if (!f->has_Mk) { f->Mk = val; f->has_Mk = 1; } }
        else if (!strcmp(key,"Ma")) { if (!f->has_Ma) { f->Ma = val; f->has_Ma = 1; } }
        else if (!strcmp(key,"Mv")) { if (!f->has_Mv) { f->Mv = val; f->has_Mv = 1; } }
        else if (!strcmp(key,"Mx")) { if (!f->has_Mx) { f->Mx = val; f->has_Mx = 1; } }   /* 实验:只谐波/只噪声 */
        else if (!strcmp(key,"Mc")) { if (!f->has_Mc) { f->Mc = val; f->has_Mc = 1; } }   /* 噪声相干化 */
        else if (!strcmp(key,"Mg")) { if (!f->has_Mg) { f->Mg = val; f->has_Mg = 1; } }   /* 噪声去颗粒(时间平滑) */
        else if (!strcmp(key,"Md")) { if (!f->has_Md) { f->Md = val; f->has_Md = 1; } }   /* 干燥度(mo) */
        else if (!strcmp(key,"Rd")) { if (!f->has_Rd) { f->Rd = val; f->has_Rd = 1; } }   /* 声门Rd缩放(原Mt) */
        else if (!strcmp(key,"Mw")) { if (!f->has_Mw) { f->Mw = val; f->has_Mw = 1; } }   /* 噪声频率平滑(原Md) */
        else if (!strcmp(key,"Mm")) { if (!f->has_Mm) { f->Mm = val; f->has_Mm = 1; } }   /* 模型插值 L1<->L2 */
        else if (!strcmp(key,"Mf")) { if (!f->has_Mf) { f->Mf = val; f->has_Mf = 1; } }   /* 共振峰调谐 */
        else if (!strcmp(key,"Ms")) { if (!f->has_Ms) { f->Ms = val; f->has_Ms = 1; } }   /* 稳定化(mo) */
        else if (!strcmp(key,"Me")) { if (!f->has_Me) { f->Me = val; f->has_Me = 1; } }   /* 强制循环(mo) */
        else if (!strcmp(key,"MC")) { if (!f->has_MC) { f->MC = val; f->has_MC = 1; } }   /* 粗糙感 */
        else if (!strcmp(key,"MG")) { if (!f->has_MG) { f->MG = val; f->has_MG = 1; } }   /* growl */
        else if (!strcmp(key,"MD")) { if (!f->has_MD) { f->MD = val; f->has_MD = 1; } }   /* 失真(快growl) */
        else if (!strcmp(key,"Mp")) { if (!f->has_Mp) { f->Mp = val; f->has_Mp = 1; } }   /* 音高随机扰动 */
        else if (!strcmp(key,"u"))  { if (!f->has_u) { f->u = val; f->has_u = 1; } }   /* 直出 */
        else if (!strcmp(key,"CR")) { if (!f->has_CR) { f->CR = val; f->has_CR = 1; } }   /* 麦乐迪内核 crisp 开关 */
        else if (!strcmp(key,"HF")) { if (!f->has_HF) { f->HF = val; f->has_HF = 1; } }   /* 神经声码器路线 */
        else if (!strcmp(key,"M"))  { if (!f->has_M) { f->M = val; f->has_M = 1; } }   /* M1=麦乐迪式调音(默认smooth) */
        else if (!strcmp(key,"Mq")) { if (!f->has_Mq) { f->Mq = val; f->has_Mq = 1; } }   /* 始终高频滚降 */
        else if (!strcmp(key,"SK")) { if (!f->has_SK) { f->SK = val; f->has_SK = 1; } }   /* 极性反相 */
        else if (!strcmp(key,"LD")) { if (!f->has_LD) { f->LD = val; f->has_LD = 1; } }   /* 降维:倒谱低阶化 */
        else if (!strcmp(key,"V"))  { if (!f->has_V) { f->V = val; f->has_V = 1; } }   /* 混合:L2谐波+L1噪声(V1启用) */
        else if (!strcmp(key,"K"))  { if (!f->has_L) { f->L = val; f->has_L = 1; } }   /* 内核选择(2026-07-06 改名:K1=LLSM1 K2=LLSM2。原 L 撞 resampler 系引擎 flag) */
        else if (!strcmp(key,"L"))  { if (!f->has_L) { f->L = val; f->has_L = 1; } }   /* 旧名兼容:=K(L1/L2) */
        else if (!strcmp(key,"PB")) { if (!f->has_PB) { f->PB = val; f->has_PB = 1; } }   /* PbP 逐脉冲合成路径 */
        else if (!strcmp(key,"P"))  { if (!f->has_P) { f->P = val; f->has_P = 1; } }   /* 峰值归一化 */
        else if (!strcmp(key,"p"))  { if (!f->has_p) { f->p = val; f->has_p = 1; } }   /* 最终峰值 dB */
        else if (!strcmp(key,"A"))  { if (!f->has_A) { f->A = val; f->has_A = 1; } }   /* 振幅-音高联动 */
        else if (!strcmp(key,"e"))  { f->has_e = 1; }   /* 强制拉伸，无数值 */
        else if (!strcmp(key,"l"))  { f->has_l = 1; }   /* 循环，无数值 */
        /* 未知 flag 忽略 */
    }
}

/* ===========================================================================
   [OK] 第 5 部分：时间映射 build_remap（移植自 timestretch.py，含 e 循环模式）
   --------------------------------------------------------------------------
   返回 n_out 个"源帧坐标"（浮点）。thop_ms = 1000*hop/fs。
   辅音区：按 velocity 等速映射（不拉伸）。
   母音区两种延长策略：
     - 默认 / e(force stretch)：按 ratio=vow_len_out/vow_len_src 拉伸/压缩（纯插值）。
                      e = 强制拉伸（优先级最高，覆盖 l），与 moresampler 的 e 同义；长音也拉伸。
     - l flag(loop) ：延长时母音以【自然速率 ratio=1】播放，超界 ping-pong 反射折返，
                      重复【真实帧】（配合主循环 PSDRES 抖动去相关）。仅"需延长且母音段够长"时生效。
   说明：实测循环常为负优化，故默认拉伸；循环改到 l flag，e 留给最常用的强制拉伸（对齐 mo 习惯）。
   =========================================================================== */
static double reflect_into(double t, double lo, double hi) {
    if (hi <= lo) return lo;
    double period = 2.0 * (hi - lo);
    double x = fmod(t - lo, period);
    if (x < 0) x += period;
    if (x > (hi - lo)) x = period - x;
    return lo + x;
}
/* 调用方 free 返回数组；*n_out 写出长度。loop_mode=1 启用母音循环模式（l flag）。
   thop_out_ms=输出帧步进(Mj 加密时=细 hop)、thop_src_ms=源帧粒度(=真实分析 hop)。
   两者分离:输出可加密采样,而 src[] 索引仍按源帧(粗 hop)计——Mj 关闭时两者相等,行为不变。 */
static double* build_remap(int n_in, double thop_out_ms, double thop_src_ms,
                           const UtauArgs* u, int loop_mode, int* n_out) {
    double total_ms = n_in * thop_src_ms;
    double vel = pow(2.0, 1.0 - u->velocity / 100.0);
    double end_ms = (u->cutoff >= 0) ? (total_ms - u->cutoff) : (u->offset - u->cutoff);
    if (end_ms < u->offset + thop_src_ms) end_ms = u->offset + thop_src_ms;
    if (end_ms > total_ms)                end_ms = total_ms;

    double con_end_ms = u->offset + u->consonant;
    if (con_end_ms > end_ms) con_end_ms = end_ms;
    double con_out_ms = u->consonant * vel;
    if (con_out_ms > u->length) con_out_ms = u->length;

    double vow_len_src = end_ms - con_end_ms; if (vow_len_src < 1e-6) vow_len_src = 1e-6;
    double vow_len_out = u->length - con_out_ms; if (vow_len_out < 0) vow_len_out = 0;
    double ratio = vow_len_out / vow_len_src;   /* >1 = 需延长 */

    /* loop 模式：仅在"需延长(vow_len_out>vow_len_src)且母音段够长(>20ms 才循环有意义)"时启用；
       此时母音以自然速率播放(eff_ratio=1)、靠 reflect 折返循环真实帧。 */
    int do_loop = loop_mode && (vow_len_out > vow_len_src) && (vow_len_src > 20.0);
    double eff_ratio = do_loop ? 1.0 : ratio;

    int N = (int)(u->length / thop_out_ms + 0.5); if (N < 1) N = 1;
    double* src = (double*)malloc(sizeof(double) * N);
    int j;
    for (j = 0; j < N; ++j) {
        double t_out = j * thop_out_ms, t_src;
        if (t_out < con_out_ms) {
            t_src = u->offset + t_out / (vel > 1e-6 ? vel : 1e-6);
        } else {
            t_src = con_end_ms + (t_out - con_out_ms) / (eff_ratio > 1e-9 ? eff_ratio : 1e-9);
            /* 超界一律 ping-pong 折返（loop 模式靠它循环；stretch 模式作越界安全兜底） */
            if (t_src > end_ms) t_src = reflect_into(t_src, con_end_ms, end_ms);
        }
        if (t_src < u->offset) t_src = u->offset;
        if (t_src > end_ms)    t_src = end_ms;
        src[j] = t_src / thop_src_ms;   /* 源帧索引=按真实分析 hop */
    }
    *n_out = N;
    return src;
}

/* ===========================================================================
   [OK] 第 6 部分：逐帧 L1 插值（自 demo-stretch.c 抄录，cos_2/sin_2/log_2 换标准库）
   =========================================================================== */
static double mag2db(double x) { return 20.0 * log10(x); }

/* 圆周插值（两个弧度值） */
static FP_TYPE linterpc(FP_TYPE a, FP_TYPE b, FP_TYPE ratio) {
    FP_TYPE ax = cos(a), ay = sin(a), bx = cos(b), by = sin(b);
    FP_TYPE cx = ax + (bx - ax) * ratio;
    FP_TYPE cy = ay + (by - ay) * ratio;
    return atan2(cy, cx);
}
static FP_TYPE linterp_(FP_TYPE a, FP_TYPE b, FP_TYPE r) { return a + (b - a) * r; }

static void interp_nmframe(llsm_nmframe* dst, llsm_nmframe* src, FP_TYPE ratio) {
    int i, b;
    for (i = 0; i < dst->npsd; ++i)
        dst->psd[i] = linterp_(dst->psd[i], src->psd[i], ratio);
    for (b = 0; b < dst->nchannel; ++b) {
        llsm_hmframe* se = src->eenv[b];
        llsm_hmframe* de = dst->eenv[b];
        dst->edc[b] = linterp_(dst->edc[b], src->edc[b], ratio);
        int bmin = se->nhar < de->nhar ? se->nhar : de->nhar;
        int bmax = se->nhar > de->nhar ? se->nhar : de->nhar;
        if (de->nhar < bmax) {
            de->ampl = (FP_TYPE*)realloc(de->ampl, sizeof(FP_TYPE) * bmax);
            de->phse = (FP_TYPE*)realloc(de->phse, sizeof(FP_TYPE) * bmax);
        }
        for (i = 0; i < bmin; ++i) {
            de->ampl[i] = linterp_(de->ampl[i], se->ampl[i], ratio);
            de->phse[i] = linterpc(de->phse[i], se->phse[i], ratio);
        }
        if (bmax == se->nhar)
            for (i = bmin; i < bmax; ++i) { de->ampl[i] = se->ampl[i]; de->phse[i] = se->phse[i]; }
        de->nhar = bmax;
    }
}

/* dst <- interp(dst, src, ratio)  （完全对照 demo-stretch.c::interp_llsm_frame） */
static void interp_llsm_frame(llsm_container* dst, llsm_container* src, FP_TYPE ratio) {
#define EPS 1e-8
    FP_TYPE dst_f0 = *((FP_TYPE*)llsm_container_get(dst, LLSM_FRAME_F0));
    FP_TYPE src_f0 = *((FP_TYPE*)llsm_container_get(src, LLSM_FRAME_F0));
    llsm_nmframe* dst_nm = llsm_container_get(dst, LLSM_FRAME_NM);
    llsm_nmframe* src_nm = llsm_container_get(src, LLSM_FRAME_NM);
    FP_TYPE* src_rd = llsm_container_get(src, LLSM_FRAME_RD);
    FP_TYPE* dst_rd = llsm_container_get(dst, LLSM_FRAME_RD);
    FP_TYPE* dst_vsphse = llsm_container_get(dst, LLSM_FRAME_VSPHSE);
    FP_TYPE* src_vsphse = llsm_container_get(src, LLSM_FRAME_VSPHSE);
    FP_TYPE* dst_vtmagn = llsm_container_get(dst, LLSM_FRAME_VTMAGN);
    FP_TYPE* src_vtmagn = llsm_container_get(src, LLSM_FRAME_VTMAGN);

    llsm_container* voiced = (dst_f0 <= 0 && src_f0 <= 0) ? NULL : (src_f0 > 0 ? src : dst);
    int bothvoiced = dst_f0 > 0 && src_f0 > 0;
    int dstnhar = dst_vsphse == NULL ? 0 : llsm_fparray_length(dst_vsphse);
    int srcnhar = src_vsphse == NULL ? 0 : llsm_fparray_length(src_vsphse);
    int maxnhar = dstnhar > srcnhar ? dstnhar : srcnhar;
    int minnhar = dstnhar < srcnhar ? dstnhar : srcnhar;
    int i;

    if (!bothvoiced && voiced == src) {
        llsm_container_attach(dst, LLSM_FRAME_F0, llsm_create_fp(src_f0), llsm_delete_fp, llsm_copy_fp);
        llsm_container_attach(dst, LLSM_FRAME_RD, llsm_create_fp(*src_rd), llsm_delete_fp, llsm_copy_fp);
    } else if (voiced == NULL) {
        llsm_container_attach(dst, LLSM_FRAME_F0, llsm_create_fp(0), llsm_delete_fp, llsm_copy_fp);
        llsm_container_attach(dst, LLSM_FRAME_RD, llsm_create_fp(1.0), llsm_delete_fp, llsm_copy_fp);
    }
    int nspec = dst_vtmagn ? llsm_fparray_length(dst_vtmagn)
                           : (src_vtmagn ? llsm_fparray_length(src_vtmagn) : 0);

    if (bothvoiced) {
        llsm_container_attach(dst, LLSM_FRAME_F0, llsm_create_fp(linterp_(dst_f0, src_f0, ratio)),
            llsm_delete_fp, llsm_copy_fp);
        llsm_container_attach(dst, LLSM_FRAME_RD, llsm_create_fp(linterp_(*dst_rd, *src_rd, ratio)),
            llsm_delete_fp, llsm_copy_fp);
        FP_TYPE* vsphse = llsm_create_fparray(maxnhar);
        FP_TYPE* vtmagn = llsm_create_fparray(nspec);
        for (i = 0; i < minnhar; ++i) vsphse[i] = linterpc(dst_vsphse[i], src_vsphse[i], ratio);
        for (i = 0; i < nspec;   ++i) vtmagn[i] = linterp_(dst_vtmagn[i], src_vtmagn[i], ratio);
        if (dstnhar < srcnhar) for (i = minnhar; i < maxnhar; ++i) vsphse[i] = src_vsphse[i];
        llsm_container_attach(dst, LLSM_FRAME_VSPHSE, vsphse, llsm_delete_fparray, llsm_copy_fparray);
        llsm_container_attach(dst, LLSM_FRAME_VTMAGN, vtmagn, llsm_delete_fparray, llsm_copy_fparray);
        dst_vtmagn = vtmagn;
    } else if (voiced == src) {
        FP_TYPE* vsphse = llsm_copy_fparray(src_vsphse);
        FP_TYPE* vtmagn = llsm_copy_fparray(src_vtmagn);
        llsm_container_attach(dst, LLSM_FRAME_VSPHSE, vsphse, llsm_delete_fparray, llsm_copy_fparray);
        llsm_container_attach(dst, LLSM_FRAME_VTMAGN, vtmagn, llsm_delete_fparray, llsm_copy_fparray);
        FP_TYPE fade = mag2db(EPS > ratio ? EPS : ratio);
        for (i = 0; i < nspec; ++i) vtmagn[i] += fade;
        dst_vtmagn = vtmagn;
    } else {
        FP_TYPE fade = mag2db(EPS > (1.0 - ratio) ? EPS : (1.0 - ratio));
        for (i = 0; i < nspec; ++i) dst_vtmagn[i] += fade;
    }
    for (i = 0; i < nspec; ++i)
        if (dst_vtmagn[i] < -80) dst_vtmagn[i] = -80;

    interp_nmframe(dst_nm, src_nm, ratio);
#undef EPS
}

/* ===========================================================================
   第 7 部分：flag 编辑（作用在单个 L1 帧上）
   --------------------------------------------------------------------------
   在 tolayer0 之前、F0 覆盖之后调用，直接修改帧的 L1 参数（=修改"中间表示"），
   原始音频/分析结果不变。VTMAGN 为 dB、线性频率轴(0..fnyq, nspec=nfft/2+1)。
   数值映射为保守初值，[CALIB] 待真实音频听感校准。
     [OK]  g  : VTMAGN 频率轴 warp（共振峰整体平移）
     [OK]  Mo : VTMAGN F1 区局部 warp（开口度，仅低频幂律弯曲）
     [OK]  ME : VTMAGN unsharp mask（共振峰强调，峰锐化/压平）
     [OK]  Mr : VTMAGN ~3kHz 高斯 dB bump（歌手共振峰）
     [OK]  Mt : LLSM_FRAME_RD（LF 声门张力，低 Rd=紧）
     [OK]  Mb : LLSM_FRAME_NM 噪声 psd(dB) 增减（气声）
   fnyq = 奈奎斯特频率(Hz)，VTMAGN 第 i bin 对应 freq = i/(n-1)*fnyq。
   =========================================================================== */
static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ===========================================================================
   L3：Melodyne式合成内核(Local Sound Synthesis / TD-PSOLA 族)。
   原理(Neubäcker)：把波形按声门周期切成"局部音色快照"，合成=在映射后的源位置取周期
   grain(2周期Hann窗)、按【目标音高的周期间距】重新排列。不重采样周期本身→共振峰保持;
   无频域模型→零模型误差、原始质感 100% 保留。无声段=固定 grain 粒化直通。
   =========================================================================== */
static void apply_growl_fx(FP_TYPE* y, int n, int fs, const PitchCurve* pc, const Flags* fl);

/* M1 用时域双二阶峰形EQ(RBJ peaking)。fc中心 gdb增益 Q带宽。 */
static void l3_biquad_peak(FP_TYPE* y, int n, int fs, double fc, double gdb, double Q) {
    double A = pow(10.0, gdb / 40.0), w = 2.0 * M_PI * fc / fs, al = sin(w) / (2.0 * Q);
    double b0 = 1 + al * A, b1 = -2 * cos(w), b2 = 1 - al * A;
    double a0 = 1 + al / A, a1 = -2 * cos(w), a2 = 1 - al / A;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0; int i;
    if (fabs(gdb) < 1e-6) return;
    for (i = 0; i < n; ++i) {
        double xx = y[i], yy = (b0 * xx + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        x2 = x1; x1 = xx; y2 = y1; y1 = yy; y[i] = (FP_TYPE)yy;
    }
}

static int l3_render(const UtauArgs* u, const PitchCurve* pc, const Flags* fl) {
    int fs = 0, nbit = 0, nx = 0, nfrm = 0, nhop = 256, i;
    FP_TYPE* x = wavread((char*)u->in_file, &fs, &nbit, &nx);
    if (!x || nx <= 0) { fprintf(stderr, "[L3] read fail\n"); return 1; }
    wcs_to_44100(&x, &nx, &fs, "M1");
    FP_TYPE* f0 = l1f0_cache_get(u->in_file, fs, nhop, &nfrm);
    if (!f0) {
        pyin_config pp = pyin_init(nhop);
        pp.fmin = 50; pp.fmax = 900; pp.trange = 24;
        pp.nf = (int)ceil(fs * 0.025); pp.w = pp.nf / 4; pp.bias = 10;
        f0 = pyin_analyze(pp, x, nx, fs, &nfrm);
        if (f0 && nfrm > 0) l1f0_cache_put(u->in_file, fs, nhop, f0, nfrm);
    }
    if (fl->has_DN && fl->DN > 0 && f0 && nfrm > 4) { int fx = f0_sanitize(f0, nfrm);
        if (fx > 0) fprintf(stderr, "[M1] f0fix: %d corrections\n", fx); }
    if (!f0 || nfrm <= 0) { fprintf(stderr, "[L3] pyin fail\n"); free(x); return 1; }

    /* 1) 源周期标记：按 f0 积分推进,并对齐局部波形峰(相位一致) */
    int cap = nx / 32 + 16, nmk = 0;
    int* mk = (int*)malloc(sizeof(int) * cap);
    double t = 0;
    while (t < nx - 1) {
        int fr = (int)(t / nhop); if (fr > nfrm - 1) fr = nfrm - 1;
        double f = f0[fr];
        if (f > 20.0) {
            int T = (int)(fs / f), c = (int)t, best = c, lo = c - T/4, hi = c + T/4;
            double bv = -1;
            if (lo < 0) lo = 0; if (hi > nx - 1) hi = nx - 1;
            for (i = lo; i <= hi; ++i) if (fabs((double)x[i]) > bv) { bv = fabs((double)x[i]); best = i; }
            if (nmk > 0) {                       /* 间距守卫:峰对齐跳错峰(偏离周期>40%)则按积分位置落 */
                int gap = best - mk[nmk-1];
                if (gap < (int)(T * 0.6) || gap > (int)(T * 1.4)) best = c;
            }
            if (nmk >= cap) { cap *= 2; mk = (int*)realloc(mk, sizeof(int) * cap); }
            mk[nmk++] = best;
            t += fs / f;
        } else t += nhop;
    }

    /* 2) 时间映射参数(与 L1/L2 一致:oto 区段+辅音保留) */
    double total_ms = 1000.0 * nx / fs;
    double vel = pow(2.0, 1.0 - u->velocity / 100.0);
    double end_ms = (u->cutoff >= 0) ? (total_ms - u->cutoff) : (u->offset - u->cutoff);
    if (end_ms > total_ms) end_ms = total_ms;
    if (end_ms < u->offset + 5) end_ms = u->offset + 5;
    double con_end = u->offset + u->consonant; if (con_end > end_ms) con_end = end_ms;
    double con_out = u->consonant * vel; if (con_out > u->length) con_out = u->length;
    double vow_src = end_ms - con_end; if (vow_src < 1e-6) vow_src = 1e-6;
    double vow_out = u->length - con_out; if (vow_out < 0) vow_out = 0;
    double ratio = vow_out / vow_src;

    /* 3) 周期同步 OLA 合成 */
    int nout = (int)(u->length / 1000.0 * fs); if (nout < 16) nout = 16;
    FP_TYPE* y  = (FP_TYPE*)calloc(nout, sizeof(FP_TYPE));
    FP_TYPE* ws = (FP_TYPE*)calloc(nout, sizeof(FP_TYPE));
    double s = 0; int hint = 0;
    while (s < nout) {
        double t_out = 1000.0 * s / fs;
        double t_src;
        if (t_out < con_out) t_src = u->offset + t_out / (vel > 1e-6 ? vel : 1e-6);
        else if ((fl->has_l || fl->has_Me) && !fl->has_e && ratio > 1.0) {   /* e 优先:强制拉伸压过循环(与 L2 一致) */
            /* 循环模式:元音自然速率 ping-pong 折返读真实周期(不稀释,长音不静止) */
            double vlen = end_ms - con_end, tt = t_out - con_out, per = 2.0 * vlen;
            double ph = fmod(tt, per); if (ph < 0) ph += per;
            t_src = con_end + (ph <= vlen ? ph : per - ph);
        } else t_src = con_end + (t_out - con_out) / (ratio > 1e-9 ? ratio : 1e-9);
        if (t_src < u->offset) t_src = u->offset;
        if (t_src > end_ms) t_src = end_ms;
        int c_src = (int)(t_src / 1000.0 * fs); if (c_src > nx - 1) c_src = nx - 1;
        int fr = c_src / nhop; if (fr > nfrm - 1) fr = nfrm - 1;
        double ft = pitchcurve_f0(pc, t_out / 1000.0);
        double f0v_ = f0[fr];                          /* 邻帧救回:单帧漏检不打断元音 */
        if (f0v_ <= 20.0) {
            if (fr > 0 && f0[fr-1] > 20.0) f0v_ = f0[fr-1];
            else if (fr < nfrm - 1 && f0[fr+1] > 20.0) f0v_ = f0[fr+1];
        }
        int pure_uv = (f0v_ <= 20.0)
            && (fr < 3 || f0[fr-3] <= 20.0) && (fr > nfrm - 4 || f0[fr+3] <= 20.0);
        int voiced = (f0v_ > 20.0 && ft > 20.0 && nmk > 1);
        int crisp = fl->has_CR;
        int glen, gc = c_src, g2 = -1; double gfr = 0.0;
        if (voiced) {
            if (crisp) {   /* M2 crisp：小窗(2周期)+就近吸附(瞬态清晰,长音偏躁) */
                while (hint > 0 && mk[hint] > c_src) --hint;   /* 回退:支持折返读取 */
                while (hint < nmk - 1 && mk[hint] < c_src) ++hint;
                if (hint > 0 && c_src - mk[hint-1] < mk[hint] - c_src) --hint;
                gc = mk[hint];
                glen = (int)(2.0 * fs / f0v_);
            } else {       /* M1 smooth：大窗(4周期)+相邻周期【连续渐变】(Melodyne smooth:
                              大处理窗+渐进过渡;不吸附冻结=去电音/去断续) */
                while (hint > 0 && mk[hint] > c_src) --hint;   /* 回退:支持折返读取 */
                while (hint < nmk - 2 && mk[hint + 1] < c_src) ++hint;
                gc = mk[hint]; g2 = mk[hint + 1 < nmk ? hint + 1 : hint];
                /* 保护:两标记间距异常(>1.6周期=跨无声空隙/段边界)时禁用渐变,退回就近吸附,
                   防止把相距很远的不相关内容混在一起(拉伸错乱感的来源)。 */
                if (g2 - gc > (int)(1.6 * fs / f0v_)) {
                    if (c_src - gc > g2 - c_src) gc = g2;
                } else {
                    /* stochastic PSOLA:按位置分数【概率选单周期】(不线性混合——自然素材相邻周期
                       有差异,混合会周期性对消=断续/AM涟漪;单周期保脉冲纯净,概率交替防冻结蜂鸣) */
                    gfr = (g2 > gc) ? (double)(c_src - gc) / (g2 - gc) : 0.0;
                    if (gfr < 0) gfr = 0; if (gfr > 1) gfr = 1;
                    if ((double)rand() / RAND_MAX < gfr) gc = g2;
                }
                g2 = -1; gfr = 0.0;
                /* 窗宽必须=2周期(PSOLA单脉冲隔离,变调相干);smooth 的平滑靠【周期渐变】而非大窗。
                   之前 4 周期大窗:grain 内源间距脉冲 vs grain 间目标间距 打架→失真,已改回。 */
                glen = (int)(2.0 * fs / f0v_);
            }
        } else {
            glen = (int)(fs * 0.023);
            if (pure_uv) gc += (rand() % (int)(fs * 0.010)) - (int)(fs * 0.005);   /* 仅纯辅音区去相关抖动 */
            if (gc < 0) gc = 0; if (gc > nx - 1) gc = nx - 1;
        }
        if (glen < 8) glen = 8;
        /* M1 简单flag：g=共振峰移动(grain内容重采样,周期间距不变=Melodyne音色轴);
           b=清辅音增益(无声grain);bh=辅音区谐波增益(辅音区有声grain)。增益乘在内容上,ws 只累窗→归一后保留。 */
        {
            double rg = (fl->has_g && fl->g != 0.0 && voiced)
                        ? pow(2.0, -clampd(fl->g, -50, 50) * 9.0 / 1200.0) : 1.0;
            double gg = 1.0;
            if (!voiced && fl->has_b) { gg *= 1.0 + 0.05 * clampd(fl->b, -20, 100); if (gg < 0) gg = 0; }
            if (voiced && fl->has_bh && fl->bh != 0.0 && t_out < con_out) {
                double q = 1.0 + 0.05 * clampd(fl->bh, -20, 100); if (q < 1e-4) q = 1e-4; gg *= q; }
            for (i = 0; i < glen; ++i) {                                      /* Hann grain OLA */
                int oi = (int)s + i - glen / 2, off = i - glen / 2;
                double sf = gc + off * rg, s2f = (g2 >= 0) ? g2 + off * rg : -1.0;
                int si = (int)floor(sf); double fa = sf - si, v;
                if (oi < 0 || oi >= nout || si < 0 || si + 1 >= nx) continue;
                v = (1.0 - fa) * x[si] + fa * x[si + 1];
                if (s2f >= 0) { int t2 = (int)floor(s2f); double fb = s2f - t2;
                    if (t2 >= 0 && t2 + 1 < nx)
                        v = (1.0 - gfr) * v + gfr * ((1.0 - fb) * x[t2] + fb * x[t2 + 1]); }
                {
                    double w = 0.5 - 0.5 * cos(2.0 * M_PI * i / (glen - 1));
                    y[oi]  += (FP_TYPE)(v * w * gg);
                    ws[oi] += (FP_TYPE)w;
                }
            }
        }
        s += voiced ? (fs / ft) : (fs * 0.0058);                          /* 目标周期间距/无声hop */
    }
    {   /* ws 2ms 滑动平均后再归一:防除法在覆盖波动/下限截断处产生电平台阶=咔哒点击 */
        int W5 = (int)(fs * 0.002), i5;
        double* cs5 = (double*)malloc(sizeof(double) * (nout + 1));
        cs5[0] = 0;
        for (i5 = 0; i5 < nout; ++i5) cs5[i5 + 1] = cs5[i5] + ws[i5];
        for (i5 = 0; i5 < nout; ++i5) {
            int a5 = i5 - W5, b5 = i5 + W5;
            if (a5 < 0) a5 = 0; if (b5 > nout) b5 = nout;
            {
                double m5 = (cs5[b5] - cs5[a5]) / (b5 > a5 ? (b5 - a5) : 1);
                y[i5] = (FP_TYPE)(y[i5] / (m5 > 0.3 ? m5 : 0.3));
            }
        }
        free(cs5);
    }

    /* 4) M1 简单谱形flag(时域EQ)：Mt=3kHz宽钟形±8dB(张力/亮暗)；Mo=开口度(900Hz±6dB+350Hz反向±4dB≈F1升降) */
    if (fl->has_Mt && fl->Mt != 0.0)
        l3_biquad_peak(y, nout, fs, 3000.0, 8.0 * clampd(fl->Mt, -100, 100) / 100.0, 0.4);
    if (fl->has_Mo && fl->Mo != 0.0) {
        double mo = clampd(fl->Mo, -100, 100) / 100.0;   /* [CALIB 2026-07-09b] →24/16dB:与谱域侧同步,原始4倍 */
        l3_biquad_peak(y, nout, fs, 900.0,  24.0 * mo, 1.0);
        l3_biquad_peak(y, nout, fs, 350.0, -16.0 * mo, 1.0);
    }

    /* 共享后效 + 归一 + 写 */
    apply_growl_fx(y, nout, fs, pc, fl);
    {
        double peak = 1e-9, Pv = fl->has_P ? clampd(fl->P, 0, 100) : 100.0;
        for (i = 0; i < nout; ++i) { double a = fabs((double)y[i]); if (a > peak) peak = a; }
        double vs = pow(0.5 / peak, Pv / 100.0) * (u->volume > 0 ? u->volume : 100.0) / 100.0;
        for (i = 0; i < nout; ++i) { double v = y[i] * vs;
            if (v > 0.99) v = 0.99; if (v < -0.99) v = -0.99; y[i] = (FP_TYPE)v; }
    }
    if (fl->has_SK) for (i = 0; i < nout; ++i) y[i] = -y[i];
    wavwrite(y, nout, fs, 16, (char*)u->out_file);
    fprintf(stderr, "[utau] M1 Melodyne-style (%s): %d marks, %d smp\n",
            fl->has_CR ? "crisp" : "smooth", nmk, nout);
    free(mk); free(y); free(ws); free(f0); free(x);
    return 0;
}

/* mo 扩展特效 MC(粗糙)/MG(growl)/MD(失真=快growl)：时域幅度调制,仅有声段(f0>20)。
   MC=逐声门周期随机增益(shimmer型粗糙)；MG=f0/2 次谐波 AM(低吼)；MD=~75Hz 快速颤动。
   在最终混音 y 上做 → 四种合成模式(L1/L2/V1/V2)全部生效。 */
static void apply_growl_fx(FP_TYPE* y, int n, int fs, const PitchCurve* pc, const Flags* fl) {
    double dC = fl->has_MC ? clampd(fl->MC, 0, 100) / 100.0 * 0.6 : 0.0;
    double dG = fl->has_MG ? clampd(fl->MG, 0, 100) / 100.0 * 0.7 : 0.0;
    double dD = fl->has_MD ? clampd(fl->MD, 0, 100) / 100.0 * 0.6 : 0.0;
    double phG = 0.0, phD = 0.0, phP = 0.0;   /* 相位累积:sub-AM / 快颤 / 周期跟踪 */
    double gC0 = 1.0, gC1 = 1.0;              /* MC 当前/下一周期随机增益 */
    int i;
    if (dC <= 0 && dG <= 0 && dD <= 0) return;
    srand(9876);
    for (i = 0; i < n; ++i) {
        double f0 = pitchcurve_f0(pc, (double)i / fs);
        double g = 1.0;
        if (f0 > 20.0) {
            if (dG > 0) { phG += 2.0 * M_PI * (f0 * 0.5) / fs;
                          g *= 1.0 + dG * sin(phG); }
            if (dD > 0) { phD += 2.0 * M_PI * 75.0 / fs;
                          g *= 1.0 + dD * sin(phD); }
            if (dC > 0) { phP += f0 / fs;
                          if (phP >= 1.0) { phP -= 1.0; gC0 = gC1;
                              gC1 = 1.0 + dC * (2.0 * rand() / RAND_MAX - 1.0); }
                          g *= gC0 + (gC1 - gC0) * phP; }
        } else { phG = phD = 0.0; }
        y[i] = (FP_TYPE)(y[i] * g);
    }
}

static void apply_flags_to_frame(llsm_container* frame, const Flags* fl, double fnyq) {
    if (!frame || !fl) return;
    int i;

    /* --- g：共振峰平移（沿 VTMAGN 线性频率轴重采样）--- */
    if (fl->has_g && fl->g != 0.0) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (vt) {
            int n = llsm_fparray_length(vt);
            /* R = 频率缩放比；g>0 → 共振峰下移(R<1)。[CALIB 2026-07] 常规刻度：
               clamp ±50，每单位≈9cents → g20≈-10%(明显变声)、g50≈-23%(完全变声,满量程) */
            double R = pow(2.0, -clampd(fl->g, -50, 50) * 9.0 / 1200.0);
            FP_TYPE* tmp = (FP_TYPE*)malloc(sizeof(FP_TYPE) * n);
            for (i = 0; i < n; ++i) {
                double si = i / (R > 1e-6 ? R : 1e-6);   /* new[i]=old[i/R] */
                int i0 = (int)floor(si); double a = si - i0;
                if (i0 < 0)         tmp[i] = vt[0];
                else if (i0 >= n-1) tmp[i] = vt[n-1];
                else                tmp[i] = (FP_TYPE)((1.0-a)*vt[i0] + a*vt[i0+1]);
            }
            memcpy(vt, tmp, sizeof(FP_TYPE) * n);
            free(tmp);
        }
    }

    /* --- Mo：开口度（仅 F1 区局部频率 warp）--------------------------------
       发声生理：嘴张得越大 → 第一共振峰 F1 越高。只在枢轴 Fp(1800Hz) 以下做幂律
       弯曲 f_src = (f/Fp)^gamma * Fp，以上原样不动（保 F2/F3）。gamma>1 → F1 上移(更开)。
       与 g(全频段等比平移) 区别：Mo 只动低频 F1，是"同一人张大嘴"而非"换体型"。 */
    if (fl->has_Mo && fl->Mo != 0.0 && fnyq > 0.0) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (vt) {
            int n = llsm_fparray_length(vt);
            double gamma = pow(2.0, (clampd(fl->Mo, -100, 100) / 100.0) * 2.0);   /* [CALIB 2026-07-09b] 0.5→1.0→2.0:两轮用户反馈力度不足,现=原始4倍(满量程γ=4/0.25) */
            int ip = (int)(1800.0 / fnyq * (n - 1));   /* 枢轴 Fp 对应 bin */
            if (ip > n - 1) ip = n - 1;
            if (ip >= 2) {
                FP_TYPE* tmp = (FP_TYPE*)malloc(sizeof(FP_TYPE) * n);
                for (i = 0; i < n; ++i) tmp[i] = vt[i];      /* Fp 以上保持原样 */
                for (i = 1; i < ip; ++i) {
                    double si = pow((double)i / ip, gamma) * ip;  /* 源 bin */
                    int i0 = (int)floor(si); double a = si - i0;
                    if (i0 < 0)         tmp[i] = vt[0];
                    else if (i0 >= n-1) tmp[i] = vt[n-1];
                    else                tmp[i] = (FP_TYPE)((1.0-a)*vt[i0] + a*vt[i0+1]);
                }
                memcpy(vt, tmp, sizeof(FP_TYPE) * n);
                free(tmp);
            }
        }
    }

    /* --- ME：共振峰强调（VTMAGN unsharp mask）------------------------------
       对包络做频域宽平滑得"趋势"，再把"原值-趋势"按 emph 放大：峰更高、谷更低 →
       共振峰对比增强(更清晰/更"有棱角")；ME<0 则压平(更糊/更柔)。emph∈[-0.8,0.8]。 */
    if (fl->has_ME && fl->ME != 0.0 && fnyq > 0.0) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (vt) {
            int n = llsm_fparray_length(vt);
            /* [CALIB 2026-07-12c] 正向 2.0+单点修正钳±12dB:无钳时大 emph 会把趋势线以下
               区域(F1以下/谷)砸掉40dB+,P归一化后频带平衡崩坏(又薄又金属)。钳制后=可闻锐化且有界。
               负向 1.0=完全压平到趋势(物理极限)。 */
            double v01 = clampd(fl->ME, -100, 100) / 100.0;
            double emph = v01 * (v01 > 0 ? 2.0 : 1.0);
            int half = (int)(1200.0 / fnyq * (n - 1));   /* ~1200Hz 平滑半窗 */
            if (half < 1) half = 1;
            if (n > 2) {
                FP_TYPE* sm = (FP_TYPE*)malloc(sizeof(FP_TYPE) * n);
                int w;
                for (i = 0; i < n; ++i) {
                    double sum = 0; int cnt = 0;
                    for (w = -half; w <= half; ++w) { int k = i + w;
                        if (k < 0 || k >= n) continue; sum += vt[k]; ++cnt; }
                    sm[i] = (FP_TYPE)(cnt > 0 ? sum / cnt : vt[i]);
                }
                for (i = 0; i < n; ++i) {
                    double d5 = emph * (vt[i] - sm[i]);
                    if (d5 > 12.0) d5 = 12.0;
                    if (d5 < -12.0) d5 = -12.0;
                    vt[i] += (FP_TYPE)d5;
                    if (vt[i] < -80) vt[i] = -80;
                }
                free(sm);
            }
        }
        /* [2026-07-12d] 同一套钳制 unsharp 同步作用到噪声 psd:只锐化谐波包络时,
           谷区被未编辑的噪声底"填回",HF(声码器自然先验再抹平)尤甚——F1 以下摆幅只剩一半。
           psd 用自己的 1200Hz 趋势,同 emph/同钳制。 */
        {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
            if (nm && nm->psd && nm->npsd > 2) {
                int np = nm->npsd;
                double v01 = clampd(fl->ME, -100, 100) / 100.0;
                double emph = v01 * (v01 > 0 ? 2.0 : 1.0);
                int half = (int)(1200.0 / fnyq * (np - 1));
                if (half < 1) half = 1;
                FP_TYPE* sm = (FP_TYPE*)malloc(sizeof(FP_TYPE) * np);
                int w;
                for (i = 0; i < np; ++i) {
                    double sum = 0; int cnt = 0;
                    for (w = -half; w <= half; ++w) { int k = i + w;
                        if (k < 0 || k >= np) continue; sum += nm->psd[k]; ++cnt; }
                    sm[i] = (FP_TYPE)(cnt > 0 ? sum / cnt : nm->psd[i]);
                }
                for (i = 0; i < np; ++i) {
                    double d5 = emph * (nm->psd[i] - sm[i]);
                    if (d5 > 12.0) d5 = 12.0;
                    if (d5 < -12.0) d5 = -12.0;
                    nm->psd[i] += (FP_TYPE)d5;
                }
                free(sm);
            }
        }
    }

    /* --- Mr：歌手共振峰（~3kHz 高斯 dB bump）-------------------------------
       在声道包络 ~3kHz 处加一个高斯凸起，模拟美声/强声的"歌手共振峰" → 穿透力、亮度。
       中心 3000Hz、带宽 sig=1200Hz、满量程 ±8dB。 */
    if (fl->has_Mr && fl->Mr != 0.0 && fnyq > 0.0) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (vt) {
            int n = llsm_fparray_length(vt);
            double gain = (clampd(fl->Mr, -100, 100) / 100.0) * 12.0; /* full-scale +-12 dB (Sundberg singer's formant +10..20dB) */
            double fc = 3000.0, sig = 1200.0;
            for (i = 0; i < n; ++i) {
                double f = (double)i / (n - 1) * fnyq;
                double d = (f - fc) / sig;
                vt[i] += (FP_TYPE)(gain * exp(-0.5 * d * d));
            }
        }
    }

    /* --- Rd(原Mt改名)：纯声门 Rd 缩放。Rd>0→更紧→Rd 降低。满量程 ±2.0 倍频程。
           clamp 文献扩展范围 [0.1,6.0](Fant 1995 正常 [0.3,2.7]，Huber/Roebel 扩展)。 --- */
    if (fl->has_Rd && fl->Rd != 0.0) {
        FP_TYPE* rd = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_RD);
        if (rd) {
            double f = pow(2.0, -(clampd(fl->Rd, -100, 100) / 100.0) * 2.0);
            *rd = (FP_TYPE)clampd((*rd) * f, 0.1, 6.0);
        }
    }

    /* --- Mt(mo语义)：张力 = 温和 Rd 变化(±1oct) + >1kHz 谱倾斜(±6dB/oct,封顶±10dB)。
           正=紧张(亮/硬)、负=松弛(柔/暗)。区别于 Rd flag(纯声门,不动包络)。 --- */
    if (fl->has_Mt && fl->Mt != 0.0) {
        double v = clampd(fl->Mt, -100, 100) / 100.0;
        FP_TYPE* rd = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_RD);
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (rd) *rd = (FP_TYPE)clampd((*rd) * pow(2.0, -v * 1.0), 0.1, 6.0);
        if (vt && fnyq > 0.0) {
            int n = llsm_fparray_length(vt);
            for (i = 0; i < n; ++i) {
                double f = (double)i / (n - 1) * fnyq;
                if (f > 1000.0) {
                    double db = clampd(v * 6.0 * log(f / 1000.0) / log(2.0), -10.0, 10.0);
                    vt[i] += (FP_TYPE)db; if (vt[i] < -80) vt[i] = -80;
                }
            }
        }
    }

    /* --- Md(mo语义)：干燥度=高频(>5kHz)气声纹理。正=加高频噪声电平(更多气纹理)、负=减(更干)。
           ±6dB 线性过渡带 4k~5k。(原"噪声频率降维"改名为 Mw。) --- */
    if (fl->has_Md && fl->Md != 0.0 && fnyq > 0.0) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
        if (nm && nm->psd) {
            double db = (clampd(fl->Md, -100, 100) / 100.0) * 6.0;
            for (i = 0; i < nm->npsd; ++i) {
                double f = (double)i / (nm->npsd - 1) * fnyq;
                double w = (f <= 4000.0) ? 0.0 : (f >= 5000.0 ? 1.0 : (f - 4000.0) / 1000.0);
                nm->psd[i] += (FP_TYPE)(db * w);
            }
        }
    }

    /* Mv（源相位时间正则化）已改到 chunk 级 smooth_vsphse_chunk（旧的单帧"缩向最小相位"
       逻辑有缠绕缺陷且方向错=电音，已移除）。见 main 中 smooth_vsphse_chunk 调用。 */


    /* --- Mx：实验/诊断。Mx1=只合成谐波(把噪声 NM 静音：psd 压到 -200dB、eenv 幅度归0)；
           Mx2=只合成噪声(把谐波 VTMAGN 压到 -200dB)。用于判断"糊"来自气声还是谐波本身。 --- */
    if (fl->has_Mx && fl->Mx != 0.0) {
        int mode = (int)(fl->Mx + 0.5);
        if (mode == 1) {                 /* 只谐波：静音噪声 */
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
            if (nm) {
                int c;
                if (nm->psd) for (i = 0; i < nm->npsd; ++i) nm->psd[i] = (FP_TYPE)-200.0;
                for (c = 0; c < nm->nchannel; ++c)
                    if (nm->eenv[c]) { int e; for (e = 0; e < nm->eenv[c]->nhar; ++e) nm->eenv[c]->ampl[e] = 0; }
            }
        } else if (mode == 2) {          /* 只噪声：静音谐波 */
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
            if (vt) { int n = llsm_fparray_length(vt); for (i = 0; i < n; ++i) vt[i] = (FP_TYPE)-200.0; }
        }
    }

    /* --- Mb：元音气声（仅有声帧 f0>0 的噪声 psd ±12dB）。全帧版见 Ab。 --- */
    if (fl->has_Mb && fl->Mb != 0.0) {
        FP_TYPE* pf0m = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_F0);
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
        if (pf0m && *pf0m > 0 && nm && nm->psd) {
            double db = (clampd(fl->Mb, -100, 100) / 100.0) * 12.0;
            for (i = 0; i < nm->npsd; ++i) nm->psd[i] += (FP_TYPE)db;
        }
    }

    /* --- NA：鼻音度（VTMAGN 上 500Hz 反共振凹陷+250Hz 鼻峰,共享 na_nasal,dB域） --- */
    if (fl->has_NA && fl->NA != 0.0 && fnyq > 0.0) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
        if (vt) na_nasal(vt, llsm_fparray_length(vt), fnyq,
                         clampd(fl->NA, -100, 100) / 100.0, 1.0);
    }

    /* --- RG：自动混声。r=sigmoid((音高st-65)/2)(换声点F4,2半音过渡),eff=r*RG/100。
           正eff=头声化(Rd松+>1kHz倾斜暗+气声),负eff=胸声保持(belt:Rd紧+亮)。低音r≈0自动无效。 --- */
    if (fl->has_RG && fl->RG != 0.0 && fnyq > 0.0) {
        FP_TYPE* pf0r = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_F0);
        if (pf0r && *pf0r > 50) {
            double st = 69.0 + 12.0 * log((double)*pf0r / 440.0) / log(2.0);
            double r = 1.0 / (1.0 + exp(-(st - 65.0) / 2.0));
            double eff = r * clampd(fl->RG, -100, 100) / 100.0;
            if (eff > 1e-4 || eff < -1e-4) {
                FP_TYPE* rd = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_RD);
                FP_TYPE* vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
                llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
                if (rd) *rd = (FP_TYPE)clampd((*rd) * pow(2.0, 0.7 * eff), 0.1, 6.0);
                if (vt) {
                    int nvt = llsm_fparray_length(vt);
                    for (i = 0; i < nvt; ++i) {
                        double f = (double)i / (nvt - 1) * fnyq;
                        if (f > 1000.0) {
                            double db = clampd(-5.0 * eff * log(f / 1000.0) / log(2.0), -8.0, 8.0);
                            vt[i] += (FP_TYPE)db; if (vt[i] < -80) vt[i] = -80;
                        }
                    }
                }
                if (nm && nm->psd && eff > 0) {   /* 头声方向补气声 */
                    double db = 5.0 * eff;
                    for (i = 0; i < nm->npsd; ++i) nm->psd[i] += (FP_TYPE)db;
                }
            }
        }
    }

    /* --- Ab：全帧气声（原 Mb 行为:所有帧噪声 psd ±12dB,含清辅音） --- */
    if (fl->has_Ab && fl->Ab != 0.0) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
        if (nm && nm->psd) {
            double db = (clampd(fl->Ab, -100, 100) / 100.0) * 12.0;
            for (i = 0; i < nm->npsd; ++i) nm->psd[i] += (FP_TYPE)db;
        }
    }

    /* --- b：清辅音增减（仅无声帧 f0<=0 的噪声能量）。对标 mo：gain=1+0.05*b，
           b=-20→gain0(完全去除)、b0→不变、b100→6倍。psd 为 dB 功率→ +=20log10(gain)。
           区别 Mb：Mb 改全部帧噪声(含浊音气声)，b 只改清辅音(/t/k/s/) 不碰元音。 --- */
    if (fl->has_b && fl->b != 0.0) {
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_F0);
        if (pf0 && *pf0 <= 0) {                    /* 仅无声(清辅音)帧 */
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
            if (nm && nm->psd) {
                double gain = 1.0 + 0.05 * clampd(fl->b, -20, 100);
                double off  = (gain > 1e-4) ? 20.0 * log10(gain) : -120.0;  /* gain~0→静音 */
                for (i = 0; i < nm->npsd; ++i) nm->psd[i] += (FP_TYPE)off;
            }
        }
    }

    /* --- Mn：噪声平滑/去颗粒。缩放 NM.eenv（脉冲同步噪声调制）幅度，
           保留 edc(稳态噪声能量)。[2026-07-12 重标定] 默认改为"原样"：
           k = 1.0 - Mn/100，范围 Mn[-100,100] → k 钳到 [0,2.0]：
             Mn=0(默认/不填)→k=1.0(录音原本纹理); Mn=50→k=0.5(旧版默认的适度平滑);
             Mn=100→k=0(完全平滑); Mn=-100→k=2.0(脉动加倍,更颗粒)。 --- */
    {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(frame, LLSM_FRAME_NM);
        if (nm) {
            double k = clampd(1.0 - clampd(fl->Mn, -100, 100) / 100.0, 0.0, 2.0);
            int c;
            for (c = 0; c < nm->nchannel; ++c) {
                llsm_hmframe* e = nm->eenv[c];
                if (e) for (i = 0; i < e->nhar; ++i) e->ampl[i] *= (FP_TYPE)k;
            }
        }
    }
}

/* ===========================================================================
   变调向上时的自适应高频滚降：对 VTMAGN(dB,频率轴) 在 pivot 以上施加 dB/oct 下倾，
   倾斜量随上变调半音数线性增大 → 高音自动变暖、压掉刺耳的高次谐波；低音/不变调不动。
   shift_st: 上变调半音(>0才生效)；coef: dB/oct per 半音；fnyq: 奈奎斯特(Hz)。
   =========================================================================== */
static void apply_hf_tilt(llsm_container* frame, double shift_st, double coef, double fnyq) {
    FP_TYPE* vt;
    int n, i;
    double tilt_per_oct;
    if (shift_st <= 0.0 || coef <= 0.0 || fnyq <= 0.0) return;
    vt = (FP_TYPE*)llsm_container_get(frame, LLSM_FRAME_VTMAGN);
    if (!vt) return;
    n = llsm_fparray_length(vt);
    if (n < 2) return;
    tilt_per_oct = -coef * shift_st;               /* 负 = 高频衰减 */
    for (i = 0; i < n; ++i) {
        double freq = (double)i / (n - 1) * fnyq;
        if (freq > HF_TILT_PIVOT)
            vt[i] += (FP_TYPE)(tilt_per_oct * log2(freq / HF_TILT_PIVOT));
    }
}

/* ===========================================================================
   参数时间平滑（硬度）：对 chunk 各帧的 Rd / VTMAGN 做对称滑动平均(窗 window) 后按
   blend 混合回去。只平滑有声帧、且只取有声邻帧（避免把清音/边界糊进来）。
   去掉逐帧微抖动 → 音色更稳/更"硬"，对标 mo 的分析规整化。
   =========================================================================== */
static void smooth_chunk_params(llsm_chunk* ch, int nfrm, int window, double blend) {
    int half = window / 2, i, j, w, nspec = 0;
    double* rd_s; FP_TYPE* vt_s; int* voiced;
    if (blend <= 0.0 || window < 2 || nfrm < 3) return;

    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (vt) { nspec = llsm_fparray_length(vt); break; }
    }
    rd_s   = (double*)malloc(sizeof(double) * nfrm);
    voiced = (int*)malloc(sizeof(int) * nfrm);
    vt_s   = nspec ? (FP_TYPE*)malloc(sizeof(FP_TYPE) * (size_t)nfrm * nspec) : NULL;
    if (!rd_s || !voiced) { free(rd_s); free(voiced); if (vt_s) free(vt_s); return; }

    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
        FP_TYPE* rd  = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_RD);
        FP_TYPE* vt  = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        voiced[j] = (pf0 && *pf0 > 0) ? 1 : 0;
        rd_s[j]   = rd ? (double)*rd : 1.0;
        if (vt_s) for (i = 0; i < nspec; ++i)
            vt_s[(size_t)j*nspec + i] = (vt && llsm_fparray_length(vt) == nspec) ? vt[i] : 0;
    }

    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* rd; FP_TYPE* vt;
        if (!voiced[j]) continue;
        rd = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_RD);
        vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (rd) {
            double sum = 0; int cnt = 0;
            for (w = -half; w <= half; ++w) { int k = j + w;
                if (k < 0 || k >= nfrm || !voiced[k]) continue; sum += rd_s[k]; ++cnt; }
            if (cnt > 0) *rd = (FP_TYPE)((1.0 - blend) * (*rd) + blend * (sum / cnt));
        }
        if (vt && vt_s && llsm_fparray_length(vt) == nspec) {
            for (i = 0; i < nspec; ++i) {
                double sum = 0; int cnt = 0;
                for (w = -half; w <= half; ++w) { int k = j + w;
                    if (k < 0 || k >= nfrm || !voiced[k]) continue; sum += vt_s[(size_t)k*nspec + i]; ++cnt; }
                if (cnt > 0) vt[i] = (FP_TYPE)((1.0 - blend) * vt[i] + blend * (sum / cnt));
            }
        }
    }
    free(rd_s); free(voiced); if (vt_s) free(vt_s);
}

/* ===========================================================================
   源相位时间正则化(circular)：对 VSPHSE 沿时间做圆周平滑，使相邻帧源相位趋于一致 →
   相邻周期波形一致 → 稳定/平滑，但【保持自然相位形状】(不强制最小相位) → 不电音。
   这才是对标 mo 的"重正则化但留自然"。区别于旧逻辑(把相位线性缩向0=最小相位=电音，
   且中间值因相位缠绕而产生乱相位)。仅平滑有声帧、只取有声邻帧；复数域均值 + 圆周混合。
   =========================================================================== */
static void smooth_vsphse_chunk(llsm_chunk* ch, int nfrm, int window, double blend) {
    int half = window / 2, j, w, h;
    FP_TYPE** cpy; int* nh; int* voiced;
    if (blend <= 0.0 || window < 2 || nfrm < 3) return;
    cpy    = (FP_TYPE**)malloc(sizeof(FP_TYPE*) * nfrm);
    nh     = (int*)malloc(sizeof(int) * nfrm);
    voiced = (int*)malloc(sizeof(int) * nfrm);
    if (!cpy || !nh || !voiced) { free(cpy); free(nh); free(voiced); return; }
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
        FP_TYPE* vs  = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VSPHSE);
        voiced[j] = (pf0 && *pf0 > 0) ? 1 : 0;
        nh[j] = vs ? llsm_fparray_length(vs) : 0;
        cpy[j] = NULL;
        if (nh[j] > 0) { cpy[j] = (FP_TYPE*)malloc(sizeof(FP_TYPE) * nh[j]);
                         if (cpy[j]) memcpy(cpy[j], vs, sizeof(FP_TYPE) * nh[j]); }
    }
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* vs;
        if (!voiced[j] || nh[j] == 0 || !cpy[j]) continue;
        vs = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VSPHSE);
        if (!vs) continue;
        for (h = 0; h < nh[j]; ++h) {
            double sre = 0, sim = 0; int cnt = 0;
            for (w = -half; w <= half; ++w) { int k = j + w;
                if (k < 0 || k >= nfrm || !voiced[k] || !cpy[k] || h >= nh[k]) continue;
                sre += cos((double)cpy[k][h]); sim += sin((double)cpy[k][h]); ++cnt; }
            if (cnt > 0) {
                FP_TYPE sm = (FP_TYPE)atan2(sim, sre);
                vs[h] = linterpc(cpy[j][h], sm, (FP_TYPE)blend);
            }
        }
    }
    for (j = 0; j < nfrm; ++j) if (cpy[j]) free(cpy[j]);
    free(cpy); free(nh); free(voiced);
}

/* ===========================================================================
   LD（降维 / low-dimensional）：VTMAGN 倒谱低阶化(cepstral liftering)。
   思路：mo 用的 LLSM1 是低阶谱包络表示(每帧仅几十个数)，天然平滑。这里对每个有声帧
   的 dB 幅度谱(nspec 点)做：对称延拓→实倒谱(ifft)→只保留低 quefrency 的前 order 个
   倒谱系数(粗包络)→fft 回 dB 谱。order 越小=阶数越低=越像 LLSM1。可开关、强度可调。
   strength[0,100]：order = 16 + 184*(1-strength/100)，即 16(最强降维)~200(最弱)。
   倒谱长度约定: re/cep 用 fft buffer 大小 2*nfft；ifft 已自带 1/nfft 归一。
   =========================================================================== */
static void liften_vtmagn_chunk(llsm_chunk* ch, int nfrm, double strength) {
    int j, i, nspec = 0, nfft, order;
    FP_TYPE *re, *im, *cep, *buf;
    if (strength <= 0.0 || nfrm < 1) return;
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (vt) { nspec = llsm_fparray_length(vt); break; }
    }
    if (nspec < 16) return;
    nfft  = (nspec - 1) * 2;
    order = (int)floor(16.0 + 184.0 * (1.0 - clampd(strength, 0, 100) / 100.0) + 0.5);
    if (order < 4) order = 4;
    if (order > nspec - 2) order = nspec - 2;
    re  = (FP_TYPE*)calloc(nfft, sizeof(FP_TYPE));
    im  = (FP_TYPE*)calloc(nfft, sizeof(FP_TYPE));
    cep = (FP_TYPE*)calloc(nfft, sizeof(FP_TYPE));
    buf = (FP_TYPE*)calloc(nfft * 2, sizeof(FP_TYPE));
    if (!re || !im || !cep || !buf) { free(re); free(im); free(cep); free(buf); return; }
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
        FP_TYPE* vt  = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (!pf0 || *pf0 <= 0 || !vt || llsm_fparray_length(vt) != nspec) continue;
        for (i = 0; i < nspec; ++i) re[i] = vt[i];
        for (i = nspec; i < nfft; ++i) re[i] = vt[nfft - i];   /* 对称延拓(实偶谱) */
        ifft(re, NULL, cep, NULL, nfft, buf);                  /* -> 实倒谱 */
        for (i = order + 1; i < nfft - order; ++i) cep[i] = 0; /* 低阶截断(保留粗包络) */
        fft(cep, NULL, re, im, nfft, buf);                     /* -> 回 dB 谱(取实部) */
        for (i = 0; i < nspec; ++i) vt[i] = re[i];
    }
    free(re); free(im); free(cep); free(buf);
}

/* 通用：对"每帧一条 len 长 FP 数组"的轨道做时间滑动平均 + blend 混合(内部快照,不污染邻居)。
   track[j]=NULL 表示该帧缺失(跳过/不参与平均)。half=窗半宽。 */
static void smooth_fp_tracks(FP_TYPE** track, int nfrm, int len, int half, double blend) {
    int j, i, w;
    FP_TYPE** cpy;
    if (len <= 0 || nfrm < 3) return;
    cpy = (FP_TYPE**)malloc(sizeof(FP_TYPE*) * nfrm);
    if (!cpy) return;
    for (j = 0; j < nfrm; ++j) {
        cpy[j] = NULL;
        if (track[j]) { cpy[j] = (FP_TYPE*)malloc(sizeof(FP_TYPE) * len);
                        if (cpy[j]) memcpy(cpy[j], track[j], sizeof(FP_TYPE) * len); }
    }
    for (j = 0; j < nfrm; ++j) {
        if (!track[j] || !cpy[j]) continue;
        for (i = 0; i < len; ++i) {
            double s = 0; int c = 0;
            for (w = -half; w <= half; ++w) { int k = j + w;
                if (k < 0 || k >= nfrm || !cpy[k]) continue;
                s += cpy[k][i]; ++c; }
            if (c > 0) track[j][i] = (FP_TYPE)((1.0 - blend) * cpy[j][i] + blend * (s / c));
        }
    }
    for (j = 0; j < nfrm; ++j) if (cpy[j]) free(cpy[j]);
    free(cpy);
}

/* ===========================================================================
   Mg：噪声分量时间平滑(去颗粒)。在【源 chunk】上对噪声模型沿时间滑动平均:
     - psd     : 噪声谱(dB)的帧间抖动→谱颗粒
     - PSDRES  : 残差波动(v2.1 把 PSD 波动单列于此)=颗粒/沙的主因
     - edc     : 各通道噪声包络均值(电平抖动)
     - eenv.ampl: 逐周期"脉冲噪声"包络的谐波幅度(气声的逐周期纹理)
   帧间抖动=听感颗粒/沙哑；时间平滑使噪声纹理连贯。源侧做=拉伸无关，与 Mv(相位)/Mc(输出梳)互补。
   =========================================================================== */
static void smooth_noise_chunk(llsm_chunk* ch, int nfrm, int window, double blend) {
    int half = window / 2, j, b;
    FP_TYPE** track;
    int npsd0 = 0, nchan0 = 0, reslen0 = 0;
    if (blend <= 0.0 || window < 2 || nfrm < 3) return;
    track = (FP_TYPE**)malloc(sizeof(FP_TYPE*) * nfrm);
    if (!track) return;

    /* 参考维度取首个有效帧 */
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        FP_TYPE* res = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_PSDRES);
        if (nm) { if (!npsd0 && nm->psd) npsd0 = nm->npsd; if (!nchan0) nchan0 = nm->nchannel; }
        if (res && !reslen0) reslen0 = llsm_fparray_length(res);
    }

    /* psd */
    if (npsd0 > 0) {
        for (j = 0; j < nfrm; ++j) { llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            track[j] = (nm && nm->psd && nm->npsd == npsd0) ? nm->psd : NULL; }
        smooth_fp_tracks(track, nfrm, npsd0, half, blend);
    }
    /* PSDRES */
    if (reslen0 > 0) {
        for (j = 0; j < nfrm; ++j) { FP_TYPE* res = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_PSDRES);
            track[j] = (res && llsm_fparray_length(res) == reslen0) ? res : NULL; }
        smooth_fp_tracks(track, nfrm, reslen0, half, blend);
    }
    /* edc(每通道均值，整条 nchannel 一起平滑) */
    if (nchan0 > 0) {
        for (j = 0; j < nfrm; ++j) { llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            track[j] = (nm && nm->edc && nm->nchannel == nchan0) ? nm->edc : NULL; }
        smooth_fp_tracks(track, nfrm, nchan0, half, blend);
    }
    /* Mg 只平滑 psd/PSDRES/edc(噪声谱+电平的帧间抖动=颗粒主因)，不动 eenv 脉冲形状。
       实测：加不加 eenv 平滑结果完全一致(本配置 eenv 通道少/能量小=无效)，故略去保持简单。
       全混音实测 rough(30-150Hz) 单调下降(Mg50 已拿大头)；高 Mg 偏激进，默认建议 Mg50。 */
    (void)b;
    free(track);
}

/* ===========================================================================
   Md：噪声【频率降维/去颗粒】。对 NM.psd 做【保持轮廓的 constant-Q 频谱平滑】(不是压平带!)：
   每个 bin 在【对数频率】上取 ±(半带宽) 邻域、在【dB 域】求均值。nbands 越小→等效带越宽→越平滑。
   - 为何不压成平带：piecewise-constant 用线性功率均值(峰值偏置)填回整带，会把噪声谱的【谷】(共振峰
     间/高频滚降)填到偏高平均→凭空多出广带"沙沙声"。constant-Q dB 平滑保留谷和滚降轮廓→不填谷=不加沙。
   - 用 dB 域均值(=几何均值)而非线性功率，避免峰值主导抬高谷。
   去频率维颗粒(Mg 去时间维)，配合丢 PSDRES。源侧做,拉伸无关。
   =========================================================================== */
static void band_reduce_noise_chunk(llsm_chunk* ch, int nfrm, int nbands) {
    int j, i, k;
    FP_TYPE* pfnyq = (FP_TYPE*)llsm_container_get(ch->conf, LLSM_CONF_FNYQ);
    double fnyq = pfnyq ? (double)*pfnyq : 22050.0;
    double flo = 50.0, lr, halfw;
    if (nbands < 1 || nfrm < 1 || fnyq <= flo) return;
    lr = log(fnyq / flo);
    halfw = lr / (2.0 * nbands);    /* 每个等效带的对数半宽(rad of ln) */
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        int np; double* src;
        if (!nm || !nm->psd) continue;
        np = nm->npsd; if (np < 4) continue;
        src = (double*)malloc(sizeof(double) * np);
        if (!src) return;
        for (i = 0; i < np; ++i) src[i] = (double)nm->psd[i];   /* 快照(dB) */
        for (i = 0; i < np; ++i) {
            double f = (double)i / (np - 1) * fnyq;
            double lf = (f <= flo) ? 0.0 : log(f / flo);
            double fa = flo * exp(lf - halfw), fb = flo * exp(lf + halfw);
            int ilo = (int)(fa / fnyq * (np - 1));
            int ihi = (int)(fb / fnyq * (np - 1) + 0.5);
            double s = 0; int c = 0;
            if (ilo < 0) ilo = 0; if (ihi > np - 1) ihi = np - 1; if (ihi < ilo) ihi = ilo;
            for (k = ilo; k <= ihi; ++k) { s += src[k]; ++c; }   /* dB 域均值 */
            nm->psd[i] = (FP_TYPE)(s / (c > 0 ? c : 1));
        }
        free(src);
    }
}

/* ===========================================================================
   降噪（谱减法）：估计背景底噪并从噪声谱 NM.psd[] 扣除。
   - psd 单位为 dB(10log10 功率) → 转线性功率处理再转回。
   - 底噪估计：各频率 bin 在所有帧的【10% 低分位】= 常驻背景噪声。
   - 扣除：power' = max(power - alpha*floor, beta*power)，beta 谱底防过减伪影(musical noise)。
   - strength 0..100 控制 alpha(过减量)。在时间伸缩前对源 chunk 做一次。
   =========================================================================== */
static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}
static void noise_reduce_chunk(llsm_chunk* ch, int nfrm, double strength) {
    int npsd = 0, i, j;
    double* floorp; double* col;
    double alpha, beta;
    if (strength <= 0.0 || nfrm < 4) return;
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (nm && nm->psd) { npsd = nm->npsd; break; }
    }
    if (npsd <= 0) return;
    floorp = (double*)malloc(sizeof(double) * npsd);
    col    = (double*)malloc(sizeof(double) * nfrm);
    if (!floorp || !col) { free(floorp); free(col); return; }

    /* 每个 bin 求底噪 = 时间 10% 低分位（线性功率） */
    for (i = 0; i < npsd; ++i) {
        int cnt = 0;
        for (j = 0; j < nfrm; ++j) {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            if (nm && nm->psd) col[cnt++] = pow(10.0, (double)nm->psd[i] / 10.0);
        }
        if (cnt == 0) { floorp[i] = 0.0; continue; }
        qsort(col, cnt, sizeof(double), cmp_double);
        floorp[i] = col[(int)(0.1 * cnt)];
    }

    alpha = 1.0 + strength / 100.0;   /* 过减量 1..2 */
    beta  = 0.1;                      /* 谱底：每 bin 至少保留 10% */
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (!nm || !nm->psd) continue;
        for (i = 0; i < npsd; ++i) {
            double p = pow(10.0, (double)nm->psd[i] / 10.0);
            double pn = p - alpha * floorp[i];
            double fl = beta * p;
            if (pn < fl) pn = fl;
            if (pn < 1e-12) pn = 1e-12;
            nm->psd[i] = (FP_TYPE)(10.0 * log10(pn));
        }
    }
    free(floorp); free(col);
}

/* ===========================================================================
   DN 层②·嘶声印记降噪(hiss-print)：录音链常驻底噪(磁带嘶声/编解码噪声)全程恒定,
   而要保留的"气声"跟随发声起伏——用这一人声先验把两者分开：
   ①静音帧 = 无声(f0<=0)且 1~12kHz 平均噪声功率落在无声帧最低值 +6dB 内(排除清辅音);
   ②印记 = 静音帧各 bin 线性功率的中值(纯底噪谱,零发声成分);
   ③全帧谱减 p' = max(p - alpha*print, beta*p),beta 谱底防 musical noise。
   区别于 Mz(全帧10%低分位:纯元音素材会把最轻气声帧当底噪→伤气声):印记只取真静音,
   气声帧高于印记的部分按构造保留。静音帧<3 时回退保守版(全帧5%低分位x0.7)。
   strength=DN 值,alpha=DN/100*1.5(DN1≈不减=只留层①,DN50≈0.75,DN100=1.5 轻度过减)。
   时间伸缩前对源 chunk(整段 wav,含首尾静音)做一次。
   =========================================================================== */
static double g_dn_hiss_ratio = 1.0;   /* 层②削减后/前 噪声总能量比(<=1)。V2 混音用它下修
                                          能量匹配目标,否则 sc=rL1n/rL2n 会把清掉的嘶声拉回来 */
static void hiss_print_reduce_chunk(llsm_chunk* ch, int nfrm, double strength, double fnyq, L1Edits* edp) {
    int npsd = 0, i, j, nsil = 0, nuv = 0;
    double* prnt; double* col; double* eng; unsigned char* sil;
    double alpha, beta = 0.1, emin = 1e30;
    double e_pre = 0.0, e_post = 0.0;
    double snr_db = 99.0;
    if (strength <= 0.0 || nfrm < 8) return;
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (nm && nm->psd) { npsd = nm->npsd; break; }
    }
    if (npsd <= 0) return;
    prnt = (double*)malloc(sizeof(double) * npsd);
    col  = (double*)malloc(sizeof(double) * nfrm);
    eng  = (double*)malloc(sizeof(double) * nfrm);
    sil  = (unsigned char*)calloc(nfrm, 1);
    if (!prnt || !col || !eng || !sil) { free(prnt); free(col); free(eng); free(sil); return; }

    {   /* 每帧 1~12kHz 平均线性功率(嘶声主区,避开低频隆隆) */
        int i_lo = (int)(1000.0 / fnyq * (npsd - 1));
        int i_hi = (int)(12000.0 / fnyq * (npsd - 1));
        if (i_hi >= npsd) i_hi = npsd - 1;
        if (i_lo < 0) i_lo = 0;
        if (i_lo >= i_hi) i_lo = 0;
        for (j = 0; j < nfrm; ++j) {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
            double s = 0; int c = 0;
            eng[j] = 1e30;
            if (!nm || !nm->psd || nm->npsd != npsd) continue;
            for (i = i_lo; i <= i_hi; ++i) { s += pow(10.0, (double)nm->psd[i] / 10.0); ++c; }
            if (c == 0) continue;
            eng[j] = s / c;
            if (!pf0 || *pf0 <= 0) { ++nuv; if (eng[j] < emin) emin = eng[j]; }
        }
    }
    if (nuv > 0) {   /* 静音帧 = 无声且能量在最低无声帧 +6dB 内 */
        for (j = 0; j < nfrm; ++j) {
            FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
            if ((!pf0 || *pf0 <= 0) && eng[j] < 1e29 && eng[j] <= emin * 4.0) { sil[j] = 1; ++nsil; }
        }
    }
    /* 印记必须来自真静音帧：全帧百分位"回退"在无静音的干净元音样本上会把最轻的气声
       当底噪(伪印记还让 SNR 门控失真,谐波侧被错误启用)。没有静音就整层跳过,层①不受影响。 */
    if (nsil < 3) {
        fprintf(stderr, "[DN] hiss-print: skipped (no silence frames, layer2 off)\n");
        free(prnt); free(col); free(eng); free(sil);
        return;
    }
    for (i = 0; i < npsd; ++i) {
        int cnt = 0;
        for (j = 0; j < nfrm; ++j) {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            if (!nm || !nm->psd || nm->npsd != npsd || !sil[j]) continue;
            col[cnt++] = pow(10.0, (double)nm->psd[i] / 10.0);
        }
        if (cnt == 0) { prnt[i] = 0.0; continue; }
        qsort(col, cnt, sizeof(double), cmp_double);
        prnt[i] = col[cnt / 2];   /* 静音帧中值 = 纯底噪谱 */
    }
    {   /* SNR(有声帧噪声 vs 印记,1~8kHz;在 psd 被谱减前测)+ 印记降采样打包给 L1 谐波侧 */
        int a_lo = (int)(1000.0 / fnyq * (npsd - 1)), a_hi = (int)(8000.0 / fnyq * (npsd - 1));
        double vsum = 0, psum = 0; int vc = 0;
        if (a_hi >= npsd) a_hi = npsd - 1;
        if (a_lo < 0) a_lo = 0;
        for (j = 0; j < nfrm; ++j) {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
            if (!nm || !nm->psd || nm->npsd != npsd || !pf0 || *pf0 <= 20) continue;
            for (i = a_lo; i <= a_hi; ++i) vsum += pow(10.0, (double)nm->psd[i] / 10.0);
            ++vc;
        }
        for (i = a_lo; i <= a_hi; ++i) psum += prnt[i];
        if (vc > 0 && psum > 1e-30) {
            double r = (vsum / vc / (a_hi - a_lo + 1)) / (psum / (a_hi - a_lo + 1));
            snr_db = 10.0 * log10(r > 1e-12 ? r : 1e-12);
        }
        if (edp) {
            int g, npb = 128;
            edp->dn_print_n = npb; edp->dn_print_fnyq = fnyq; edp->dn_nsnr_db = snr_db;
            for (g = 0; g < npb; ++g) {
                int b0 = (int)((double)g / npb * npsd), b1 = (int)((double)(g + 1) / npb * npsd);
                double s = 0; int c = 0;
                if (b1 > npsd) b1 = npsd;
                for (i = b0; i < b1; ++i) { s += prnt[i]; ++c; }
                s = (c > 0) ? s / c : 0.0;
                edp->dn_print_db[g] = 10.0 * log10(s > 1e-30 ? s : 1e-30);
            }
        }
    }
    alpha = strength / 100.0 * 1.5;
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (!nm || !nm->psd || nm->npsd != npsd) continue;
        for (i = 0; i < npsd; ++i) {
            double p = pow(10.0, (double)nm->psd[i] / 10.0);
            double pn = p - alpha * prnt[i];
            double pmin = beta * p;
            if (pn < pmin) pn = pmin;
            if (pn < 1e-12) pn = 1e-12;
            e_pre += p; e_post += pn;
            nm->psd[i] = (FP_TYPE)(10.0 * log10(pn));
        }
    }
    g_dn_hiss_ratio = (e_pre > 1e-12) ? e_post / e_pre : 1.0;
    fprintf(stderr, "[DN] hiss-print: %d silence frames, noise %.1f dB\n", nsil,
            10.0 * log10(g_dn_hiss_ratio > 1e-12 ? g_dn_hiss_ratio : 1e-12));
    {   /* 谐波/包络侧(L2)：嘶声也被谐波分析拟合进高次谐波幅度(=VTMAGN 高频采样),
           psd 清了 y_sin 还带着。轨迹=VTMAGN 各 bin(固定频率,dB 域),
           印记自校准+SNR 门控见 dn_harm_floor;合成 tolayer0 时从 VTMAGN 重烘焙 ampl,
           故此处编辑对 L2/V1 谐波与 HF 导出都生效。 */
        int nv = 0;
        for (j = 0; j < nfrm; ++j) {
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
            if (vt) { nv = llsm_fparray_length(vt); break; }
        }
        if (nv >= 64 && edp) {
            double* pw = (double*)malloc(sizeof(double) * (size_t)nfrm * nv);
            double* ftrk = (double*)malloc(sizeof(double) * nv);
            double* gn = (double*)malloc(sizeof(double) * nv);
            if (pw && ftrk && gn) {
                for (i = 0; i < nv; ++i) ftrk[i] = (double)i / (nv - 1) * fnyq;
                for (j = 0; j < nfrm; ++j) {
                    FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
                    FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
                    int ok = (vt && pf0 && *pf0 > 20 && llsm_fparray_length(vt) == nv);
                    for (i = 0; i < nv; ++i)
                        pw[(size_t)j * nv + i] = ok ? pow(10.0, (double)vt[i] / 10.0) : -1.0;
                }
                if (dn_harm_gain(pw, nfrm, nv, ftrk, edp->dn_print_db, edp->dn_print_n,
                                 edp->dn_print_fnyq, snr_db, strength / 100.0, gn)) {
                    for (j = 0; j < nfrm; ++j) {
                        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
                        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
                        if (!vt || !pf0 || *pf0 <= 20 || llsm_fparray_length(vt) != nv) continue;
                        for (i = 0; i < nv; ++i)
                            vt[i] += (FP_TYPE)(10.0 * log10(gn[i]));
                    }
                    fprintf(stderr, "[DN] harm-floor(L2): on (snr=%.0f dB)\n", snr_db);
                } else {
                    fprintf(stderr, "[DN] harm-floor(L2): off (snr=%.0f dB)\n", snr_db);
                }
            }
            free(pw); free(ftrk); free(gn);
        }
    }
    free(prnt); free(col); free(eng); free(sil);
}

/* ===========================================================================
   BX 频宽扩展(bandwidth extension)：低质量素材普遍被有损编码砍掉高频
   (金坷垃全员 ~12.7kHz 截止,70dB 悬崖),听感发闷。真实人声 12k+ 以气声/擦音的
   宽带噪声为主,谐波结构基本消失——因此只外推噪声 psd(谐波不动,避免金属声)。
   ①悬崖检测:平均谱(帧均线性功率→dB)上找 800Hz 窗前后落差最大处,
     落差<20dB=全频段素材→整层不动作(与 DN 同哲学:自动无副作用);
   ②逐帧外推:截止前 [fc-3k, fc-200] 真实频段线性拟合(电平+斜率,斜率钳 [-6,0]dB/kHz),
     从 fc 沿拟合斜率外推填充,fc±200Hz 交叉过渡,线性域取 max(原值,填充);
   ③strength=BX 值:填充电平偏移 20log10(BX/100),BX100=贴合自然斜率,BX50≈-6dB 保守。
   逐帧拟合→填充电平跟随源频段(擦音亮、静音帧填充极低,不会重新引入嘶声;宜配 DN 用,
   放在 DN 之后:先去底噪再补带)。
   =========================================================================== */
/* 悬崖检测(在 DN 之前对原始谱做:DN 谱减会把噪声整体压低、削浅悬崖测度)：
   平均谱上宽窗对比(崖前2kHz均值 vs 崖后500Hz起2kHz均值)量全深度——psd 模型会把
   波形谱上的陡崖抹缓,窄窗局部斜率测不出来。天然滚降在此测度下 ~6-10dB,
   编解码悬崖实测 21-33dB(金坷垃),阈值 18dB 可分。返回膝点 bin,无悬崖返回 -1。 */
static int bx_detect_cutoff(llsm_chunk* ch, int nfrm, double fnyq, double* drop_out) {
    int npsd = 0, i, j, i_c = -1;
    double* avg;
    double drop_best = 0.0;
    *drop_out = 0.0;
    if (nfrm < 4) return -1;
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (nm && nm->psd) { npsd = nm->npsd; break; }
    }
    if (npsd < 64) return -1;
    avg = (double*)malloc(sizeof(double) * npsd);
    if (!avg) return -1;
    for (i = 0; i < npsd; ++i) {   /* 平均谱(线性功率帧均→dB) */
        double s = 0; int c = 0;
        for (j = 0; j < nfrm; ++j) {
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
            if (nm && nm->psd && nm->npsd == npsd) { s += pow(10.0, (double)nm->psd[i] / 10.0); ++c; }
        }
        s = (c > 0) ? s / c : 0.0;
        avg[i] = 10.0 * log10(s > 1e-30 ? s : 1e-30);
    }
    {
        int wb = (int)(2000.0 / fnyq * (npsd - 1)), w;
        int gap = (int)(500.0 / fnyq * (npsd - 1));
        int lo = (int)(6000.0 / fnyq * (npsd - 1));
        int hi = npsd - gap - wb - 1;
        int hi2 = (int)(0.88 * (npsd - 1));
        if (hi > hi2) hi = hi2;
        if (wb < 4) wb = 4;
        for (i = lo; i < hi; ++i) {
            double before = 0, after = 0, d;
            if (i - wb < 0) continue;
            for (w = 1; w <= wb; ++w) { before += avg[i - w]; after += avg[i + gap + w]; }
            d = (before - after) / wb;
            if (d > drop_best) { drop_best = d; i_c = i; }
        }
        if (i_c >= 0 && drop_best >= 18.0) {   /* 膝点回溯:定位谱离开"崖前电平-10dB"处 */
            double ref = 0;
            int k;
            for (w = 1; w <= wb; ++w) ref += avg[i_c - w];
            ref /= wb;
            k = i_c;
            if (avg[k] < ref - 10.0) { while (k > 0 && avg[k] < ref - 10.0) --k; }
            else { while (k < npsd - 1 && avg[k] >= ref - 10.0) ++k; --k; }
            i_c = k;
        }
    }
    free(avg);
    *drop_out = drop_best;
    return (i_c >= 0 && drop_best >= 18.0) ? i_c : -1;
}
static void bandwidth_extend_chunk(llsm_chunk* ch, int nfrm, double strength, double fnyq,
                                   int i_c, double drop_db) {
    int npsd = 0, i, j;
    double f_c, gain_off;
    int s_lo, s_hi, r_lo;
    if (strength <= 0.0 || nfrm < 4 || i_c < 0) return;
    for (j = 0; j < nfrm; ++j) {
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        if (nm && nm->psd) { npsd = nm->npsd; break; }
    }
    if (npsd < 64 || i_c >= npsd) return;
    f_c = (double)i_c / (npsd - 1) * fnyq;
    /* 源区间必须避开编解码低通的过渡带(膝点前 ~1.5kHz 已在滚降上,拟合会沿滚降外推=填了个寂寞)：
       取 [fc-4.5k, fc-1.5k] 的干净频段,跨过渡带线性外推。 */
    s_lo = (int)((f_c - 4500.0) / fnyq * (npsd - 1));
    s_hi = (int)((f_c - 1500.0) / fnyq * (npsd - 1));
    r_lo = s_hi;
    if (s_lo < 0) s_lo = 0;
    if (s_hi - s_lo < 8) return;
    gain_off = 20.0 * log10((strength < 1 ? 1 : strength) / 100.0) - 3.0;   /* -3dB:线性外推在凸形谱上略过冲,压回接缝连续 */
    for (j = 0; j < nfrm; ++j) {   /* 逐帧:源频段线性拟合→外推填充 */
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        double sx = 0, sy = 0, sxx = 0, sxy = 0, det, bb, aa, lvl, slope;
        int n = 0;
        if (!nm || !nm->psd || nm->npsd != npsd) continue;
        for (i = s_lo; i <= s_hi; ++i) {
            double x = (double)i / (npsd - 1) * fnyq, y = (double)nm->psd[i];
            sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
        }
        det = n * sxx - sx * sx;
        if (fabs(det) < 1e-9 || n < 8) continue;
        bb = (n * sxy - sx * sy) / det;
        aa = (sy - bb * sx) / n;
        lvl = aa + bb * f_c;              /* 拟合在 fc 处的电平(dB) */
        slope = bb * 1000.0;              /* dB/kHz;钳 [-6,-1.5]:真实空气带必然下落,拟合走平/上翘时强制最小滚降 */
        if (slope > -1.5) slope = -1.5;
        if (slope < -6.0) slope = -6.0;
        for (i = r_lo; i < npsd; ++i) {
            double f = (double)i / (npsd - 1) * fnyq;
            double w = (f - (f_c - 1500.0)) / 1000.0;   /* [fc-1.5k, fc-0.5k] 渐入,覆盖过渡带 */
            double fill_db, pf, po;
            if (w <= 0.0) continue;
            if (w > 1.0) w = 1.0;
            fill_db = lvl + slope * ((f - f_c) / 1000.0) + gain_off;
            pf = w * pow(10.0, fill_db / 10.0);
            po = pow(10.0, (double)nm->psd[i] / 10.0);
            if (pf > po) nm->psd[i] = (FP_TYPE)(10.0 * log10(pf));
        }
    }
    fprintf(stderr, "[BX] cutoff %.1f kHz (drop %.0f dB), air band extended\n", f_c / 1000.0, drop_db);
}

/* ===========================================================================
   TM 音色对齐(bank 档案修复,"说话人先验"第一档)：同一文件夹=同一个人——
   扫描音源文件夹全部 wav,每个算 LTAS(活动帧长期平均谱,64 带)+质量分(有效带宽+SNR),
   质量最好的 1/4 平均为"参考音色档案"(200~3k 电平归一=纯形状);渲染时当前音的 LTAS
   与档案做形状差 → 平滑校正 EQ(钳±10dB),同时施加到谐波包络(VTMAGN/L1 vt)与噪声 psd。
   与 DN/BX 零重叠：DN=减性噪声,BX=截止以上补带,TM=活带内乘性染色校正;
   死带(低于档案 >25dB,编码砍死的)校正斜坡归零=让给 BX;当前音≈档案时校正≈0(自门控)。
   档案缓存 <dir>\wcsndm_bank.prof:一次建档(数秒),档案缺当前文件或 G flag 时重建。
   =========================================================================== */
#define TM_NB   64
#define TM_MAXF 500
typedef struct { char name[96]; int src_fs, src_ch; float bw_hz, snr_db; float ltas[TM_NB]; } TMEnt;

static int tm_wav_ch_(const char* path) {   /* 声道数(标准 RIFF 布局;解析失败=1,仅影响分组粒度) */
    FILE* f = fopen(path, "rb");
    unsigned char h[64]; size_t got; int ch = 1;
    if (!f) return 1;
    got = fread(h, 1, sizeof(h), f);
    fclose(f);
    if (got >= 24 && !memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVE", 4) && !memcmp(h + 12, "fmt ", 4))
        ch = h[22] | (h[23] << 8);
    if (ch < 1 || ch > 8) ch = 1;
    return ch;
}

static const char* tm_basename(const char* p) {
    const char* b = p; const char* q;
    for (q = p; *q; ++q) if (*q == '\\' || *q == '/') b = q + 1;
    return b;
}
static void tm_norm_(const float* lt, double* out) {   /* 200~3k 电平归一(减均值)=纯形状 */
    int b, b0 = (int)(200.0 / 22050.0 * TM_NB), b1 = (int)(3000.0 / 22050.0 * TM_NB), c = 0;
    double m = 0;
    for (b = b0; b <= b1; ++b) { m += lt[b]; ++c; }
    m /= (c ? c : 1);
    for (b = 0; b < TM_NB; ++b) out[b] = lt[b] - m;
}
/* 单文件 LTAS+质量分+源格式。统一 44.1k、至多取前 3 秒。活动帧=RMS>15%峰值,静默帧=<5%。 */
static int tm_file_ltas(const char* path, float* ltas_db, float* bw_hz, float* snr_db,
                        int* src_fs, int* src_ch) {
    int fs = 0, nbit = 0, nx = 0, i, j, b;
    const int nwin = 1024, hop = 512;
    int nfrm, nact = 0, nqt = 0, rc = 0;
    FP_TYPE* x = wavread((char*)path, &fs, &nbit, &nx);
    double *acta = NULL, *qta = NULL, *rms = NULL, *wnd = NULL;
    FP_TYPE *re = NULL, *im = NULL, *buf = NULL;
    double maxr = 0, athr, qthr;
    if (!x || nx <= 0) { free(x); return 0; }
    *src_fs = fs;
    *src_ch = tm_wav_ch_(path);
    if (nx > fs * 3) nx = fs * 3;
    if (fs != 44100) {
        int ny = 0; FP_TYPE* y = wcs_resample_44k(x, nx, fs, &ny);
        if (y && ny > 0) { free(x); x = y; nx = ny; fs = 44100; } else free(y);
    }
    nfrm = (nx - nwin) / hop + 1;
    if (nfrm < 4) { free(x); return 0; }
    acta = (double*)calloc(TM_NB, sizeof(double));
    qta  = (double*)calloc(TM_NB, sizeof(double));
    rms  = (double*)malloc(sizeof(double) * nfrm);
    wnd  = (double*)malloc(sizeof(double) * nwin);
    re   = (FP_TYPE*)malloc(sizeof(FP_TYPE) * nwin);
    im   = (FP_TYPE*)malloc(sizeof(FP_TYPE) * nwin);
    buf  = (FP_TYPE*)malloc(sizeof(FP_TYPE) * nwin * 2);
    if (!acta || !qta || !rms || !wnd || !re || !im || !buf) goto done;
    for (j = 0; j < nwin; ++j) wnd[j] = 0.5 - 0.5 * cos(2.0 * M_PI * j / (nwin - 1));
    for (i = 0; i < nfrm; ++i) {
        double s = 0;
        for (j = 0; j < nwin; ++j) { double v = x[i * hop + j]; s += v * v; }
        rms[i] = sqrt(s / nwin);
        if (rms[i] > maxr) maxr = rms[i];
    }
    athr = 0.15 * maxr; qthr = 0.05 * maxr;
    for (i = 0; i < nfrm; ++i) {
        int cls = (rms[i] >= athr) ? 1 : ((rms[i] <= qthr) ? 2 : 0);
        if (!cls) continue;
        for (j = 0; j < nwin; ++j) { re[j] = (FP_TYPE)(x[i * hop + j] * wnd[j]); im[j] = 0; }
        fft(re, NULL, re, im, nwin, buf);
        for (j = 0; j < nwin / 2; ++j) {
            double p = (double)re[j] * re[j] + (double)im[j] * im[j];
            double f = (double)j * fs / nwin;
            b = (int)(f / 22050.0 * TM_NB); if (b >= TM_NB) b = TM_NB - 1;
            if (cls == 1) acta[b] += p; else qta[b] += p;
        }
        if (cls == 1) ++nact; else ++nqt;
    }
    if (nact < 3) goto done;
    for (b = 0; b < TM_NB; ++b)
        ltas_db[b] = (float)(10.0 * log10(acta[b] / nact + 1e-20));
    {   /* snr: 1~8kHz 活动/静默 */
        double sa = 0, sq = 0;
        int b0 = (int)(1000.0 / 22050.0 * TM_NB), b1 = (int)(8000.0 / 22050.0 * TM_NB);
        for (b = b0; b <= b1; ++b) { sa += acta[b] / nact; if (nqt) sq += qta[b] / nqt; }
        *snr_db = (nqt >= 3 && sq > 1e-20) ? (float)(10.0 * log10(sa / sq)) : 99.0f;
    }
    {   /* bw: 0.5~4k 平均电平 -45dB 以内的最高带 */
        double m = 0; int c = 0, bb = 0;
        int b0 = (int)(500.0 / 22050.0 * TM_NB), b1 = (int)(4000.0 / 22050.0 * TM_NB);
        for (b = b0; b <= b1; ++b) { m += ltas_db[b]; ++c; }
        m /= (c ? c : 1);
        for (b = 0; b < TM_NB; ++b) if (ltas_db[b] >= m - 45.0) bb = b;
        *bw_hz = (float)((bb + 0.5) / (double)TM_NB * 22050.0);
    }
    rc = 1;
done:
    free(acta); free(qta); free(rms); free(wnd); free(re); free(im); free(buf); free(x);
    return rc;
}
static int tm_prof_save(const char* dir, TMEnt* ents, int n, const double* ref) {
    char p[1200]; FILE* f; int hdr[3], b; float reff[TM_NB];
    snprintf(p, sizeof(p), "%s\\wcsndm_bank.prof", dir);
    f = fopen(p, "wb"); if (!f) return 0;
    hdr[0] = 0x54503257; hdr[1] = n; hdr[2] = TM_NB;
    for (b = 0; b < TM_NB; ++b) reff[b] = (float)ref[b];
    fwrite(hdr, 4, 3, f); fwrite(reff, 4, TM_NB, f); fwrite(ents, sizeof(TMEnt), n, f);
    fclose(f); return 1;
}
static int tm_prof_load(const char* dir, TMEnt** ents_out, int* n_out, double* ref_out) {
    char p[1200]; FILE* f; int hdr[3], b; float reff[TM_NB]; TMEnt* e;
    snprintf(p, sizeof(p), "%s\\wcsndm_bank.prof", dir);
    f = fopen(p, "rb"); if (!f) return 0;
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x54503257 || hdr[2] != TM_NB
        || hdr[1] <= 0 || hdr[1] > TM_MAXF) { fclose(f); return 0; }
    if (fread(reff, 4, TM_NB, f) != TM_NB) { fclose(f); return 0; }
    e = (TMEnt*)malloc(sizeof(TMEnt) * hdr[1]);
    if (!e || fread(e, sizeof(TMEnt), hdr[1], f) != (size_t)hdr[1]) { free(e); fclose(f); return 0; }
    fclose(f);
    for (b = 0; b < TM_NB; ++b) ref_out[b] = reff[b];
    *ents_out = e; *n_out = hdr[1];
    return 1;
}
/* 建档:扫描 dir 下 *.wav(至多 TM_MAXF),选优 1/4(得分=带宽+50Hz/dB*SNR)平均成参考档案 */
static int tm_build_profile(const char* dir, TMEnt** ents_out, int* n_out, double* ref_out) {
    WIN32_FIND_DATAA fd; HANDLE h;
    char pat[1200], full[1400];
    TMEnt* ents = (TMEnt*)malloc(sizeof(TMEnt) * TM_MAXF);
    int n = 0, i, b;
    if (!ents) return 0;
    snprintf(pat, sizeof(pat), "%s\\*.wav", dir);
    h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) { free(ents); return 0; }
    do {
        float lt[TM_NB], bw = 0, snr = 0;
        int sfs = 0, sch = 1;
        if (n >= TM_MAXF) break;
        snprintf(full, sizeof(full), "%s\\%s", dir, fd.cFileName);
        if (!tm_file_ltas(full, lt, &bw, &snr, &sfs, &sch)) continue;
        memset(&ents[n], 0, sizeof(TMEnt));
        strncpy(ents[n].name, fd.cFileName, sizeof(ents[n].name) - 1);
        ents[n].src_fs = sfs; ents[n].src_ch = sch;
        ents[n].bw_hz = bw; ents[n].snr_db = snr;
        memcpy(ents[n].ltas, lt, sizeof(lt));
        ++n;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    if (n < 4) { free(ents); return 0; }
    {
        int ngood = n / 4, k;
        int* idx = (int*)malloc(sizeof(int) * n);
        double* sc = (double*)malloc(sizeof(double) * n);
        if (!idx || !sc) { free(idx); free(sc); free(ents); return 0; }
        if (ngood < 3) ngood = 3;
        if (ngood > 60) ngood = 60;
        if (ngood > n) ngood = n;
        for (i = 0; i < n; ++i) { idx[i] = i;
            sc[i] = ents[i].bw_hz + 50.0 * (ents[i].snr_db < 40 ? ents[i].snr_db : 40.0); }
        for (i = 0; i < ngood; ++i) { int mx = i;
            for (k = i + 1; k < n; ++k) if (sc[idx[k]] > sc[idx[mx]]) mx = k;
            { int t = idx[i]; idx[i] = idx[mx]; idx[mx] = t; } }
        for (b = 0; b < TM_NB; ++b) ref_out[b] = 0;
        for (i = 0; i < ngood; ++i) {
            double nrm[TM_NB];
            tm_norm_(ents[idx[i]].ltas, nrm);
            for (b = 0; b < TM_NB; ++b) ref_out[b] += nrm[b];
        }
        for (b = 0; b < TM_NB; ++b) ref_out[b] /= ngood;
        fprintf(stderr, "[TM] bank profile built: %d files, ref=top %d (best bw %.1fk snr %.0f dB)\n",
                n, ngood, ents[idx[0]].bw_hz / 1000.0, ents[idx[0]].snr_db);
        free(idx); free(sc);
    }
    *ents_out = ents; *n_out = n;
    return 1;
}
/* TM 校正曲线计算(建档/加载→批次分组→组均差→保护/钳制/平滑)。与所用内核无关:
   L2 路径算完施加到 chunk(VTMAGN+psd),L1 路径算完经 L1Edits 传入(vt+噪声谱)。
   返回 1=有校正(sm_out[TM_NB],dB,线性频轴 0..22050),0=无(单批次/已最亮/无档案)。 */
static int tm_compute_corr(const char* in_file, int force, double strength01, double* sm_out) {
    char dirbuf[1100];
    const char* base = tm_basename(in_file);
    TMEnt* ents = NULL; int nent = 0, ei = -1, b, rc = 0;
    double ref[TM_NB];
    size_t dl = (size_t)(base - in_file);
    if (dl > 0 && dl < sizeof(dirbuf)) { memcpy(dirbuf, in_file, dl - 1); dirbuf[dl - 1] = 0; }
    else snprintf(dirbuf, sizeof(dirbuf), ".");
    if (force || !tm_prof_load(dirbuf, &ents, &nent, ref)) { free(ents); ents = NULL; nent = 0; }
    if (ents) { int i5; for (i5 = 0; i5 < nent; ++i5) if (!_stricmp(ents[i5].name, base)) { ei = i5; break; } }
    if (ei < 0) {   /* 无档案/档案缺当前文件 → 重建 */
        free(ents); ents = NULL; nent = 0;
        if (tm_build_profile(dirbuf, &ents, &nent, ref)) {
            int i5; tm_prof_save(dirbuf, ents, nent, ref);
            for (i5 = 0; i5 < nent; ++i5) if (!_stricmp(ents[i5].name, base)) { ei = i5; break; }
        }
    }
    if (ei >= 0) {
        /* 按录音批次分组(源采样率+声道=转录代际指纹),组均 LTAS 相减：
           两侧都平均了整套音素 → 音素内容抵消,剩下纯通道染色(逐文件对比会把音素身份
           误当染色 → 元音同化,已废弃)。单一批次 → 无染色差可测,自动不动作。 */
        enum { TM_MAXG = 8 };
        int gfs[TM_MAXG], gch[TM_MAXG], gcnt[TM_MAXG], ng = 0;
        double gsum[TM_MAXG][TM_NB], gbr[TM_MAXG];
        int gcur = -1, gref = -1, i5, g;
        for (i5 = 0; i5 < nent; ++i5) {
            for (g = 0; g < ng; ++g)
                if (gfs[g] == ents[i5].src_fs && gch[g] == ents[i5].src_ch) break;
            if (g == ng) {
                if (ng >= TM_MAXG) continue;
                gfs[ng] = ents[i5].src_fs; gch[ng] = ents[i5].src_ch; gcnt[ng] = 0;
                for (b = 0; b < TM_NB; ++b) gsum[ng][b] = 0;
                ++ng;
            }
            {
                double nrm[TM_NB];
                tm_norm_(ents[i5].ltas, nrm);
                for (b = 0; b < TM_NB; ++b) gsum[g][b] += nrm[b];
                ++gcnt[g];
            }
        }
        for (g = 0; g < ng; ++g) {   /* 组均+亮度(6~14k 均值,选参考批次用) */
            int b0 = (int)(6000.0 / 22050.0 * TM_NB), b1 = (int)(14000.0 / 22050.0 * TM_NB), c = 0;
            gbr[g] = 0;
            for (b = 0; b < TM_NB; ++b) gsum[g][b] /= (gcnt[g] ? gcnt[g] : 1);
            for (b = b0; b <= b1; ++b) { gbr[g] += gsum[g][b]; ++c; }
            gbr[g] /= (c ? c : 1);
            if (gfs[g] == ents[ei].src_fs && gch[g] == ents[ei].src_ch) gcur = g;
        }
        for (g = 0; g < ng; ++g)
            if (gcnt[g] >= 8 && (gref < 0 || gbr[g] > gbr[gref])) gref = g;
        if (gcur < 0 || gref < 0 || gref == gcur || gbr[gref] - gbr[gcur] < 1.0) {
            fprintf(stderr, "[TM] %s, no correction (batches=%d)\n",
                    ng < 2 ? "single batch" : "file already in best batch", ng);
        } else {
            double corr[TM_NB], msum = 0;
            for (b = 0; b < TM_NB; ++b) {
                double c = gsum[gref][b] - gsum[gcur][b];
                double fb = (b + 0.5) / TM_NB * 22050.0;
                double wlf = (fb - 800.0) / 1000.0;   /* <800Hz 不动(低频无染色),1.8k 起全量 */
                if (wlf < 0) wlf = 0;
                if (wlf > 1) wlf = 1;
                if (c > 15.0) {   /* 死带斜坡:缺口>15dB 渐衰,>25dB 归零(编码砍死的带让给 BX) */
                    double w = (25.0 - c) / 10.0;
                    if (w < 0) w = 0;
                    if (w > 1) w = 1;
                    c *= w;
                }
                if (c > 10.0) c = 10.0;
                if (c < -10.0) c = -10.0;
                corr[b] = c * wlf;
            }
            for (b = 0; b < TM_NB; ++b) {   /* ±2 带平滑(抹残余共振峰差,留宽带染色) + 强度 */
                double t = 0; int cc = 0, k;
                for (k = b - 2; k <= b + 2; ++k) { if (k < 0 || k >= TM_NB) continue; t += corr[k]; ++cc; }
                sm_out[b] = t / cc * strength01;
                msum += fabs(sm_out[b]);
            }
            fprintf(stderr, "[TM] batch corr: %dHz/%dch(n=%d) -> ref %dHz/%dch(n=%d), mean |%.1f| dB\n",
                    gfs[gcur], gch[gcur], gcnt[gcur], gfs[gref], gch[gref], gcnt[gref], msum / TM_NB);
            rc = 1;
        }
    } else {
        fprintf(stderr, "[TM] no bank profile available, skip\n");
    }
    free(ents);
    return rc;
}

/* ===========================================================================
   Mi 神经蒸馏预设(参数蒸馏)：把 Seed-VC(或任意神经修复)对本 wav 的效果,离线蒸馏成
   "原始分析 vs 修复后分析"的逐帧 64 带对数谱差量(dB),存 <wav>.nfx。渲染时引擎正常分析
   原始 wav,再按强度把差量叠加到 VTMAGN(谐波包络)+psd(噪声)——纯参数数学,微秒级,无 daemon。
   参数内核(谐波由包络重烘焙)天然只用得上这些量:波形级修复(相位/瞬态)本就被丢弃,故近无损。
   与音频影子缓存相比:每文件几十 KB、可调强度(差量×Mi/100)、确定可逆。仅 M1(时域)享受不到。
   预设按【源时间轴】索引:apply 在源 chunk(伸缩前),帧 i 时间 = i*thop_ms,查预设同刻差量。
   **差量在引擎参数域算**(蒸馏工具用 @DUMP 拿原始/修复的 vt_db+psd 各自 diff),vt→VTMAGN、psd→psd 分开应用,
   才与施加对易(v1 用波形 STFT 差量不对易已废弃)。
   .nfx v2: magic'WNF2' nfrm nband=64 + float thop_ms fnyq + int16 vt[nfrm*64] + int16 psd[nfrm*64](dB*100)。 */
static void apply_nfx_chunk(llsm_chunk* ch, int nfrm, const char* wav, double strength01, double thop_ms) {
    char path[1200]; FILE* f; int hdr[3], pn, pb, i, j, b;
    float ft[2]; short* qv = NULL; short* qp = NULL; double* bv = NULL; double* bp = NULL;
    double thop_p, fnyq; size_t cnt;
    if (strength01 <= 0.0) return;
    snprintf(path, sizeof(path), "%s.nfx", wav);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[Mi] no preset (%s), skip\n", path); return; }
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x32464E57 || hdr[1] <= 0 || hdr[1] > 100000
        || hdr[2] != 64 || fread(ft, 4, 2, f) != 2) { fclose(f); return; }
    pn = hdr[1]; pb = hdr[2]; thop_p = ft[0]; fnyq = ft[1];
    cnt = (size_t)pn * pb;
    qv = (short*)malloc(sizeof(short) * cnt);
    qp = (short*)malloc(sizeof(short) * cnt);
    bv = (double*)malloc(sizeof(double) * pb);
    bp = (double*)malloc(sizeof(double) * pb);
    if (!qv || !qp || !bv || !bp
        || fread(qv, sizeof(short), cnt, f) != cnt || fread(qp, sizeof(short), cnt, f) != cnt) {
        free(qv); free(qp); free(bv); free(bp); fclose(f); return;
    }
    fclose(f);
    if (thop_p <= 0) thop_p = thop_ms;
    for (i = 0; i < nfrm; ++i) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[i], LLSM_FRAME_VTMAGN);
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[i], LLSM_FRAME_NM);
        double tp = (i * thop_ms) / thop_p;         /* 源帧 i 对应的预设帧坐标(时间对齐) */
        int p0 = (int)tp; double fr = tp - p0;
        int p1 = p0 + 1;
        if (p0 < 0) { p0 = 0; fr = 0; }
        if (p0 > pn - 1) { p0 = pn - 1; fr = 0; }
        if (p1 > pn - 1) p1 = pn - 1;
        for (b = 0; b < pb; ++b) {                   /* 预设两帧线性插值 → 本帧 64 带 dB × 强度 */
            bv[b] = (qv[p0 * pb + b] * (1.0 - fr) + qv[p1 * pb + b] * fr) / 100.0 * strength01;
            bp[b] = (qp[p0 * pb + b] * (1.0 - fr) + qp[p1 * pb + b] * fr) / 100.0 * strength01;
        }
        if (vt) { int nv = llsm_fparray_length(vt);   /* vt 差量 → 谐波包络 */
            for (j = 0; j < nv; ++j)
                vt[j] += (FP_TYPE)dn_interp_db_(bv, pb, fnyq, (double)j / (nv - 1) * fnyq); }
        if (nm && nm->psd) { int np = nm->npsd;       /* psd 差量 → 噪声谱 */
            for (j = 0; j < np; ++j)
                nm->psd[j] += (FP_TYPE)dn_interp_db_(bp, pb, fnyq, (double)j / (np - 1) * fnyq); }
    }
    fprintf(stderr, "[Mi] preset applied: %d frames, strength %.0f%%\n", pn, strength01 * 100.0);
    free(qv); free(qp); free(bv); free(bp);
}

/* ===========================================================================
   子谐波抑制（suppress-subharmonics）：痰 = 有声帧里谐波之间塞满了子谐波/噪声(HNR低)，
   这些 inter-harmonic 能量在我们模型里落在【噪声谱 NM.psd】里。
   做法：仅对【有声帧】(f0>0) 把噪声谱压低 cut_db，拉高 HNR → 去痰；清音擦音帧不动(保气声)。
   这正是它区别于 Mb(无差别砍全帧)的地方，符合 mo"对嘶吼有用、略损气声嗓"。
   strength 0..100 → cut_db 0..18dB。在伸缩前对源 chunk 做一次。
   =========================================================================== */
static void suppress_subharmonics_chunk(llsm_chunk* ch, int nfrm, double strength, double fnyq) {
    int j, i;
    double cut_db;
    if (strength <= 0.0) return;
    cut_db = (strength / 100.0) * 18.0;
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_F0);
        llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(ch->frames[j], LLSM_FRAME_NM);
        int np;
        if (!pf0 || *pf0 <= 0 || !nm || !nm->psd) continue;   /* 仅有声帧 */
        np = nm->npsd;
        /* 频率加权：主体(<3kHz,谐波主导区)重压=干净/稳，高频(>3kHz,空气感)完全不削=全保留。
           对齐 mo 的噪声剖面(低净高空气)——主体清干净换周期稳定，HF 宽带空气整段保住。
           w: <1kHz=2.0, 1-3kHz 线性 2.0→0, >3kHz=0。 */
        for (i = 0; i < np; ++i) {
            double f = (np > 1) ? (double)i / (np - 1) * fnyq : 0.0;
            double w;
            if (f < 1000.0)      w = 2.0;
            else if (f > 3000.0) w = 0.0;
            else                 w = 2.0 + (f - 1000.0) / 2000.0 * (0.0 - 2.0);
            nm->psd[i] -= (FP_TYPE)(cut_db * w);
        }
    }
}

/* ===========================================================================
   抗失真（anti-distortion）：低音量/量化误差使 VTMAGN 高频出现伪峰，变调向上被放大成
   "发尖/发毛"。修法：对 VTMAGN 在【频率方向】做平滑——平滑窗随频率增大(低频≈不动,保住
   共振峰；高频窗大,抹掉伪峰)。按 blend 混合。即 mo 所说"略微模糊语音"。
   与 MH(砍高频电平) 机制不同：这里抹的是高频伪细节、保留平滑趋势。
   strength 0..100 → blend 0..1。在伸缩前对源 chunk 做一次。
   =========================================================================== */
static void anti_distortion_chunk(llsm_chunk* ch, int nfrm, double strength) {
    int j, i, w, nspec = 0;
    double blend, wmax;
    double* tmp;
    if (strength <= 0.0) return;
    blend = strength / 100.0;
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (vt) { nspec = llsm_fparray_length(vt); break; }
    }
    if (nspec < 16) return;
    wmax = nspec / 32.0;                 /* 最高频处的平滑半窗(bin) */
    tmp  = (double*)malloc(sizeof(double) * nspec);
    if (!tmp) return;
    for (j = 0; j < nfrm; ++j) {
        FP_TYPE* vt = (FP_TYPE*)llsm_container_get(ch->frames[j], LLSM_FRAME_VTMAGN);
        if (!vt || llsm_fparray_length(vt) != nspec) continue;
        for (i = 0; i < nspec; ++i) {
            int half = (int)(1.0 + (double)i / nspec * wmax);  /* 窗随频率线性增大 */
            double sum = 0; int cnt = 0;
            for (w = -half; w <= half; ++w) { int k = i + w;
                if (k < 0 || k >= nspec) continue; sum += vt[k]; ++cnt; }
            tmp[i] = (cnt > 0) ? sum / cnt : vt[i];
        }
        for (i = 0; i < nspec; ++i)
            vt[i] = (FP_TYPE)((1.0 - blend) * vt[i] + blend * tmp[i]);
    }
    free(tmp);
}

/* ===========================================================================
   主流程
   =========================================================================== */
int main(int argc, char** argv) {
    if (argc < 13) {
        printf("Usage: %s in.wav out.wav pitch velocity flags offset length "
               "consonant cutoff volume modulation tempo pitchbend\n", argv[0]);
        return 1;
    }
    UtauArgs u;
    u.in_file   = argv[1];
    u.out_file  = argv[2];
    u.pitch_note= argv[3];
    u.velocity  = atof(argv[4]);
    u.flags     = argv[5];
    {   /* 调试取证:把本次调用参数写到 exe 旁 wcsndm_lastcall.txt(覆盖式,1KB 内)。
           用于诊断"UTAU 实际传了什么 flag"(编辑器吃字符/缓存不重渲染类问题)。 */
        char exed[1040], lp[1140]; FILE* lf; int ai;
        GetModuleFileNameA(NULL, exed, sizeof(exed));
        { char* p9 = strrchr(exed, '\\'); if (p9) *p9 = 0; }
        snprintf(lp, sizeof(lp), "%s\\wcsndm_lastcall.txt", exed);
        lf = fopen(lp, "w");
        if (lf) {
            fprintf(lf, "flags=[%s]\nin=%s\npitch=%s vel=%s off=%s len=%s con=%s cut=%s vol=%s\n",
                    argv[5], argv[1], argv[3], argv[4], argv[6], argv[7], argv[8], argv[9], argv[10]);
            for (ai = 11; ai < argc && ai < 14; ++ai) fprintf(lf, "arg%d=%.60s\n", ai, argv[ai]);
            fclose(lf);
        }
    }
    u.offset    = atof(argv[6]);
    u.length    = atof(argv[7]);
    u.consonant = atof(argv[8]);
    u.cutoff    = atof(argv[9]);
    u.volume    = atof(argv[10]);
    u.modulation= atof(argv[11]);
    u.tempo     = atof(argv[12][0] == '!' ? argv[12] + 1 : argv[12]);
    u.pitchbend = argc > 13 ? argv[13] : "";

    int nhop = DEFAULT_HOP;

    /* --- 解析 flags & pitchbend --- */
    Flags fl; parse_flags(u.flags, &fl);
    PitchCurve pc;
    pc.base_midi = note_to_midi(u.pitch_note);
    if (fl.has_t) pc.base_midi += fl.t / 100.0;     /* t flag：cents → 半音 */
    pc.cents = decode_pitchbend(u.pitchbend, &pc.n);
    pc.dt    = 60.0 / (u.tempo * 96.0);

    /* Ms(mo稳定化 0~10)：映射为 源相位正则 Mv=Ms*10 + 参数平滑 MS=Ms*10(不覆盖显式值)。 */
    if (fl.has_Ms) {
        double s10 = clampd(fl.Ms, 0, 10) * 10.0;
        if (!fl.has_Mv) { fl.Mv = s10; fl.has_Mv = 1; }
        if (!fl.has_MS) { fl.MS = s10; fl.has_MS = 1; }
    }
    /* Me(mo强制循环) = l flag 同义 */
    if (fl.has_Me) fl.has_l = 1;

    /* Mp：音高随机扰动(带限,~50ms平滑,满量程±25cents)。改写 pc.cents → 全模式(L1/L2/V)自动生效。 */
    if (fl.has_Mp && fl.Mp > 0.0) {
        int need = (int)((u.length / 1000.0) / pc.dt) + 4, i2, w2;
        int n0 = pc.n;
        double* nc = (double*)calloc(need, sizeof(double));
        double* wn = (double*)malloc(sizeof(double) * need);
        int half = (int)(0.05 / pc.dt); if (half < 1) half = 1;
        double amp = clampd(fl.Mp, 0, 100) / 100.0 * 25.0;
        srand(12345);   /* 固定种子:同一音符可重渲一致 */
        for (i2 = 0; i2 < need; ++i2) wn[i2] = (2.0 * rand() / RAND_MAX - 1.0);
        for (i2 = 0; i2 < need; ++i2) {
            double s = 0; int c = 0;
            for (w2 = -half; w2 <= half; ++w2) { int q = i2 + w2;
                if (q < 0 || q >= need) continue; s += wn[q]; ++c; }
            nc[i2] = (pc.cents && i2 < n0 ? pc.cents[i2] : (pc.cents && n0 > 0 ? pc.cents[n0-1] : 0.0))
                   + amp * (c ? s / c : 0.0) * sqrt((double)(2*half+1));
        }
        free(wn); free(pc.cents); pc.cents = nc; pc.n = need;
    }

    /* u(mo直出)：跳过一切修改。原区段 [offset,end] 直接输出(×音量,截/补到 length)。 */
    if (fl.has_u) {
        int ufs = 0, unbit = 0, unx = 0;
        FP_TYPE* ux = wavread(u.in_file, &ufs, &unbit, &unx);
        if (ux && unx > 0) {
            wcs_to_44100(&ux, &unx, &ufs, "utau");
            double total_ms = 1000.0 * unx / ufs;
            double end_ms = (u.cutoff >= 0) ? (total_ms - u.cutoff) : (u.offset - u.cutoff);
            if (end_ms > total_ms) end_ms = total_ms;
            if (end_ms < u.offset + 1) end_ms = u.offset + 1;
            int i0 = (int)(u.offset / 1000.0 * ufs), i1 = (int)(end_ms / 1000.0 * ufs);
            int nout = (int)(u.length / 1000.0 * ufs), i3;
            if (nout < 1) nout = i1 - i0;
            FP_TYPE* uy = (FP_TYPE*)calloc(nout > 0 ? nout : 1, sizeof(FP_TYPE));
            double vg = (u.volume > 0 ? u.volume : 100.0) / 100.0;
            for (i3 = 0; i3 < nout && (i0 + i3) < i1 && (i0 + i3) < unx; ++i3)
                uy[i3] = (FP_TYPE)clampd(ux[i0 + i3] * vg, -0.99, 0.99);
            wavwrite(uy, nout, ufs, 16, (char*)u.out_file);
            fprintf(stderr, "[utau] u: direct output (%d smp)\n", nout);
            free(uy); free(ux);
            return 0;
        }
        free(ux);
    }

    /* --- 合成模式选择 ---
       L1 → LLSM1 后端；L2 → 纯 L2；V1 → L2谐波+L1噪声；V2 → L1谐波+L2噪声。
       缺省(什么模式flag都不填) = V2(当前最佳)。synth_mode: -1=L1后端 / 0=纯L2 / 1=V1 / 2=V2。 */
    int Lval = fl.has_L ? (int)(fl.L + 0.5) : 0;
    int synth_mode;
    if (Lval == 1)          synth_mode = -1;                 /* L1 */
    else if (fl.has_M && (int)(fl.M+0.5)==1)
                            synth_mode = -3;                 /* M1=麦乐迪式(默认smooth,CR切crisp) */
    else if (fl.has_V)      synth_mode = (int)(fl.V + 0.5);  /* V1 / V2 显式 */
    else if (Lval == 2)     synth_mode = 0;                  /* L2 纯 */
    else                    synth_mode = 2;                  /* 缺省 = V2 */
    if (synth_mode != 1 && synth_mode != 2 && synth_mode != -1 && synth_mode != -3) synth_mode = 0;

    /* --- L3：Melodyne式(Local Sound Synthesis/PSOLA)时域内核,无模型直接周期重排。 --- */
    if (synth_mode == -3) return l3_render(&u, &pc, &fl);

    /* 共通编辑量(L1 内核吃的全套：批次1 g/Mo/ME/Mr + 批次2 Mt/MH/A/bh/Mb/b) */
    L1Edits edc;
    edc.has_g=fl.has_g;   edc.g=fl.g;
    edc.has_Mo=fl.has_Mo; edc.Mo=fl.Mo;
    edc.has_ME=fl.has_ME; edc.ME=fl.ME;
    edc.has_Mr=fl.has_Mr; edc.Mr=fl.Mr;
    edc.has_Mt=fl.has_Mt; edc.Mt=fl.Mt;
    edc.has_MH=fl.has_MH; edc.MH=fl.MH;
    edc.has_A =fl.has_A;  edc.A =fl.A;
    edc.has_bh=fl.has_bh; edc.bh=fl.bh;
    edc.has_Mb=fl.has_Mb; edc.Mb=fl.Mb;
    edc.has_Ab=fl.has_Ab; edc.Ab=fl.Ab;
    edc.has_NA=fl.has_NA; edc.NA=fl.NA;
    edc.has_RG=fl.has_RG; edc.RG=fl.RG;
    edc.has_DN=fl.has_DN; edc.DN=fl.DN;
    edc.has_b =fl.has_b;  edc.b =fl.b;
    edc.has_Mf=fl.has_Mf; edc.Mf=fl.Mf;
    edc.dn_print_n = 0; edc.dn_print_fnyq = 0.0; edc.dn_nsnr_db = 99.0;   /* 层② 印记:V2/V1 在 L2 分析后填入 */
    edc.tm_n = 0; edc.tm_fnyq = 0.0;   /* TM 校正:L2 分析后按 bank 档案填入 */

    /* --- L1 后端：取分量→共享后效(MC/MG/MD)→P/音量归一→写。逐输出帧跟随音高曲线。 --- */
    if (synth_mode == -1) {
        FP_TYPE *l1s = NULL, *l1n = NULL; int l1ny = 0, l1fs = 0, i2;
        fprintf(stderr, "[utau] LLSM1 backend (L%g): base=%.1fHz len=%.0fms pb=%s\n",
                fl.L, midi_to_hz(pc.base_midi), u.length, (pc.cents && pc.n) ? "on" : "flat");
        /* TM 音色对齐:校正计算与内核无关,提前算好经 edc 传入(L1 侧应用到 vt 包络+噪声谱) */
        if (fl.has_TM && fl.TM > 0.0) {
            double sm[TM_NB]; int b2;
            int force1 = (strchr(u.flags ? u.flags : "", 'G') != NULL);
            if (tm_compute_corr(u.in_file, force1, clampd(fl.TM, 0, 100) / 100.0, sm)) {
                edc.tm_n = TM_NB; edc.tm_fnyq = 22050.0;
                for (b2 = 0; b2 < TM_NB; ++b2) edc.tm_corr_db[b2] = sm[b2];
            }
        }
        int mj_cap = (fl.has_Mj && fl.Mj >= 1 && !fl.has_HF) ? (int)clampd(fl.Mj, 1, 6) : 0;
        if (llsm1_render(u.in_file, "", u.length, u.consonant, u.velocity, u.volume,
                         u.offset, u.cutoff, &edc, l1_f0_cb, &pc, &l1s, &l1n, &l1ny, &l1fs, mj_cap) != 0
            || !l1s || !l1n || l1ny <= 0) { fprintf(stderr, "[utau] L1 render failed\n"); return 1; }
        for (i2 = 0; i2 < l1ny; ++i2) l1s[i2] = (FP_TYPE)(l1s[i2] + l1n[i2]);   /* 混音 */
        apply_growl_fx(l1s, l1ny, l1fs, &pc, &fl);                              /* MC/MG/MD */
        {   /* P(默认100)+音量+限幅(与 L1 内部原逻辑一致,P 可调) */
            double peak = 1e-9, Pv = fl.has_P ? clampd(fl.P, 0, 100) : 100.0;
            for (i2 = 0; i2 < l1ny; ++i2) { double a = fabs(l1s[i2]); if (a > peak) peak = a; }
            double vs = pow(0.5 / peak, Pv / 100.0) * (u.volume > 0 ? u.volume : 100.0) / 100.0;
            for (i2 = 0; i2 < l1ny; ++i2) { double v = l1s[i2] * vs;
                if (v > 0.99) v = 0.99; if (v < -0.99) v = -0.99; l1s[i2] = (FP_TYPE)v; }
        }
        if (fl.has_SK) for (i2 = 0; i2 < l1ny; ++i2) l1s[i2] = -l1s[i2];
        wavwrite(l1s, l1ny, l1fs, 16, (char*)u.out_file);
        free(l1s); free(l1n);
        return 0;
    }

    /* --- 1~4. 取得 tolayer1 后的 L1 chunk：优先命中 .llsm 缓存，否则分析并写缓存 ---
       缓存路径 = <in_file>.llsm（仿 moresampler）。'G' flag 或 wav 比缓存新则强制重分析。 */
    int fs = 0, nfrm = 0, from_cache = 0;
    llsm_chunk* chunk = NULL;
    char cache_path[1100];
    int force = (strchr(u.flags ? u.flags : "", 'G') != NULL);
    snprintf(cache_path, sizeof(cache_path), "%s.llsm2", u.in_file);  /* .llsm2:避免与 mo 的 .llsm 缓存互相覆盖 */

    if (!force &&
        file_mtime(cache_path) >= 0 &&
        file_mtime(cache_path) >= file_mtime(u.in_file)) {
        chunk = llsm_blob_load(cache_path);
        if (chunk) from_cache = 1;
        if (chunk) {   /* 统一 44.1k 之前生成的异采样率旧缓存作废,强制重分析 */
            FP_TYPE* pfq = (FP_TYPE*)llsm_container_get(chunk->conf, LLSM_CONF_FNYQ);
            int cfs = pfq ? (int)(2.0 * (*pfq) + 0.5) : 0;
            if (cfs != 44100) {
                fprintf(stderr, "[utau] cache fs=%d != 44100, re-analyze\n", cfs);
                llsm_delete_chunk(chunk); chunk = NULL; from_cache = 0;
            }
        }
    }
    if (!chunk) {
        int nbit = 0, nx = 0;
        FP_TYPE* x = wavread(u.in_file, &fs, &nbit, &nx);
        if (!x || nx <= 0) { fprintf(stderr, "[utau] cannot read %s\n", u.in_file); return 1; }
        wcs_to_44100(&x, &nx, &fs, "utau");
        pyin_config pcfg = pyin_init(nhop);
        pcfg.fmin = 50.0; pcfg.fmax = 800.0; pcfg.trange = 24; pcfg.bias = 2;
        pcfg.nf = (int)(fs * 0.025);
        FP_TYPE* f0 = pyin_analyze(pcfg, x, nx, fs, &nfrm);
        if (fl.has_DN && fl.DN > 0 && f0 && nfrm > 4) { int fx = f0_sanitize(f0, nfrm);
            if (fx > 0) fprintf(stderr, "[L2] f0fix: %d corrections\n", fx); }
        if (!f0 || nfrm <= 0) { fprintf(stderr, "[utau] pYIN failed\n"); free(x); return 1; }
        llsm_aoptions* aopt = llsm_create_aoptions();
        aopt->thop = (FP_TYPE)nhop / fs;
        aopt->npsd = 1025; aopt->maxnhar = 2048; aopt->maxnhar_e = 5;
        aopt->hm_method = LLSM_AOPTION_HMPP;
        chunk = llsm_analyze(aopt, x, nx, (FP_TYPE)fs, f0, nfrm, NULL);
        free(x); free(f0);
        if (!chunk) { fprintf(stderr, "[utau] analyze failed\n"); llsm_delete_aoptions(aopt); return 1; }
        llsm_chunk_tolayer1(chunk, 2048);
        /* 写缓存（tolayer1 后快照），供后续调用命中 */
        if (llsm_blob_save(cache_path, chunk, aopt->thop, 2048, 5, LLSM_AOPTION_HMPP) == 0)
            fprintf(stderr, "[utau] cached -> %s\n", cache_path);
        llsm_delete_aoptions(aopt);
    }

    /* 统一从 conf 取 fs / thop / nfrm（缓存与分析两条路径一致） */
    {
        FP_TYPE* pfnyq = (FP_TYPE*)llsm_container_get(chunk->conf, LLSM_CONF_FNYQ);
        FP_TYPE* pthop = (FP_TYPE*)llsm_container_get(chunk->conf, LLSM_CONF_THOP);
        fs   = pfnyq ? (int)(2.0 * (*pfnyq) + 0.5) : 44100;
        nfrm = blob_cgi(chunk->conf, LLSM_CONF_NFRM, nfrm);
        { double thop_sec = pthop ? (double)*pthop : (double)nhop / fs;
          nhop = (int)(thop_sec * fs + 0.5); }
    }
    fprintf(stderr, "[utau] %s  nfrm=%d fs=%d nhop=%d\n",
            from_cache ? "cache HIT" : "analyzed", nfrm, fs, nhop);

    /* @DUMP 参数导出模式(供 Mi 蒸馏):把源 chunk 的【原始分析】参数(f0/psd含PSDRES折回/vt_db)
       写二进制到 out_file,不合成、不启声码器。与 Mi 施加同域(源帧 VTMAGN+psd),差量才对易。
       @DUMP 是非字母前缀,parse_flags 自动忽略;仅此处 strstr 触发。 */
    if (u.flags && strstr(u.flags, "@DUMP")) {
        FILE* df = fopen(u.out_file, "wb");
        if (!df) { fprintf(stderr, "[dump] cannot write %s\n", u.out_file); return 1; }
        int nspec0 = 0, npsd0 = 0, i2, dj;
        for (dj = 0; dj < nfrm; ++dj) {
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_VTMAGN);
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_NM);
            if (vt && !nspec0) nspec0 = llsm_fparray_length(vt);
            if (nm && nm->psd && !npsd0) npsd0 = nm->npsd;
        }
        { int hdr[4]; float ft[2];
          hdr[0] = 0x504C4C57; hdr[1] = nfrm; hdr[2] = npsd0; hdr[3] = nspec0;
          ft[0] = (float)(1000.0 * nhop / fs); ft[1] = (float)(0.5 * fs);
          fwrite(hdr, 4, 4, df); fwrite(ft, 4, 2, df); }
        for (dj = 0; dj < nfrm; ++dj) {
            FP_TYPE* pf = (FP_TYPE*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_F0);
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_NM);
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_VTMAGN);
            FP_TYPE* res = (FP_TYPE*)llsm_container_get(chunk->frames[dj], LLSM_FRAME_PSDRES);
            int nres = res ? llsm_fparray_length(res) : 0;
            float f0v = (pf && *pf > 0) ? (float)*pf : 0.0f;
            fwrite(&f0v, 4, 1, df);
            for (i2 = 0; i2 < npsd0; ++i2) {   /* psd + PSDRES 折回(与 HF 导出/合成一致) */
                double v = (nm && nm->psd && i2 < nm->npsd) ? (double)nm->psd[i2] : -200.0;
                if (res && i2 < nres) v += (double)res[i2] - (0.375 / 2.3025851 * 10.0);
                float fv = (float)v; fwrite(&fv, 4, 1, df);
            }
            for (i2 = 0; i2 < nspec0; ++i2) {
                float fv = (vt && i2 < llsm_fparray_length(vt)) ? (float)vt[i2] : -200.0f;
                fwrite(&fv, 4, 1, df);
            }
        }
        fclose(df);
        fprintf(stderr, "[dump] %d frames (npsd=%d nspec=%d) -> %s\n", nfrm, npsd0, nspec0, u.out_file);
        llsm_delete_chunk(chunk);
        return 0;
    }

    llsm_soptions* sopt = llsm_create_soptions((FP_TYPE)fs);

    /* BX 悬崖检测:在 DN 谱减之前对原始谱测(DN 会把噪声整体压低、削浅悬崖测度) */
    int bx_ic = -1; double bx_drop = 0.0;
    if (fl.has_BX && fl.BX > 0.0)
        bx_ic = bx_detect_cutoff(chunk, nfrm, 0.5 * fs, &bx_drop);

    /* DN 层②：嘶声印记降噪(静音帧提常驻底噪谱,全帧只减这个底,保气声)。层①=f0 修复在分析处。
       放在 Mz 之前:印记从原始谱估计;两者可叠加。 */
    if (fl.has_DN && fl.DN > 0.0)
        hiss_print_reduce_chunk(chunk, nfrm, clampd(fl.DN, 0, 100), 0.5 * fs, &edc);

    /* BX 频宽扩展填充:放在 DN 之后(先去底噪再补带,填充电平跟随已清理的谱)。无悬崖自动跳过。 */
    if (fl.has_BX && fl.BX > 0.0) {
        if (bx_ic >= 0)
            bandwidth_extend_chunk(chunk, nfrm, clampd(fl.BX, 0, 100), 0.5 * fs, bx_ic, bx_drop);
        else
            fprintf(stderr, "[BX] no codec cutoff detected (max drop %.1f dB), skip\n", bx_drop);
    }

    /* TM 音色对齐:批次染色校正 EQ 施加到 VTMAGN+psd,并经 edc 传给 L1 侧(V2 谐波+L1 噪声谱)。
       放最后:在干净化(DN)+补带(BX)之后校形状。 */
    if (fl.has_TM && fl.TM > 0.0) {
        double sm[TM_NB];
        if (tm_compute_corr(u.in_file, force, clampd(fl.TM, 0, 100) / 100.0, sm)) {
            int j5, i5, b;
            for (j5 = 0; j5 < nfrm; ++j5) {
                FP_TYPE* vt = (FP_TYPE*)llsm_container_get(chunk->frames[j5], LLSM_FRAME_VTMAGN);
                llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(chunk->frames[j5], LLSM_FRAME_NM);
                if (vt) { int nv5 = llsm_fparray_length(vt);
                    for (i5 = 0; i5 < nv5; ++i5)
                        vt[i5] += (FP_TYPE)dn_interp_db_(sm, TM_NB, 0.5 * fs, (double)i5 / (nv5 - 1) * 0.5 * fs); }
                if (nm && nm->psd) { int np5 = nm->npsd;
                    for (i5 = 0; i5 < np5; ++i5)
                        nm->psd[i5] += (FP_TYPE)dn_interp_db_(sm, TM_NB, 0.5 * fs, (double)i5 / (np5 - 1) * 0.5 * fs); }
            }
            edc.tm_n = TM_NB; edc.tm_fnyq = 0.5 * fs;
            for (b = 0; b < TM_NB; ++b) edc.tm_corr_db[b] = sm[b];
        }
    }

    /* Mi 神经蒸馏预设:把 <wav>.nfx 逐帧 64 带差量按强度叠加到源 chunk 的 VTMAGN+psd。
       放在 DN/BX/TM 之后:预设是"原始→修复"的差量,叠在引擎自身修复之上(可叠加/独立)。 */
    if (fl.has_Mi && fl.Mi > 0.0)
        apply_nfx_chunk(chunk, nfrm, u.in_file, clampd(fl.Mi, 0, 100) / 100.0, 1000.0 * nhop / fs);

    /* 降噪（Mz>0）：在伸缩前对源 chunk 的噪声谱做谱减，扣背景底噪 → 更干净 */
    if (fl.has_Mz && fl.Mz > 0.0)
        noise_reduce_chunk(chunk, nfrm, clampd(fl.Mz, 0, 100));

    /* 子谐波抑制（去糊/清洁，**默认即生效**）：仅对有声帧压噪声谱 → 拉高 HNR。
       对齐 mo 的 analysis-suppress-subharmonics 默认 ON：未填 Mk 时用默认 25(≈mo 出厂清洁度)，
       Mk0 显式关闭(还原最自然/略糊)，Mk50+ 更干净。实测默认 25 把 HNR 39.6→~44≈mo(43.1)。 */
    {
        double mk_amt = fl.has_Mk ? clampd(fl.Mk, 0, 100) : 25.0;
        if (mk_amt > 0.0)
            suppress_subharmonics_chunk(chunk, nfrm, mk_amt, 0.5 * fs);
    }

    /* 抗失真（Ma>0）：VTMAGN 频域平滑，抹高频伪峰 → 减变调发尖/发毛 */
    if (fl.has_Ma && fl.Ma > 0.0)
        anti_distortion_chunk(chunk, nfrm, clampd(fl.Ma, 0, 100));

    /* 解相位（与非缓存流程一致：tolayer1 后 → phasepropagate(-1)） */
    llsm_chunk_phasepropagate(chunk, -1);

    /* Mv 源侧相位正则化（在【源帧】上做，拉伸无关）：解相位后 VSPHSE 已是相对相位域，
       此处对真实相邻周期做圆周平滑，再交给插值/拉伸。修复"输出侧 Mv 在重拉伸下失效"——
       重拉伸时相邻输出帧来自同一源帧对(已是平滑斜坡)，输出侧平滑无物可平。源侧平滑覆盖全程。 */
    if (fl.has_Mv && fl.Mv > 0.0)
        smooth_vsphse_chunk(chunk, nfrm, 7, clampd(fl.Mv, 0, 100) / 100.0);

    /* Mg 噪声去颗粒：同样在源侧(真实帧)对噪声模型沿时间平滑，去帧间抖动=去颗粒/沙。拉伸无关。 */
    if (fl.has_Mg && fl.Mg > 0.0)
        smooth_noise_chunk(chunk, nfrm, 7, clampd(fl.Mg, 0, 100) / 100.0);

    /* Mw(原Md) 噪声频率constant-Q平滑：值=等效带数。同时下面会跳过 PSDRES 复制(丢残差波动)。 */
    if (fl.has_Mw && fl.Mw >= 1.0)
        band_reduce_noise_chunk(chunk, nfrm, (int)(clampd(fl.Mw, 1, 256) + 0.5));

    /* --- 5. 建伸缩后的新 chunk --- */
    double thop_ms = 1000.0 * nhop / fs;
    int n_out = 0;
    /* e=强制拉伸(优先) / l=循环 / 默认拉伸 */
    int loop_mode = (!fl.has_e && fl.has_l);

    /* Mj 转音平滑:快速 pitchbend 下,谐波合成的"每帧恒定频率阶梯"在帧重叠处失去相位相干,
       中高次谐波产生断点/沙哑(误差 ∝ (k+1)·(df0/dt)·thop²)。加密合成帧(thop/F)→误差降 1/F²。
       自动因子:扫音高曲线取峰值 半音/hop,F=ceil(st/0.4),钳到 [1, Mj值(上限,默认4)]。
       普通音符(平缓音高)F=1 不掉速;HF(神经声码器无此阶梯问题)不介入。 */
    int mjF = 1;
    if (fl.has_Mj && fl.Mj >= 1 && !fl.has_HF) {
        int cap = (int)clampd(fl.Mj, 1, 6);
        double dt = thop_ms / 1000.0, maxst = 0.0, t;
        for (t = 0.0; t < u.length / 1000.0; t += dt) {
            double a = pitchcurve_f0(&pc, t), b = pitchcurve_f0(&pc, t + dt);
            if (a > 50 && b > 50) { double st = fabs(12.0 * log2(b / a)); if (st > maxst) maxst = st; }
        }
        mjF = (int)ceil(maxst / 0.4);
        if (mjF < 1) mjF = 1;
        if (mjF > cap) mjF = cap;
        if (mjF > 1) fprintf(stderr, "[Mj] transition smoothing: %dx frames (peak %.2f st/hop, cap %d)\n",
                             mjF, maxst, cap);
    }
    double thop_out_ms = thop_ms / mjF;

    double* remap = build_remap(nfrm, thop_out_ms, thop_ms, &u, loop_mode, &n_out);

    /* 辅音固定区在输出时间轴上的边界(ms)，供 bh 判定（与 build_remap 内一致） */
    double con_out_ms = u.consonant * pow(2.0, 1.0 - u.velocity / 100.0);
    if (con_out_ms > u.length) con_out_ms = u.length;

    llsm_container* conf_new = llsm_copy_container(chunk->conf);
    llsm_container_attach(conf_new, LLSM_CONF_NFRM, llsm_create_int(n_out),
        llsm_delete_int, llsm_copy_int);
    /* Mj 加密:输出帧步进变细 → conf 的 THOP 必须同步设为细 hop,
       否则 phasepropagate(cumsum(f0)*thop) 与合成器(nwin=2·thop、baseidx=i·thop·fs)会按粗 hop 错算。 */
    if (mjF > 1)
        llsm_container_attach(conf_new, LLSM_CONF_THOP,
            llsm_create_fp((FP_TYPE)(thop_out_ms / 1000.0)), llsm_delete_fp, llsm_copy_fp);
    llsm_chunk* out_chunk = llsm_create_chunk(conf_new, 0);
    llsm_delete_container(conf_new);

    int j;

    /* --- A flag 预备：振幅↔音高联动（颤音增强）---------------------------------
       原理：真实歌声的颤音不仅是音高摆动，还伴随同步的幅度起伏。这里先按各输出帧时刻
       采样目标 log2(F0)，再做移动平均得到"基线"(滤掉颤音振荡、只留音符主音高与缓慢滑音)，
       每帧相对基线的偏离量(半音) = 颤音瞬时分量。主循环里据此调制该帧 VTMAGN 电平。
       用"偏离基线"而非"偏离音符"，可只联动颤音/抖动、不被整体滑音/弯音误触发。 */
    double* amp_dev_st = NULL;
    if (fl.has_A && fl.A != 0.0) {
        double* lf = (double*)malloc(sizeof(double) * n_out);
        double* sm = (double*)malloc(sizeof(double) * n_out);
        int half = (int)(0.18 / (thop_out_ms / 1000.0));   /* 基线半窗 ~180ms（>1 个颤音周期） */
        int w;
        amp_dev_st = (double*)malloc(sizeof(double) * n_out);
        if (half < 1) half = 1;
        for (j = 0; j < n_out; ++j) {
            double f0 = pitchcurve_f0(&pc, j * (thop_out_ms / 1000.0));
            lf[j] = (f0 > 0) ? log2(f0) : 0.0;
        }
        for (j = 0; j < n_out; ++j) {
            double s = 0; int c = 0;
            for (w = -half; w <= half; ++w) { int k = j + w;
                if (k < 0 || k >= n_out) continue; s += lf[k]; ++c; }
            sm[j] = c > 0 ? s / c : lf[j];
            amp_dev_st[j] = 12.0 * (lf[j] - sm[j]);    /* 半音偏离 */
        }
        free(lf); free(sm);
    }

    for (j = 0; j < n_out; ++j) {
        double s = remap[j];
        int base = (int)floor(s);
        if (base < 0) base = 0;
        if (base > nfrm - 2) base = nfrm - 2;
        if (base < 0) base = 0;
        double ratio = s - (int)floor(s);
        int nxt = base + 1; if (nxt > nfrm - 1) nxt = nfrm - 1;

        out_chunk->frames[j] = llsm_copy_container(chunk->frames[base]);
        interp_llsm_frame(out_chunk->frames[j], chunk->frames[nxt], (FP_TYPE)ratio);

        /* 覆盖 F0 = UTAU 音高曲线（仅有声帧），并记录上变调量供高频滚降 */
        FP_TYPE* pf0 = (FP_TYPE*)llsm_container_get(out_chunk->frames[j], LLSM_FRAME_F0);
        double shift_st = 0.0;
        if (pf0 && *pf0 > 0) {
            double src_f0 = *pf0;                   /* 覆盖前=插值后的源 f0 */
            double t_out = j * (thop_out_ms / 1000.0);
            double tgt_f0 = pitchcurve_f0(&pc, t_out);
            *pf0 = (FP_TYPE)tgt_f0;
            if (src_f0 > 0 && tgt_f0 > 0) shift_st = 12.0 * log2(tgt_f0 / src_f0);
        }

        /* flag 编辑 */
        apply_flags_to_frame(out_chunk->frames[j], &fl, 0.5 * fs);

        /* Mf 共振峰调谐(共享 mf_formant_tune,dB域)：上移=F1对齐最近谐波;下移>2st=去轰+变暗(用 shift_st)。 */
        if (fl.has_Mf && fl.Mf > 0.0 && pf0 && *pf0 > 0) {
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(out_chunk->frames[j], LLSM_FRAME_VTMAGN);
            if (vt) mf_formant_tune(vt, llsm_fparray_length(vt), 0.5 * fs, (double)*pf0,
                                    clampd(fl.Mf, 0, 100) / 100.0, 1.0, shift_st);
        }

        /* 自适应高频滚降（去高音刺耳）；MH flag 缩放强度，MH<=-100 关闭 */
        {
            double coef = HF_TILT_COEF * (1.0 + clampd(fl.MH, -100, 100) / 100.0);
            if (coef > 0.0) apply_hf_tilt(out_chunk->frames[j], shift_st, coef, 0.5 * fs);
        }

        /* Mq：始终生效的高次谐波滚降(不依赖变调)。>3kHz 按 tilt dB/oct 压低，
           对标 mo 对顶部谐波(>5kHz)的强滚降 → 去掉我们多出的高频"沙/刺/毛"。 */
        if (fl.has_Mq && fl.Mq > 0.0) {
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(out_chunk->frames[j], LLSM_FRAME_VTMAGN);
            if (vt) {
                int n = llsm_fparray_length(vt), ii;
                double tilt = -(clampd(fl.Mq, 0, 100) / 100.0) * 10.0;
                for (ii = 0; ii < n; ++ii) {
                    double fr = (double)ii / (n - 1) * 0.5 * fs;
                    if (fr > 3000.0) {
                        vt[ii] += (FP_TYPE)(tilt * log2(fr / 3000.0));
                        if (vt[ii] < -80) vt[ii] = -80;
                    }
                }
            }
        }

        /* A：振幅↔音高联动（仅有声帧）。偏离基线 1 半音、A=100 → ±3dB；A<0 反相。
           gdb 钳到 ±12dB 防极端弯音爆音。叠加到 VTMAGN(整体抬/压该帧谐波电平)。 */
        if (amp_dev_st && pf0 && *pf0 > 0) {
            double gdb = clampd((clampd(fl.A, -100, 100) / 100.0) * 3.0 * amp_dev_st[j], -12.0, 12.0);
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(out_chunk->frames[j], LLSM_FRAME_VTMAGN);
            if (vt) { int n = llsm_fparray_length(vt), ii;
                for (ii = 0; ii < n; ++ii) { vt[ii] += (FP_TYPE)gdb; if (vt[ii] < -80) vt[ii] = -80; } }
        }

        /* bh：辅音区"谐波部分"响度。仅辅音固定区(t_out<con_out_ms)内的有声帧，缩放 VTMAGN
           (dB幅度) += 20log10(gain)，gain=1+0.05*bh。与 b 互补：b=清辅音噪声、bh=(浊)辅音谐波。
           gain~0 时(bh=-20)谐波趋静音。仅作用辅音区，不碰后面的元音稳态。 */
        if (fl.has_bh && fl.bh != 0.0 && pf0 && *pf0 > 0 && (j * thop_out_ms) < con_out_ms) {
            double gain = 1.0 + 0.05 * clampd(fl.bh, -20, 100);
            double off  = (gain > 1e-4) ? 20.0 * log10(gain) : -120.0;
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(out_chunk->frames[j], LLSM_FRAME_VTMAGN);
            if (vt) { int n = llsm_fparray_length(vt), ii;
                for (ii = 0; ii < n; ++ii) { vt[ii] += (FP_TYPE)off; if (vt[ii] < -80) vt[ii] = -80; } }
        }

        /* 复制残差 PSDRES：
           仅【循环模式】才用 ±2 帧随机抖动去相关（防循环机械感）；
           非循环时取相干的 base 帧——逐帧随机抽换会破坏噪声时间连贯性，听感"湿/有痰"。 */
        int residx = base;
        if (loop_mode) {
            residx = base + (rand() % 5 - 2);
            if (residx < 0) residx = 0; if (residx > nfrm - 1) residx = nfrm - 1;
        }
        /* Mw 频率平滑时丢弃 PSDRES（残差波动=颗粒），噪声只剩平滑轮廓，根除颗粒（仿 mo 无残差）。 */
        if (!(fl.has_Mw && fl.Mw >= 1.0)) {
            FP_TYPE* resvec = llsm_container_get(chunk->frames[residx], LLSM_FRAME_PSDRES);
            if (resvec)
                llsm_container_attach(out_chunk->frames[j], LLSM_FRAME_PSDRES,
                    llsm_copy_fparray(resvec), llsm_delete_fparray, llsm_copy_fparray);
        }
    }

    /* --- 5.5 参数时间平滑（MS>0）：对 out_chunk 的 Rd/VTMAGN 做几帧滑动平均，
           去掉 LLSM2 忠实还原的逐帧微抖动 → 更硬/更干净（对标 mo 的规整化）。 --- */
    if (fl.has_MS && fl.MS > 0.0)
        smooth_chunk_params(out_chunk, n_out, 5, clampd(fl.MS, 0, 100) / 100.0);

    /* Mv：源相位时间正则化已移到【源侧】(phasepropagate(-1) 后，见上)。原"输出侧平滑"在重拉伸下
       失效(相邻输出帧来自同一源帧对=已平滑斜坡,无物可平)，故移除。源侧做一次=全程一致正则化。 */

    /* LD：降维(倒谱低阶化)。在 tolayer0 前对 VTMAGN 低阶平滑→类 LLSM1 低维包络。可开关。 */
    if (fl.has_LD && fl.LD > 0.0)
        liften_vtmagn_chunk(out_chunk, n_out, clampd(fl.LD, 0, 100));

    /* PB：Pulse-by-Pulse 合成路径(可开关)。每个有声帧的脉冲由 LF 声门模型(Rd)在频域生成、经声道
       VTMAGN 滤波后 OLA 叠加(类 WORLD)——脉冲形状来自声门模型而非 min-phase 重建，可能更圆/更干净。
       需 use_l1=1（L0 路径不含 PbP），并给每有声帧挂 PBPSYN=1（库内自动在谐波/PbP 间交叉淡入）。 */
    if (fl.has_PB) {
        int jj;
        sopt->use_l1 = 1;
        for (jj = 0; jj < n_out; ++jj) {
            FP_TYPE* pf = (FP_TYPE*)llsm_container_get(out_chunk->frames[jj], LLSM_FRAME_F0);
            if (pf && *pf > 0)
                llsm_container_attach(out_chunk->frames[jj], LLSM_FRAME_PBPSYN,
                    llsm_create_int(1), llsm_delete_int, llsm_copy_int);
        }
    }

    /* --- 6. 回 L0 + 重传播相位 --- */
    llsm_chunk_tolayer0(out_chunk);
    llsm_chunk_phasepropagate(out_chunk, 1);

    /* --- HF：神经声码器路线。把【编辑后】的逐帧参数导出为 legacy CSV(llsm_csv.py 格式)，
       交给外挂钩子 F:/UTAU/hf_backend/hf_render.bat <csv> <out.wav>(logmel_physics_v3 + NSF-HiFiGAN)。
       f0 列=含 pitchbend/颤音的目标曲线;全部 flag 编辑已在参数里。 --- */
    FP_TYPE* hfy = NULL; int hfn = 0, hffs = 0, hf_hybrid = 0, wnb0_ = 0;
    if (fl.has_HF) {
        char csvp[1100]; FILE* cf;
        snprintf(csvp, sizeof(csvp), "%s.llsm.csv", u.out_file);
        cf = fopen(csvp, "w");
        if (!cf) { fprintf(stderr, "[utau] HF: cannot write %s\n", csvp); return 1; }
        fprintf(cf, "frame,f0,vuv,nhar,npsd,ampl...,phse...,psd...,vt_db...\n");
        for (j = 0; j < n_out; ++j) {
            llsm_container* fr = out_chunk->frames[j];
            FP_TYPE* pf = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_F0);
            llsm_hmframe* hm = (llsm_hmframe*)llsm_container_get(fr, LLSM_FRAME_HM);
            llsm_nmframe* nm = (llsm_nmframe*)llsm_container_get(fr, LLSM_FRAME_NM);
            FP_TYPE* vt = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_VTMAGN);
            double f0v = (pf && *pf > 0) ? (double)*pf : 0.0;
            int nh = (f0v > 0 && hm) ? hm->nhar : 0;
            int np2 = (nm && nm->psd) ? nm->npsd : 0;
            int nv = vt ? llsm_fparray_length(vt) : 0, i2;
            fprintf(cf, "%d,%.6g,%d,%d,%d", j, f0v, f0v > 0 ? 1 : 0, nh, np2);
            for (i2 = 0; i2 < nh; ++i2) fprintf(cf, ",%.6g", (double)hm->ampl[i2]);
            for (i2 = 0; i2 < nh; ++i2) fputs(",0", cf);   /* phse: mel路径不需要,写0提速解析 */
            {   /* PSDRES 折回 psd(与库合成一致:psd+res-LOG2IN(LOGRESBIAS)),保住辅音瞬态细节 */
                FP_TYPE* res = (FP_TYPE*)llsm_container_get(fr, LLSM_FRAME_PSDRES);
                int nres = res ? llsm_fparray_length(res) : 0;
                for (i2 = 0; i2 < np2; ++i2) {
                    double v = (double)nm->psd[i2];
                    if (res && i2 < nres) v += (double)res[i2] - (0.375 / 2.3025851 * 10.0)   /* LOG2IN(LOGRESBIAS=0.375) */;
                    fprintf(cf, ",%.6g", v);
                }
            }
            for (i2 = 0; i2 < nv; ++i2) fprintf(cf, ",%.6g", (double)vt[i2]);
            fprintf(cf, "\n");
        }
        fclose(cf);
        {
            /* C 客户端直连 daemon(127.0.0.1:51765);连不上则拉起 hf_daemon.py 后重试(最长180s)。
               免去 python 客户端+cmd 启动(~0.3s/音符)。协议:"csv|out\n" -> "OK"。 */
            WSADATA wd; SOCKET sk = INVALID_SOCKET; int tries, ok = 0, spawned = 0;
            struct sockaddr_in ad; char msg[2300]; char rb[8];
            fprintf(stderr, "[utau] HF: params -> %s (%d frames)\n", csvp, n_out);
            WSAStartup(MAKEWORD(2,2), &wd);
            memset(&ad, 0, sizeof(ad)); ad.sin_family = AF_INET;
            ad.sin_port = htons(51765); ad.sin_addr.s_addr = inet_addr("127.0.0.1");
            snprintf(msg, sizeof(msg), "%s|%s\n", csvp, u.out_file);
            for (tries = 0; tries < 40 && !ok; ++tries) {   /* 40s:冷启约20s;无python时快速失败 */
                sk = socket(AF_INET, SOCK_STREAM, 0);
                if (connect(sk, (struct sockaddr*)&ad, sizeof(ad)) == 0) {
                    int rn; DWORD tmo = 300000;
                    setsockopt(sk, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof(tmo));
                    send(sk, msg, (int)strlen(msg), 0);
                    rn = recv(sk, rb, sizeof(rb) - 1, 0);
                    ok = (rn >= 2 && rb[0] == 'O' && rb[1] == 'K');
                    closesocket(sk);
                    if (ok) break;
                    fprintf(stderr, "[utau] HF daemon returned ERR\n");
                    WSACleanup(); return 1;
                }
                closesocket(sk);
                if (!spawned) {
                    /* 后端随 exe 定位:<exe目录>\\hf_backend\\hf_daemon.py(打包=引擎+声码器同目录) */
                    char exed[1040], dpy[1140], cmd2[1400]; struct stat sd;
                    GetModuleFileNameA(NULL, exed, sizeof(exed));
                    { char* p9 = strrchr(exed, '\\'); if (p9) *p9 = 0; }
                    snprintf(dpy, sizeof(dpy), "%s\\\\hf_backend\\\\hf_daemon.py", exed);
                    if (stat(dpy, &sd) != 0) {
                        fprintf(stderr, "[utau] HF: extension not installed (need %s)\n", dpy);
                        closesocket(sk); WSACleanup(); return 1;
                    }
                    snprintf(cmd2, sizeof(cmd2), "start /min \"HFdaemon\" pythonw \"%s\"", dpy);
                    system(cmd2);
                    fprintf(stderr, "[utau] HF: starting daemon (%s)...\n", dpy);
                    spawned = 1;
                }
                Sleep(1000);
            }
            WSACleanup();
            hf_hybrid = (ok && (int)(fl.HF + 0.5) >= 2);
            if (hf_hybrid) {
                hfy = wavread((char*)u.out_file, &hffs, &wnb0_, &hfn);
                if (!hfy || hfn <= 0 || hffs != fs) {
                    fprintf(stderr, "[utau] HF2: readback failed, fallback plain HF\n");
                    free(hfy); hfy = NULL; hf_hybrid = 0;
                } else fprintf(stderr, "[utau] HF2 hybrid: consonant=V2, vowel=HF\n");
            }
            if (!hf_hybrid) {
            /* HF 后效：daemon 的 wav 回读,补时域 flag(MC/MG/MD growl、SK)与 P/音量归一+限幅。
               谱/噪声类 flag(g/Mo/Mt/Mr/ME/Mf/Mb/b/bh/A/Mk/Mg/Mw/Md/LD 等)已在导出前烙进参数,无需处理。 */
            if (ok) {
                int wfs = 0, wnb = 0, wn = 0, i3;
                FP_TYPE* wy = wavread((char*)u.out_file, &wfs, &wnb, &wn);
                if (wy && wn > 0) {
                    apply_growl_fx(wy, wn, wfs, &pc, &fl);
                    if (fl.has_SK) for (i3 = 0; i3 < wn; ++i3) wy[i3] = -wy[i3];
                    {
                        double peak = 1e-9, Pv = fl.has_P ? clampd(fl.P, 0, 100) : 100.0;
                        double vsc;
                        for (i3 = 0; i3 < wn; ++i3) { double a2 = fabs((double)wy[i3]); if (a2 > peak) peak = a2; }
                        vsc = pow(0.5 / peak, Pv / 100.0) * (u.volume > 0 ? u.volume : 100.0) / 100.0;
                        for (i3 = 0; i3 < wn; ++i3) { double v2 = wy[i3] * vsc;
                            if (v2 > 0.99) v2 = 0.99; if (v2 < -0.99) v2 = -0.99; wy[i3] = (FP_TYPE)v2; }
                    }
                    wavwrite(wy, wn, wfs, 16, (char*)u.out_file);
                }
                free(wy);
            }
            if (!ok) { fprintf(stderr, "[utau] HF: daemon failed to start - is Python installed? (see README) CSV kept: %s\n", csvp); return 1; }
            return 0;
            }  /* end if(!hf_hybrid) */
        }
    }

    /* --- 7. 合成 --- */
    llsm_output* out = llsm_synthesize(sopt, out_chunk);
    if (!out || !out->y) { fprintf(stderr, "[utau] synth failed\n"); return 1; }

    /* --- V1/V2：跨库混合合成。V1 = L2 谐波 + L1 干净噪声(去颗粒,保 L2 谐波细节)；
           V2 = L1 谐波 + L2 噪声。都按 L1 自身的 谐波/噪声 能量比校准(L1 噪声作电平参考)。 --- */
    if ((synth_mode == 1 || synth_mode == 2 || (fl.has_Mm && synth_mode == 0))
        && out->y_sin && out->y_noise) {
        int mode = synth_mode;   /* 1=V1 / 2=V2 / 0+Mm=模型插值 */
        FP_TYPE *l1sin = NULL, *l1nos = NULL; int l1ny = 0, l1fs = 0;
        int want_sin = (mode == 2 || fl.has_Mm);
        /* V2=L1谐波:加密修转音断点;V1=L1噪声:加密无害。cap 与 L2 侧一致 */
        int mj_cap = (fl.has_Mj && fl.Mj >= 1 && !fl.has_HF) ? (int)clampd(fl.Mj, 1, 6) : 0;
        int rc = llsm1_render(u.in_file, "", u.length, u.consonant, u.velocity, u.volume,
                              u.offset, u.cutoff, &edc, l1_f0_cb, &pc,
                              want_sin ? &l1sin : NULL, &l1nos, &l1ny, &l1fs, mj_cap);
        if (rc == 0 && l1nos && l1fs == fs) {
            int M = (out->ny < l1ny) ? out->ny : l1ny, i;
            double sL2n = 0, sL1n = 0, sL2s = 0, sL1s = 0;
            for (i = 0; i < M; ++i) { sL2n += (double)out->y_noise[i]*out->y_noise[i];
                                      sL1n += (double)l1nos[i]*l1nos[i];
                                      sL2s += (double)out->y_sin[i]*out->y_sin[i];
                                      if (l1sin) sL1s += (double)l1sin[i]*l1sin[i]; }
            double rL2n = sqrt(sL2n/(M>0?M:1)), rL1n = sqrt(sL1n/(M>0?M:1));
            double rL2s = sqrt(sL2s/(M>0?M:1)), rL1s = l1sin ? sqrt(sL1s/(M>0?M:1)) : 0;
            if (fl.has_Mm && mode == 0 && l1sin) {
                /* Mm 模型插值(mo语义)：0=纯L1 ↔ 100=纯L2。分量级能量匹配后线性混合。 */
                double m = clampd(fl.Mm, 0, 100) / 100.0;
                double ss = (rL1s > 1e-9) ? rL2s/rL1s : 1.0;   /* L1谐波→L2谐波电平 */
                double sn = (rL1n > 1e-9) ? rL2n/rL1n : 1.0;
                for (i = 0; i < M; ++i)
                    out->y[i] = (FP_TYPE)( (1.0-m)*(ss*l1sin[i] + sn*l1nos[i])
                                          + m*(out->y_sin[i] + out->y_noise[i]) );
                fprintf(stderr, "[utau] Mm model interp: %.0f%% L2 (ss=%.3f sn=%.3f)\n", m*100, ss, sn);
            } else if (mode == 2 && l1sin) {
                double sc = (rL2n > 1e-9) ? rL1n/rL2n : 1.0;   /* L2 噪声匹配到 L1 噪声能量 */
                /* DN 层②后 L2 噪声已被谱减,但 rL1n(L1 噪声,未降噪)仍含全部嘶声——
                   匹配目标同步下修 sqrt(能量比),否则清掉的底噪会被拉回来 */
                sc *= sqrt(g_dn_hiss_ratio);
                for (i = 0; i < M; ++i) out->y[i] = l1sin[i] + (FP_TYPE)(sc * out->y_noise[i]);
                fprintf(stderr, "[utau] V2 hybrid: L1 harmonic + L2 noise (scale=%.3f, M=%d)\n", sc, M);
            } else {
                double sc = (rL1n > 1e-9) ? rL2n/rL1n : 1.0;   /* L1 噪声匹配到 L2 噪声能量 */
                for (i = 0; i < M; ++i) out->y[i] = out->y_sin[i] + (FP_TYPE)(sc * l1nos[i]);
                fprintf(stderr, "[utau] V1 hybrid: L2 harmonic + L1 noise (scale=%.3f, M=%d)\n", sc, M);
            }
        } else {
            fprintf(stderr, "[utau] V%d hybrid: L1 render failed (rc=%d fs=%d/%d), keep L2\n", mode, rc, l1fs, fs);
        }
        free(l1sin); free(l1nos);
    }

    /* --- Mc：噪声相干化（梳状滤波，周期=基音周期）。递归梳: ync[n]=(1-g)*noise[n]+g*ync[n-T]，
           传递函数在谐波频率处增益=1、谐波间陷波 → 噪声逐周期重复、锁到谐波 = mo 的相干"电流音"
           噪声(不爆音)。仅有声段(f0>0)生效；清辅音保持原样。T 分数延迟用线性插值。 --- */
    if (fl.has_Mc && fl.Mc > 0.0 && out->y_sin && out->y_noise && synth_mode == 0) {
        double g = clampd(fl.Mc, 0, 100) / 100.0 * 0.92;
        int N = out->ny, n;
        FP_TYPE* ync = (FP_TYPE*)calloc(N > 0 ? N : 1, sizeof(FP_TYPE));
        if (ync) {
            for (n = 0; n < N; ++n) {
                double f0n = pitchcurve_f0(&pc, (double)n / fs);
                double Tn  = (f0n > 20.0) ? (double)fs / f0n : 0.0;
                if (Tn >= 2.0 && (double)n - Tn >= 1.0) {
                    double idx = (double)n - Tn;
                    int i0 = (int)floor(idx); double a = idx - i0;
                    double delayed = (i0 >= 0 && i0 + 1 < N) ? ((1.0 - a) * ync[i0] + a * ync[i0 + 1]) : 0.0;
                    ync[n] = (FP_TYPE)((1.0 - g) * out->y_noise[n] + g * delayed);
                } else {
                    ync[n] = out->y_noise[n];
                }
            }
            for (n = 0; n < N; ++n) out->y[n] = out->y_sin[n] + ync[n];
            free(ync);
        }
    }

    /* mo 扩展特效 MC/MG/MD(粗糙/growl/失真)：最终混音上做,V/Mm 混合后 → 全模式一致。 */
    /* HF2 拼接 v3：①互相关相位对齐(±半周期最佳滞后,重叠区相干化,消梳状)
       ②RMS 匹配改在拼接点邻域(~3 窗)局部做——全段匹配时两侧能量包络(衰减形状)不同,
         单一系数在拼接点局部对不上=交接台阶;
       ③按重叠区实测相关系数 rho 做精确功率补偿:除以 sqrt(ga^2+gb^2+2*rho*ga*gb)。
         旧版等功率淡化(cos/sin)只对不相关信号成立,而①刚把两路对齐成高度相关,
         中点合成幅度≈1.414 倍=拼接中心 +3dB 隆起(用户反馈的异常响度)。
         rho=1 自动退化为等幅淡化,rho=0 退化为等功率,任意相关度下功率恒定。 */
    if (hf_hybrid && hfy) {
        int M2 = (out->ny < hfn) ? out->ny : hfn, i4;
        int ib = (int)((con_out_ms + 10.0) / 1000.0 * fs), nf = (int)(0.060 * fs);
        double f0b_ = pitchcurve_f0(&pc, (con_out_ms + 40.0) / 1000.0);
        int L4 = (f0b_ > 50) ? (int)(fs / f0b_ / 2) : (int)(fs * 0.002), best = 0; double bc = -1e30;   /* ±半周期:纯相位对齐,零时移 */
        double sA = 0, sB = 0, sAB = 0; int c4 = 0, im;
        if (ib < 0) ib = 0;
        if (ib < M2 - nf - L4 - 1 && M2 > nf + 2 * L4 + 16) {
            int lag, k4;
            for (lag = -L4; lag <= L4; ++lag) {
                double cc = 0;
                for (k4 = ib; k4 < ib + nf; ++k4) {
                    int q4 = k4 + lag;
                    if (q4 < 0 || q4 >= hfn) continue;
                    cc += (double)out->y[k4] * hfy[q4];
                }
                if (cc > bc) { bc = cc; best = lag; }
            }
            im = ib + 3 * nf; if (im > M2) im = M2;   /* 局部匹配窗:拼接点起 ~180ms */
            for (i4 = ib; i4 < im; ++i4) {
                int q4 = i4 + best;
                double h4 = (q4 >= 0 && q4 < hfn) ? hfy[q4] : 0.0;
                sA += (double)out->y[i4] * out->y[i4]; sB += h4 * h4;
                ++c4;
            }
            (void)sAB;
            if (c4 > 8 && sB > 1e-12) {
                double sc4 = sqrt(sA / sB);
                double shaped = 0.0;
                for (i4 = 0; i4 < M2; ++i4) {
                    int q4 = i4 + best;
                    double h4 = (q4 >= 0 && q4 < hfn) ? sc4 * hfy[q4] : 0.0;
                    double t4 = (double)(i4 - ib) / (nf > 0 ? nf : 1);
                    double w4 = (t4 <= 0) ? 0.0 : (t4 >= 1.0 ? 1.0 : t4);
                    double ga = cos(0.5 * 3.14159265358979 * w4);
                    double gb = sin(0.5 * 3.14159265358979 * w4);
                    out->y[i4] = (FP_TYPE)(ga * out->y[i4] + gb * h4);
                }
                {   /* 拼接区响度整形(测量驱动):淡化数学无论怎么选(等幅/等功率/rho补偿)都
                       挡不住两侧相关度逐点变化——隆起、凹陷、台阶都可能出现(用户实测反馈)。
                       直接测淡化窗内 5ms 粒度短时能量,校正到两端 30ms 锚点的 dB 线性过渡:
                       ±3dB 封顶、窗缘斜坡回零(不与窗外产生新台阶)、逐样本插值(防拉链)。 */
                    int hop5 = (int)(0.005 * fs), win20 = (int)(0.020 * fs), aw = (int)(0.030 * fs);
                    double eA = 0, eB = 0; int ca = 0, cb = 0, k5;
                    for (k5 = ib - aw; k5 < ib; ++k5) if (k5 >= 0) { eA += (double)out->y[k5] * out->y[k5]; ++ca; }
                    for (k5 = ib + nf; k5 < ib + nf + aw && k5 < M2; ++k5) { eB += (double)out->y[k5] * out->y[k5]; ++cb; }
                    if (ca > 16 && cb > 16 && eA > 1e-12 && eB > 1e-12 && hop5 > 0) {
                        double dbA = 10.0 * log10(eA / ca), dbB = 10.0 * log10(eB / cb);
                        double gh[20];
                        int nh = 0, t5, k6;
                        for (t5 = ib; t5 < ib + nf && nh < 20; t5 += hop5, ++nh) {
                            double e = 0, u5, tgt, act, gdb, edge;
                            int ce = 0;
                            gh[nh] = 0.0;
                            for (k6 = t5 - win20 / 2; k6 < t5 + win20 / 2; ++k6)
                                if (k6 >= 0 && k6 < M2) { e += (double)out->y[k6] * out->y[k6]; ++ce; }
                            if (ce < 16) continue;
                            u5 = (double)(t5 + hop5 / 2 - ib) / nf;
                            act = 10.0 * log10(e / ce + 1e-20);
                            tgt = dbA + (dbB - dbA) * u5;
                            gdb = tgt - act;
                            if (gdb > 3.0) gdb = 3.0;
                            if (gdb < -3.0) gdb = -3.0;
                            edge = (u5 < 0.2) ? u5 / 0.2 : (u5 > 0.8 ? (1.0 - u5) / 0.2 : 1.0);
                            if (edge < 0) edge = 0;
                            gh[nh] = gdb * edge;
                            if (fabs(gh[nh]) > shaped) shaped = fabs(gh[nh]);
                        }
                        if (nh >= 2) {
                            for (i4 = ib; i4 < ib + nf && i4 < M2; ++i4) {
                                double pos = (double)(i4 - ib) / hop5 - 0.5;
                                int i0 = (int)floor(pos);
                                double fr5 = pos - i0, g0, g1;
                                g0 = (i0 < 0) ? gh[0] : (i0 >= nh - 1 ? gh[nh - 1] : gh[i0]);
                                g1 = (i0 + 1 < 0) ? gh[0] : (i0 + 1 >= nh - 1 ? gh[nh - 1] : gh[i0 + 1]);
                                out->y[i4] = (FP_TYPE)(out->y[i4] * pow(10.0, (g0 * (1.0 - fr5) + g1 * fr5) / 20.0));
                            }
                        }
                    }
                }
                fprintf(stderr, "[utau] HF2 splice v3: lag=%d scale=%.2f shaped=%.1fdB fade=60ms\n", best, sc4, shaped);
            }
        }
        free(hfy); hfy = NULL;
    }

    apply_growl_fx(out->y, out->ny, fs, &pc, &fl);

    /* --- SK：极性反相（整段 ×-1），对齐 mo 的声门脉冲极性约定。极性人耳基本不可闻，
           主要让波形外观/下游处理与 mo 一致。在归一化前做，符号不影响后续峰值计算。 --- */
    if (fl.has_SK) {
        int i;
        for (i = 0; i < out->ny; ++i) out->y[i] = -out->y[i];
    }

    /* --- 8. P 峰值归一化(UTAU标准,默认100) + volume + p 最终峰值 + 安全限幅 → 写 wav --- */
    {
        int i;
        double peak, Pv, vscale;
        /* 1) P：把峰值往 0.5(半满刻度) 拉，指数强度 (0.5/peak)^(P/100)。P=0 关闭。 */
        Pv = fl.has_P ? clampd(fl.P, 0, 100) : 100.0;   /* 默认 100(完全归一) */
        if (Pv > 0.0) {
            peak = 0.0;
            for (i = 0; i < out->ny; ++i) { double a = fabs((double)out->y[i]); if (a > peak) peak = a; }
            if (peak > 1e-6) {
                double gain = pow(0.5 / peak, Pv / 100.0);
                for (i = 0; i < out->ny; ++i) out->y[i] *= (FP_TYPE)gain;
            }
        }
        /* 2) volume */
        vscale = u.volume / 100.0;
        for (i = 0; i < out->ny; ++i) out->y[i] *= (FP_TYPE)vscale;
        /* 3) p：最终峰值归一到 10^(-p/20)（可选）。p 越大越安静。 */
        if (fl.has_p) {
            double tgt = pow(10.0, -clampd(fl.p, -1, 6) / 20.0);
            peak = 0.0;
            for (i = 0; i < out->ny; ++i) { double a = fabs((double)out->y[i]); if (a > peak) peak = a; }
            if (peak > 1e-6) for (i = 0; i < out->ny; ++i) out->y[i] *= (FP_TYPE)(tgt / peak);
        }
        /* 4) 安全限幅：峰值 > 0.99 则等比缩回，防 16bit 削波 */
        peak = 0.0;
        for (i = 0; i < out->ny; ++i) { double a = fabs((double)out->y[i]); if (a > peak) peak = a; }
        if (peak > 0.99) for (i = 0; i < out->ny; ++i) out->y[i] *= (FP_TYPE)(0.99 / peak);
    }
    wavwrite(out->y, out->ny, fs, 16, u.out_file);
    printf("[utau] wrote %s (%d samples @ %d Hz, %d frames)\n",
           u.out_file, out->ny, fs, n_out);

    free(remap);
    if (amp_dev_st) free(amp_dev_st);
    if (pc.cents) free(pc.cents);
    llsm_delete_output(out);
    llsm_delete_chunk(chunk);
    llsm_delete_chunk(out_chunk);
    llsm_delete_soptions(sopt);
    return 0;
}

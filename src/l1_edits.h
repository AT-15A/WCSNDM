/* l1_edits.h —— L1/L2 共通的"直观编辑量"载体。
   主引擎(llsm_utau.c)从 Flags 填好，传给 L1 后端(l1_path.c)在 L1 结构上应用同一套编辑。
   只含基本类型，两个 TU 都能安全 include(不牵涉任一库的结构体)。
   批次1：声道/共振峰(g/Mo/ME/Mr)。批次2(mo 全参数化)：Mt(张力,tilt部分)/MH(高频倾斜)/
   A(颤音幅度联动)/bh(辅音区谐波)/Mb(气声)/b(清辅音噪声)。 */
#ifndef L1_EDITS_H
#define L1_EDITS_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* L1/M1 共享 pyin f0 缓存(<wav>.l1f0)。两内核 pyin 参数一致(nhop256),分析最慢处即 pyin。
   格式: magic'L1F0' fs nhop nfrm + FP_TYPE[nfrm]。wav 比缓存新则失效。 */
static FP_TYPE* l1f0_cache_get(const char* wav, int fs, int nhop, int* nfrm_out) {
  char pth[1100]; struct stat sw, sc; FILE* f; int hdr[4]; FP_TYPE* arr;
  snprintf(pth, sizeof(pth), "%s.l1f0", wav);
  if (stat(wav,&sw)!=0 || stat(pth,&sc)!=0 || sc.st_mtime < sw.st_mtime) return NULL;
  f = fopen(pth, "rb"); if (!f) return NULL;
  if (fread(hdr,4,4,f)!=4 || hdr[0]!=0x4C314630 || hdr[1]!=fs || hdr[2]!=nhop || hdr[3]<=0 || hdr[3]>2000000) { fclose(f); return NULL; }
  arr = (FP_TYPE*)malloc(sizeof(FP_TYPE)*hdr[3]);
  if (!arr || fread(arr,sizeof(FP_TYPE),hdr[3],f)!=(size_t)hdr[3]) { free(arr); fclose(f); return NULL; }
  fclose(f); *nfrm_out = hdr[3]; return arr;
}
static void l1f0_cache_put(const char* wav, int fs, int nhop, FP_TYPE* f0, int nfrm) {
  char pth[1100]; FILE* f; int hdr[4];
  snprintf(pth, sizeof(pth), "%s.l1f0", wav);
  f = fopen(pth, "wb"); if (!f) return;
  hdr[0]=0x4C314630; hdr[1]=fs; hdr[2]=nhop; hdr[3]=nfrm;
  fwrite(hdr,4,4,f); fwrite(f0,sizeof(FP_TYPE),nfrm,f); fclose(f);
}
typedef struct {
  int    has_g, has_Mo, has_ME, has_Mr;
  double g, Mo, ME, Mr;
  int    has_Mt, has_MH, has_A, has_bh, has_Mb, has_b;
  double Mt, MH, A, bh, Mb, b;
  int    has_Mf, has_Ab, has_NA, has_RG, has_DN;
  double Mf, Ab, NA, RG, DN;
  /* DN 层② 跨模型传递:L2 静音印记(降采样常驻底噪谱)+SNR,供 L1 谐波侧做绝对底噪参考。
     dn_print_n=0 表示无印记(纯 L1 模式等),谐波侧自动关闭。 */
  int    dn_print_n;
  double dn_print_fnyq;
  double dn_nsnr_db;        /* 有声帧噪声/印记 信噪比(dB),越小素材越脏 */
  double dn_print_db[128];
  /* TM 音色对齐("第一档"说话人先验):主引擎从 bank 档案算出的染色校正 EQ(dB,线性频轴),
     L1 侧在 vt 包络上同步施加。tm_n=0 表示未启用。 */
  int    tm_n;
  double tm_fnyq;
  double tm_corr_db[64];
} L1Edits;

static int dn_cmp_d_(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/* 统一采样率防线:UTAU 生态事实标准是 44.1k——wavtool 按第一块的头拼接整轨,
   混采样率音源(如 48k 立体声转录)的音符会被拼成静音/变速块;HF 声码器也固定 44.1k 输出。
   引擎在 wavread 后统一重采样到 44100,所有内核输出恒 44.1k。
   24 阶 Hann 窗 sinc 归一化插值(直流增益=1),48k->44.1k 混叠抑制约 50dB,对人声足够。 */
static FP_TYPE* wcs_resample_44k(const FP_TYPE* x, int nx, int fs_in, int* ny_out) {
  const double PI_ = 3.14159265358979323846;
  const int HT = 12;
  double ratio = 44100.0 / fs_in;
  double fc = (fs_in > 44100) ? 0.45 * 44100.0 / fs_in : 0.45;
  int ny = (int)(nx * ratio), i, k;
  FP_TYPE* y;
  if (ny < 1) ny = 1;
  y = (FP_TYPE*)malloc(sizeof(FP_TYPE) * ny);
  if (!y) { *ny_out = 0; return NULL; }
  for (i = 0; i < ny; ++i) {
    double t = i / ratio;
    int t0 = (int)t;
    double s = 0, wsum = 0;
    for (k = t0 - HT + 1; k <= t0 + HT; ++k) {
      double d, w, snc;
      if (k < 0 || k >= nx) continue;
      d = t - k;
      w = 0.5 + 0.5 * cos(PI_ * d / HT);
      snc = (fabs(d) < 1e-9) ? 2.0 * fc : sin(2.0 * PI_ * fc * d) / (PI_ * d);
      s += x[k] * snc * w;
      wsum += snc * w;
    }
    y[i] = (FP_TYPE)(fabs(wsum) > 1e-12 ? s / wsum : 0.0);
  }
  *ny_out = ny;
  return y;
}
static int wcs_to_44100(FP_TYPE** px, int* pnx, int* pfs, const char* tag) {
  FP_TYPE* y; int ny = 0;
  if (!px || !*px || *pnx <= 0 || *pfs <= 0 || *pfs == 44100) return 0;
  y = wcs_resample_44k(*px, *pnx, *pfs, &ny);
  if (!y || ny <= 0) { free(y); return 0; }
  fprintf(stderr, "[%s] resample %d -> 44100\n", tag, *pfs);
  free(*px);
  *px = y; *pnx = ny; *pfs = 44100;
  return 1;
}
static double dn_interp_db_(const double* pdb, int n, double fnyq, double f) {
  double pos; int i0; double fr;
  if (!pdb || n < 2 || fnyq <= 0) return 0.0;
  pos = f / fnyq * (n - 1);
  if (pos < 0) pos = 0; if (pos > n - 1) pos = n - 1;
  i0 = (int)pos; fr = pos - i0;
  if (i0 >= n - 1) { i0 = n - 2; fr = 1.0; }
  return pdb[i0] * (1.0 - fr) + pdb[i0 + 1] * fr;
}
static double dn_print_lin_(const double* pdb, int n, double fnyq, double f) {
  if (!pdb || n < 2 || fnyq <= 0) return 0.0;
  return pow(10.0, dn_interp_db_(pdb, n, fnyq, f) / 10.0);
}
/* DN 层②·谐波/包络侧共用核心：嘶声不只在噪声模型里——谐波分析会把常驻嘶声拟合成
   高次"谐波"幅度(贴着一条不随发声起伏的底线),只清 psd 听不出来。
   以静音印记为绝对参考,给每条"固定频率轨迹"(L2=VTMAGN 各 bin,L1=各谐波序号)算恒定增益：
   ①自校准 c:轨迹逐帧功率近指数分布(p90/p10 天然~13dB),不能按平坦度识别;
     取带内(5~11.5kHz)轨迹【中值功率/印记功率】之比的 20% 低分位=c(带内最安静轨迹=嘶声,
     真谐波比值更高不污染校准),把印记换算到本模型拟合域(吸收窗长/归一化差异);
   ②SNR 门控 w=clamp((26-snr)/8,0,1):干净素材(印记远低于气声)强度平滑归零,常开无副作用;
   ③每轨迹恒定功率增益 g=clamp(1-alpha*c*print(f_t)/med_t, beta,1)：嘶声轨迹(med≈底)整条
     压到 beta=-10dB,真谐波(med>>底)g≈1;恒定增益不随帧抖动,杜绝 musical noise。
   pw[i*ntrk+t]=第 i 帧轨迹 t 线性功率(<0=缺失)。gain_out[t]=功率域增益。返回 1=可用。 */
static int dn_harm_gain(const double* pw, int nfrm, int ntrk, const double* ftrk,
                        const double* print_db, int print_n, double print_fnyq,
                        double snr_db, double strength01, double* gain_out) {
  int t, i, ncal = 0;
  double w, alpha, c, beta = 0.05;   /* 恒定增益无 musical noise 风险,可比 psd 侧(0.1)更深 */
  double* col; double* cal; double* med;
  if (!pw || !ftrk || !gain_out || nfrm < 8 || ntrk < 4
      || !print_db || print_n < 2 || print_fnyq <= 0) return 0;
  w = (26.0 - snr_db) / 8.0;
  if (w > 1.0) w = 1.0;
  if (w <= 0.0) return 0;
  alpha = 1.5 * strength01 * w;
  if (alpha <= 1e-3) return 0;
  col = (double*)malloc(sizeof(double) * nfrm);
  cal = (double*)malloc(sizeof(double) * ntrk);
  med = (double*)malloc(sizeof(double) * ntrk);
  if (!col || !cal || !med) { free(col); free(cal); free(med); return 0; }
  for (t = 0; t < ntrk; ++t) {
    int cnt = 0; double pl;
    med[t] = -1.0;
    for (i = 0; i < nfrm; ++i) { double v = pw[(size_t)i * ntrk + t]; if (v >= 0) col[cnt++] = v; }
    if (cnt < 6) continue;
    qsort(col, cnt, sizeof(double), dn_cmp_d_);
    med[t] = col[cnt / 2];
    pl = dn_print_lin_(print_db, print_n, print_fnyq, ftrk[t]);
    if (ftrk[t] >= 5000.0 && ftrk[t] <= 11500.0 && ftrk[t] <= 0.92 * print_fnyq
        && med[t] > 1e-30 && pl > 1e-30)
      cal[ncal++] = med[t] / pl;
  }
  if (ncal < 8) {
    fprintf(stderr, "[DN] harm-floor: calib insufficient (ncal=%d)\n", ncal);
    free(col); free(cal); free(med); return 0;
  }
  qsort(cal, ncal, sizeof(double), dn_cmp_d_);
  c = cal[ncal / 2];   /* 中位数:低质素材带内轨迹以嘶声为主,真谐波在高尾不拖动中位 */
  for (t = 0; t < ntrk; ++t) {
    double pl = dn_print_lin_(print_db, print_n, print_fnyq, ftrk[t]);
    double g = 1.0;
    if (med[t] > 1e-30 && pl > 1e-30) {
      g = 1.0 - alpha * (c * pl) / med[t];
      if (g < beta) g = beta;
      if (g > 1.0) g = 1.0;
    }
    gain_out[t] = g;
  }
  fprintf(stderr, "[DN] harm-floor: calib ncal=%d c=%.3g alpha=%.2f\n", ncal, c, alpha);
  free(col); free(cal); free(med);
  return 1;
}

/* Mf 共振峰调谐(resonance tuning,参照 logmel_for_hifigan_physics_correct_v3.py)：
   每帧估计 F1(200-1500Hz 平滑峰)→目标=最近谐波 k*f0,λ=强度*接近度门控(谐波离 F1 越近越强)；
   高斯搬移(+新位置-旧位置)+低频段(<=1500Hz)均值守恒。env=对数幅度(线性频率轴 0..fnyq)，
   dbscale=每 dB 对应的 env 单位(L2 dB 域=1,L1 自然对数域=1/8.6859)。纯数学,L1/L2 共用。 */
static void mf_formant_tune(FP_TYPE* env, int n, double fnyq, double f0,
                            double strength01, double dbscale, double d_st) {
  int i, i_lo, i_hi, ip = -1;
  double best = -1e30;
  if (!env || n < 32 || f0 <= 40 || fnyq <= 0 || strength01 <= 0) return;
  i_lo = (int)(200.0 / fnyq * (n - 1)); i_hi = (int)(1500.0 / fnyq * (n - 1));
  if (i_hi >= n) i_hi = n - 1; if (i_lo < 1) i_lo = 1;
  { /* ~100Hz 平滑窗内求最强峰 = F1 */
    int half = (int)(50.0 / fnyq * (n - 1)), w; if (half < 1) half = 1;
    for (i = i_lo; i <= i_hi; ++i) {
      double s = 0; int c = 0;
      for (w = -half; w <= half; ++w) { int q = i + w; if (q < 0 || q >= n) continue; s += env[q]; ++c; }
      s /= (c ? c : 1);
      if (s > best) { best = s; ip = i; }
    }
  }
  if (ip < 0) return;
  {
    double F1 = (double)ip / (n - 1) * fnyq;
    int k;
    /* 下移>2半音：对称下半——去轰(F1区温和高斯衰减,~1.5dB/oct下移)+变暗(>2kHz轻倾斜,~1dB/oct)。
       补偿谐波密度效应(降调后F1带能量+3dB/oct=轰)并模拟低音源协变(松弛=更暗)。无能量守恒(衰减即目的)。 */
    if (d_st < -2.0) {
      double oct = (-d_st - 2.0) / 12.0;
      double att = 1.5 * oct * strength01; if (att > 4.5) att = 4.5;
      double tlt = 1.0 * oct * strength01; if (tlt > 3.0) tlt = 3.0;
      for (i = 0; i < n; ++i) {
        double f = (double)i / (n - 1) * fnyq;
        double dd = (f - F1) / 300.0;
        double cut = att * exp(-0.5 * dd * dd);
        if (f > 2000.0) { double tb = tlt * log(f / 2000.0) / log(2.0); if (tb > 3.0) tb = 3.0; cut += tb; }
        env[i] -= (FP_TYPE)(cut * dbscale);
      }
    }
    /* 大幅转音修复：f0 连续滑动时 round(F1/f0) 会整数跳变→目标谐波跳 f0 宽→逐帧间断。
       改为评估相邻两候选谐波(k_lo/k_hi),有效强度=λ_max−λ_min：远离切换点=原行为;
       两谐波等距(切换点)时强度连续衰减到0=不搬移→过渡无跳变(无状态,数学连续)。 */
    k = (int)(F1 / f0); if (k < 1) k = 1;
    double d_lo = ((double)k * f0 - F1) / 250.0;
    double d_hi = ((double)(k + 1) * f0 - F1) / 250.0;
    double lam_lo = strength01 * exp(-0.5 * d_lo * d_lo);
    double lam_hi = strength01 * exp(-0.5 * d_hi * d_hi);
    double drive = (lam_lo >= lam_hi) ? (double)k * f0 : (double)(k + 1) * f0;
    double lam = (lam_lo >= lam_hi) ? (lam_lo - lam_hi) : (lam_hi - lam_lo);
    double rt, sig = 160.0, gdb, m_old = 0, m_new = 0; int cb = 0;
    if (lam < 1e-3) return;
    rt = (1.0 - lam) * F1 + lam * drive;
    if (rt < 200) rt = 200; if (rt > 1500) rt = 1500;
    gdb = 14.0 * lam;   /* 满量程 8->14dB(更激进) */
    for (i = 0; i < n; ++i) {
      double f = (double)i / (n - 1) * fnyq;
      double dn = (f - rt) / sig, doo = (f - F1) / sig;
      if (f <= 1500.0) { m_old += env[i]; ++cb; }
      env[i] += (FP_TYPE)(gdb * (exp(-0.5*dn*dn) - exp(-0.5*doo*doo)) * dbscale);
      if (f <= 1500.0) m_new += env[i];
    }
    if (cb > 0) { /* 低频段能量守恒 */
      double corr = (m_old - m_new) / cb;
      for (i = 0; i < n; ++i) { double f = (double)i/(n-1)*fnyq; if (f <= 1500.0) env[i] += (FP_TYPE)corr; }
    }
  }
}

/* NA 鼻音度(±1)：鼻道耦合声学近似——500Hz 反共振凹陷(满量程-12dB,σ180)+250Hz 鼻腔哼鸣峰(+4dB,σ80)。
   正=更鼻音(闭口哼鸣感),负=去鼻音。env=对数幅度,dbscale 同 mf_formant_tune。 */
static void na_nasal(FP_TYPE* env, int n, double fnyq, double v01, double dbscale) {
  int i;
  if (!env || n < 16 || fnyq <= 0 || v01 == 0.0) return;
  for (i = 0; i < n; ++i) {
    double f = (double)i / (n - 1) * fnyq;
    double dz = (f - 500.0) / 180.0, dp = (f - 250.0) / 80.0;
    double db = v01 * (-12.0 * exp(-0.5 * dz * dz) + 4.0 * exp(-0.5 * dp * dp));
    env[i] += (FP_TYPE)(db * dbscale);
  }
}

/* f0 轨迹人声先验修复(DN栈·先验5)：人声音高必然连续——
   ①八度错误吸附(2x/0.5x 且折回后贴近邻域中值) ②野点(偏离邻域中值>20%)替换为中值
   ③单帧漏检(两侧有声)线性补插 ④孤立单帧误检(两侧无声)清零。
   只动异常帧,干净轨迹按构造不受影响。返回修正帧数。 */
static int f0_sanitize(FP_TYPE* f0, int n) {
  int i, j, k, fixes = 0;
  if (!f0 || n < 5) return 0;
  for (i = 0; i < n; ++i) {                       /* ①② 邻域中值(±3,排除自身,只取有声) */
    double nb[6]; int c = 0;
    if (f0[i] <= 20) continue;
    for (j = i - 3; j <= i + 3; ++j) {
      if (j == i || j < 0 || j >= n || f0[j] <= 20) continue;
      nb[c++] = f0[j]; if (c >= 6) break;
    }
    if (c < 2) continue;
    for (j = 1; j < c; ++j) { double t = nb[j]; k = j - 1;   /* 插入排序 */
      while (k >= 0 && nb[k] > t) { nb[k+1] = nb[k]; --k; } nb[k+1] = t; }
    {
      double med = (c & 1) ? nb[c/2] : 0.5 * (nb[c/2-1] + nb[c/2]);
      double v = f0[i];
      if (v > 1.8 * med && fabs(v * 0.5 - med) < 0.25 * med) { f0[i] = (FP_TYPE)(v * 0.5); ++fixes; }
      else if (v < 0.55 * med && fabs(v * 2.0 - med) < 0.25 * med) { f0[i] = (FP_TYPE)(v * 2.0); ++fixes; }
      else if (fabs(v - med) > 0.20 * med) { f0[i] = (FP_TYPE)med; ++fixes; }
    }
  }
  for (i = 1; i < n - 1; ++i) {                   /* ③ 单帧漏检补插 */
    if (f0[i] <= 20 && f0[i-1] > 20 && f0[i+1] > 20) { f0[i] = (FP_TYPE)(0.5 * (f0[i-1] + f0[i+1])); ++fixes; }
  }
  for (i = 1; i < n - 1; ++i) {                   /* ④ 孤立单帧误检清零 */
    if (f0[i] > 20 && f0[i-1] <= 20 && f0[i+1] <= 20) { f0[i] = 0; ++fixes; }
  }
  return fixes;
}
#endif

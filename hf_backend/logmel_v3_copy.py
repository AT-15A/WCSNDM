# -*- coding: utf-8 -*-
"""
logmel_for_hifigan_physics_correct_v3.py
=========================================
在 v2 基础上新增：

Opt 7c 自适应谐波扩散权重（--harm_spread_adaptive 1）
    根据当前帧 F0 所对应的"谐波间距（FFT bin 数）"自动在
    [0, 1, 0]（极低音，谐波密集）和 --harm_spread（高音，谐波稀疏）
    之间线性插值，解决固定权重在不同音区的矛盾：
      • 低音区（bins/谐波 ≤ bins_low）：侧边权重→0，消除谐波串扰
      • 高音区（bins/谐波 ≥ bins_high）：使用 harm_spread，保留亚bin插值
      • 过渡区：线性插值
    参数：
      --harm_spread_adaptive  0/1（默认 0=关闭）
      --harm_spread_bins_low  过渡起点 bins/谐波（默认 4.0，≈ 86 Hz）
      --harm_spread_bins_high 过渡终点 bins/谐波（默认 6.0，≈ 129 Hz）

Opt 7b 谐波幅度累加模式（--harm_accum_mode amp）
    将谐波在 STFT bin 上的累加从"功率相加"改为"幅度相加"：
        power 模式（默认）：harm_acc[k] += (mag_h × w)^2   → 非相干叠加
        amp   模式：        harm_acc[k] += mag_h × w       → 相干叠加后平方
    幅度累加后平方得功率，再与噪声功率叠加，与训练时 |STFT|
    作为幅度谱送入 mel 滤波器的方式在物理上更一致。
    参数：
      --harm_accum_mode  power/amp（默认 power）

Opt 7a 高频段谐波包络平滑（LPC）（--hf_harm_lpc_enable）
    对每帧谐波幅度在对数域拟合 LPC 全极点模型，用平滑后的包络
    替换高频段（>= hf_start_hz）的原始谐波幅度，再以 strength
    比例混合回原始值：
        log_har_lpc[h]  = LPC 包络在谐波 h 处的估计值
        har_out[h]      = exp( (1−strength)·log_har[h]
                               + strength·log_har_lpc[h] )
    仅对 freq_h >= hf_start_hz 的谐波施加；低频谐波不变。
    LPC 阶数越高→越接近原始细节；阶数越低→包络越平滑。
    参数：
      --hf_harm_lpc_enable      0/1（默认 0）
      --hf_harm_lpc_order       LPC 阶数（默认 12）
      --hf_harm_lpc_start_hz    高频起点 Hz（默认 2000.0）
      --hf_harm_lpc_strength    混合强度 0–1（默认 1.0）

继承自 v2 的优化：
Opt 1  高频源参考混合（--hf_blend_enable）
Opt 2  噪声 PSD 高频倾斜（--noise_tilt_enable）
Opt 5  逐频带均值对齐（--perband_align_enable）
Opt 6  平滑谱包络修正（--src_env_correct_enable）

所有新功能默认关闭（=0）。Opt 1/5/6 依赖 --target_mel_npy。
原有功能完整保留，向下兼容 v2/v1 的所有 CLI 参数。
"""

import argparse
import csv
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from librosa.filters import mel as librosa_mel_fn
except Exception:
    librosa_mel_fn = None

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_prominences


# ═══════════════════════════════════════════════════════════════════════════════
#  基础工具
# ═══════════════════════════════════════════════════════════════════════════════

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

def mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    m_min = hz_to_mel(np.asarray([fmin], dtype=np.float32))[0]
    m_max = hz_to_mel(np.asarray([fmax], dtype=np.float32))[0]
    m = np.linspace(m_min, m_max, n_mels, dtype=np.float32)
    return mel_to_hz(m).astype(np.float32)

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    n_freqs = 1 + n_fft // 2
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float32)
    m_min = hz_to_mel(np.asarray([fmin], dtype=np.float32))[0]
    m_max = hz_to_mel(np.asarray([fmax], dtype=np.float32))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float32)
    f_pts = mel_to_hz(m_pts).astype(np.float32)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:  center = left + 1
        if right  <= center: right  = center + 1
        center = min(center, n_freqs - 1)
        right  = min(right,  n_freqs - 1)
        if center > left:
            fb[i, left:center]   = (fft_freqs[left:center]   - f_pts[i])     / max(f_pts[i+1]-f_pts[i],   1e-6)
        if right > center:
            fb[i, center:right]  = (f_pts[i+2] - fft_freqs[center:right])    / max(f_pts[i+2]-f_pts[i+1], 1e-6)
        bw = float(f_pts[i+2] - f_pts[i])
        if bw > 1e-6:
            fb[i, :] *= 2.0 / bw
    return np.maximum(fb, 0.0)

def db_to_amp(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, db / 20.0, dtype=np.float32)

def compute_harm_stft_gain(n_fft: int, mode: str, custom_gain: float = 1.0) -> float:
    """时域正弦幅度 A → Hann 窗 STFT 峰值 bin 幅度的换算系数。"""
    mode = str(mode).lower()
    if mode == "hann":
        win = np.hanning(int(n_fft)).astype(np.float64)
        return float(np.sum(win) / 2.0)
    if mode == "nfft4":   return float(n_fft) / 4.0
    if mode == "custom":  return float(custom_gain)
    if mode == "none":    return 1.0
    raise ValueError("Unknown harm_stft_gain_mode: %s" % mode)



# ═══════════════════════════════════════════════════════════════════════════════
#  CSV 解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_legacy_llsm_csv(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            _header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty: %s" % path)
        for row_num, row in enumerate(reader, start=2):
            if not row: continue
            if len(row) < 5:
                raise ValueError("Bad CSV line %d in %s" % (row_num, path))
            frame_idx = int(float(row[0]))
            f0   = float(row[1])
            nhar = int(float(row[3]))
            npsd = int(float(row[4]))
            data = row[5:]
            need = 2 * nhar + npsd
            if len(data) < need:
                raise ValueError("Bad CSV line %d in %s: data_len=%d < %d" % (row_num, path, len(data), need))
            har    = np.asarray([float(x) for x in data[:nhar]],              dtype=np.float32)
            psd_db = np.asarray([float(x) for x in data[2*nhar:2*nhar+npsd]], dtype=np.float32)
            vt_tail = data[2*nhar+npsd:]
            vt_db = np.asarray([float(x) for x in vt_tail], dtype=np.float32) if vt_tail else None
            out[frame_idx] = {"f0": f0, "nhar": nhar, "npsd": npsd,
                              "har": har, "psd_db": psd_db, "vt_db": vt_db}
    if not out:
        raise ValueError("No data rows in CSV: %s" % path)
    return out


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:    return float(str(x).strip())
    except: return default

def _safe_int(x: Any, default: int = 0) -> int:
    try:    return int(float(str(x).strip()))
    except: return default

def parse_hn_ratio_csv(path: str) -> Dict[int, Dict[str, float]]:
    if not path: return {}
    if not os.path.exists(path):
        raise FileNotFoundError("H/N ratio CSV not found: %s" % path)
    out: Dict[int, Dict[str, float]] = {}
    header = None
    has_vuv_col = False   # 记录 CSV 是否真实存在 vuv 列
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            if str(row[0]).strip().startswith("#"): continue
            if header is None:
                header = [str(x).strip() for x in row]
                has_vuv_col = "vuv" in header   # ← 读到表头后才能判断
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rec   = {header[i]: row[i] for i in range(len(header))}
            frame = _safe_int(rec.get("frame", ""), default=-1)
            if frame < 0: continue
            hn_ratio = _safe_float(rec.get("hn_ratio", ""))
            hn_db    = _safe_float(rec.get("hn_db", ""))
            harm_p   = _safe_float(rec.get("harmonic_power", ""))
            noise_p  = _safe_float(rec.get("noise_power", ""))
            # 若 CSV 无 vuv 列，默认全部视为浊音（1.0），避免校正被静默跳过
            if has_vuv_col:
                vuv = _safe_float(rec.get("vuv", ""), default=0.0)
            else:
                vuv = 1.0
            if not np.isfinite(hn_ratio) or hn_ratio <= 0.0:
                if np.isfinite(hn_db):
                    hn_ratio = float(10.0 ** (hn_db / 10.0))
                elif np.isfinite(harm_p) and np.isfinite(noise_p) and noise_p > 0.0:
                    hn_ratio = float(harm_p / noise_p)
            out[frame] = {"hn_ratio": float(hn_ratio), "vuv": float(vuv)}
    if not out:
        raise ValueError("H/N ratio CSV has no usable rows: %s" % path)
    if not has_vuv_col:
        print("[HN-CSV] ⚠ 未找到 'vuv' 列，所有帧将视为浊音（vuv=1）以保证校正生效")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  共振峰估计 & 共振峰补偿（与 noiseboost 相同）
# ═══════════════════════════════════════════════════════════════════════════════

def semitone_shift(f0_ref: float, f0_new: float) -> float:
    if f0_ref > 0.0 and f0_new > 0.0:
        return 12.0 * math.log(f0_new / f0_ref, 2.0)
    return 0.0

def _interp_cross_x(x1, y1, x2, y2, y):
    if abs(y2 - y1) < 1e-12: return x1
    return x1 + (y - y1) / (y2 - y1) * (x2 - x1)

def half_power_bandwidth(freq_hz, power, peak_idx):
    half = float(power[peak_idx]) * 0.5
    n = int(power.size)
    l = int(peak_idx)
    while l > 0 and float(power[l]) > half: l -= 1
    f_left = (float(freq_hz[peak_idx]) if l == peak_idx
              else float(_interp_cross_x(freq_hz[l], power[l], freq_hz[l+1], power[l+1], half)))
    r = int(peak_idx)
    while r < n-1 and float(power[r]) > half: r += 1
    f_right = (float(freq_hz[peak_idx]) if r == peak_idx
               else float(_interp_cross_x(freq_hz[r-1], power[r-1], freq_hz[r], power[r], half)))
    return max(0.0, f_right - f_left), f_left, f_right

def estimate_formant_from_vt_peaks(vt_db, vt_bin_hz, fmin, fmax, smooth_size, prom_ratio, bw_floor_hz):
    if vt_db is None or vt_db.size < 16: return None, None
    band = np.where((vt_bin_hz >= fmin) & (vt_bin_hz <= fmax))[0]
    if band.size < 16: return None, None
    power = np.power(10.0, vt_db[band].astype(np.float32) / 10.0)
    if smooth_size and smooth_size > 1:
        power = uniform_filter1d(power, size=int(smooth_size), mode="nearest")
    peaks, _ = find_peaks(power)
    if peaks.size == 0: return None, None
    prom      = peak_prominences(power, peaks)[0].astype(np.float32)
    peak_vals = power[peaks].astype(np.float32)
    keep = (prom / np.maximum(peak_vals, 1e-12)) >= float(prom_ratio)
    peaks2 = peaks[keep] if np.any(keep) else peaks
    prom2  = prom[keep]  if np.any(keep) else prom
    best = int(peaks2[int(np.argmax(prom2))])
    bw, _, _ = half_power_bandwidth(vt_bin_hz[band], power, best)
    return float(vt_bin_hz[band[best]]), float(max(bw, bw_floor_hz))

def estimate_formant_from_vt_simple(vt_db, vt_bin_hz, fmin, fmax, smooth_size, bw_floor_hz):
    if vt_db is None or vt_db.size < 16: return None, None
    band = np.where((vt_bin_hz >= fmin) & (vt_bin_hz <= fmax))[0]
    if band.size < 16: return None, None
    x = vt_db[band].astype(np.float32)
    if smooth_size and smooth_size > 1:
        x = uniform_filter1d(x, size=int(smooth_size), mode="nearest")
    return float(vt_bin_hz[band[int(np.argmax(x))]]), float(bw_floor_hz)

def distance_weight(hz_target, center_hz, bw_hz, weight_func):
    if bw_hz <= 1e-6: return 0.0
    d = (hz_target - center_hz) / bw_hz
    if weight_func == "gauss": return float(math.exp(-0.5 * d * d))
    return float(1.0 / (1.0 + d * d))

def apply_resonance_tuning_bump(vt_db, vt_bin_hz, r_ref, r_target, lam, gain_db, sigma_hz, keep_band_hz):
    if vt_db is None: return vt_db
    lam = float(np.clip(lam, 0.0, 1.0))
    if lam <= 0.0: return vt_db
    f     = vt_bin_hz.astype(np.float32)
    g_new = np.exp(-0.5 * ((f - float(r_target)) / float(sigma_hz)) ** 2).astype(np.float32)
    g_old = np.exp(-0.5 * ((f - float(r_ref))    / float(sigma_hz)) ** 2).astype(np.float32)
    vt2   = vt_db.astype(np.float32) + float(lam * gain_db) * (g_new - g_old)
    band  = f <= float(keep_band_hz)
    if np.any(band):
        vt2 = vt2 - float(np.mean(vt2[band])) + float(np.mean(vt_db[band]))
    return vt2

def _solve_capped_energy_redistribution(target_shape, upper, target_power):
    if target_power <= 1e-12: return np.zeros_like(target_shape, dtype=np.float32)
    target_shape = np.maximum(np.asarray(target_shape, dtype=np.float64), 0.0)
    upper        = np.maximum(np.asarray(upper,        dtype=np.float64), 0.0)
    max_power = float(np.sum(upper ** 2))
    if max_power <= 1e-12:              return np.zeros_like(target_shape, dtype=np.float32)
    if max_power <= target_power + 1e-9: return upper.astype(np.float32)
    def shaped_power(a):
        return float(np.sum(np.minimum(a * target_shape, upper) ** 2))
    lo, hi = 0.0, 1.0
    while shaped_power(hi) < target_power and hi < 1e6: hi *= 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if shaped_power(mid) >= target_power: hi = mid
        else:                                 lo = mid
    return np.minimum(hi * target_shape, upper).astype(np.float32)

def apply_resonance_harmonic_compensation(har_current, har_old, f0_ref, f0_use,
                                          f1_ref, b1, f2_ref, b2, args):
    x = np.asarray(har_current, dtype=np.float32).copy()
    if int(args.resonance_harmonic_comp_enable) != 1: return x
    if x.size == 0 or float(f0_ref) <= 0.0 or float(f0_use) <= 0.0: return x
    d_st = semitone_shift(float(f0_ref), float(f0_use))
    if int(args.resonance_harmonic_comp_disable_downshift) == 1 and d_st < 0.0: return x
    centers = []
    if int(args.resonance_harmonic_comp_use_f1) == 1 and f1_ref is not None and b1 is not None:
        centers.append((float(f1_ref),
                        float(max(float(b1) * float(args.resonance_harmonic_comp_bw_scale),
                                  float(args.resonance_harmonic_comp_min_bw_hz)))))
    if int(args.resonance_harmonic_comp_use_f2) == 1 and f2_ref is not None and b2 is not None:
        centers.append((float(f2_ref),
                        float(max(float(b2) * float(args.resonance_harmonic_comp_bw_scale),
                                  float(args.resonance_harmonic_comp_min_bw_hz)))))
    if not centers: return x
    nhar  = int(x.size)
    freqs = float(f0_use) * (np.arange(nhar, dtype=np.float32) + 1.0)
    weights = np.zeros(nhar, dtype=np.float32)
    for center, sigma in centers:
        g = np.exp(-0.5 * ((freqs - center) / max(sigma, 1e-6)) ** 2).astype(np.float32)
        weights = np.maximum(weights, g)
    if float(np.max(weights)) <= 1e-6: return x
    strength     = max(float(args.resonance_harmonic_comp_strength), 0.0)
    target_shape = x.astype(np.float64) * (1.0 + strength * weights.astype(np.float64))
    cap_linear   = 10.0 ** (max(float(args.resonance_harmonic_comp_gain_cap_db), 0.0) / 20.0)
    upper        = x.astype(np.float64) * cap_linear
    target_power = float(np.sum(np.asarray(har_old[:nhar], dtype=np.float64) ** 2))
    if target_power <= 1e-12: return x
    return _solve_capped_energy_redistribution(target_shape, upper, target_power)


# ═══════════════════════════════════════════════════════════════════════════════
#  H/N 功率域校正（直接解析，无需迭代）
# ═══════════════════════════════════════════════════════════════════════════════

def apply_hn_power_calibration(
    harm_power_spec: np.ndarray,
    noise_power_spec: np.ndarray,
    frames: List[int],
    hn_info: Dict[int, Dict[str, float]],
    mode: str,
    gain_min_db: float,
    gain_max_db: float,
    smooth_size: int,
    eps: float,
    debug_csv_path: str,
) -> np.ndarray:
    if mode == "none" or not hn_info:
        return harm_power_spec.copy()

    T = min(harm_power_spec.shape[0], len(frames))

    target_ratio = np.full(T, np.nan, dtype=np.float64)
    target_vuv   = np.zeros(T, dtype=np.float64)
    for ti in range(T):
        rec = hn_info.get(int(frames[ti]))
        if rec is None: continue
        target_ratio[ti] = float(rec.get("hn_ratio", float("nan")))
        target_vuv[ti]   = float(rec.get("vuv", 0.0))

    harm_sum  = np.maximum(np.sum(harm_power_spec[:T],  axis=1), eps)
    noise_sum = np.maximum(np.sum(noise_power_spec[:T], axis=1), eps)
    current_ratio = harm_sum / noise_sum

    valid = (
        np.isfinite(target_ratio) &
        (target_ratio > eps) &
        (target_vuv > 0.0) &
        np.isfinite(current_ratio)
    )

    if not np.any(valid):
        vuv_nonzero = int(np.sum(target_vuv > 0.0))
        ratio_valid = int(np.sum(np.isfinite(target_ratio) & (target_ratio > eps)))
        print("[HN-Calib-Power] ⚠ 没有有效浊音帧，跳过校正。"
              "（vuv>0 的帧=%d，hn_ratio 有效帧=%d）" % (vuv_nonzero, ratio_valid))
        if vuv_nonzero == 0:
            print("[HN-Calib-Power]   → hn_ratio CSV 可能缺少 'vuv' 列，"
                  "或全部帧的 vuv=0（parse_hn_ratio_csv 应已给出警告）")
        return harm_power_spec.copy()

    gain_min_linear = 10.0 ** (float(gain_min_db) / 10.0)
    gain_max_linear = 10.0 ** (float(gain_max_db) / 10.0)
    desired_gain = np.where(valid,
                            target_ratio / np.maximum(current_ratio, eps),
                            1.0)
    desired_gain = np.clip(desired_gain, gain_min_linear, gain_max_linear)

    power_gain = np.ones(T, dtype=np.float64)
    if mode == "global":
        power_gain[:] = float(np.clip(np.median(desired_gain[valid]),
                                      gain_min_linear, gain_max_linear))
    elif mode == "frame":
        valid_idx  = np.where(valid)[0]
        valid_vals = desired_gain[valid]
        if valid_idx.size == 1:
            power_gain[:] = float(valid_vals[0])
        else:
            power_gain[:] = np.interp(np.arange(T), valid_idx, valid_vals)
        power_gain = np.clip(power_gain, gain_min_linear, gain_max_linear)
        if smooth_size and smooth_size > 1 and T > 1:
            log_g = np.log(np.maximum(power_gain, 1e-300))
            log_g = uniform_filter1d(log_g, size=int(smooth_size), mode="nearest")
            power_gain = np.clip(np.exp(log_g), gain_min_linear, gain_max_linear)
    else:
        raise ValueError("Unknown hn_calib_mode: %s" % mode)

    harm2 = harm_power_spec.copy()
    harm2[:T] *= power_gain[:, None]

    after_ratio = np.sum(harm2[:T], axis=1) / noise_sum
    def _median_db(arr, mask):
        v = float(np.median(arr[mask]))
        return float("nan") if v <= 0.0 else 10.0 * math.log10(max(v, 1e-300))
    print("[HN-Calib-Power] mode=%s  valid=%d/%d" % (mode, int(np.sum(valid)), T))
    print("  目标  H/N: %.2f dB" % _median_db(target_ratio,  valid))
    print("  校正前H/N: %.2f dB" % _median_db(current_ratio, valid))
    print("  校正后H/N: %.2f dB" % _median_db(after_ratio,   valid))
    print("  power_gain: median=%.4g  min=%.4g  max=%.4g" % (
          float(np.median(power_gain[valid])),
          float(np.min(power_gain[valid])),
          float(np.max(power_gain[valid]))))

    if debug_csv_path:
        def _db10(x):
            return float(10.0 * math.log10(max(float(x), 1e-300))) if (np.isfinite(x) and x > 0.0) else float("nan")
        with open(debug_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "valid", "vuv",
                        "target_hn_ratio", "target_hn_db",
                        "before_hn_ratio", "before_hn_db",
                        "after_hn_ratio",  "after_hn_db",
                        "power_gain_linear", "power_gain_db"])
            for ti in range(T):
                w.writerow([int(frames[ti]), int(valid[ti]), float(target_vuv[ti]),
                            float(target_ratio[ti]),  _db10(float(target_ratio[ti])),
                            float(current_ratio[ti]), _db10(float(current_ratio[ti])),
                            float(after_ratio[ti]),   _db10(float(after_ratio[ti])),
                            float(power_gain[ti]),    _db10(float(power_gain[ti]))])
        print("[HN-Calib-Power] 调试 CSV 已写入:", debug_csv_path)

    return harm2


# ═══════════════════════════════════════════════════════════════════════════════
#  分布对齐目标统计（扩展版：同时返回逐 bin 统计）
# ═══════════════════════════════════════════════════════════════════════════════

def build_target_stats(target_mel_npy, default_mean, default_std):
    """兼容旧版接口（仅返回全局统计）。"""
    res = build_target_stats_v2(target_mel_npy, default_mean, default_std)
    return {"mean": res["mean"], "std": res["std"],
            "min":  res["min"],  "max": res["max"]}


def build_target_stats_v2(target_mel_npy, default_mean, default_std):
    """
    加载源参考 mel（target_mel_npy），返回全局统计 + 逐 bin 统计。
    若 target_mel_npy 为空或不存在，bin_mean / bin_std 返回 None。
    """
    if target_mel_npy and os.path.exists(target_mel_npy):
        mel = np.load(target_mel_npy).astype(np.float32)
        return {
            "mean":     float(np.mean(mel)),
            "std":      float(np.std(mel)),
            "min":      float(np.min(mel)),
            "max":      float(np.max(mel)),
            "bin_mean": np.mean(mel, axis=0).astype(np.float32),  # [n_mels]
            "bin_std":  np.std(mel,  axis=0).astype(np.float32),  # [n_mels]
        }
    return {
        "mean":     float(default_mean),
        "std":      float(default_std),
        "min":      float("nan"),
        "max":      float("nan"),
        "bin_mean": None,
        "bin_std":  None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 2 —— 噪声 PSD 高频倾斜校正
# ═══════════════════════════════════════════════════════════════════════════════

def apply_noise_spectral_tilt(
    noise_power_spec: np.ndarray,
    sr: int,
    n_fft: int,
    tilt_db_per_oct: float,
    ref_hz: float,
) -> np.ndarray:
    """
    对噪声功率谱施加频谱倾斜（dB/octave）。

    gain_power[k] = 10^( tilt_db_per_oct × log2(freq_k / ref_hz) / 10 )

    正的 tilt_db_per_oct → 高频提升（增加亮度/气声）。
    ref_hz 处增益 = 0 dB（不改变该频率的能量）。
    DC bin (freq≈0) 不施加倾斜（增益固定为 1）。
    """
    npsd = noise_power_spec.shape[1]
    fft_freqs = np.linspace(0.0, float(sr) / 2.0, npsd, dtype=np.float64)
    ref_hz = max(float(ref_hz), 1.0)
    # log2(freq/ref)；DC 和极低频 bin 设为 0 增益
    safe_freq = np.where(fft_freqs > 0.0, fft_freqs, ref_hz)
    log2_r    = np.log2(safe_freq / ref_hz)
    log2_r[fft_freqs < 1.0] = 0.0
    gain_db    = float(tilt_db_per_oct) * log2_r          # dB in power domain
    power_gain = np.power(10.0, gain_db / 10.0).astype(np.float64)
    return noise_power_spec * power_gain[None, :]


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 1 —— 高频源参考混合修正
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hf_blend_correction(
    logmel: np.ndarray,         # [T, n_mels]
    src_bin_mean: np.ndarray,   # [n_mels]  源参考 mel 的逐 bin 均值
    mel_freqs: np.ndarray,      # [n_mels]  每个 mel bin 的中心频率（Hz）
    start_hz: float,            # 混合起始频率（此处权重=0）
    end_hz: float,              # 混合终止频率（此处权重=alpha_max）
    alpha: float,               # 最大混合强度（0=不修正，1=完全对齐到源均值）
) -> np.ndarray:
    """
    Opt 1：高频段逐 bin 混合修正。

    对每个 mel bin 计算偏移量：correction[b] = (src_mean[b] - syn_mean[b]) × weight[b]
    weight[b] 在 [start_hz, end_hz] 从 0 线性增至 alpha，end_hz 以上保持 alpha。
    返回 [n_mels] 修正量，调用方执行：logmel += correction[None, :]

    效果：补偿 LLSM nhar 覆盖上限导致的高频亮度不足（5 kHz 以上谱峰不足）。
    """
    n_mels       = logmel.shape[1]
    syn_bin_mean = np.mean(logmel, axis=0).astype(np.float32)   # [n_mels]
    raw_diff     = src_bin_mean.astype(np.float32) - syn_bin_mean

    start_hz = float(start_hz)
    end_hz   = max(float(end_hz), start_hz + 1.0)
    alpha    = float(np.clip(alpha, 0.0, 1.0))

    weights = np.zeros(n_mels, dtype=np.float32)
    for b in range(n_mels):
        f = float(mel_freqs[b])
        if f <= start_hz:
            weights[b] = 0.0
        elif f <= end_hz:
            weights[b] = float((f - start_hz) / (end_hz - start_hz)) * alpha
        else:
            weights[b] = alpha

    correction = raw_diff * weights
    n_affected = int(np.sum(weights > 0.0))
    if n_affected > 0:
        print("[HFBlend] 高频混合修正：影响 %d/%d 个 mel bin（>= %.0f Hz）"
              % (n_affected, n_mels, start_hz))
        print("  correction: mean=%.4f  max=%.4f  (高频段 bin %d–%d)"
              % (float(np.mean(correction[weights > 0.0])),
                 float(np.max(np.abs(correction[weights > 0.0]))),
                 int(np.argmax(weights > 0.0)), n_mels - 1))
    return correction


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 5 —— 逐频带均值对齐修正
# ═══════════════════════════════════════════════════════════════════════════════

def compute_perband_correction(
    logmel: np.ndarray,         # [T, n_mels]
    src_bin_mean: np.ndarray,   # [n_mels]
    n_bands: int,               # 频带数量
    alpha: float,               # 修正强度（0–1）
) -> np.ndarray:
    """
    Opt 5：逐频带均值对齐修正。

    将 n_mels 个 mel bin 均匀分成 n_bands 段；
    每段内施加相同偏移量：segment_offset = α × (src_band_mean - syn_band_mean)。
    返回 [n_mels] 修正量。

    效果：比全局单一均值对齐更精细，消除各频段的系统性幅度偏差。
    """
    n_mels       = logmel.shape[1]
    syn_bin_mean = np.mean(logmel, axis=0).astype(np.float32)   # [n_mels]
    correction   = np.zeros(n_mels, dtype=np.float32)
    alpha        = float(np.clip(alpha, 0.0, 2.0))
    n_bands      = max(int(n_bands), 1)
    band_edges   = np.linspace(0, n_mels, n_bands + 1, dtype=int)

    print("[PerbandAlign] 逐频带均值对齐：%d 个频带，alpha=%.2f" % (n_bands, alpha))
    for i in range(n_bands):
        lo, hi = int(band_edges[i]), int(band_edges[i + 1])
        if lo >= hi: continue
        src_m = float(np.mean(src_bin_mean[lo:hi]))
        syn_m = float(np.mean(syn_bin_mean[lo:hi]))
        diff  = (src_m - syn_m) * alpha
        correction[lo:hi] = diff
        print("  Band %2d [bin %3d–%3d]: src_mean=%.4f  syn_mean=%.4f  corr=%.4f"
              % (i, lo, hi - 1, src_m, syn_m, diff))
    return correction


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 6 —— 平滑谱包络修正
# ═══════════════════════════════════════════════════════════════════════════════

def compute_smooth_env_correction(
    logmel: np.ndarray,         # [T, n_mels]
    src_bin_mean: np.ndarray,   # [n_mels]
    smooth_bins: int,           # 移动平均窗口宽度（mel bin 数量）
    alpha: float,               # 修正强度（0–1）
) -> np.ndarray:
    """
    Opt 6：平滑谱包络修正。

    计算源参考 mel 与合成 mel 的逐 bin 均值差值，
    经 mel-bin 方向移动平均平滑后施加：
        raw_diff[b]    = src_bin_mean[b] - syn_bin_mean[b]
        correction[b]  = smooth(raw_diff)[b] × alpha

    通过平滑消除逐 bin 噪声，只修正宽带谱包络偏差，
    不干扰谐波峰的精细结构（相当于 LPC 级别的包络修正）。
    返回 [n_mels] 修正量。
    """
    n_mels       = logmel.shape[1]
    syn_bin_mean = np.mean(logmel, axis=0).astype(np.float64)
    raw_diff     = src_bin_mean.astype(np.float64) - syn_bin_mean
    smooth_bins  = max(int(smooth_bins), 1)

    if smooth_bins > 1:
        smoothed = uniform_filter1d(raw_diff, size=smooth_bins, mode="nearest")
    else:
        smoothed = raw_diff.copy()

    correction = (smoothed * float(np.clip(alpha, 0.0, 2.0))).astype(np.float32)

    # 诊断输出
    raw_rms    = float(np.sqrt(np.mean(raw_diff ** 2)))
    smooth_rms = float(np.sqrt(np.mean(smoothed ** 2)))
    print("[SrcEnvCorrect] 平滑谱包络修正：smooth_bins=%d  alpha=%.2f" % (smooth_bins, alpha))
    print("  raw_diff rms=%.4f → 平滑后 rms=%.4f  correction max=%.4f"
          % (raw_rms, smooth_rms, float(np.max(np.abs(correction)))))
    return correction


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 7 —— 高频段谐波包络平滑（LPC）
# ═══════════════════════════════════════════════════════════════════════════════

def apply_hf_harmonic_lpc_smooth(
    har: np.ndarray,        # 线性谐波幅度 [nhar_use]，float32
    f0_hz: float,           # 基频 Hz（用于计算各谐波频率）
    hf_start_hz: float,     # 高频起点：仅对 freq >= hf_start_hz 的谐波施加平滑
    lpc_order: int,         # LPC 阶数（自动 clamp 到 min(order, nhar-2)）
    strength: float,        # 混合强度：0=不变，1=纯 LPC 包络
) -> np.ndarray:
    """
    对谐波幅度在对数域拟合 LPC 全极点模型，用平滑包络替换高频段谐波幅度。

    算法：
      1. x[h] = log(A[h] + eps)，h = 0..nhar-1
      2. 对 x 拟合 LPC(order)：a, e = scipy.signal.lpc(x, order)
      3. 在各谐波归一化频率处评估 LPC 全极点幅度谱（对数域）：
             log_env[h] = 0.5·log(e) − log|A(e^{jω_h})|
             ω_h = π·(h+1)/(nhar+1)  （均匀分布于 (0, π)）
      4. DC 修正：log_env += mean(x) − mean(log_env)（保持整体电平）
      5. 仅对 f0·(h+1) >= hf_start_hz 的谐波进行混合：
             x_out[h] = (1−strength)·x[h] + strength·log_env[h]
      6. 返回 exp(x_out)

    LPC 阶数建议：8–16（越高越接近原始，越低越平滑）。
    典型使用：hf_start_hz=2000, order=12, strength=1.0。
    """
    nhar = len(har)
    # 自动 clamp 阶数
    order = int(np.clip(lpc_order, 1, max(1, nhar - 2)))
    if nhar < order + 2 or strength <= 0.0:
        return har.copy()

    eps = 1e-12
    log_har = np.log(np.maximum(har.astype(np.float64), eps))  # [nhar]

    # ── 1. LPC 拟合 ──────────────────────────────────────────────────────────
    try:
        from scipy.signal import lpc as _scipy_lpc
        a, e = _scipy_lpc(log_har, order)       # a[0]=1，e=预测误差方差
        e = float(max(e, 1e-40))
    except Exception:
        return har.copy()

    # ── 2. LPC 包络评估 ──────────────────────────────────────────────────────
    # 归一化角频率：harmonic h (0-indexed) → ω = π·(h+1)/(nhar+1) ∈ (0, π)
    h_idx  = np.arange(nhar, dtype=np.float64)
    omega  = np.pi * (h_idx + 1.0) / (nhar + 1.0)   # [nhar]

    # A(e^{jω}) = Σ_{m=0}^{p} a[m]·e^{−jmω}
    m_arr = np.arange(len(a), dtype=np.float64)       # [order+1]
    # shape: [nhar, order+1]
    phase_mat = np.outer(-omega, m_arr)               # ω_h × m
    Az = np.sum(a[np.newaxis, :] * np.exp(1j * phase_mat), axis=1)  # [nhar]

    log_env = 0.5 * math.log(e) - np.log(np.abs(Az) + 1e-20)       # [nhar]

    # ── 3. DC 修正（保持整体电平） ───────────────────────────────────────────
    dc_shift = float(np.mean(log_har)) - float(np.mean(log_env))
    log_env += dc_shift

    # ── 4. 高频 mask 与混合 ──────────────────────────────────────────────────
    harm_freqs = f0_hz * (h_idx + 1.0)               # [nhar]
    hf_mask    = harm_freqs >= float(hf_start_hz)

    x_out = log_har.copy()
    if np.any(hf_mask):
        alpha = float(np.clip(strength, 0.0, 1.0))
        x_out[hf_mask] = (1.0 - alpha) * log_har[hf_mask] + alpha * log_env[hf_mask]

    return np.exp(x_out).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Opt 2 辅助 —— 高频亮度亏缺诊断
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_hf_deficit(
    logmel: np.ndarray,
    src_bin_mean: np.ndarray,
    mel_freqs: np.ndarray,
    auto_fmin: float,
    noise_tilt_ref_hz: float,
) -> None:
    """
    打印高频亮度亏缺诊断信息，并建议 noise_tilt_db_per_oct 的参考值。
    此函数仅输出建议，不修改任何数据。
    """
    auto_fmin = float(auto_fmin)
    hf_mask   = mel_freqs >= auto_fmin
    if not np.any(hf_mask):
        print("[NoiseTilt-Diag] 无高频 mel bins（>= %.0f Hz），跳过诊断。" % auto_fmin)
        return

    syn_bin_mean = np.mean(logmel, axis=0).astype(np.float32)
    src_hf_mean  = float(np.mean(src_bin_mean[hf_mask]))
    syn_hf_mean  = float(np.mean(syn_bin_mean[hf_mask]))

    # log-mel 域差值 → amplitude dB
    deficit_logmel  = src_hf_mean - syn_hf_mean
    deficit_amp_db  = deficit_logmel * 20.0 / math.log(10.0)

    # 从 auto_fmin 到 fmax 的倍频程数（估算）
    fmax_approx = float(mel_freqs[-1])
    n_oct       = math.log2(max(fmax_approx, auto_fmin + 1.0) / max(auto_fmin, 1.0))
    # 假设倾斜从 ref_hz 开始，在 [auto_fmin, fmax] 内的平均增益约为 tilt × (n_oct/2)
    ref_hz  = max(float(noise_tilt_ref_hz), 1.0)
    oct_ref = math.log2(max(auto_fmin, ref_hz * 0.5) / ref_hz)  # 高频区起点 vs ref 的倍频程
    # 在高频区 [auto_fmin, fmax]，倾斜产生的平均 log-mel 提升 ≈ tilt × (oct_ref + n_oct/2) × ln(10)/20
    ln10_20 = math.log(10.0) / 20.0
    mean_oct = oct_ref + n_oct / 2.0
    if abs(mean_oct) > 1e-3:
        suggested_tilt = deficit_amp_db / (mean_oct * 10.0)  # power dB/oct
        # convert to amplitude dB/oct for human readability:  amp_dB/oct = power_dB/oct / 2
        # But noise_tilt_db_per_oct is power dB/oct in this implementation
        suggested_tilt_amp = suggested_tilt / 2.0
    else:
        suggested_tilt = suggested_tilt_amp = 0.0

    print("[NoiseTilt-Diag] 高频亮度诊断（>= %.0f Hz）：" % auto_fmin)
    print("  源参考均值=%.4f  合成均值=%.4f  亏缺=%.4f (log) / %.2f dB (amplitude)"
          % (src_hf_mean, syn_hf_mean, deficit_logmel, deficit_amp_db))
    print("  建议 --noise_tilt_db_per_oct ≈ %.1f（功率 dB/oct，枢轴 %.0f Hz）"
          % (suggested_tilt, ref_hz))
    print("  （此值仅供参考；下次实验时设置后重新运行以验证效果）")


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLSM CSV → log-Mel for NSF-HiFiGAN（物理正确版 v2：含高频优化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 输入/输出 ────────────────────────────────────────────────────────────
    ap.add_argument("--ref_csv",  required=True, help="参考 CSV（原始 F0）")
    ap.add_argument("--edit_csv", default="",    help="编辑 CSV（修改后 F0），pitch_shift=1 时必须提供")
    ap.add_argument("--orig_csv", default="",
                    help="可选：原始（未编辑）CSV，用于提供 vt_db_base。"
                         "若提供，则 vt_db_base 从此 CSV 读取，vt_db_tuned 从 ref_csv 读取。"
                         "har_vt_comp 自动用 vt_warped/vt_orig 比值修正谐波振幅 → "
                         "支持'只 warp vt' 的编辑工作流（UTAU/WORLD 风格）")
    ap.add_argument("--out_npy",  default="input.npy", help="输出 mel .npy 路径")
    ap.add_argument("--save_pre_align_npy", default="", help="可选：对齐前的 mel 保存路径（调试用）")

    ap.add_argument("--pitch_shift", type=int, default=0, choices=[0, 1],
                    help="0=保持原音高，1=使用 edit_csv 的 F0 变调")

    # ── 编辑模式（UTAU-style gender/formant 编辑）─────────────────────────────
    ap.add_argument("--edit_mode", type=int, default=0, choices=[0, 1],
                    help="0=默认（match-original 高保真重合成模式，所有补偿模块按 CLI 参数生效）；"
                         "1=编辑模式（关闭所有 match-original 补偿，最大化 per-harmonic 编辑精度，"
                         "适用于配合 --orig_csv 做 gender/formant 编辑）。"
                         "保留：har_vt_comp（应用 vt 编辑）、噪声 psd 处理、mel filterbank。"
                         "关闭：hn_calib、harm_spread、resonance_comp、align、所有 *_correct 模块。")

    # ── 谐波聚焦模式（让 per-harmonic 编辑 1:1 反映到 mel）──────────────────
    ap.add_argument("--harmonic_focus", type=int, default=0, choices=[0, 1],
                    help="0=默认（谐波经 STFT → 三角 mel 滤波器扩散，每根谐波能量分到 3-5 个 mel bin）；"
                         "1=谐波聚焦模式（每根谐波 ampl^2 直接累加到最近的 mel bin，"
                         "完全跳过 STFT 滤波器扩散，每根谐波只贡献到一个 mel bin）。"
                         "效果：PHG 测量的 per-harmonic 传递率从 ~0.5 跳到 ~1.0。"
                         "代价：mel 频谱不再「自然平滑」，HiFiGAN 听感可能有量化感（适合验证不适合产品）。"
                         "噪声部分仍按原方式处理（STFT → mel filter）。")

    # ── ★ 改进 9/11：导出 H/N 分量供测量端「无反演直读」─────────────────────
    ap.add_argument("--save_harm_mel_npy", type=str, default=None,
                    help="路径（.npy）：导出 harm_mel_focus（谐波 mel 功率，纯 H）。"
                         "仅 harmonic_focus=1 时有效。测量端读取后可跳过 H+N 反演公式。")
    ap.add_argument("--save_noise_mel_npy", type=str, default=None,
                    help="路径（.npy）：导出 noise_mel_pow（噪声 mel 功率，纯 N）。"
                         "测量端可作为「真值噪声底」用于反演精化。")

    # ── 声学参数 ─────────────────────────────────────────────────────────────
    ap.add_argument("--sr",     type=int,   default=44100)
    ap.add_argument("--n_mels", type=int,   default=128)
    ap.add_argument("--fmin",   type=float, default=40.0)
    ap.add_argument("--fmax",   type=float, default=16000.0)
    ap.add_argument("--max_hz", type=float, default=16000.0, help="忽略高于此频率的谐波")
    ap.add_argument("--harm_spread", type=float, nargs=3, default=[0.25, 0.5, 0.25],
                    help="谐波在 FFT bin 上的 3 点扩散权重 [-1,0,+1]（自适应模式中用作高音区权重）")
    ap.add_argument("--harm_accum_mode", type=str, default="power",
                    choices=["power", "amp"],
                    help="谐波累加模式：power=功率累加（默认，(mag*w)^2），"
                         "amp=幅度累加（mag*w，与Hann扩散组合时有实质差异）")
    # ── [Opt 7c] 自适应扩散权重 ──────────────────────────────────────────────
    ap.add_argument("--harm_spread_adaptive", type=int, default=0, choices=[0, 1],
                    help="[Opt 7c] 自适应扩散：根据 F0 在 [0,1,0] 与 harm_spread 间插值（默认 0=关闭）")
    ap.add_argument("--harm_spread_bins_low", type=float, default=4.0,
                    help="[Opt 7c] 过渡起点 bins/谐波（默认 4.0，约 86 Hz）；低于此值侧边权重→0")
    ap.add_argument("--harm_spread_bins_high", type=float, default=6.0,
                    help="[Opt 7c] 过渡终点 bins/谐波（默认 6.0，约 129 Hz）；高于此值使用 harm_spread")

    # ── STFT 单位换算 ─────────────────────────────────────────────────────────
    ap.add_argument("--harm_stft_gain_mode", type=str, default="hann",
                    choices=["hann", "nfft4", "custom", "none"])
    ap.add_argument("--harm_stft_gain", type=float, default=1.0)

    # ── H/N 校正 ─────────────────────────────────────────────────────────────
    ap.add_argument("--hn_csv",           default="")
    ap.add_argument("--hn_calib_mode",    default="frame",
                    choices=["none", "global", "frame"])
    ap.add_argument("--hn_gain_min_db",   type=float, default=-40.0)
    ap.add_argument("--hn_gain_max_db",   type=float, default=40.0)
    ap.add_argument("--hn_gain_smooth",   type=int,   default=5)
    ap.add_argument("--hn_calib_eps",     type=float, default=1e-12)
    ap.add_argument("--save_hn_debug_csv", default="")

    # ── 共振峰估计 ────────────────────────────────────────────────────────────
    ap.add_argument("--formant_method", default="peaks", choices=["peaks", "simple"])
    ap.add_argument("--f1_band", type=float, nargs=2, default=[200.0, 1200.0])
    ap.add_argument("--f2_band", type=float, nargs=2, default=[800.0, 3500.0])
    ap.add_argument("--vt_smooth",        type=int,   default=5)
    ap.add_argument("--peak_prom_ratio",  type=float, default=0.10)
    ap.add_argument("--bw_floor_hz",      type=float, default=60.0)
    ap.add_argument("--print_formants",   type=int,   default=0, choices=[0, 1])

    # ── 共振峰频移 bump（可选） ───────────────────────────────────────────────
    ap.add_argument("--r1_tuning",          type=int,   default=0, choices=[0, 1])
    ap.add_argument("--r2_tuning",          type=int,   default=0, choices=[0, 1])
    ap.add_argument("--k_harm",             type=float, default=1.0)
    ap.add_argument("--tuning_weight_mode", default="distance", choices=["distance", "legacy"])
    ap.add_argument("--distance_weight_func", default="lorentz", choices=["lorentz", "gauss"])
    ap.add_argument("--lambda_max",         type=float, default=0.8)
    ap.add_argument("--use_pitch_gate",     type=int,   default=0, choices=[0, 1])
    ap.add_argument("--shift_start_st",     type=float, default=2.0)
    ap.add_argument("--shift_full_st",      type=float, default=10.0)
    ap.add_argument("--enable_downshift",   action="store_true")
    ap.add_argument("--delta_start_hz",     type=float, default=120.0)
    ap.add_argument("--delta_width_hz",     type=float, default=300.0)
    ap.add_argument("--gain_db",            type=float, default=6.0)
    ap.add_argument("--sigma_hz",           type=float, default=180.0)
    ap.add_argument("--keep_band_hz",       type=float, default=4000.0)

    # ── 谐波区域能量补偿 ─────────────────────────────────────────────────────
    ap.add_argument("--resonance_harmonic_comp_enable",          type=int,   default=0, choices=[0, 1])
    ap.add_argument("--resonance_harmonic_comp_strength",        type=float, default=1.0)
    ap.add_argument("--resonance_harmonic_comp_gain_cap_db",     type=float, default=4.0)
    ap.add_argument("--resonance_harmonic_comp_disable_downshift",type=int,  default=1, choices=[0, 1])
    ap.add_argument("--resonance_harmonic_comp_use_f1",          type=int,   default=1, choices=[0, 1])
    ap.add_argument("--resonance_harmonic_comp_use_f2",          type=int,   default=1, choices=[0, 1])
    ap.add_argument("--resonance_harmonic_comp_bw_scale",        type=float, default=1.0)
    ap.add_argument("--resonance_harmonic_comp_min_bw_hz",       type=float, default=120.0)
    ap.add_argument("--resonance_harmonic_comp_debug",           type=int,   default=0, choices=[0, 1])

    # ── 声道幅度补偿 ─────────────────────────────────────────────────────────
    ap.add_argument("--har_vt_comp_mix",  type=float, default=1.0)
    ap.add_argument("--preserve_har_power", type=int, default=1, choices=[0, 1])
    ap.add_argument("--vt_comp_time_smooth", type=int, default=0,
                    help="VT 谱时间平滑窗口帧数：\n"
                         "  0  = 逐帧（默认，保留每帧 VT 特征，鼻音/辅音过渡准确）\n"
                         "  -1 = 全局平均（已不推荐：会破坏鼻音、辅音帧的谱特征）\n"
                         "  N>0= 时间滑动平均 N 帧（适度平滑分析噪声，建议 3–7）")

    # ── 时间平滑 ─────────────────────────────────────────────────────────────
    ap.add_argument("--time_smooth", type=int, default=3)
    ap.add_argument("--noise_time_smooth", type=int, default=0)

    # ── 噪声增益 ─────────────────────────────────────────────────────────────
    ap.add_argument("--nonharmonic_gain_db", type=float, default=0.0)

    # ── 后处理（全局对齐） ────────────────────────────────────────────────────
    ap.add_argument("--max_logmel",      type=float, default=math.log(50.0))
    ap.add_argument("--global_gain_db",  type=float, default=0.0)
    ap.add_argument("--align_mode",      default="mean",
                    choices=["auto", "none", "mean", "meanstd"])
    ap.add_argument("--target_mel_npy",  default="",
                    help="源参考 mel .npy（用于分布对齐和 Opt 1/5/6 谱包络修正）")
    ap.add_argument("--match_ref_stats", type=int, default=1, choices=[0, 1])
    ap.add_argument("--ref_mean",        type=float, default=-6.047)
    ap.add_argument("--ref_std",         type=float, default=2.4656184)
    ap.add_argument("--clip_min",        type=float, default=-15.0)
    ap.add_argument("--clip_max",        type=float, default=5.0)
    ap.add_argument("--clip_after_gain", type=int,   default=1, choices=[0, 1])
    ap.add_argument("--print_mel_stats", type=int,   default=1, choices=[0, 1])

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] Opt 2 —— 噪声 PSD 高频倾斜修正
    # ════════════════════════════════════════════════════════════════════════
    ap.add_argument("--noise_tilt_enable", type=int, default=0, choices=[0, 1],
                    help="[Opt 2] 启用噪声 PSD 高频倾斜修正（默认关闭）")
    ap.add_argument("--noise_tilt_db_per_oct", type=float, default=2.0,
                    help="[Opt 2] 噪声倾斜量（功率 dB/octave，正值=高频提升）。"
                         "建议范围 1–4；对低音区谐波覆盖不足时可适当增大。")
    ap.add_argument("--noise_tilt_ref_hz", type=float, default=2000.0,
                    help="[Opt 2] 噪声倾斜枢轴频率（Hz，此处增益=0 dB）。"
                         "建议设为 1000–3000 Hz。")
    ap.add_argument("--noise_tilt_auto_diag", type=int, default=1, choices=[0, 1],
                    help="[Opt 2] 合成后打印高频亮度亏缺诊断及建议倾斜量（不修改数据）。"
                         "需要 --target_mel_npy。")
    ap.add_argument("--noise_tilt_auto_fmin", type=float, default=5000.0,
                    help="[Opt 2] 诊断时统计高频亮度的起始频率（Hz）。")

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] Opt 1 —— 高频源参考混合修正
    # ════════════════════════════════════════════════════════════════════════
    ap.add_argument("--hf_blend_enable", type=int, default=0, choices=[0, 1],
                    help="[Opt 1] 启用高频段源参考混合修正（默认关闭）。需要 --target_mel_npy。")
    ap.add_argument("--hf_blend_start_hz", type=float, default=4000.0,
                    help="[Opt 1] 高频混合起始频率（Hz）；低于此频率不修正。")
    ap.add_argument("--hf_blend_end_hz", type=float, default=8000.0,
                    help="[Opt 1] 高频混合达到最大权重的频率（Hz）。")
    ap.add_argument("--hf_blend_alpha", type=float, default=0.8,
                    help="[Opt 1] 最大混合强度（0=不修正，1=完全对齐到源参考均值）。")

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] Opt 5 —— 逐频带均值对齐
    # ════════════════════════════════════════════════════════════════════════
    ap.add_argument("--perband_align_enable", type=int, default=0, choices=[0, 1],
                    help="[Opt 5] 启用逐频带均值对齐（默认关闭）。需要 --target_mel_npy。")
    ap.add_argument("--perband_n_bands", type=int, default=8,
                    help="[Opt 5] 频带分段数量（建议 4–16）。")
    ap.add_argument("--perband_alpha", type=float, default=1.0,
                    help="[Opt 5] 逐频带修正强度（0–1）。")

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] Opt 6 —— 平滑谱包络修正
    # ════════════════════════════════════════════════════════════════════════
    ap.add_argument("--src_env_correct_enable", type=int, default=0, choices=[0, 1],
                    help="[Opt 6] 启用平滑谱包络修正（默认关闭）。需要 --target_mel_npy。")
    ap.add_argument("--src_env_smooth_bins", type=int, default=12,
                    help="[Opt 6] 跨 mel-bin 的移动平均窗口宽度（bin 数量）。"
                         "越大越平滑（只修正宽带包络），建议 8–20。")
    ap.add_argument("--src_env_alpha", type=float, default=1.0,
                    help="[Opt 6] 平滑谱包络修正强度（0–1）。")

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] Opt 7 —— 高频段谐波包络平滑（LPC）
    # ════════════════════════════════════════════════════════════════════════
    ap.add_argument("--hf_harm_lpc_enable", type=int, default=0, choices=[0, 1],
                    help="[Opt 7] 启用高频段谐波包络 LPC 平滑（默认关闭）。"
                         "在每帧谐波幅度合成前，用 LPC 全极点包络平滑高频段。")
    ap.add_argument("--hf_harm_lpc_order", type=int, default=12,
                    help="[Opt 7] LPC 阶数（默认 12）。越低→越平滑，越高→越接近原始。建议 8–20。")
    ap.add_argument("--hf_harm_lpc_start_hz", type=float, default=2000.0,
                    help="[Opt 7] 高频平滑起点 Hz（默认 2000.0）。"
                         "低于此频率的谐波不受影响。")
    ap.add_argument("--hf_harm_lpc_strength", type=float, default=1.0,
                    help="[Opt 7] 混合强度（0–1，默认 1.0）。"
                         "0=完全保留原始，1=完全用 LPC 包络替换高频段。")

    args = ap.parse_args()

    # ══════════════════════════════════════════════════════════════════════════
    #  EDIT MODE：UTAU-style 编辑模式（关闭所有 match-original 补偿）
    # ══════════════════════════════════════════════════════════════════════════
    # 设计动机：
    #   原版 13 个补偿模块（hn_calib / harm_spread / align / perband / src_env /
    #   hf_blend / noise_tilt / resonance_comp / hf_harm_lpc / vt_comp_time_smooth
    #   ...）目的是"让合成 mel 在 vocoder 训练分布内"。但这些模块会**逐帧、
    #   逐谐波地与编辑信号产生交叉污染**，导致 per-harmonic 编辑精度被噪声底
    #   抹平（实测 PHG slope < 0.1）。
    #
    #   编辑模式下，关闭所有这些补偿，保留：
    #     ✓ har_vt_comp（需要它把 vt 编辑应用到谐波 ampl）
    #     ✓ 噪声 psd → noise_power 处理（音质必需）
    #     ✓ mel filterbank（HiFiGAN 输入格式必需）
    #     ✓ log + clip（HiFiGAN 输入格式必需）
    #
    #   关闭：
    #     ✗ hn_calib_mode → "none"（per-frame uniform gain 是 PHG 噪声底主因）
    #     ✗ harm_spread → [0, 1, 0]（单 STFT bin 放置，无邻 bin 串扰）
    #     ✗ harm_spread_adaptive → 0（不再 f0 自适应）
    #     ✗ resonance_harmonic_comp_enable → 0
    #     ✗ noise_tilt_enable → 0
    #     ✗ hf_blend_enable → 0
    #     ✗ perband_align_enable → 0
    #     ✗ src_env_correct_enable → 0
    #     ✗ hf_harm_lpc_enable → 0
    #     ✗ align_mode → "none"
    #     ✗ vt_comp_time_smooth → 0（用逐帧 vt，不用全局平均）
    #
    #   适用场景：CSV 编辑（特别是 gender / formant warp）的下游 mel 合成
    #   配合：apply_formant_warp.py --vt-only + 本脚本 --orig_csv + --edit_mode=1
    if args.edit_mode == 1:
        print("=" * 76)
        print("[EditMode] ★ 编辑模式启用 ★ 关闭所有 match-original 补偿")
        print("           (适用于 CSV 编辑场景，最大化 per-harmonic 编辑精度)")
        print("=" * 76)
        _overrides = {
            "hn_calib_mode":                  ("none", str(args.hn_calib_mode)),
            "harm_spread":                    ([0.0, 1.0, 0.0], list(args.harm_spread)),
            "harm_spread_adaptive":           (0, int(args.harm_spread_adaptive)),
            "resonance_harmonic_comp_enable": (0, int(args.resonance_harmonic_comp_enable)),
            "noise_tilt_enable":              (0, int(args.noise_tilt_enable)),
            "hf_blend_enable":                (0, int(args.hf_blend_enable)),
            "perband_align_enable":           (0, int(args.perband_align_enable)),
            "src_env_correct_enable":         (0, int(args.src_env_correct_enable)),
            "hf_harm_lpc_enable":             (0, int(args.hf_harm_lpc_enable)),
            "align_mode":                     ("none", str(args.align_mode)),
            "vt_comp_time_smooth":            (0, int(args.vt_comp_time_smooth)),
        }
        for k, (new_v, old_v) in _overrides.items():
            if str(new_v) != str(old_v):
                print(f"[EditMode]   {k}: {old_v}  →  {new_v}")
            setattr(args, k, new_v)
        print("[EditMode] 保留：har_vt_comp_mix=%.2f（应用 vt 编辑必需）, "
              "噪声 psd, mel filterbank, log+clip" % float(args.har_vt_comp_mix))
        if not args.orig_csv:
            print("[EditMode] ⚠ 警告：未提供 --orig_csv！vt 编辑不会通过 har_vt_comp 生效。")
            print("[EditMode]    建议配合 apply_formant_warp.py --vt-only 和 --orig_csv 一起用。")
        print("=" * 76)

    # ── 加载 H/N CSV ─────────────────────────────────────────────────────────
    hn_info = parse_hn_ratio_csv(args.hn_csv) if args.hn_csv else {}
    if hn_info:
        print("[HN] 已加载 H/N 比例帧数:", len(hn_info))

    # ── 加载主 CSV ───────────────────────────────────────────────────────────
    if (args.r1_tuning == 1 or args.r2_tuning == 1) and args.pitch_shift == 0:
        if args.edit_csv and os.path.exists(args.edit_csv):
            print("[Info] r1/r2 tuning 已开启，自动设置 pitch_shift=1")
            args.pitch_shift = 1
        else:
            print("[Warn] r1/r2 tuning 已开启但 edit_csv 不存在，保持 pitch_shift=0")

    ref = parse_legacy_llsm_csv(args.ref_csv)
    if args.pitch_shift == 0:
        edit   = ref
        frames = sorted(ref.keys())
    else:
        if not args.edit_csv or not os.path.exists(args.edit_csv):
            raise FileNotFoundError("pitch_shift=1 需要有效的 --edit_csv，未找到: %s" % args.edit_csv)
        edit   = parse_legacy_llsm_csv(args.edit_csv)
        frames = sorted(set(ref.keys()) & set(edit.keys()))
    if not frames:
        raise ValueError("ref_csv 与 edit_csv 没有交叠帧")

    # ── 可选：加载原始 CSV，用于 vt_db_base ─────────────────────────────────
    # 设计意图：当 --orig_csv 提供时，vt_db_base 从原始 CSV 读取，
    # vt_db_tuned 从 ref_csv 读取（应为已 warp 过的 CSV）。
    # 这样 har_vt_comp 会自动应用 (vt_tuned/vt_base) = (vt_warped/vt_orig)
    # 作为谐波振幅修正 → 实现"只 warp vt 即可生效"的 UTAU 风格编辑。
    orig: Optional[Dict[int, Dict[str, Any]]] = None
    if args.orig_csv and os.path.exists(args.orig_csv):
        orig = parse_legacy_llsm_csv(args.orig_csv)
        print("[OrigCSV] 已加载原始 CSV: %s (%d 帧)" % (args.orig_csv, len(orig)))
        print("[OrigCSV] vt_db_base 将从原始 CSV 读取，vt_db_tuned 从 ref_csv 读取")
        # 帧集取交集，确保 orig 包含所有要处理的帧
        frames = sorted(set(frames) & set(orig.keys()))
        if not frames:
            raise ValueError("orig_csv 与 ref_csv 没有交叠帧")

    first = frames[0]
    npsd  = int(ref[first]["npsd"])
    n_fft = (npsd - 1) * 2
    T     = len(frames)

    vt_len    = int(ref[first]["vt_db"].shape[0]) if ref[first]["vt_db"] is not None else 0
    vt_bin_hz = (np.linspace(0.0, args.sr / 2.0, vt_len, dtype=np.float32)
                 if vt_len > 0 else None)

    # ── VT 谱时间预平滑 ────────────────────────────────────────────────────
    vt_db_smoothed_map: Dict[int, Optional[np.ndarray]] = {}
    vt_comp_smooth = int(args.vt_comp_time_smooth)

    if vt_len > 0 and vt_comp_smooth != 0 and float(args.har_vt_comp_mix) > 0.0:
        vt_db_mat_full = np.zeros((T, vt_len), dtype=np.float32)
        # 若提供 orig_csv，base 用 orig 的 vt（这样 vt_db_smoothed 反映原始包络）
        vt_src_dict = orig if orig is not None else ref
        for ti, fr in enumerate(frames):
            vdb = vt_src_dict[fr]["vt_db"]
            if vdb is not None and int(vdb.size) == vt_len:
                vt_db_mat_full[ti] = vdb.astype(np.float32)

        if vt_comp_smooth == -1:
            mean_vt = np.mean(vt_db_mat_full, axis=0, keepdims=True)
            vt_db_smoothed = np.repeat(mean_vt, T, axis=0)
            print("[VTSmooth] ⚠ 全局平均 VT 谱（已不推荐：会破坏鼻音/辅音帧的谱特征）")
        else:
            vt_db_smoothed = uniform_filter1d(
                vt_db_mat_full, size=int(vt_comp_smooth), axis=0, mode="nearest"
            )
            print("[VTSmooth] VT 谱时间平滑 window=%d 帧" % vt_comp_smooth)

        vt_db_smoothed_map = {int(frames[ti]): vt_db_smoothed[ti] for ti in range(T)}
    else:
        if vt_len > 0:
            print("[VTSmooth] VT 谱逐帧模式（vt_comp_time_smooth=0）")

    # ── Mel 滤波器（librosa，与 HiFiGAN 一致） ───────────────────────────────
    if librosa_mel_fn is None:
        raise ImportError("需要安装 librosa 以构建 mel 滤波器")

    mel_filter_ours = mel_filterbank(args.sr, n_fft, args.n_mels, args.fmin, args.fmax)
    mel_filter_lib  = librosa_mel_fn(sr=args.sr, n_fft=n_fft, n_mels=args.n_mels,
                                     fmin=args.fmin, fmax=args.fmax).astype(np.float32)
    if mel_filter_ours.shape == mel_filter_lib.shape:
        max_abs = float(np.max(np.abs(mel_filter_ours - mel_filter_lib)))
        print("[MelCheck] ours_vs_librosa max_abs=%.8g" % max_abs)
    mel_filter = mel_filter_lib

    # ── 噪声功率谱 [T, npsd]（物理正确：PSD × df） ───────────────────────────
    df = float(args.sr) / float(n_fft)
    psd_src = edit if args.pitch_shift == 1 else ref
    psd_db_mat = np.zeros((T, npsd), dtype=np.float32)
    for ti, fr in enumerate(frames):
        psd_db_mat[ti] = psd_src[fr]["psd_db"]

    noise_power_spec = (np.power(10.0, psd_db_mat / 10.0) * df).astype(np.float64)
    print("[STFTUnit] 噪声: sqrt(PSD × df)，df=%.6g Hz/bin" % df)

    # ════════════════════════════════════════════════════════════════════════
    #  [Opt 2] 噪声 PSD 高频倾斜修正（在合并谐波前，功率域施加）
    # ════════════════════════════════════════════════════════════════════════
    if args.noise_tilt_enable:
        noise_power_spec = apply_noise_spectral_tilt(
            noise_power_spec,
            sr            = int(args.sr),
            n_fft         = int(n_fft),
            tilt_db_per_oct = float(args.noise_tilt_db_per_oct),
            ref_hz        = float(args.noise_tilt_ref_hz),
        )
        print("[NoiseTilt] [Opt 2] 已施加噪声 PSD 倾斜：%.2f 功率dB/oct，枢轴 %.0f Hz"
              % (args.noise_tilt_db_per_oct, args.noise_tilt_ref_hz))

    # ── 谐波 STFT 增益 ───────────────────────────────────────────────────────
    harm_stft_gain = compute_harm_stft_gain(n_fft, args.harm_stft_gain_mode, args.harm_stft_gain)
    print("[STFTUnit] 谐波 STFT 增益=%.6g（模式=%s）" % (harm_stft_gain, args.harm_stft_gain_mode))

    # ── 谐波累加谱 [T, npsd]（power 模式存功率，amp 模式存幅度）──────────────
    harm_acc_spec        = np.zeros((T, npsd), dtype=np.float64)
    harm_accum_mode      = str(args.harm_accum_mode)
    spread_w_high        = np.asarray(args.harm_spread, dtype=np.float32)   # 高音区权重
    harm_spread_adaptive = (int(args.harm_spread_adaptive) == 1)
    spread_bins_low      = float(args.harm_spread_bins_low)
    spread_bins_high     = float(args.harm_spread_bins_high)
    bin_hz              = float(args.sr) / float(n_fft)
    eps                 = 1e-6
    debug_print_left    = 10

    # ── ★ harmonic_focus 模式：mel 域谐波直接累加（每根谐波 → 唯一 mel bin） ─
    # 不经过 STFT 三角滤波器扩散，让 per-harmonic 编辑在 mel 上 1:1 反映
    harm_mel_focus = None
    mel_freqs_for_focus = None
    if args.harmonic_focus == 1:
        harm_mel_focus = np.zeros((T, int(args.n_mels)), dtype=np.float64)
        # mel bin 的中心频率（用于"最近 bin"查找）
        mel_freqs_for_focus = mel_frequencies(int(args.n_mels),
                                              float(args.fmin),
                                              float(args.fmax))
        print("=" * 76)
        print("[HarmonicFocus] ★ 谐波聚焦模式启用 ★")
        print("[HarmonicFocus]   每根谐波 ampl^2 直接累加到最近的 mel bin")
        print("[HarmonicFocus]   完全跳过 STFT → mel filter 三角扩散")
        print("[HarmonicFocus]   per-harmonic 编辑在 mel 域 1:1 反映")
        print("[HarmonicFocus]   噪声部分仍按原方式处理（STFT → mel filter）")
        print("=" * 76)

    if harm_spread_adaptive:
        print("[SpreadAdapt] [Opt 7c] 自适应扩散已启用："
              " bins_low=%.1f（约%.0fHz）→[0,1,0]"
              " bins_high=%.1f（约%.0fHz）→[%.2f,%.2f,%.2f]" % (
              spread_bins_low,  spread_bins_low  * bin_hz,
              spread_bins_high, spread_bins_high * bin_hz,
              spread_w_high[0], spread_w_high[1], spread_w_high[2]))
    else:
        print("[SpreadFixed] 固定扩散权重 [%.2f, %.2f, %.2f]" % tuple(spread_w_high))

    for ti, fr in enumerate(frames):
        f0_ref  = float(ref[fr]["f0"])
        f0_use  = float(edit[fr]["f0"]) if args.pitch_shift == 1 else f0_ref

        nhar_ref = int(ref[fr]["nhar"])
        nhar_new = int(edit[fr]["nhar"])
        nhar     = int(min(nhar_ref, nhar_new))

        if f0_use <= 0.0 or nhar <= 0:
            continue

        har_old = ref[fr]["har"]
        if har_old is None or har_old.size < nhar:
            continue

        pitch_gate = 1.0
        if args.use_pitch_gate == 1:
            d_st = semitone_shift(f0_ref, f0_use)
            if (not args.enable_downshift) and (d_st < 0.0):
                pitch_gate = 0.0
            else:
                x = abs(d_st)
                if args.shift_full_st <= args.shift_start_st:
                    pitch_gate = 1.0 if x >= args.shift_start_st else 0.0
                else:
                    pitch_gate = float(np.clip((x - args.shift_start_st) /
                                               (args.shift_full_st - args.shift_start_st), 0.0, 1.0))

        # 当 --orig_csv 提供时：base 来自 orig（原始 vt），tuned 来自 ref（编辑后的 vt）
        # 否则：二者都来自 ref（向后兼容）
        if orig is not None:
            vt_db_base_src  = vt_db_smoothed_map.get(int(fr), orig[fr]["vt_db"]) \
                              if vt_db_smoothed_map else orig[fr]["vt_db"]
            vt_db_base  = vt_db_base_src
            vt_db_tuned = ref[fr]["vt_db"]      # 编辑后的 vt 作为 tuned
        else:
            vt_db_base  = vt_db_smoothed_map.get(int(fr), ref[fr]["vt_db"]) \
                          if vt_db_smoothed_map else ref[fr]["vt_db"]
            vt_db_tuned = vt_db_base
        f1_ref = b1 = f2_ref = b2 = None
        if vt_db_base is not None and vt_bin_hz is not None and vt_db_base.size == vt_len > 0:
            fn = (estimate_formant_from_vt_peaks if args.formant_method == "peaks"
                  else estimate_formant_from_vt_simple)
            kw = dict(vt_db=vt_db_base, vt_bin_hz=vt_bin_hz,
                      smooth_size=int(args.vt_smooth), bw_floor_hz=float(args.bw_floor_hz))
            if args.formant_method == "peaks":
                kw["prom_ratio"] = float(args.peak_prom_ratio)
            f1_ref, b1 = fn(fmin=float(args.f1_band[0]), fmax=float(args.f1_band[1]), **kw)
            f2_ref, b2 = fn(fmin=float(args.f2_band[0]), fmax=float(args.f2_band[1]), **kw)
            if args.print_formants == 1 and debug_print_left > 0:
                debug_print_left -= 1
                print("[Formant] frame=%d f0_ref=%.2f f0_use=%.2f F1=%s B1=%s F2=%s B2=%s" % (
                      int(fr), f0_ref, f0_use,
                      "%.1f" % f1_ref if f1_ref else "N/A", "%.1f" % b1 if b1 else "N/A",
                      "%.1f" % f2_ref if f2_ref else "N/A", "%.1f" % b2 if b2 else "N/A"))

            if vt_db_base is not None:
                for tuning_on, f_ref, b_ref, k_harm_mult, f_min_clamp, f_max_clamp in [
                    (args.r1_tuning == 1, f1_ref, b1, float(args.k_harm), 200.0, 1500.0),
                    (args.r2_tuning == 1, f2_ref, b2, 2.0,               600.0, 4500.0),
                ]:
                    if not tuning_on or f_ref is None or b_ref is None:
                        continue
                    if args.tuning_weight_mode == "distance":
                        w = distance_weight(k_harm_mult * f0_use, f_ref, b_ref, args.distance_weight_func)
                        lam = float(np.clip(args.lambda_max * w * pitch_gate, 0.0, 1.0))
                    else:
                        drive   = k_harm_mult * f0_use
                        lam_near = (drive - (f_ref - args.delta_start_hz)) / max(args.delta_width_hz, 1e-6)
                        lam     = float(np.clip(args.lambda_max * float(np.clip(lam_near, 0.0, 1.0)) * pitch_gate, 0.0, 1.0))
                    r_target = float(np.clip((1.0 - lam) * f_ref + lam * k_harm_mult * f0_use,
                                             f_min_clamp, f_max_clamp))
                    vt_db_tuned = apply_resonance_tuning_bump(
                        vt_db_tuned, vt_bin_hz, f_ref, r_target, lam,
                        float(args.gain_db), float(args.sigma_hz), float(args.keep_band_hz))

        nhar_use = sum(1 for i in range(nhar) if f0_use * (i + 1) <= float(args.max_hz))
        if nhar_use <= 0:
            continue

        vt_amp_orig   = db_to_amp(vt_db_base.astype(np.float32))  if vt_db_base  is not None and vt_bin_hz is not None else None
        vt_amp_tuned  = db_to_amp(vt_db_tuned.astype(np.float32)) if vt_db_tuned is not None and vt_bin_hz is not None else None

        har_vt_mix = float(np.clip(args.har_vt_comp_mix, 0.0, 1.0))
        har_final  = har_old[:nhar_use].astype(np.float32).copy()

        if vt_amp_orig is not None and vt_amp_tuned is not None and f0_ref > 0.0 and har_vt_mix > 0.0:
            har_sf = np.zeros(nhar_use, dtype=np.float32)
            for i in range(nhar_use):
                fo = float(f0_ref) * (i + 1)
                fn = float(f0_use) * (i + 1)
                a_old = max(float(np.interp(fo, vt_bin_hz, vt_amp_orig)), eps)
                a_new = float(np.interp(fn, vt_bin_hz, vt_amp_tuned))
                har_sf[i] = float(har_old[i]) * (a_new / a_old)
            har_final = (1.0 - har_vt_mix) * har_final + har_vt_mix * har_sf
            if args.preserve_har_power == 1:
                base_p = float(np.sum((har_old[:nhar_use].astype(np.float32)) ** 2))
                new_p  = float(np.sum((har_final.astype(np.float32)) ** 2))
                if base_p > 0.0 and new_p > 0.0:
                    har_final *= math.sqrt(base_p / new_p)

        har_final = apply_resonance_harmonic_compensation(
            har_final, har_old[:nhar_use].astype(np.float32),
            f0_ref, f0_use, f1_ref, b1, f2_ref, b2, args)

        # ── [Opt 7] 高频段谐波包络 LPC 平滑 ─────────────────────────────────
        if args.hf_harm_lpc_enable and nhar_use >= 4:
            har_final = apply_hf_harmonic_lpc_smooth(
                har            = har_final,
                f0_hz          = float(f0_use),
                hf_start_hz    = float(args.hf_harm_lpc_start_hz),
                lpc_order      = int(args.hf_harm_lpc_order),
                strength       = float(args.hf_harm_lpc_strength),
            )

        # ── [Opt 7c] 自适应扩散：每帧根据 f0 计算侧边权重 ─────────────────────
        if harm_spread_adaptive:
            spacing_bins = f0_use / bin_hz
            alpha  = float(np.clip(
                (spacing_bins - spread_bins_low) /
                max(spread_bins_high - spread_bins_low, 1e-6),
                0.0, 1.0))
            w_side   = float(spread_w_high[0]) * alpha   # 低音区→0，高音区→harm_spread[0]
            w_center = 1.0 - 2.0 * w_side
            spread_w = np.array([w_side, w_center, w_side], dtype=np.float32)
        else:
            spread_w = spread_w_high

        for i in range(nhar_use):
            freq  = float(f0_use) * (i + 1)
            mag_h = float(har_final[i]) * float(harm_stft_gain)
            if mag_h <= 0.0: continue

            k = int(round(freq / bin_hz))
            if k < 0 or k >= npsd: continue
            for off, w in zip([-1, 0, 1], spread_w):
                kk = k + off
                if 0 <= kk < npsd:
                    if harm_accum_mode == "amp":
                        harm_acc_spec[ti, kk] += mag_h * float(w)
                    else:
                        harm_acc_spec[ti, kk] += float((mag_h * float(w)) ** 2)

            # ── ★ harmonic_focus 模式：同时把谐波能量直接放进最近的 mel bin ─────
            # 跳过 STFT 三角滤波器扩散，让 per-harmonic 编辑 1:1 反映到单一 mel bin
            if args.harmonic_focus == 1:
                # 在 mel-Hz 空间找最近的 mel bin
                mel_bin_idx = int(np.argmin(np.abs(mel_freqs_for_focus - freq)))
                if 0 <= mel_bin_idx < int(args.n_mels):
                    # 功率累加（如果多个谐波恰好落在同一 mel bin，自然相加）
                    harm_mel_focus[ti, mel_bin_idx] += float(mag_h ** 2)

    # ─────────────────────────────────────────────────────────────────────────
    #  后处理：平滑 → 校正 → 叠加 → 转幅度 → Mel → Log
    # ─────────────────────────────────────────────────────────────────────────

    # 1. 时间平滑
    #    harmonic_focus=1 时谐波已直接累加到 mel 域（harm_mel_focus），
    #    必须平滑 harm_mel_focus；平滑 harm_acc_spec 对该路径无任何效果。
    #    harmonic_focus=0（标准路径）时平滑 STFT 域的 harm_acc_spec。
    if args.time_smooth and args.time_smooth > 1:
        if args.harmonic_focus == 1:
            harm_mel_focus = uniform_filter1d(harm_mel_focus, size=int(args.time_smooth),
                                              axis=0, mode="nearest")
            print("[Smooth] 谐波 mel 功率时间平滑 window=%d（harmonic_focus 模式）"
                  % int(args.time_smooth))
        else:
            harm_acc_spec = uniform_filter1d(harm_acc_spec, size=int(args.time_smooth),
                                             axis=0, mode="nearest")
            print("[Smooth] 谐波%s时间平滑 window=%d" % (
                  "幅度谱" if harm_accum_mode == "amp" else "功率谱", int(args.time_smooth)))

    # 累加谱 → 功率谱
    if harm_accum_mode == "amp":
        harm_power_spec = np.square(harm_acc_spec)
        print("[AccumMode] 幅度累加：已平方转功率谱（与Hann扩散组合时谐波相干叠加）")
    else:
        harm_power_spec = harm_acc_spec

    if args.noise_time_smooth and args.noise_time_smooth > 1:
        noise_power_spec = uniform_filter1d(noise_power_spec, size=int(args.noise_time_smooth),
                                            axis=0, mode="nearest")
        print("[Smooth] 噪声功率谱时间平滑 window=%d" % args.noise_time_smooth)
    else:
        print("[Smooth] 噪声功率谱未平滑（保留气息纹理）")

    # 2. H/N 功率域校正
    if args.hn_csv and args.hn_calib_mode != "none":
        harm_power_spec = apply_hn_power_calibration(
            harm_power_spec.astype(np.float64),
            noise_power_spec.astype(np.float64),
            frames,
            hn_info,
            mode          = str(args.hn_calib_mode),
            gain_min_db   = float(args.hn_gain_min_db),
            gain_max_db   = float(args.hn_gain_max_db),
            smooth_size   = int(args.hn_gain_smooth),
            eps           = float(args.hn_calib_eps),
            debug_csv_path= str(args.save_hn_debug_csv),
        )

    # 3. 噪声增益
    if abs(float(args.nonharmonic_gain_db)) > 1e-9:
        noise_power_gain = 10.0 ** (float(args.nonharmonic_gain_db) / 10.0)
        noise_power_spec = noise_power_spec * noise_power_gain
        print("[NoiseGain] 噪声功率 ×%.4g（%.2f amplitude dB）" % (noise_power_gain, float(args.nonharmonic_gain_db)))

    if args.harmonic_focus == 1:
        # ── ★ harmonic_focus 路径：谐波 mel 已聚焦，噪声照旧通过 STFT 滤波器 ─
        # 谐波部分：harm_mel_focus[t, m] = sum of ampl[i]^2 直接放进最近 mel bin
        # 噪声部分：noise_power_spec → sqrt → mel_filter → noise_mel_amp
        # 合并：mel_mag[t, m] = sqrt(harm_mel_focus[t, m] + noise_mel_amp[t, m]^2)
        noise_mag_spec = np.sqrt(np.clip(noise_power_spec, 1e-20, None)).astype(np.float32)
        noise_mel_amp  = np.dot(noise_mag_spec, mel_filter.T).astype(np.float64)
        noise_mel_pow  = noise_mel_amp ** 2

        # 谐波 mel 功率 + 噪声 mel 功率 → mel 幅度
        # 注意：harm_mel_focus 已是功率（ampl^2 累加）
        mel_total_pow = harm_mel_focus + noise_mel_pow
        mel_total_pow = np.clip(mel_total_pow, 1e-20, None)
        mel_mag = np.sqrt(mel_total_pow).astype(np.float32)
        mel_mag = np.clip(mel_mag, 1e-10, None)

        print("[HarmonicFocus] mel 合成: 谐波直接进 mel bin（无 STFT 扩散），"
              "噪声 STFT → mel filter")

        # ── ★ 改进 9/11：导出 H/N 分量供测量端无反演直读 ──────────────────
        if args.save_harm_mel_npy:
            try:
                import os as _os
                _os.makedirs(_os.path.dirname(args.save_harm_mel_npy) or ".", exist_ok=True)
                np.save(args.save_harm_mel_npy, harm_mel_focus.astype(np.float32))
                print("[SaveHarmMel] 已导出谐波 mel 功率到: %s  shape=%s" %
                      (args.save_harm_mel_npy, harm_mel_focus.shape))
            except Exception as _e:
                print("[SaveHarmMel] 失败: %s" % _e)
        if args.save_noise_mel_npy:
            try:
                import os as _os
                _os.makedirs(_os.path.dirname(args.save_noise_mel_npy) or ".", exist_ok=True)
                np.save(args.save_noise_mel_npy, noise_mel_pow.astype(np.float32))
                print("[SaveNoiseMel] 已导出噪声 mel 功率到: %s  shape=%s" %
                      (args.save_noise_mel_npy, noise_mel_pow.shape))
            except Exception as _e:
                print("[SaveNoiseMel] 失败: %s" % _e)
    else:
        # ── 原始路径：谐波和噪声都通过 STFT，再统一应用 mel filter ───────────
        # 4. 功率域叠加
        total_power_spec = harm_power_spec + noise_power_spec
        total_power_spec = np.clip(total_power_spec, 1e-20, None)

        # 5. 转幅度
        total_mag_spec = np.sqrt(total_power_spec).astype(np.float32)

        # 6. Mel 滤波器作用于幅度谱
        mel_mag = np.dot(total_mag_spec, mel_filter.T).astype(np.float32)
        mel_mag = np.clip(mel_mag, 1e-10, None)

    # 7. 对数压缩
    logmel = np.log(mel_mag).astype(np.float32)

    if args.save_pre_align_npy:
        np.save(args.save_pre_align_npy, logmel)
        print("已保存对齐前 mel:", args.save_pre_align_npy)

    if args.print_mel_stats == 1:
        mu0, sd0, mn0, mx0 = float(np.mean(logmel)), float(np.std(logmel)), float(np.min(logmel)), float(np.max(logmel))
        print("Mel before align: mean=%.6f std=%.6f min=%.6f max=%.6f" % (mu0, sd0, mn0, mx0))

    # 8. 加载源参考 mel 统计量（全局 + 逐 bin，一次加载，多步复用）
    target = build_target_stats_v2(args.target_mel_npy, args.ref_mean, args.ref_std)
    print("Target stats: mean=%.6f std=%.6f" % (target["mean"], target["std"]))
    src_bin_mean: Optional[np.ndarray] = target["bin_mean"]  # [n_mels] or None
    mel_freqs_hz: Optional[np.ndarray] = (
        mel_frequencies(args.n_mels, args.fmin, args.fmax)
        if src_bin_mean is not None else None
    )

    # 9. 全局分布对齐
    align_mode = args.align_mode
    if align_mode == "auto":
        align_mode = "meanstd" if args.match_ref_stats == 1 else "mean"

    mu, sd = float(np.mean(logmel)), float(np.std(logmel))
    if   align_mode == "meanstd":
        logmel = (logmel - mu) / (sd + 1e-8) * float(target["std"]) + float(target["mean"])
    elif align_mode == "mean":
        logmel = logmel - mu + float(target["mean"])
    elif align_mode == "none":
        pass
    else:
        raise ValueError("Unknown align_mode: %s" % align_mode)

    logmel = np.clip(logmel, float(args.clip_min), float(args.clip_max))

    if args.print_mel_stats == 1:
        mu1, sd1, mn1, mx1 = float(np.mean(logmel)), float(np.std(logmel)), float(np.min(logmel)), float(np.max(logmel))
        print("Mel after align:  mean=%.6f std=%.6f min=%.6f max=%.6f" % (mu1, sd1, mn1, mx1))

    # ════════════════════════════════════════════════════════════════════════
    #  [新增] 谱包络后处理修正（Opt 6 → Opt 5 → Opt 1，从宽到窄）
    #  注：以下修正均在全局对齐之后施加，修正的是对齐后的"残差"偏差。
    # ════════════════════════════════════════════════════════════════════════

    specenv_applied = False

    # [Opt 6] 平滑谱包络修正（最先：修正宽带包络偏差）
    if args.src_env_correct_enable:
        if src_bin_mean is not None:
            corr6 = compute_smooth_env_correction(
                logmel,
                src_bin_mean    = src_bin_mean,
                smooth_bins     = int(args.src_env_smooth_bins),
                alpha           = float(args.src_env_alpha),
            )
            logmel += corr6[None, :]
            specenv_applied = True
        else:
            print("[SrcEnvCorrect] [Opt 6] 跳过：未提供 --target_mel_npy。")

    # [Opt 5] 逐频带均值对齐（中等粒度）
    if args.perband_align_enable:
        if src_bin_mean is not None:
            corr5 = compute_perband_correction(
                logmel,
                src_bin_mean = src_bin_mean,
                n_bands      = int(args.perband_n_bands),
                alpha        = float(args.perband_alpha),
            )
            logmel += corr5[None, :]
            specenv_applied = True
        else:
            print("[PerbandAlign] [Opt 5] 跳过：未提供 --target_mel_npy。")

    # [Opt 1] 高频源参考混合（最后：精细高频修正）
    if args.hf_blend_enable:
        if src_bin_mean is not None and mel_freqs_hz is not None:
            corr1 = compute_hf_blend_correction(
                logmel,
                src_bin_mean = src_bin_mean,
                mel_freqs    = mel_freqs_hz,
                start_hz     = float(args.hf_blend_start_hz),
                end_hz       = float(args.hf_blend_end_hz),
                alpha        = float(args.hf_blend_alpha),
            )
            logmel += corr1[None, :]
            specenv_applied = True
        else:
            print("[HFBlend] [Opt 1] 跳过：未提供 --target_mel_npy。")

    # 应用谱包络修正后再次 clip（防止修正后超出范围）
    if specenv_applied:
        logmel = np.clip(logmel, float(args.clip_min), float(args.clip_max))
        if args.print_mel_stats == 1:
            mu2, sd2, mn2, mx2 = float(np.mean(logmel)), float(np.std(logmel)), float(np.min(logmel)), float(np.max(logmel))
            print("Mel after specenv: mean=%.6f std=%.6f min=%.6f max=%.6f" % (mu2, sd2, mn2, mx2))

    # [Opt 2 诊断] 打印高频亮度亏缺及建议的 noise_tilt_db_per_oct
    if args.noise_tilt_auto_diag and src_bin_mean is not None and mel_freqs_hz is not None:
        diagnose_hf_deficit(
            logmel,
            src_bin_mean       = src_bin_mean,
            mel_freqs          = mel_freqs_hz,
            auto_fmin          = float(args.noise_tilt_auto_fmin),
            noise_tilt_ref_hz  = float(args.noise_tilt_ref_hz),
        )

    # ─────────────────────────────────────────────────────────────────────────

    # 10. 全局增益（amplitude dB）
    if args.global_gain_db and abs(args.global_gain_db) > 1e-9:
        logmel = logmel + float(args.global_gain_db) * (math.log(10.0) / 20.0)
        print("[Gain] global_gain_db=%.2f (amplitude dB)" % float(args.global_gain_db))

    if args.clip_after_gain == 1:
        logmel = np.clip(logmel, float(args.clip_min), float(args.clip_max))

    if args.max_logmel is not None:
        logmel = np.clip(logmel, a_min=None, a_max=float(args.max_logmel))

    np.save(args.out_npy, logmel.astype(np.float32))
    print("已保存:", args.out_npy, "shape=", tuple(logmel.shape))


if __name__ == "__main__":
    main()

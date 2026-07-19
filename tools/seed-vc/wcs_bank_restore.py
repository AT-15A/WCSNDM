# -*- coding: utf-8 -*-
"""WCSNDM 音源修复批处理("第三档"AI 音色模仿路线,Seed-VC 零样本 VC)。

用同一音源里质量好的样本做 reference,把整个文件夹的 wav 过一遍 Seed-VC
(f0 条件模型,44.1kHz 输出,保留源音高与时长) → 输出"修复版音源文件夹"。
离线一次性预处理,不进引擎渲染路径;oto.ini 等配置原样复制,文件名不变。

用法(在 seed-vc 目录下,用 venv 的 python):
  venv\\Scripts\\python.exe wcs_bank_restore.py --bank "F:\\UTAU\\voice\\XXX" --out "F:\\UTAU\\voice\\XXX_fix"
      [--ref auto | 逗号分隔的wav路径]  [--steps 30] [--cfg 0.7] [--limit N] [--pattern glob]

注意:
- 输出裁齐/补零到与源完全等长(oto 毫秒标定不失效);逐文件 RMS 匹配到源(保音源内部响度平衡)。
- reference=auto 时取库内最长的若干 wav 拼接(约 20s;长样本多为持续元音,音色代表性好)。
  更好的做法是手动指定好批次的干净样本: --ref "a.wav,b.wav,c.wav"(相对 bank 目录或绝对路径)。
- 首次运行会从 HuggingFace 下载模型(whisper-small/DiT/BigVGAN/RMVPE,约 3-4GB);
  国内网络先设 HF_ENDPOINT=https://hf-mirror.com。
"""
import os, sys, argparse, glob, shutil, time, types

os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")

import numpy as np
import torch
import torchaudio
import librosa

import inference as inf   # 复用 seed-vc 官方加载/推理组件

SR = 44100
HOP = 512


def build_reference(bank, ref_arg, max_sec=20.0):
    if ref_arg and ref_arg != "auto":
        paths = []
        for p in ref_arg.split(","):
            p = p.strip()
            if not os.path.isabs(p):
                p = os.path.join(bank, p)
            paths.append(p)
    else:
        wavs = glob.glob(os.path.join(bank, "*.wav"))
        wavs.sort(key=lambda p: -os.path.getsize(p))
        paths = wavs[:12]
    segs, total = [], 0.0
    for p in paths:
        try:
            x, _ = librosa.load(p, sr=SR, mono=True)
        except Exception:
            continue
        segs.append(x)
        total += len(x) / SR
        if total >= max_sec:
            break
    if not segs:
        raise RuntimeError("no usable reference audio")
    ref = np.concatenate(segs)[: int(max_sec * SR)]
    print(f"[ref] {len(paths)} files -> {len(ref)/SR:.1f}s reference")
    return ref


def snr_est(x, fs=SR):
    """粗 SNR:最响 25% 帧均值 vs 最静 10% 帧均值(20ms 帧)。"""
    fr = int(0.02 * fs)
    n = len(x) // fr
    if n < 6:
        return 99.0
    rms = np.sort(np.array([np.sqrt((x[i * fr:(i + 1) * fr] ** 2).mean()) for i in range(n)]))
    floor = rms[: max(1, n // 10)].mean() + 1e-9
    act = rms[-max(1, n // 4):].mean() + 1e-9
    return float(20 * np.log10(act / floor))


def envelope_ceiling(y, src, fs=SR, headroom=1.4):
    """源包络天花板:逐帧(10ms)限制输出 RMS <= 源帧 RMS*headroom。
    扩散模型会把数字静音"再生"成自然呼吸/房间底噪 → 干净样本 SNR 崩(实测 45->14dB)。
    源里静音的地方输出必须还是静音;有声帧输出本就 ≈ 源电平,几乎无操作。增益帧间线性插值。"""
    fr = int(0.010 * fs)
    n = min(len(y), len(src)) // fr
    if n < 2:
        return y
    g = np.ones(n)
    for i in range(n):
        rs = np.sqrt((src[i * fr:(i + 1) * fr] ** 2).mean()) + 1e-9
        ry = np.sqrt((y[i * fr:(i + 1) * fr] ** 2).mean()) + 1e-9
        g[i] = min(1.0, rs * headroom / ry)
    g = np.convolve(g, np.ones(3) / 3, mode="same")   # 3 帧平滑防泵振
    env = np.interp(np.arange(len(y)), np.arange(n) * fr + fr / 2, g)
    return y * env


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="auto")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--fp16", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("--match-rms", default=True)
    ap.add_argument("--skip-clean", type=float, default=30.0,
                    help="SNR(dB)高于此值的文件视为干净,原样复制不转换(0=全转)")
    a = ap.parse_args()

    largs = types.SimpleNamespace(f0_condition=True, checkpoint=None, config=None,
                                  fp16=(str(a.fp16).lower() != "false"))
    print("[load] models (first run downloads ~3-4GB) ...")
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = inf.load_models(largs)
    device = inf.device
    print(f"[load] done, device={device}")

    max_context_window = SR // HOP * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * HOP

    def make_ref_features(ref_wave):
        rt = torch.tensor(ref_wave[: SR * 25]).unsqueeze(0).float().to(device)
        o16 = torchaudio.functional.resample(rt, SR, 16000)
        S_o = semantic_fn(o16)
        m2 = mel_fn(rt)
        t2 = torch.LongTensor([m2.size(2)]).to(device)
        f2 = torchaudio.compliance.kaldi.fbank(o16, num_mel_bins=80, dither=0, sample_frequency=16000)
        f2 = f2 - f2.mean(dim=0, keepdim=True)
        st2 = campplus_model(f2.unsqueeze(0))
        F0o = torch.from_numpy(f0_fn(o16[0], thred=0.03)).to(device)[None]
        pc, _, _, _, _ = model.length_regulator(S_o, ylens=t2, n_quantizers=3, f0=F0o)
        return m2, st2, pc, max_context_window - m2.size(2)

    # ---- reference:固定参考只算一次;self 模式(每文件用自己当参考=纯再生修复,
    #      说话人身份精确保留,靠扩散模型的干净人声先验去噪,多人混装音源也适用)逐文件算 ----
    self_ref = (a.ref == "self")
    if not self_ref:
        mel2, style2, prompt_condition, max_source_window = make_ref_features(build_reference(a.bank, a.ref))

    os.makedirs(a.out, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(a.bank, a.pattern)))
    wavs = [w for w in wavs if w.lower().endswith(".wav")]
    if a.limit > 0:
        wavs = wavs[: a.limit]
    print(f"[run] {len(wavs)} files, steps={a.steps} cfg={a.cfg}")

    t0 = time.time()
    for wi, wp in enumerate(wavs):
        name = os.path.basename(wp)
        try:
            src, _ = librosa.load(wp, sr=SR, mono=True)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        if len(src) < SR // 20:
            print(f"[copy] {name}: too short")
            shutil.copy(wp, os.path.join(a.out, name))
            continue
        if a.skip_clean > 0:
            s0 = snr_est(src)
            if s0 >= a.skip_clean:
                print(f"[copy] {name}: clean (SNR {s0:.0f} dB)")
                shutil.copy(wp, os.path.join(a.out, name))
                continue
        if self_ref:
            mel2, style2, prompt_condition, max_source_window = make_ref_features(src)
        src_t = torch.tensor(src).unsqueeze(0).float().to(device)
        w16 = torchaudio.functional.resample(src_t, SR, 16000)
        S_alt = semantic_fn(w16)   # 音源样本均 <30s,单次前向

        mel1 = mel_fn(src_t)
        t1len = torch.LongTensor([mel1.size(2)]).to(device)

        F0_alt = torch.from_numpy(f0_fn(w16[0], thred=0.03)).to(device)[None]
        shifted_f0_alt = F0_alt   # 不做自动移调:严格保留源音高

        cond, _, _, _, _ = model.length_regulator(S_alt, ylens=t1len, n_quantizers=3, f0=shifted_f0_alt)

        processed = 0
        chunks = []
        prev = None
        while processed < cond.size(1):
            chunk_cond = cond[:, processed: processed + max_source_window]
            is_last = processed + max_source_window >= cond.size(1)
            cat = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if inf.fp16 else torch.float32):
                vc = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device),
                                         mel2, style2, None, a.steps, inference_cfg_rate=a.cfg)
                vc = vc[:, :, mel2.size(-1):]
            wav = vocoder_fn(vc.float()).squeeze()[None, :]
            if processed == 0 and is_last:
                chunks.append(wav[0].cpu().numpy())
                break
            if processed == 0:
                chunks.append(wav[0, :-overlap_wave_len].cpu().numpy())
                prev = wav[0, -overlap_wave_len:]
            elif is_last:
                chunks.append(inf.crossfade(prev.cpu().numpy(), wav[0].cpu().numpy(), overlap_wave_len))
            else:
                chunks.append(inf.crossfade(prev.cpu().numpy(), wav[0, :-overlap_wave_len].cpu().numpy(),
                                            overlap_wave_len))
                prev = wav[0, -overlap_wave_len:]
            processed += vc.size(2) - overlap_frame_len
        y = np.concatenate(chunks)

        # 精确对齐源长度(oto 不失效) + RMS 匹配(保音源内部响度平衡)
        if len(y) >= len(src):
            y = y[: len(src)]
        else:
            y = np.pad(y, (0, len(src) - len(y)))
        if str(a.match_rms).lower() != "false":
            rs = float(np.sqrt((src ** 2).mean()) + 1e-9)
            ry = float(np.sqrt((y ** 2).mean()) + 1e-9)
            g = min(max(rs / ry, 0.25), 4.0)
            y = y * g
        y = envelope_ceiling(y, src)
        y = np.clip(y, -0.99, 0.99)
        torchaudio.save(os.path.join(a.out, name), torch.tensor(y).unsqueeze(0).float(),
                        SR, encoding="PCM_S", bits_per_sample=16)
        el = time.time() - t0
        print(f"[{wi+1}/{len(wavs)}] {name}  ({el/(wi+1):.1f}s/file)")

    # 配置文件原样复制(引擎缓存/frq 跳过,渲染时自动重建)
    skip_ext = {".wav", ".frq", ".llsm", ".llsm2", ".l1f0", ".pmk", ".mdd", ".bak", ".prof"}
    for p in glob.glob(os.path.join(a.bank, "*")):
        if os.path.isdir(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in skip_ext or p.endswith("wcsndm_bank.prof"):
            continue
        shutil.copy(p, os.path.join(a.out, os.path.basename(p)))
    print("[done] config files copied, bank ready:", a.out)


if __name__ == "__main__":
    main()

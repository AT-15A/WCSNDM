# -*- coding: utf-8 -*-
"""WCSNDM 神经蒸馏预设生成("参数蒸馏"路线,配合引擎 Mi flag)。

对音源里每个 wav 跑 Seed-VC(self-ref 纯再生修复,内存里做),再用 STFT 对比
"原始 vs 修复"的逐帧 64 带对数谱差量(dB) → 写 <wav>.nfx(每文件几十 KB)。
**不新建/不覆盖任何 wav**;渲染时引擎 Mi flag 分析原始 wav 再叠加此差量(纯参数,微秒级)。

用法(seed-vc 目录,venv 的 python):
  venv\\Scripts\\python.exe wcs_distill.py --bank "F:\\UTAU\\voice\\XXX" [--steps 30] [--skip-clean 30] [--limit N]

- 差量按引擎分析 hop(512@44100)对齐;.nfx 与 wav 同目录、同名+.nfx。
- 干净文件(SNR≥skip-clean)不写 .nfx(引擎 Mi 自动不动作)。
- 引擎侧: 音符 flags 写 Mi100(满强度) / Mi50(半量) 等。删 .nfx 或不写 Mi 即完全回原始。
"""
import os, sys, argparse, glob, types, subprocess, tempfile, struct, soundfile as sf
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
import numpy as np
import torch, torchaudio, librosa
import inference as inf

SR = 44100
HOP = 512          # 必须与引擎 L2 分析 hop 一致
NBAND = 64
NFX_MAGIC = 0x32464E57   # 'WNF2' LE (v2: vt+psd 参数域差量)
FNYQ = SR / 2.0
ENGINE = r"F:\UTAU\WCSNDM.exe"


def dump_params(wav_path, llp_path):
    """调引擎 @DUMP:分析 wav 写二进制参数(f0/psd/vt_db,每帧)。返回(nfrm,npsd,nspec,thop,fnyq,f0[],psd[nfrm,npsd],vt[nfrm,nspec])。"""
    subprocess.run([ENGINE, wav_path, llp_path, "C4", "100", "@DUMP",
                    "0", "9999", "0", "0", "100", "0", "100"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    with open(llp_path, "rb") as f:
        magic, nfrm, npsd, nspec = struct.unpack("<4i", f.read(16))
        thop, fnyq = struct.unpack("<2f", f.read(8))
        assert magic == 0x504C4C57, "bad WLLP magic"
        rec = 1 + npsd + nspec
        data = np.fromfile(f, dtype="<f4", count=nfrm * rec).reshape(nfrm, rec)
    f0 = data[:, 0]
    psd = data[:, 1:1 + npsd]
    vt = data[:, 1 + npsd:]
    return nfrm, npsd, nspec, thop, fnyq, f0, psd, vt


def to_bands(spec_db, fnyq):
    """[nfrm, nbin] dB → [nfrm, 64] 线性带均值(第 b 带中心 freq=b/63*fnyq,与引擎 dn_interp_db_ 同)。"""
    nbin = spec_db.shape[1]
    freqs = np.linspace(0, fnyq, nbin)
    bidx = np.clip((freqs / fnyq * (NBAND - 1) + 0.5).astype(int), 0, NBAND - 1)
    out = np.full((spec_db.shape[0], NBAND), 0.0)
    cnt = np.zeros(NBAND);
    for b in range(NBAND):
        m = bidx == b
        if m.any():
            out[:, b] = spec_db[:, m].mean(axis=1)
        else:
            out[:, b] = np.nan
    # 填补空带(线性插值)
    for i in range(out.shape[0]):
        r = out[i]; nanm = np.isnan(r)
        if nanm.any():
            r[nanm] = np.interp(np.flatnonzero(nanm), np.flatnonzero(~nanm), r[~nanm])
    return out


def snr_est(x, fs=SR):
    fr = int(0.02 * fs); n = len(x) // fr
    if n < 6: return 99.0
    rms = np.sort(np.array([np.sqrt((x[i*fr:(i+1)*fr]**2).mean()) for i in range(n)]))
    return float(20*np.log10((rms[-max(1,n//4):].mean()+1e-9)/(rms[:max(1,n//10)].mean()+1e-9)))


def write_nfx(path, dvt, dpsd, thop_ms, fnyq):
    nfrm = dvt.shape[0]
    qv = np.clip(np.round(dvt * 100.0), -32000, 32000).astype('<i2')
    qp = np.clip(np.round(dpsd * 100.0), -32000, 32000).astype('<i2')
    with open(path, "wb") as f:
        np.array([NFX_MAGIC, nfrm, NBAND], dtype='<i4').tofile(f)
        np.array([thop_ms, fnyq], dtype='<f4').tofile(f)
        qv.tofile(f); qp.tofile(f)


@torch.no_grad()
def convert(a, model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, steps, cfg):
    """self-ref 再生:每文件用自己当参考,长度/音高保持。返回同长修复音频。"""
    device = inf.device
    src = a
    def feats(w):
        rt = torch.tensor(w[:SR*25]).unsqueeze(0).float().to(device)
        o16 = torchaudio.functional.resample(rt, SR, 16000)
        m2 = mel_fn(rt); t2 = torch.LongTensor([m2.size(2)]).to(device)
        f2 = torchaudio.compliance.kaldi.fbank(o16, num_mel_bins=80, dither=0, sample_frequency=16000)
        f2 = f2 - f2.mean(dim=0, keepdim=True)
        st2 = campplus_model(f2.unsqueeze(0))
        F0o = torch.from_numpy(f0_fn(o16[0], thred=0.03)).to(device)[None]
        pc,_,_,_,_ = model.length_regulator(semantic_fn(o16), ylens=t2, n_quantizers=3, f0=F0o)
        return m2, st2, pc, (SR//HOP*30) - m2.size(2)
    mel2, style2, prompt_condition, max_src = feats(src)
    st = torch.tensor(src).unsqueeze(0).float().to(device)
    w16 = torchaudio.functional.resample(st, SR, 16000)
    S_alt = semantic_fn(w16)
    mel1 = mel_fn(st); t1 = torch.LongTensor([mel1.size(2)]).to(device)
    F0a = torch.from_numpy(f0_fn(w16[0], thred=0.03)).to(device)[None]
    cond,_,_,_,_ = model.length_regulator(S_alt, ylens=t1, n_quantizers=3, f0=F0a)
    ov = 16 * HOP; proc = 0; chunks = []; prev = None
    while proc < cond.size(1):
        cc = cond[:, proc:proc+max_src]; last = proc+max_src >= cond.size(1)
        cat = torch.cat([prompt_condition, cc], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if inf.fp16 else torch.float32):
            vc = model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(device), mel2, style2, None, steps, inference_cfg_rate=cfg)
            vc = vc[:, :, mel2.size(-1):]
        wav = vocoder_fn(vc.float()).squeeze()[None, :]
        if proc == 0 and last: chunks.append(wav[0].cpu().numpy()); break
        if proc == 0: chunks.append(wav[0,:-ov].cpu().numpy()); prev = wav[0,-ov:]
        elif last: chunks.append(inf.crossfade(prev.cpu().numpy(), wav[0].cpu().numpy(), ov))
        else: chunks.append(inf.crossfade(prev.cpu().numpy(), wav[0,:-ov].cpu().numpy(), ov)); prev = wav[0,-ov:]
        proc += vc.size(2) - 16
    y = np.concatenate(chunks)
    if len(y) >= len(src): y = y[:len(src)]
    else: y = np.pad(y, (0, len(src)-len(y)))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--skip-clean", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=0, help="差量频带方向平滑(±带,0=不平滑;参数域vt本就平滑,平滑会欠修正)")
    ap.add_argument("--clamp", type=float, default=18.0, help="差量幅度上限(dB)")
    ap.add_argument("--keep-conv", default="", help="调试:把该次转换的修复 wav 存到此目录(同名)")
    ap.add_argument("--pattern", default="*.wav", help="文件名 glob(默认全部 wav)")
    ap.add_argument("--vt-mode", default="off", choices=["off", "shape"],
                    help="off=只蒸馏 psd 降噪(推荐,稳);shape=也含 vt 形状(实验,timbre-align 用)")
    a = ap.parse_args()

    largs = types.SimpleNamespace(f0_condition=True, checkpoint=None, config=None, fp16=True)
    print("[load] models ...")
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, _ = inf.load_models(largs)
    print("[load] done, device", inf.device)

    wavs = sorted(glob.glob(os.path.join(a.bank, a.pattern)))
    if a.limit > 0: wavs = wavs[:a.limit]
    print("[run] %d files, steps=%d" % (len(wavs), a.steps))
    nwritten = 0
    for wi, wp in enumerate(wavs):
        name = os.path.basename(wp)
        try:
            src, _ = librosa.load(wp, sr=SR, mono=True)
        except Exception as e:
            print("[skip] %s: %s" % (name, e)); continue
        if len(src) < SR // 20:
            print("[skip] %s: too short" % name); continue
        if a.skip_clean > 0 and snr_est(src) >= a.skip_clean:
            print("[clean] %s (no preset)" % name); continue
        y = convert(src, model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, a.steps, a.cfg)
        # 修复音频写临时 wav,引擎 @DUMP 分析原始+修复,在参数域(vt_db/psd)算差量
        tmpd = tempfile.mkdtemp()
        try:
            tconv = os.path.join(tmpd, "conv.wav")
            sf.write(tconv, np.clip(y, -0.99, 0.99).astype(np.float32), SR, subtype="PCM_16")
            if a.keep_conv:
                os.makedirs(a.keep_conv, exist_ok=True)
                sf.write(os.path.join(a.keep_conv, name), np.clip(y,-0.99,0.99).astype(np.float32), SR, subtype="PCM_16")
            lo = os.path.join(tmpd, "o.llp"); lc = os.path.join(tmpd, "c.llp")
            no, npsd, nspec, thop, fnyq, f0o, po, vo = dump_params(wp, lo)
            nc, _, _, _, _, f0c, pc, vc = dump_params(tconv, lc)
            n = min(no, nc)
            # psd 差量:噪声底电平是真实量(去噪=psd 降),保留绝对值——参数域实测完美复现(0.4dB)。
            po_c = np.clip(po[:n], -120, 60); pc_c = np.clip(pc[:n], -120, 60)
            dpsd = to_bands(pc_c - po_c, fnyq)
            # vt 差量:合成时 hm.ampl=vt×源(源归一到1),vt 绝对电平=谐波电平、不是自由 gauge;
            # 而 orig/conv 把"谐波=vt×源(Rd)"分解不同(vt 均值可差 30dB+),vt 差脱离配套 Rd 差
            # 无法移植(实测渲染整体压暗)。self-ref 降噪不改谐波音色→vt 形状差≈0,余下全是 gauge 垃圾。
            # 故只蒸馏 psd(降噪),vt 置零。TODO:timbre-align ref 需谐波域方案。
            if a.vt_mode == "off":
                dvt = np.zeros_like(dpsd)
            else:
                fcb0 = max(8, int(10000.0/fnyq*(NBAND-1)))
                dvt = to_bands(vc[:n] - vo[:n], fnyq)
                dvt = dvt - dvt[:, :fcb0].mean(axis=1, keepdims=True)
        finally:
            import shutil; shutil.rmtree(tmpd, ignore_errors=True)
        if a.smooth > 0:
            k = 2*a.smooth+1
            sm = lambda D: np.apply_along_axis(lambda r: np.convolve(r, np.ones(k)/k, mode='same'), 1, D)
            dvt = sm(dvt); dpsd = sm(dpsd)
        # 截止以上归零:原始信号那里是编解码死区,分析出的 vt/psd 是垃圾值,差量无意义(交给引擎 BX)。
        # 用原始波形长时谱找截止(相对 0.5-4k 主体 -45dB 处),在 [fc, fc+2k] 平滑 taper 到 0。
        S = np.abs(np.fft.rfft(src * np.hanning(len(src)) if len(src)>=2048 else np.pad(src,(0,2048-len(src)))))**2
        ff = np.fft.rfftfreq(max(len(src),2048), 1.0/SR)
        body = 10*np.log10(S[(ff>=500)&(ff<=4000)].mean()+1e-12)
        prof = np.array([10*np.log10(S[(ff>=b/(NBAND-1)*fnyq-fnyq/(NBAND-1)/2)&(ff<b/(NBAND-1)*fnyq+fnyq/(NBAND-1)/2)].mean()+1e-12) if ((ff>=b/(NBAND-1)*fnyq-fnyq/(NBAND-1)/2)&(ff<b/(NBAND-1)*fnyq+fnyq/(NBAND-1)/2)).any() else -120 for b in range(NBAND)])
        alive = prof >= body - 45.0
        fc_b = np.max(np.flatnonzero(alive)) if alive.any() else NBAND-1
        taper = np.ones(NBAND)
        tb = max(1, int(2000.0/fnyq*(NBAND-1)))   # 2k 过渡
        for b in range(NBAND):
            if b > fc_b: taper[b] = max(0.0, 1.0 - (b-fc_b)/tb)
        dvt *= taper[None,:]; dpsd *= taper[None,:]
        dvt = np.clip(dvt, -a.clamp, a.clamp); dpsd = np.clip(dpsd, -a.clamp, a.clamp)
        print("   cutoff band %d (~%.1fk), taper applied" % (fc_b, fc_b/(NBAND-1)*fnyq/1000))
        write_nfx(wp + ".nfx", dvt, dpsd, thop, fnyq)
        nwritten += 1
        print("[%d/%d] %s -> .nfx (%d frm, |dvt| %.1f |dpsd| %.1f dB)" % (
            wi+1, len(wavs), name, n, np.abs(dvt).mean(), np.abs(dpsd).mean()))
    print("[done] %d presets written under %s" % (nwritten, a.bank))


if __name__ == "__main__":
    main()

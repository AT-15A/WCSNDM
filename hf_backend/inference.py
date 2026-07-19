import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from nsf_hifigan import NsfHifiGAN

# 文件路径
model_path = Path("model.ckpt")
mel_path = "input.npy"
f0_path = "f0.npy"
output_path = "output.wav"

# 加载模型
model = NsfHifiGAN(model_path)
model.to_device("cpu")

# 加载梅尔谱和 F0
mel = np.load("input.npy")   # shape [T, 128]
f0 = np.load("f0.npy")       # shape [T]

mel_tensor = torch.from_numpy(mel.T).unsqueeze(0).float()  # [1, 128, T]
f0_tensor = torch.from_numpy(f0).unsqueeze(0).float()      # [1, T]


# 合成
wav = model.model(mel_tensor, f0_tensor).view(-1)

# 输出
sf.write(output_path, wav.detach().cpu().numpy(), model.h.sampling_rate)
print("✅ 合成完成：output.wav")

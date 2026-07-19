# WCSNDM

 UTAU 重采样引擎（resampler）：六个合成模式 + 可选神经声码器（NSF-HiFiGAN），
内置针对低质量人声素材的一定修复能力。
主要面向人力 VOCALOID / 鬼畜调音场景设计。
大部分代码由AI编写。


## 快速开始

1. 下载 Release 中的配布包（含编译好的 `WCSNDM.exe`），或自行编译（见下）。
2. UTAU 工程设置：Tool2（resample）指向 `WCSNDM.exe`。
3. 直接使用即可

### HF 神经声码器（可选）

需要 Python 3.10+ 与 `pip install torch soundfile scipy numpy`。
**声码器模型 `model.ckpt` 不随仓库分发**（第三方模型，许可条款见
`hf_backend/NOTICE.txt` / `STATEMENTS.txt`），请从 Release 附件获取并放入
`hf_backend/` 目录。不装扩展不影响其它内核。

## 从源码编译

Windows + MSVC（Visual Studio Build Tools）：

```powershell
cd src
powershell -ExecutionPolicy Bypass -File build.ps1
```

产物为 `WCSNDM.exe`（单文件，静态编译）。

## 许可

本项目基于 [libllsm2](https://github.com/Sleepwalking/libllsm2) 与 libllsm
（Kanru Hua）二次开发，依 **GPLv3** 分发，全文见 [LICENSE](LICENSE)。
完整源码位于 `src/`。

`hf_backend/model.ckpt`（pc_nsf_hifigan 声码器模型）版权归其原作者，
许可条款见 `hf_backend/NOTICE.txt`，不适用本仓库的 GPLv3。

## 反馈

问题反馈请附：声库名、音符 flags、现象描述联系b站AT-15A。

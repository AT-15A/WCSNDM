WCSNDM 测试版 beta3 (2026-07-13)
====================================

多内核 UTAU 重采样引擎:六个原生内核 + 可选神经声码器(HF)。
内置人声针对性修复:智能降噪 / 频宽扩展 / 音色对齐 / 转音平滑 / 神经蒸馏降噪。


【缓存说明】
引擎会在声库目录生成 .llsm2 / .l1f0 缓存(加速二次渲染)、TM 建的
wcsndm_bank.prof(音色档案)、Mi 用的 .nfx(神经降噪预设)。均可随时删除。

【许可】
- 本引擎基于 libllsm2 与 libllsm(Kanru Hua, GPLv3)。依 GPLv3 分发:
  完整源码见 src\ 目录;GPLv3 全文: https://www.gnu.org/licenses/gpl-3.0.html
- hf_backend 内 model.ckpt(pc_nsf_hifigan 声码器模型)版权归其原作者,
  许可条款见 hf_backend\NOTICE.txt / STATEMENTS.txt(如再分发请先确认其条款)。
- tools\seed-vc 的两个脚本为本项目原创;Seed-VC 本体与模型请依其各自许可获取。
- 本包为测试版,不作任何保证;问题反馈请附:声库名、音符 flags、现象描述。

【已知事项】
- 默认 V2 内核每音符跑双分析(约 2 倍耗时),追求速度可显式填 K2。
- HF 的辅音由 mel 分辨率限制,建议用 HF2。
- M1 长音建议 M1l(循环模式);M1 不支持 Mf/BX/TM/Mi 等谱域 flag。
- 48kHz/立体声音源自动重采样到 44.1kHz 单声道(与 UTAU 生态一致)。

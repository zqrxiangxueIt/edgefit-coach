# EdgeFit Coach — 边缘协同的 AI 健身教练

两设备视频流计算系统：**树莓派端侧 (A)** 做初步计算后通过 Tailscale 将帧流发送至**边缘 PC (B)**，B 上的姿态估计神经网络实时完成**哑铃二头弯举计数与动作纠错**。

> Working Code Trumps All Hype.

## 系统架构

```
┌────────── 树莓派 Pi 5 (A · 4GB · Ubuntu 24.04) ──────────┐       ┌────────── 边缘 PC (B · 9070GRE · Windows) ──────────┐
│  USB 摄像头 / 视频文件 (Source 抽象)                      │       │  ZMQ SUB 收帧 (JPEG + JSON 协议信封)                │
│   ├─ raw      全帧采样压缩直传（对照组）                   │ ZMQ   │   ├─ YOLO-pose 推理 (DirectML/OpenVINO/CPU 可切换) │
│   ├─ event    帧差运动门控 + 自适应 JPEG 质量              │─────→ │   ├─ One-Euro 关键点滤波                            │
│   ├─ cascade  LiteRT 人形检测级联初筛（A 端 NN 初步计算）  │ :5555 │   ├─ 弯举状态机：计数 + 幅度/借力纠错               │
│   └─ privacy  只发人形 ROI，其余区域脱敏                   │       │   └─ FastAPI 仪表盘：画面/计数/曲线/模式热切换      │
└──────────────────────────────────────────────────────────┘       └────────────────────────────────────────────────────┘
```

设计依据（调研结论）：Pi 5 **没有硬件视频编码器**（仅 HEVC 解码），软编 H.264 会挤占级联检测 CPU；按帧 JPEG 与"逐帧独立决策"（事件门控/级联门控/质量自适应/ROI 脱敏）天然契合，也与国际顶会的端侧过滤路线一致（Reducto, NSDI'20 / E4, AAAI'25）。

## 快速开始

```bash
# B 端（PC）
python -m venv .venv && .venv\Scripts\pip install -r requirements-server.txt
.venv\Scripts\pytest                # 单元测试
.venv\Scripts\python server\app.py  # 仪表盘 http://127.0.0.1:8000

# A 端（树莓派）
bash scripts/setup_pi.sh && bash scripts/deploy_pi.sh
python edge/edge_main.py --mode event --source camera
```

## 流模式

| 模式 | A 端行为 | 目的 |
|---|---|---|
| `raw` | 均匀采样 + JPEG 固定质量 | 对照基线 |
| `event` | 帧差运动检测门控 + 质量自适应 | 静景省带宽 |
| `cascade` | LiteRT MobileNet-SSD int8 人形初筛 | 端侧 NN 协同，无人才不传 |
| `privacy` | 仅发送人形 ROI，其余打码 | 隐私保护演示 |

## 量化实验（✅ 见 [docs/benchmarks.md](docs/benchmarks.md)）

真实跨机链路实测（Pi 5 → PC，四模式热切换 × 令牌桶限速模拟）：

| 场景 | raw | event | cascade | privacy |
|---|---|---|---|---|
| 空场景 60% + 间歇运动 | 137.7 KB/s | **1.9 KB/s（-98.6%）** | **0（无人即静默）** | **0（连环境都不上传）** |
| 持续有人运动 | 1529 KB/s | 1300 KB/s | 1550 KB/s | 2280 KB/s（换来隐私+数字变焦）|
| 全链路 e2e（采集→PC 收到）| 24ms | 33ms | 46ms | 57ms（不含摄像头缓冲）|

- A 端神经网络初筛（LiteRT SSD int8）在 Pi 5 上实测 **~21ms/帧**
- B 端 DirectML（RX 9070GRE）纯推理 **5.6ms** vs CPU 25.8ms（**4.6×**）
- 模型对比：yolo11n-pose（默认）≈ yolo26n-pose，前者计数更稳

## 里程碑

- [x] M0 仓库骨架 + B 端离线管线（测试视频 → 姿态 → 弯举计数）✅ 2026-11-22：DirectML 5.8ms/帧，离线视频 6 reps 计数正确
- [x] M1 ZMQ 跨机链路（Pi ↔ PC over Tailscale）+ 时钟同步 ✅ 2026-11-22：30fps 零丢帧，传输 p50 20ms，全链路 ~34ms；控制信道 NTP 式测偏移（Pi 时钟 -769ms）
- [x] M2 A 端 event 模式（运动门控 + 自适应压缩）✅ 静止段带宽 -98.6%（真实链路实测）
- [x] M3 USB 摄像头 + Web 仪表盘 ✅ 四模式热切换、实时曲线、浏览器实测验收
- [x] M4 cascade/privacy 模式 + 限速量化实验 ✅ 详见 docs/benchmarks.md
- [ ] 实拍演示视频（收尾）

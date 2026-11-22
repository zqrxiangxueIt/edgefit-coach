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

## 量化实验（进行中）

- 四模式 × 三档限速（模拟 5G 上行/WiFi/窄带）下的带宽、端到端延迟、有效帧率对比
- 推理后端对比：DirectML (RX 9070GRE) vs OpenVINO (125H) vs CPU
- 模型对比：yolo11n-pose vs yolo26n-pose

结果将更新在 `docs/benchmarks.md`。

## 里程碑

- [x] M0 仓库骨架 + B 端离线管线（测试视频 → 姿态 → 弯举计数）
- [ ] M1 ZMQ 跨机链路（Pi ↔ PC over Tailscale）+ 时钟同步
- [ ] M2 A 端 event 模式（运动门控 + 自适应压缩）
- [ ] M3 USB 摄像头 + Web 仪表盘
- [ ] M4 cascade/privacy 模式 + 限速量化实验 + 实拍演示视频

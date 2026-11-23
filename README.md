# EdgeFit Coach — 边缘协同的 AI 健身教练

两设备视频流计算系统：**树莓派端侧 (A)** 做初步计算（含端侧神经网络人形初筛）后经 Tailscale 将帧流送至**边缘 PC (B)**，B 上的姿态估计神经网络实时完成**哑铃二头弯举计数、幅度/节奏/借力纠错与训练计划管理**。

> Working Code Trumps All Hype. —— 所有数字来自真机实测，方法见 [docs/benchmarks.md](docs/benchmarks.md)。

## 系统架构

```
┌────────── 树莓派 Pi 5 (A · 4GB · Ubuntu 24.04) ──────────┐       ┌────────── 边缘 PC (B · RX 9070GRE · Windows) ──────────┐
│  USB 摄像头(全局快门 720p MJPG) → FrameGrabber 线程       │       │  ZMQ SUB 收帧 → (裁剪变焦待定) yolo11n-pose 推理      │
│   ├─ raw      全帧采样压缩直传（对照组）                   │ ZMQ   │   (DirectML) → 视角补偿(肩线偏航反投影)               │
│   ├─ event    帧差运动门控 + 自适应 JPEG 质量              │─────→ │   → One-Euro 滤波(真实 dt) → 双臂迟滞计数             │
│   ├─ cascade  LiteRT SSD 人形初筛（A 端 NN 协同）          │ :5555 │   → 幅度/节奏/借力/幽灵计数防护                       │
│   ├─ privacy  只发人形 ROI（脱敏+数字变焦）                │       │   → 训练会话(组×次×休息) → SQLite 训练历史            │
│  └─ set_rate_limit 令牌桶上行预算（闭环自适应种子）         │       │  FastAPI 仪表盘: ROM表盘/逐次深度柱图/TTS 语音教练     │
│  电源管理：SSH 编排，网页 ⏻ 一键开关整套系统               │       │  /历史图表/演示模式；网页热切换模式与限速               │
└──────────────────────────────────────────────────────────┘       └────────────────────────────────────────────────────┘
```

设计依据：Pi 5 **没有硬件视频编码器**（仅 HEVC 解码），软编 H.264 会挤占级联检测 CPU；按帧 JPEG 与"逐帧独立决策"（事件门控/级联门控/质量自适应/ROI 脱敏/上行预算）天然契合，也与国际同路线系统一致（Reducto, NSDI'20 / E4, AAAI'25；闭环自适应对标 AccMPEG, MLSys'22）。

## 快速开始

### 无硬件 10 分钟体验（评审友好）

```bash
git clone https://github.com/zqrxiangxueIt/edgefit-coach.git
cd edgefit-coach
python -m venv .venv
.venv\Scripts\pip install -r requirements-server.txt   # Windows
.venv\Scripts\pytest                                    # 69 个单元/集成测试
# 离线管线：测试视频 → 姿态 → 弯举计数（应输出 5 reps，4 全程 1 半程）
.venv\Scripts\python scripts\offline_pipeline.py videos\curl_a.mp4
# 本机回环全系统（无需树莓派）：终端1 推流 + 终端2 仪表盘
.venv\Scripts\python -m edge.edge_main --bind tcp://127.0.0.1:5555 --video videos\curl_a.mp4 --loop
set EDGE_CONNECT=tcp://127.0.0.1:5555 && .venv\Scripts\python server\app.py
# 浏览器 http://127.0.0.1:8000 → 点 ⏻ 旁的"开始训练"前先点模式按钮体验
```

注：`onnxruntime-directml` 仅支持 Windows（AMD/Intel GPU）；Linux/macOS 请改 `onnxruntime` 并用 `--backend cpu`。

### 完整双机部署

```bash
# ① Pi 侧（首次，在树莓派上执行）：装环境（cascade/privacy 需要 --with-litert）
bash scripts/setup_pi.sh --with-litert
# ② PC 侧（每次部署）：同步代码+模型+视频到 Pi
bash scripts/deploy_pi.sh
# ③ 启动：双击 scripts/start_system.bat（或 python server\app.py 后网页点 ⏻）
#   关停：网页点 ⏻，或双击 scripts/stop_system.bat（释放摄像头/Pi/PC 全部资源）
```

### 环境变量（可选覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `EDGE_CONNECT` | `tcp://100.93.84.79:5555` | B 端连接的 A 端帧流地址 |
| `EDGE_MODEL / EDGE_BACKEND` | yolo11n-pose / dml | 推理模型与后端（dml/cpu/openvino）|
| `EDGE_RECORD / EDGE_DB` | 空 / data/edgefit.db | 录像输出 / 训练历史库 |
| `PI_USER / PI_HOST / PI_KEY / PI_DIR` | saltz / 100.93.84.79 / ~/.ssh/id_rsa / edgefit | 电源管理 SSH 参数 |

## 量化实验（✅ 见 [docs/benchmarks.md](docs/benchmarks.md)）

真实跨机链路实测（四模式热切换 × 令牌桶限速）：

| 场景 | raw | event | cascade | privacy |
|---|---|---|---|---|
| 空场景 60% + 间歇运动 | 138.1 KB/s | **2.3 KB/s（-98.4%）** | **≈0（无人即静默）** | **0（连环境都不上传）** |
| 持续有人运动 | 1558 KB/s | 1480 KB/s | 1555 KB/s | 2372 KB/s（换来隐私+数字变焦）|
| 全链路 e2e（采集→PC 收到）| 25ms | 33ms | 47ms | 60ms（不含摄像头缓冲）|

- A 端神经网络初筛（LiteRT SSD int8）Pi 5 实测 **~19ms/帧**；A 端 a_proc 经采集线程分离优化 raw 模式仅 **~4ms**
- B 端 DirectML（9070GRE）纯推理 **5.6ms** vs CPU 25.8ms（**4.6×**）
- **任务精度-带宽曲线**（计数误差 vs 送达带宽）：见 benchmarks.md —— 把带宽节省兑换成任务货币
- 模型 A/B：yolo26n-pose 虽论文 mAP 更高，实测少计（7→4、5→3）——**证据保留 yolo11n-pose**

## 核心功能

- **四流模式热切换**（网页按钮/快捷键 1-4）：raw / event（运动门控）/ cascade（端侧 NN）/ privacy（ROI 脱敏）
- **训练会话**：组×次计划、3-2-1 倒计时、组间休息（可跳过/语音播报）、完成评分
- **计数质量**：双臂独立迟滞状态机、One-Euro 滤波（真实 dt）、**视角补偿**（肩线偏航反投影，正视时诚实提示换机位）、手腕行程防幽灵计数、躯干借力检测、**节奏反馈**（离心过快/向心吃力）
- **仪表盘**：ROM 半圆表盘（双指针实时肘角）、逐次深度柱图、全宽警告横幅、TTS 语音教练、左右对称、SQLite 训练历史与图表、"超越上次"横幅、演示模式（拍屏幕友好）、键盘快捷键
- **电源管理**：网页 ⏻ 一键开关整套系统（SSH 编排 + 预杀防双启），空闲零资源占用

## 开发文档

- [docs/benchmarks.md](docs/benchmarks.md) —— 量化实验全数据与方法
- [docs/demo_video_plan.md](docs/demo_video_plan.md) —— 演示视频拍摄脚本
- [docs/prototype_dashboard.html](docs/prototype_dashboard.html) —— 仪表盘设计原型（模拟数据可独立打开）

## 里程碑

- [x] M0 骨架+离线管线（DirectML 5.8ms/帧）
- [x] M1 跨机链路（30fps 零丢帧、NTP 式时钟补偿、传输 p50 20ms）
- [x] M2 event 模式（静止段 -98.4%）
- [x] M3 仪表盘（四模式热切换浏览器验收）
- [x] M4 cascade/privacy + 限速量化实验 + 电源管理 + 训练会话/历史
- [x] 优化轮：时基秒制化、视角补偿、tempo/ROM/TTS、任务精度曲线、四轮对抗审查（69 测试）
- [ ] 实拍演示视频

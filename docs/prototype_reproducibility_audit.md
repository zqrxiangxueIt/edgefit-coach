# 陌生克隆者可复现性审计（只读审计，2026-11-22）

> 视角：刚 `git clone` 本仓库的课程评审人，只有 README 与仓库内容，没有任何作者环境信息。
> 方法：逐步走查 README「快速开始」+ `git ls-files` 入库对照 + 脚本逐行阅读。**未修改任何文件、未启动任何进程**；唯一的实验是用本仓库 `.venv` 里的 pytest 9.2.0 在系统临时目录对一个同构玩具项目做了导入机制验证（不执行仓库代码）。

## 总分：5.5 / 10

| 维度 | 得分 | 说明 |
|---|---|---|
| 代码与模型入库完整性 | 9/10 | onnx/tflite/测试视频全入库，离线可跑管线，这是最大亮点 |
| README 快速开始真实性 | 3/10 | B 端 pytest 命令**实测必失败**；A 端两行命令顺序错误；无"无硬件体验"路径 |
| 评审可自行看到"它在工作"吗 | 2/10 | 仪表盘必须 SSH 到作者 Pi 才能点亮（PowerManager 设计如此），README 未给替代路径 |
| 文档与代码一致性 | 5/10 | README 数字与 docs/benchmarks.md **不同步**；power/session 两个已交付功能未入架构图与里程碑 |
| 环境假设披露 | 4/10 | 硬编码 IP/用户/密钥路径多数可 env 覆盖但**从未在 README 披露**；stop_system.bat 不可覆盖 |

---

## 1. README「快速开始」逐步走查

### B 端（PC）

| 步骤 | 命令 | 判定 | 详情 |
|---|---|---|---|
| B1 | `python -m venv .venv && .venv\Scripts\pip install -r requirements-server.txt` | ⚠️ 部分能过 | ① Windows+DX12 GPU：能装；`onnxruntime-directml` 实际只有 Windows wheel，**Linux/macOS 评审在第一步 pip install 即失败**（README 未提示平台限制）。② `ultralytics>=8.3.0` 连带 torch 全家桶 ~2GB+，而运行时根本不需要（推理走纯 ORT；ultralytics 只服务 `scripts/export_models.py` 与模型对比）——下载 20 分钟的第一印象代价。③ 安装顺序陷阱已写注释且 requirements 顺序正确（directml 放最后），这点做得好 |
| B2 | `.venv\Scripts\pytest` | ❌ **必失败（已验证机制）** | 仓库无根级 `conftest.py`、无 `pyproject.toml`、`tests/` 无 `__init__.py`，而 `tests/test_backends.py:10` 等直接 `from server... import ...`。pytest 控制台脚本（prepend 模式）只把 `tests/` 插入 sys.path，不插仓库根。**验证**：用同一 `.venv` 的 pytest 9.2.0 对同构布局（包+无 `__init__` 的 tests 目录）裸跑 → `ModuleNotFoundError`；在根加一个空 `conftest.py` → 通过。作者本机大概率一直用 `python -m pytest`（把 CWD 加进 sys.path）才没暴露。**新克隆者按 README 敲命令，10 秒内收获一个红色 ImportError** |
| B3 | `.venv\Scripts\python server\app.py` → 打开 127.0.0.1:8000 | ⚠️ 起得来，点不亮 | 仪表盘进程能起（模型懒加载，设计好）；但页面核心是 ⏻ 电源按钮 → `PowerManager.on()` **先 SSH 到 `saltz@100.93.84.79`**（`server/power.py:17-23`），SSH 失败即返回 off，本地管线一律不启动。评审人画面=永久"系统已关停"。`EDGE_CONNECT/EDGE_BACKEND` 等 env 覆盖存在但 README 只字未提 |
| B3' | （README 未写的替代路径，代码里其实有） | — | `python -m edge.edge_main --bind tcp://127.0.0.1:5555 --video videos/curl_a.mp4` + `python -m server.receiver --connect tcp://127.0.0.1:5555`（`edge/edge_main.py` 文档字符串自己写了 loopback 用法）和 `scripts/offline_pipeline.py`（零网络出带标注 mp4）都能让评审看到系统工作——**全部没进 README** |

### A 端（树莓派）

| 步骤 | 命令 | 判定 | 详情 |
|---|---|---|---|
| A1 | `bash scripts/setup_pi.sh && bash scripts/deploy_pi.sh` | ❌ 顺序错误 | `setup_pi.sh` 头注释明说"Run ON the Raspberry Pi"，`deploy_pi.sh` 头注释明说"Run FROM the PC (Git Bash)"——两个不同机器的脚本被 README 串成一条命令：在 PC 上跑=建错 venv + scp 因缺 `~/.ssh/id_rsa`（默认指 `/c/Users/ZHENG QIAORI/...`）而失败；在 Pi 上跑=deploy 的 scp/key 路径全不对。`deploy_pi.sh` 自己的提示输出其实写了正确顺序，README 没采纳 |
| A2 | `python edge/edge_main.py --mode event --source camera` | ⚠️ 小问题 | 漏 venv 前缀（Pi 上系统 python 无 opencv）；应为 `.venv/bin/python -m edge.edge_main ...`（demo_video_plan.md 里反而是对的）；cascade/privacy 还需 `setup_pi.sh --with-litert`，README 未提 |

### 其他隐藏依赖

- `dashboard/index.html:6` 从 jsdelivr CDN 拉 chart.js——离线/内网环境仪表盘无图表（低危，但"离线可跑"卖点打折）
- 无 LICENSE 文件；无 CI；无 pyproject/锁定版本（`>=` 全浮动，半年后 pip 解析出的一切未验证）

## 2. git 入库状态与 .gitignore 对照

**应入库而缺失（按 `git ls-files` 核实）**

| 文件 | 影响 |
|---|---|
| 根级 `conftest.py`（空文件即可） | **B2 必失败的根因**，5 分钟修复 |
| LICENSE | 课程场景低危，但开源仓库惯例缺失 |
| CI（如 `.github/workflows/test.yml`） | 无法向评审自证"tests green"，锦上添花 |

**已入库（必要的，做得对）**：`models/detect.tflite`（7.0MB）、`models/yolo11n-pose.onnx`（11.3MB）、`models/yolo26n-pose.onnx`（11.6MB）——cascade/privacy/B 端推理开箱即用；`videos/*.mp4`（37MB）——离线管线与复现实验的输入，保留合理；`docs/benchmark_data_*.json` 原始数据入库，极佳。

**不该入库但进了仓库**

| 文件 | 大小 | 问题 |
|---|---|---|
| 根目录 `yolo11n-pose.pt` / `yolo26n-pose.pt` | 6.2 + 7.9 MB | `.gitignore` 写的是 `models/*.pt`（意图=不入库 .pt），但这两个文件躺在**根目录**绕过了规则；它们可由 `scripts/export_models.py` 经 ultralytics 自动下载再生成。白增 14MB 克隆体积，且暴露了 ignore 规则与实际意图脱节 |
| `runs/`（~260MB 工作区文件） | — | 已被 ignore、未入库，✅ 无问题（仅提醒别 `git add -f`） |

仓库总对象 72.8MB，单文件均 <100MB，GitHub 直接克隆可行；若把根目录 .pt 移除可到 ~59MB。

## 3. 环境假设清单（对外人复现的阻碍分级）

| 级别 | 假设 | 位置 | 可否绕过 |
|---|---|---|---|
| 🔴 阻断 | Tailscale IP `100.93.84.79` + 用户 `saltz` + 密钥 `…/.ssh/id_rsa`（作者私人身份三元组） | `server/power.py:17-19`、`scripts/deploy_pi.sh:9-11`、`scripts/estimate_offset.sh:5-7`（均可 env 覆盖但未披露）；**`scripts/stop_system.bat:7` 硬编码且不可覆盖** | 不持有作者 Pi 与密钥则**双机路径整体不可复现**；唯一活路（loopback/offline）未写进 README |
| 🔴 阻断 | Windows + DX12 GPU（`onnxruntime-directml`，默认 `EDGE_BACKEND=dml`，strict 不回退） | `requirements-server.txt`、`server/app.py:32`、`backends.py`（strict 设计本身是好工程） | 可 `EDGE_BACKEND=cpu` 绕过，但 README 未写 |
| 🟠 高 | pytest 控制台脚本 + 无根 conftest | README B2 | `python -m pytest` 可绕过（README 未写对命令） |
| 🟠 高 | `start_system.bat` 写死 `cd /d E:\multi_system_design` | `scripts/start_system.bat:3` | 评审克隆到任何其他路径即失效；应 `cd /d %~dp0..` |
| 🟡 中 | Pi 端 Python 3.12 / aarch64 wheel（ai-edge-litert） | `requirements-edge.txt`（已注释说明，好） | 已披露 |
| 🟡 中 | dashboard 依赖 CDN（chart.js） | `dashboard/index.html:6` | 离线演示无图表 |
| 🟢 低 | Git Bash（跑 .sh）、README 的 bash 代码块里用 Windows 反斜杠路径 | 排版层面 | 无实害，观感分 |

## 4. README 结构审计

1. **架构图过时**：图中 B 端仅有"FastAPI 仪表盘：画面/计数/曲线/模式热切换"，未包含最近两个提交交付的 **⏻ 电源管理（SSH 拉起 Pi 流+管线生命周期）** 与 **训练计划状态机（组×次、3-2-1 倒计时、组间休息）**——这两恰是"工程完成度"证据；`server/session.py`、`server/power.py` 在里程碑列表中也无条目（最后一项还停在 M4）。
2. **量化数字与 benchmarks.md 不同步**（评审交叉核对必翻车）：

| 指标 | README | docs/benchmarks.md |
|---|---|---|
| 空场景 raw | 137.7 KB/s | 138.1 KB/s |
| 空场景 event | 1.9 KB/s（-98.6%） | 2.3 KB/s（-98.4%） |
| 有人运动 raw/event/privacy | 1529 / 1300 / 2280 | 1558 / 1480 / 2372 |
| e2e 四模式 | 24/33/46/57 ms | 25.0/32.6/46.7/59.9 ms |

（benchmarks.md 自身注明由 d843ec8 一轮重测生成；README 是旧一轮数据，两处"事实源"不一致。）
3. **quick start 与实际命令不一致**：B2 命令错（见上）、A 端顺序错、漏 venv 前缀、未提 `--with-litert`、未提 env 变量、未提 loopback。
4. **缺一张 GIF/截图**：README 纯文字+ASCII 架构图；首屏没有可视化，视频出来前评审第一眼抓不到"工作品"证据。

## 5. 修复清单（按"评审第一印象"排序）

| # | 优先级 | 修复 | 改动量 | 预期效果 |
|---|---|---|---|---|
| 1 | P0 | 根目录加**空 `conftest.py`**（或 `pyproject.toml` 声明 `pythonpath="."`） | 1 个空文件 | README 的 pytest 命令立刻从"红"变"绿" |
| 2 | P0 | README 增设 **「无硬件 10 分钟体验」** 小节：loopback 双终端命令（edge_main 127.0.0.1 + `server.receiver`）+ `scripts/offline_pipeline.py videos/curl_a.mp4`（不依赖 Pi/SSH/GPU，CPU 即可） | 纯文档 | 评审从"点不亮的网页"变成"亲眼看它数出 6 个弯举"；顺带解决无 DX12 GPU 的死路 |
| 3 | P0 | **同步 README 量化表与 docs/benchmarks.md**（或声明数据快照日期），统一"事实源" | 纯文档 | 消除最伤可信度的自相矛盾 |
| 4 | P1 | 修正 A 端快速开始：`bash scripts/deploy_pi.sh`（PC 侧，Git Bash）→ `ssh pi 'cd ~/edgefit && bash scripts/setup_pi.sh --with-litert'` → `.venv/bin/python -m edge.edge_main …`；并在 README 披露 `PI_USER/PI_HOST/PI_KEY/EDGE_CONNECT/EDGE_BACKEND` 环境变量表 | 纯文档 | 双机路径至少"照着敲能走到正确的报错" |
| 5 | P1 | `git rm --cached yolo11n-pose.pt yolo26n-pose.pt` 并把 .gitignore 改为 `*.pt`（或补根目录规则） | 2 行 | 仓库 -14MB，规则与意图一致 |
| 6 | P1 | `start_system.bat` 用 `cd /d %~dp0..`；`stop_system.bat` 的 IP/密钥改为读 env（与 power.py 一致） | 3 行 | 消除两处"换个目录就坏"的死硬编码 |
| 7 | P2 | 架构图与里程碑补 power/session（新增 M5 条目）；README 顶部加 15s GIF/视频入口（demo_video_plan.md 本就计划了） | 文档 | 首屏印象与"完成度"叙事 |
| 8 | P2 | `dashboard/index.html` 把 chart.js 改为本地 vendor（`dashboard/vendor/chart.umd.min.js`） | 1 文件 | 离线可演示 |
| 9 | P2 | `ultralytics` 拆到 `requirements-export.txt`（导出/对比才需要），主 requirements 减重 ~2GB 下载 | 1 文件 | 新机器 5 分钟装完 |
| 10 | P3 | LICENSE + GitHub Actions 跑测试矩阵（win/ubuntu, cpu backend） | 小 | 自证 tests green |

**做完 #1~#4（约半天，纯文档+1 空文件）预计得分 5.5 → 8.5**：评审能在无任何作者硬件的前提下，10 分钟内看到测试全绿 + 离线视频计数正确 + loopback 带宽对比，剩下的"必须有 Pi"部分变成诚实披露而非隐性撞墙。

## 6. 值得保留的亮点（审计中也应被看见）

- 模型与测试资产全入库（多数课程仓库做不到）；`docs/benchmark_data_*.json` 原始数据可复核
- `backends.py` strict 拒绝静默 CPU 回退 + 可移植 skipif 测试——防"假基准"的工程自觉
- requirements-server.txt 顶部的 onnxruntime 遮蔽坑注释（YOLO_AUTOINSTALL）是真实踩坑沉淀
- `deploy_pi.sh`/`power.py` 的 env 覆盖设计已为陌生环境留了口子，只差 README 把口子写出来
- 18 个提交均有量化验收结论（"54 tests green""-98.6% 实测"），git 历史本身就是"Working Code Trumps All Hype"的证据链

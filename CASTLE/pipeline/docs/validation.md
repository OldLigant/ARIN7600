# 验证记录 — 2026-09-16

本文件保留 2026-09-16/17 各次实验的条件和结果。“本轮未提交”“尚未验证”只描述对应实验，不代表当前部署状态或当前测试数量。后续运行见带时间戳的 STATUS.md；2026-09-19 launcher 修复的 222 项测试记录见 [本轮记录](launcher-collection-2026-09-19.md)。

## 媒体解码并行度验证（2026-09-17）

背景：HF Job `Ligant/6aaae5a0f76d6a098a71184f`（Batch 准备阶段，cpu-basic）实测每片段约 73 秒（`batch_clip_prepared` 相邻间隔 71–75 秒，105 个片段约 105 分钟）。定位到 `MediaExtractor` 的解码线程数与并发槽位是硬编码 1。

测量方法：合成 40 秒 3840×2160 H.264 High / 50 fps 源（与真实素材同规格；真实素材 10.79 GB、约 23.8 Mbps），每个配置取 30 张 1 fps 采样帧。

```powershell
python _bench\framesel_bench.py --source _bench\src\synth4k50.mp4 --work _bench\tmp --make-source _bench\src\synth4k50.mp4 --source-seconds 40 --fps 50
foreach($t in 1,2,4,8){ python _bench\decode_cost.py --source _bench\src\synth4k50.mp4 --work "_bench\tt$t" --threads $t --seconds 30 }
```

结果（秒，越小越好）：

| `-threads` | D0 纯解码 | D1 解码+缩放 | D2 解码+缩放+JPEG q:v2 | D4 +pad+drawtext | P2 Python 串行盖章 | P3 Python 并行盖章 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.5 | 66.8 | 65.9 | 63.7 | 6.5 | 0.46 |
| 2 | 29.4 | — | 31.4 | — | 6.8 | 0.46 |
| 4 | 11.3 | — | 10.9 | — | 6.6 | 0.47 |
| 8 | 5.7 | 4.9 | 5.4 | 4.9 | 6.8 | 0.44 |

结论：

1. **成本几乎全在解码**：D0 与 D2 在各档线程数下都落在噪声范围内，缩放与 JPEG 编码不是瓶颈。因此"少编码一次"不是主要收益，"多线程解码"才是。
2. **线程扩展接近线性到 8 线程**（1→8 为 12.2x），该规格素材有足够并行度，8 vCPU 机器用得上。
3. Python 盖章串行 6.5 s / 30 帧，线程池后 0.46 s（14x）。这部分与解码无关，独立可省。

改动与验证：`MediaExtractor` 增加 `threads`／`decode_slots`／`footer_workers`，`RunConfig` 与两个 CLI 暴露对应参数，`batch_pipeline.prepare_media` 由逐片段串行改为 `--prepare-workers` 并发，并新增按解码缓冲的准入检查。

- `tests/test_media.py` 新增：不同 `footer_workers`（1/3/8）逐字节一致、`threads` 真正进入 ffmpeg argv、非法调优值被拒、同一源两个并发片段的产物互不干扰。
- `tests/test_batch_prepare.py` 新增：并发与串行产生相同 clip_id 顺序与相同帧时间／原生帧引用，两次同配置并发运行逐字节一致（FFmpeg 的 JPEG 输出只对固定线程数稳定，故不跨线程数比较字节）；解码缓冲准入检查与 `--memory-hint-gib` 的拒绝／放行。
- `tests/test_cli_jobs.py` 新增：`--media-threads`／`--footer-workers` 渲染为 `CASTLE_*` 环境变量，缺省时不下发。
- `tests/test_runner.py` 新增：`RunConfig` 拒绝不可用取值，`Pipeline` 从配置构造 extractor。

全套：`_test/runtime/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp _test/pytest-parallel-run5`，结果 **195 passed in 18.15s**。

**尚未验证：** 本机为 24 核 i9-13980HX，测试只说明扩展性与正确性，不代表 cpu-basic／cpu-upgrade 上的绝对耗时；本轮未提交任何付费 Job，也未在真实 4K 素材上跑完整准备阶段。

## 后续日志增强验证

新增逐请求／逐阶段事件、HTTP或规范化错误码、input/output/thought/cached/total token统计、AIMD变化、独立600秒心跳及events.jsonl。默认新运行使用standard，未提交新Job。

命令：`_test/runtime/Scripts/python.exe -m pytest tests -q --basetemp _test/logging-final-20260916`；结果 **99 passed in 11.51s**。compileall通过。新增测试验证429→退避→成功的请求关联、失败码脱敏、无效JSON的已消耗tokens、阶段复用、工作线程阻塞时仍产生心跳、控制台／文件日志失败不中断标注、KeyboardInterrupt释放API槽、遥测故障后心跳继续。

独立检查发现并修复了日志异常可能占住并发槽的问题。skill已同步standard偏好和日志说明并通过官方验证。以下为此前基础管线验证记录。

验证环境：Windows 11，本机 Python 3.13；生产 Docker 配置为 Python 3.12-slim。独立测试虚拟环境位于 `_test/runtime`，安装了 requirements.txt 中固定版本的 SDK。未执行真实 Gemini 生成、付费 HF Jobs、代码／输出 bucket 上传或云端挂载测试。

## 自动化测试

最终命令：

```powershell
& '_test/runtime/Scripts/python.exe' -m pytest tests -q --basetemp _test/pytest-final-sdk
& '_test/runtime/Scripts/python.exe' -m compileall -q castle_pipeline run_pipeline.py jobs.py
```

结果：**83 passed in 11.54s**，无警告；compileall 退出码为 0。

覆盖范围：

- 真实 FFmpeg 50 fps 采样，30 秒边界、亚秒尾部、时间标记附加边栏、原生坐标裁剪、单解码器互斥。
- 音轨比视频长、音轨延迟开始、片段起点非零、内部音频数据空档、纯静音区间；WAV 保持视频 clip-local 时间轴。
- 媒体准备失败时清理已生成的已知临时帧／音频，保留不认识的文件。
- 多人／重叠事件、证据模态和来源、时间范围、人物引用、wearer 动作链、按 ID 修订与显式 null／空列表。
- 有界惰性调度、指纹变更、阶段断点复用、复核失败后只重做复核、相同 source/config 的输出独占锁。
- Vertex 请求限速、AIMD、共享退避、重试上限、错误类型、载荷检查、用量和错误脱敏。
- 真实 google-genai 2.23.0 的离线客户端构造：禁止 socket 连接，验证 Vertex endpoint、service-bound key／ADC 分支、Flex headers、毫秒超时、SDK attempts=1。没有用模型调用代替配置测试。
- 清单去除 novideo、source-level 分片不重叠、保留 commit、hour 过滤；Jobs 默认只打印命令、持久化结果挂载、secret 名称传递、解析后的项目传递。

独立代码检查发现并修复了音频起点偏移、非法证据模态、清单 commit 丢失、失败临时文件泄漏、wearer ID 提示缺失、未应用 hour 过滤及项目参数转发等问题。检查后未留下已知阻塞项。

## 只读远程接口验证

```powershell
& '_test/runtime/Scripts/python.exe' run_pipeline.py list --day day1 --stream Allie --output _test/hf-manifest-allie-20260916.jsonl
```

返回 commit `c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4`、11 个 MP4。此项只读取仓库目录，没有下载小时视频。

## 合成 4K／50 fps 内存实验

复现脚本为 `tests/benchmark_memory.py`，要求显式指定一个新的输出目录：

```bash
python tests/benchmark_memory.py --output _test/memory-benchmark-NEW-ID
```

本次报告保存在 `_test/memory-benchmark-20260916/final/report.json`，完整命令在该目录 `commands.txt`。报告还记录运行时模块哈希，可区别实验版本与之后的配置／诊断代码调整。

两个实验均使用 3 个在途 clip、1 个 FFmpeg 解码器、1 fps、30 秒片段、1440 最大边长和音频；模型替身会实际读入图片并生成 raw/base64/JSON 载荷，以包含请求准备内存，但不会调用 API。

| 源视频长度 | 处理片段数 | 峰值进程树 RSS | 用时 |
|---|---:|---:|---:|
| 30 秒 | 1 | 153.24 MiB | 9.201 秒 |
| 180 秒 | 6 | 179.34 MiB | 46.193 秒 |

六倍源时长对应约 1.17 倍峰值内存，实验结束后的 GC 后 RSS 约 39 MiB。每片段 JPEG 总大小约 773,378 bytes，最大帧 25,832 bytes，PCM WAV 约 960,078 bytes；模拟请求峰值约 1.28 MB。

**局限：**合成蓝色画面和 ultrafast H.264 都比真实 CASTLE 简单，低估 JPEG 大小和某些解码缓冲；短实验只有 1 个 clip，长实验才覆盖 3 workers 的并发；未包括真实 SDK 网络请求、远端排队、bucket 文件系统和 Linux page cache。此实验验证结构上不累积整段视频，不能承诺实际 CASTLE RSS 只有这些数值。README 中另列按较大 JPEG 和解码器余量计算的约 1.49 GiB 工程预算；二者都不能替代首次 cpu-basic 实测。

## Skill

`skills/castle-caption-jobs/SKILL.md` 通过官方 quick_validate.py，并已复制安装至 `C:/Users/ZHANG Jiachang/.codex/skills/castle-caption-jobs/SKILL.md`。

另用独立 agent 检查了“旧 Job 被 kill、audio 成功日志、输出未挂 bucket、曾发生429、只准备续跑命令”的场景：正确区分日志与已保存检查点，不承诺恢复丢失数据，保持指纹相关设置，降低配额，且不带 --submit 启动付费任务。这是使用说明的行为检查，不表示已在真实账号提交或恢复过 Job。

首次真实验证仍需：一个源文件的少量片段、真实音画同步和细动作质量、生成 JSON 合规率、实际 token 用量／服务层级支持，以及持久化 bucket 上的原子写入与终止后恢复。

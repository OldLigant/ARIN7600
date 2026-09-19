# CASTLE 视频标注管线

面向 Hugging Face Jobs 的 CPU 管线：**每次一份源视频落盘 → 每片段 30 秒、1 fps → 原始音频独立标注 → 视觉密集标注 → 按需原生画面裁剪复核 → 阶段检查点与流式 JSONL 汇总**。

在线模式默认 3 个在途 clip、1 个 FFmpeg 解码进程，所有模型请求共用 AIMD 与 RPM 限制。代码保持源小时视频在磁盘，不把整小时的图片／PCM／caption 全部装入 RAM。原 Aria 脚本未改动。

新增 **Vertex Batch 模式**：`batch_pipeline.py` 负责持久化、分阶段提交和一次性检查，`batch_jobs.py` 生成一次性 submit / 每小时 tick 的 HF Jobs。详细配置、ADC/GCS 权限、运行示例、恢复与停止定时任务见 [Batch 使用说明](docs/batch.md)。需要 GCS 和 Google ADC；现有在线 API key 不能替代。真实运行进度见带探测时间戳的 [STATUS.md](STATUS.md)。以下其余章节描述在线 `run_pipeline.py`。

多 GCP 账号的 run 与 SA 文件、执行项目、HF 输出位置关联，使用本地私有 `credentials/run-bindings.jsonl`，通过 `private_runs.py` 登记；不要写入公开 ledger 或云端产物。操作者（包括 LLM）的凭据选择与记账流程见 [私有绑定说明](credentials/README.md)。

首次配置请阅读 [GCP 环境配置](docs/gcp-setup.md) 和 [Hugging Face 准备](docs/huggingface-setup.md)。完整阅读路线见 [文档导航](docs/README.md)，LLM agent 从 [AGENTS.md](AGENTS.md) 开始。

## 对输入的处理

- **音频先行。** 从当前 30 秒截取 mono / 16 kHz / PCM16 WAV，Gemini 单独判断语音与非语音声音。音频结果交给视觉阶段；没有音轨则明确无音频。声源／说话人不因声音大或距离近而自动归属于 wearer。
- **当前不使用 HF 原始转录。** 自动转录中已经观察到重复识别；为避免以它引导新的识别，音频阶段默认从原始声音重新判断。`prompts/audio.md` 的可选转录说明为输入扩展留口，当前 CLI 不下载或注入旧 transcript。新生成结果也可能出错，仍需抽样审核。
- **OCR 由视觉模型直接完成。** 第一遍看 1440 像素长边的采样图片，遇到影响判断的文字、手—物交互或物体状态细节，可以请求最多 4 个原生帧裁剪，第二遍精读；不额外运行 OCR 引擎。裁剪提高空间分辨率，无法恢复 1 fps 遗漏的短暂动作。
- **时间标记是我们添加的。** 每张模型输入图在底部新增 24 像素黑边栏，写 clip-local 时间，原画面不被覆盖；上下文另提供 frame_index 和数值秒。源视频文件名表示整点，clip offset 保存在结果中，不能把时区硬写成 HKT。采样使用请求时刻后的第一张可用源帧，50 fps 下量化误差通常小于一帧；标记不是更精确的采集时钟。
- **不使用 FACE 建立人物身份。** CSV 字段尚有疑点，相机人脸检测 ID 也不是稳定的跨 clip／跨机位身份。本实现只输出本片段内的 wearer/person_N/speaker_N，没有添加人脸追踪框、真实姓名推断或跨片段身份关联。
- **测试图交由标注识别。** 没有预先部署测试图分类器；提示词要求输出视觉不可用区间，不能将它解释为看电视。不可见不意味着音频缺失。需要强保证的过滤应在首轮真实样本审核后再加检测器。

## 安装与离线检查

Python 3.12+，FFmpeg 和 ffprobe 在 PATH。`requirements.txt` 固定了本次验证的直接依赖版本。

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --help
python run_pipeline.py estimate --source-seconds 3600 --workers 3
```

运行真实标注必须显式提供可用的 Vertex 模型 ID。没有默认猜测模型名；音频能力、输入限制和 Flex 支持也必须与该模型相符。当前 SDK 为 `google-genai 2.23.0`，使用 `genai.Client(vertexai=True, project=..., location=...)`；已做禁止 socket 连接的客户端构造测试。

认证支持 ADC 或环境变量 `GOOGLE_API_KEY` 中的 **Vertex 服务绑定密钥**。它不是 Gemini Developer API key。项目通过 `--project` 或 `GOOGLE_CLOUD_PROJECT` 指定。HF Jobs 里需提供可用的 ADC 凭据配置或相应密钥；不能把本机 ADC 自动视为已带进 Job。不要把密钥写入命令参数、代码 bucket 或日志。

## 先跑一个小时中的少量片段

下面是 Bash 示例。替换项目、模型和输出路径；PowerShell 可写为单行。该命令会实际调用付费 Vertex API：

所有 smoke 示例须改成全新的 `_test/<unique-id>` 输出/暂存路径；HF 测试也使用独立的新输出前缀，不要直接复用下文的示意目录。正式续跑才使用原来的持久化输出。

```bash
python run_pipeline.py run \
  --source main/day1/Allie/video/08.mp4 \
  --revision c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4 \
  --model YOUR_VERTEX_MODEL --project YOUR_GCP_PROJECT \
  --start-clip 10 --max-clips 3 \
  --workers 3 --initial-concurrency 2 --rpm 30 \
  --output-dir /output/castle-smoke --scratch-dir /scratch/castle
```

`start-clip=10` 在 30 秒设置下从源视频 300 秒处开始；此处不是保证无测试图的区间。每个 clip 完成前面阶段后才进入后面阶段，但不同 clip 可以流水并发。`--max-clips` **按每个源视频生效**，不是多文件任务的总预算。默认启用复核，但没有请求区域时自动跳过。`--no-review` 可用于成本对照；它改变配置指纹，不会被当作同一次标注继续。

本地视频也可输入：

```bash
python run_pipeline.py run --local-video /data/08.mp4 \
  --day day1 --stream Allie --hour 8 \
  --model YOUR_VERTEX_MODEL --project YOUR_GCP_PROJECT \
  --max-clips 3 --output-dir /output/local-smoke --scratch-dir /scratch/castle
```

固定机位可用精确 HF 路径或本地输入的 `--viewpoint exo`；提示词会明确没有 wearer。研究主线为第一人称，固定机位可作为后续对照。

## 清单、版本与分片

```bash
python run_pipeline.py list --day day1 --output day1-manifest.jsonl
python run_pipeline.py run --manifest day1-manifest.jsonl \
  --viewpoint ego --shard-index 0 --shard-count 4 \
  --model YOUR_VERTEX_MODEL --project YOUR_GCP_PROJECT \
  --output-dir /output/castle --scratch-dir /scratch/castle
```

list 默认固定到已调查的 commit，也可以显式 `--revision main`；输出清单记录解析后的不可变 commit。run 沿用清单版本，显式版本与清单不一致会拒绝，避免无意处理别的数据版本。原调查的 `hour_manifest.jsonl` 也可用；它没有逐行 revision，因此需保留默认已调查 commit 或明确指定正确版本。

分片是在过滤后的源视频路径排序列表上按序号取模，每个小时视频只属于一个 shard。不同 Job 必须使用同一份清单、过滤条件、shard_count，并分配不同 shard_index。不要让两个活跃 Job 处理同一输出目录下的同一 source/config；本地 `run.lock` 只能在支持原子独占创建的文件系统上防冲突，不能承诺远程 bucket 挂载具有跨 Job 分布式锁语义。

每次只下载一个源 MP4，处理后默认仅删除本次下载的那个 MP4，不删除原始本地视频、用户文件或整个缓存树。读取前检查剩余磁盘是否足够容纳源文件并留 2 GiB。`--keep-source` 会保留源文件，长任务可能耗尽磁盘；`--keep-media` 会保留采样帧／裁剪，也只应限于调试。

## 输出与续跑

模型主标注返回后先做有限的规范化：重叠事件按时间排序，非片段边缘的ongoing降为uncertain（不改起止时间）。原始输出保存在阶段结果raw_data，normalization列出修正；边界修正标review_required。越界时间、缺失证据、未知引用等仍拒绝。真实失败样本复验见 [day1-allie-08-v1诊断](docs/job-6aa9bfc3-diagnosis.md)。

```text
output/
  <configuration-fingerprint>/
    <source-id-hash>/
      run.json
      run.lock                 # 运行时存在；正常结束移除
      summary.json
      captions.jsonl           # 已完成 clip，一行一个；流式写出
      clips/
        00010/
          audio.json
          annotation.json
          review.json          # 只有实际需要复核时才有
          final.json
          error.json           # 某次失败诊断；旧错误可与已恢复 final 同时存在
          annotation.invalid.json  # 若模型返回可解析但不合规 JSON，保存供诊断，不作为成功结果
```

audio／annotation／review 检查点封装为 `{fingerprint,ok,result}`；final.json 顶层直接包含 source、clip、annotation、usage、review_required 等字段，以及 fingerprint/ok。`captions.jsonl` 按 clip_index 顺序汇总，不依赖并发完成顺序。语义输出沿用先前设计中的 `castle-caption-v1`；音频结果、原始模型段输出和复核修改都可回查。

同一命令、同一输出根目录重跑时：

1. 验证配置指纹与 JSON 结构、时间范围、人物和证据引用。
2. final 已完整则跳过；否则逐阶段复用。例如复核失败，只重跑复核模型请求。
3. 媒体位于临时盘时，重启仍可能需要下载源文件并重新抽帧；复用指的是成功的模型阶段，不是保证所有本地 I/O 都省略。
4. 指纹包括源身份／数据 commit、模型、service tier、采样设置、提示词及管线代码。改变它们会产生新输出分支；workers、RPM、重试次数和选取范围不影响既有标注身份，可以为恢复降低它们。

```bash
python run_pipeline.py status --output-dir /output/castle
```

检查点必须位于持久化挂载，只有本地 `/tmp` 或 Job 短期磁盘的成功日志并不能保证重启后还能恢复。强制 kill 后可能遗留 run.lock：先读其中 job_id/pid/host/nonce，确认旧 Job 已经终止，再移除**那一个**锁文件；不得为了续跑直接删除 output 树。不要移除仍有活跃 Job 持有的锁。

复核只进行按 segment_id 的完整 replacement，保留显式 null／空数组修正；不能用数组位置碰运气。需要重新分段／更改人物归属时，结果会标 `review_required` 和 `resegmentation_requests`。这表示该 clip 已完成自动处理，但仍需人工或后续时序复核，不能把它当成已验证真值。

## Hugging Face Jobs

`cpu-basic` 的官方配置为 2 vCPU、16 GB RAM、50 GB 磁盘；运行前以 `hf jobs hardware` 核对当前规格。代码和结果使用两个独立的 bucket 挂载：代码只读 `/workspace`，结果读写 `/output`；下载、解码在 `/scratch`。所有 stage 的结果直接写入持久化输出，不能只在任务结束时上传。

运行代码使用独立、已发布的版本目录。已登记 release 的发布由 release.py 从 Git tag 按 manifest 白名单取文件；不要手工同步整个工作区，也不要覆盖已有前缀。发布前可先做本地检查（NEW_RELEASE 为待发布且已有 tag / manifest 的版本）：

```bash
python release.py publish --release NEW_RELEASE --verify-only
```

`--verify-only` 不上传，也不核验云端内容。完整约定见 [发布规范](docs/release-process.md)；历史在线 castle-v1/v2 尚无现有 Batch 格式的清单，不能假定上述工具已经补齐了它们的发布历史。凭据留在 `credentials/`，不属于发布文件集。

保持发布的代码目录不变；同一版本目录在 Job 运行时被改写会破坏复现。准备输出 bucket 后，用 launcher 先检查命令：

```bash
python jobs.py \
  --code-volume hf://buckets/OWNER/CODE_BUCKET/castle-v1 \
  --output-volume hf://buckets/OWNER/OUTPUT_BUCKET/castle-v1 \
  --name castle-smoke --project YOUR_GCP_PROJECT \
  --secret GOOGLE_API_KEY --flavor cpu-basic --timeout 2h \
  -- \
  --source main/day1/Allie/video/08.mp4 --model YOUR_VERTEX_MODEL \
  --start-clip 10 --max-clips 3 --workers 3 --rpm 30
```

默认只输出 JSON argv，**不会提交**。确认本次运行的范围和费用已获授权后，在 `--` 之前添加 `--submit`。secret 参数只传环境变量名，由 HF CLI 读取值并加密传递；本 launcher 不显示或接收 secret=value。公开 CASTLE 下载不强制 HF_TOKEN；只有其他明确需要的私有资源才传相应令牌。

默认使用 `python:3.12-slim`，启动时安装 FFmpeg 和固定版本依赖。亦提供 Dockerfile，可构建有相同依赖的自有镜像，并通过 `--image` 指定；当前 helper 的 bootstrap 要求 Debian-compatible shell/image。代码 bucket 须直接包含 run_pipeline.py、castle_pipeline/、prompts/、requirements.txt。

```bash
hf jobs inspect JOB_ID
hf jobs logs JOB_ID
```

是否完成应看任务退出码和持久化 summary/final，不以某条“完成一个 clip”日志替代整体状态。Job 名称不是唯一 ID；监控和取消必须使用返回的 JOB_ID。

## 配额与失败处理

本节只适用于在线模式；大规模标注目前优先使用前述 Batch 流程。在线新运行使用 **standard**（也是在线 CLI 默认值）。用户已取消慢速 Flex 测试，不自动重提它；只有显式要求时才使用 Flex。切换 service tier 会改变标注指纹，不能把旧 Flex 检查点悄悄当作 standard 的结果。

- 每个进程共用一个请求级 limiter，音频、主标注、复核及重试都计入；不是只限制视觉请求。
- 默认最多 3 个 clip / 3 个同时 API 请求，初始 AIMD 窗口 2，成功后逐步增加，429／暂时服务故障／网络错误减半并共享退避；等待退避时释放请求槽。
- RPM 默认 30，用于平滑请求起始频率。Vertex 也可能按 token／动态容量限流，RPM 不保证没有 429。多 Jobs 的 limiter 互不通信，总额度需由提交计划划分，例如总 30 RPM 分 3 个 Job，则每个不超过 10 RPM。
- SDK 自动重试关闭，管线有界重试。auth／model／billing／配置问题停止继续调度；quota、server、network 只在次数范围内重试；无效 JSON／契约校验失败记录失败并留待后续修正或续跑，避免无界重复付费。
- `--service-tier flex --location global` 才允许 Flex，并使用官方两个请求头。默认 standard；不要在恢复时偷偷改变 tier。
- 网络超时后服务端可能已经计算并计费；检查点只能避免已收到且保存的成功阶段重跑，无法保证网络层 exactly-once 计费。

## 媒体解码并行度（只影响速度，不改变标注身份）

4K H.264 解码是每片段准备阶段的主要成本，而它默认只用一个核。实测同一段 30 秒 4K 50fps 素材（3840×2160、H.264 High、约 24 Mbps）、取 30 张 1 fps 采样帧：

| FFmpeg `-threads` | 纯解码 | 解码+缩放+JPEG | 相对 1 线程 |
|---:|---:|---:|---:|
| 1 | 69.5 s | 65.9 s | 1.0x |
| 2 | 29.4 s | 31.4 s | 2.2x |
| 4 | 11.3 s | 10.9 s | 6.0x |
| 8 | 5.7 s | 5.4 s | 12.2x |

缩放与 JPEG 编码相对解码可以忽略（两列几乎相同），因此在这组实验中，增加 `--media-threads` 是主要加速手段。Python 侧的缩略图与时间戳盖章是另一段串行成本，实测 30 帧 6.5 s，改用线程池后 0.46 s（约 14x），由 `--footer-workers` 控制。

| 参数 | 默认 | 环境变量 | 作用 |
|---|---|---|---|
| `--media-threads` | CPU 数推导，上限 4 | `CASTLE_MEDIA_THREADS` | 每次 ffmpeg 调用的解码／滤镜线程数 |
| `--decode-slots` | 1 | — | 允许几个片段同时解码 |
| `--footer-workers` | CPU 数推导，上限 4 | `CASTLE_FOOTER_WORKERS` | 时间戳盖章线程池大小 |

Batch 的 `batch_pipeline.py prepare` 另有 `--prepare-workers`（并发准备的片段数，环境变量 `CASTLE_PREPARE_WORKERS`）与 `--memory-hint-gib`。

**当前在线指纹不包含这些执行调优值。** 相同代码及其它身份输入下，在线恢复可以调整线程和并发并复用检查点。测试验证了不同 footer_workers 下盖章结果逐字节一致，但没有保证改变 FFmpeg 解码线程数后 JPEG 字节不变。Batch 的调优参数在 prepare 时存入 state，当前 tick 没有中途调整接口；不要把在线恢复能力等同于 Batch 配置可改。`source_start`（在线）与 `batch_media_tuning`（Batch）记录实际值。

**还需核对内存、CPU 配额与磁盘容量。** 4K H.264 解码线程需要参考帧缓冲，总并发解码线程数约为 `media_threads × decode_slots`。Batch 准备阶段在启动时按每线程约 40 MiB 做准入检查，超出 `--memory-hint-gib`（默认 16）就拒绝而不是等 OOM；在线模式在超出内存软上限 25% 时发出 `media_tuning_warning`。容器里 `os.cpu_count()` 看到的是宿主机核数而不是 cgroup 配额，所以默认值刻意保守（上限 4）；HF Jobs 通过 `jobs.py --media-threads / --footer-workers` 写成 `CASTLE_*` 环境变量，使渲染出的 argv 自身可复现。

推荐值：`cpu-basic`（2 vCPU/16 GB）用 `--media-threads 2 --decode-slots 2`；`cpu-upgrade`（8 vCPU/32 GB）用 `--media-threads 4 --decode-slots 3`。完整测量命令与边界见 `docs/validation.md`。

## 实时日志与十分钟进度

所有结构化事件以 JSON 单行立即 flush 到 stdout，可在 HF Job 日志页面查看。开始处理源视频后，还会追加到相应输出目录的 `events.jsonl`；下载阶段输出到 Job 控制台。时间戳 `timestamp_utc` 是日志产生的 UTC 时间，视频内部时间仍用 clip-local 秒。

| 事件 | 可见信息 |
|---|---|
| download_start / download_success / download_failed | 源文件、下载耗时、字节数或安全错误类型 |
| source_start | 模型、service tier、选中的 clip 数、心跳间隔 |
| stage_start / stage_success / stage_failed | clip ID、prepare/audio/annotation/crops/review 阶段、耗时 |
| stage_reused / stage_skipped | 复用检查点，或无音轨／无裁剪／关闭复核 |
| request_queued | 已准备好逻辑请求，等待限速／并发许可 |
| request_start | request_id、clip_id、phase、attempt、排队耗时、模型及 tier |
| request_success | 本次调用与整个逻辑请求耗时、input/output/thought/cached/total tokens、traffic_type |
| request_failed | HTTP 状态码（若有），否则规范化码；category、error_type、attempt、fatal、will_retry、重试等待时间 |
| retry_scheduled | 下一次尝试编号、共享退避时间与失败码 |
| aimd_change | 实际并发上限改变时的 old_limit、new_limit、原因及当时在途数 |
| clip_completed / clip_failed / clip_reused | 一个 clip 的最终处理状态 |
| progress | 独立线程定时汇总，见下文 |
| source_completed | 当前源文件汇总及本进程 telemetry |

每个逻辑请求有一个稳定 request_id；重试沿用 ID 并增加 attempt，方便把429、退避和最终成功关联起来。HTTP错误保留如429／401／503的数字码，非HTTP错误用NETWORK、PAYLOAD、INVALID_OUTPUT等规范化码，不输出原始 SDK错误文本、密钥、prompt 或转录正文。返回了可解析 JSON 的 request_success 不代表最终语义契约通过；随后仍可能出现 stage_failed。

日志示意（省略部分公共字段）：

```json
{"event":"request_start","clip_id":"source:10","phase":"audio","attempt":1,"queue_wait_sec":0.0}
{"event":"request_success","clip_id":"source:10","phase":"audio","attempt":1,"elapsed_sec":12.3,"usage":{"input_tokens":1200,"output_tokens":300,"thought_tokens":0}}
{"event":"aimd_change","old_limit":2,"new_limit":3,"reason":"success"}
```

默认 `--progress-interval-sec 600`。心跳独立于 clip／API完成事件：即使所有 worker 在等待服务端或退避，仍可每10分钟看到 selected/completed/failed/reused/pending、当前各clip阶段及已持续时间、在途和等待的request、AIMD当前上限／退避、进程树RSS，以及本进程累计请求和tokens。调试可改为60秒，修改该间隔本身不改变配置身份。

统计仅累加当前进程**实际收到的 usage**，不会重复把 stage_success 或已复用检查点的tokens计入新消费；收到无效JSON但带usage时仍计入。失败或超时没有usage时保留未知计数，不宣称零计费。summary.json 的 telemetry 与心跳使用同一统计口径。

日志写入故障不会让已付费的结果丢失，也不会占住限流槽；累计 logging_errors 可用于发现日志通道异常。文件写入失败后仍尝试输出到控制台；周期性遥测失败会记录 progress_unavailable 并在下一周期继续。

本次日志实现改动了代码，因此按已有规则会产生新的代码指纹。使用新版代码时应保留旧输出，不修改旧指纹以强行复用。未自动上传、重启或提交任何 Job。

## CPU 内存与磁盘预算

4K RGB 原始帧是 `3840×2160×3 = 24,883,200 bytes`，约 23.73 MiB。一小时若全部展开：50 fps 约 **4,171 GiB**，即使先降到 1 fps，仍约 **83.43 GiB**。缩小为 1440×810 后再一次保留 3,600 张 RGB，也约 11.73 GiB，尚未算 Python、SDK、音频和解码缓冲。

原 Aria 代码实际在 frames_data 中保留 JPEG 字节，不能把上述 RGB 估计直接说成其精确 RSS；但“整条录制先准备完，再调用模型”的内存随录制时长增长，确实是需要避免的结构。长视频 OOM 还可能包含 JPEG 大小、并发、底层缓冲等因素，不能只凭读源码宣称诊断了历史那次 OOM。

本实现按每 JPEG 平均 0.35 MiB 的预算假设，30 张约 10.5 MiB；考虑 SDK/base64/JSON 多份副本、单张解码图和音频，给每个在途片段约 82.5 MiB，另给单个 FFmpeg 768 MiB、运行时 512 MiB。3 workers 的工作集预算约 **1.49 GiB**，这是工程预算，**不是实测保证**。source 文件大小只影响磁盘；一个 37.6 GB 的大文件仍需先通过剩余磁盘检查。

`summary.json` 记录峰值进程树 RSS；在 Linux 可用时也记录 cgroup memory.current 峰值。RSS 会重复计算共享页，cgroup 值包含可回收文件缓存，两者不是同一口径。12 GiB 软件阈值仅在新 clip 准备前检查进程树 RSS，它不是内核硬限制，也不能拦截瞬时峰值。固定在途数量、单解码器、输入尺寸限制才是主要内存约束。

复现实测脚本：`tests/benchmark_memory.py`。它只使用合成视频和本地模型边界替身，不调用 Vertex；运行时必须明确输出专用测试目录。实测结果另见 `docs/validation.md`。不能把本机 Windows 的测试等同于已经通过 cpu-basic 16 GB 云端运行。

## 测试与后续验证

```bash
python -m pip install pytest
mkdir -p _test
python -m pytest tests -q --basetemp _test/pytest-NEW-RUN-ID
```

每次使用新的 basetemp，避免 pytest 清理既有数据。测试覆盖真实 FFmpeg 抽帧／音轨、尾段、采样量化、并发解码、输出契约、阶段续跑、只替换指定段、共享限流与失败分类、SDK 客户端构造、Jobs 命令生成。模型边界使用替身测试，不代表真实字幕质量。

当前的真实运行状态（已发布的 release、各 Batch run 的 pin／阶段／计数、已提交的 Job）见仓库根的 **`STATUS.md`**，它由 `tools/report.py` 从云端只读探测生成并带探测时间戳；本文件不再手写任何云端状态断言。首次真实 smoke 仍建议只选一个源文件的 2–3 个 clip，确认时间／音画同步、测试图、人物归属和环境变化后，再扩到一小时。支持人工检查的 agent skill 位于 `skills/castle-caption-jobs/SKILL.md`。

官方接口依据：[HF Jobs 配置](https://huggingface.co/docs/hub/jobs-configuration)、[Jobs SDK](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)、[Google Gen AI SDK](https://googleapis.github.io/python-genai/)、[Flex PayGo](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/flex-paygo)。

# CASTLE：Vertex Batch 与短 HF Jobs

截至 2026-09-19 的运行记录，GCS、ADC 和真实 Batch 已投入使用。**实况见仓库根
`STATUS.md`**——它由 `tools/report.py` 从云端只读探测生成并带探测时间戳；历史 Job 备注不能替代当前 run 状态。
下文描述的是接口与运维约定，不是当前进度。

## 是否符合“提交后关掉脚本，过一会再取”

是。输入 JSONL、图片和音频先放入 GCS，Vertex 接收 Batch 后独立执行，HF 提交任务即可退出。后续 HF worker 每次检查一次状态；未完成就退出，完成则收集结果并提交下一阶段。

这里是两种 HF worker，而非整个流程恰好只有两个 Job 实例：

1. `submit`：下载一个源视频到本地磁盘 → 按 clip 提取并上传媒体 → 提交 audio Batch → 退出。
2. `hourly-tick`：每小时启动一个短任务，检查 → 收集 → 推进。audio → annotation → 有需要才执行的 review，分别是独立 Batch。没有音轨时跳过 audio，没有 crop 请求时跳过 review。

GCS `state.json` 是状态依据，HF `/output` 保存最终标注、日志和状态快照；v5 外围 tick 还镜像已收集的阶段 JSON（见下文）。这个 state 在全部媒体预处理完成后才初始化；各阶段的 Vertex Job 完整资源名保存在 `batches.audio.job.name`、`batches.annotation.job.name`、`batches.review.job.name`，当前阶段为 `current_stage`。Job 重启不需要重新处理整段视频。Batch 是独立执行方式，没有 `standard/flex` 参数，也不使用在线请求的 AIMD。

2026-09-23 起，在线与 Batch 的**请求构造共享同一实现**（`castle_pipeline/request_spec.py`）：同一份提示词后缀、context 键序、媒体标注与 part 顺序、默认 token 预算。规范形以 v1–v6 已发布 Batch run 实际发出的请求为准，重构后 Batch 请求字节经黄金样本比对不变；在线 `run_pipeline.py` 由此成为同素材 Batch 请求的预演（此前在线硬编码 16384 token 且各处文案独立漂移）。差异只剩传输层与限流重试策略。该改动改变核心 `code_hash`，下一条新 Batch run 需要新 release；旧 run 继续钉在原版本，不受影响。

后续开发版本又为三个模型阶段加上共享的 Vertex `responseSchema`。每条 Batch JSONL 请求在 `generationConfig` 中带自己的 schema；在线 standard/flex 请求使用同一份 schema。该变更会再次改变核心 `code_hash`，须发布新版本后才用于新的 Batch run；旧 run 仍使用它们各自的发布字节。服务端 schema 约束输出形状，本地语义校验仍负责时间、证据与引用关系。

这里的“收集”不是重新调用模型：worker 读取 GCS 中的原始预测，按 request_id 对应 clip，校验后保存阶段结果；不需要下一阶段的 clip 生成 final，并写入 GCS 和 HF 输出 bucket。当前 `tick` 会在收集后立即提交需要的下一阶段，所以 audio 的 tick 可以产生新的 annotation Batch，annotation 的 tick 可以产生 review Batch。只想查看状态时使用只读探测工具，不要执行 tick。

本地 launcher 的 `--code-volume auto` 用于已经有 state 的 run；首次 submit 必须明确指定发布版本。续跑兼容性按 `code_hash` 判断：`castle-batch-v3/v4/v5` 核心身份相同，显式选择任一版本都可使用，并保留所选挂载。此 launcher 修复于 2026-09-19 在本地实施，未覆盖云端已经发布的 v4。

2026-09-20 起，本地 launcher 还允许显式指定版本的 `hourly-tick` 在 state 尚不存在时创建 schedule。等待检查随命令保存在 HF schedule 中，因此可继续挂载已有 v4，不改变核心 `code_hash`，也不需要覆盖或重新发布该目录。通过旧 launcher 创建的 schedule 不会自动更新；该改动已随新的不可变 v5 发布，v4 保留原样。发布内容及核验记录见 [v5 发布记录](batch-v5.md)。

`castle-batch-v5` 增加外围入口 `batch_tick.py`，它不属于核心身份文件。使用新版 launcher 并挂载 v5 时，手动 `tick` 和 `hourly-tick` 都会在同一容器内运行原 tick，然后把已收集的阶段结果复制到 HF。v5 的核心身份和提示词与 v3/v4 一致，因此可续跑同哈希的 run；v1/v2 仍不能改挂 v5。release 名区分完整发布物，`code_hash` 决定核心兼容性；辅助文件也由 manifest 单独记录哈希，不能因为不属于核心身份就覆盖旧发布物。

### v5 的中间结果镜像

GCS `results/audio/<clip_id>.json`、`results/annotation/<clip_id>.json`、`results/review/<clip_id>.json` 会按同样相对路径写入该 run 的 HF output prefix。保持原始阶段 JSON 的字节，包括 `data`、`usage`、annotation 的归一化信息；没有执行的 review 不会生成虚构结果。无音轨的 skipped audio 只要已存在于 GCS，也会被复制。

同步扫描已有阶段文件，因此首次使用 v5 tick 可以补齐之前完成的阶段。相同内容跳过；同名不同内容拒绝覆盖并让 worker 返回失败。下载使用 GCS generation 和已有 SHA-256 元数据核验，写入采用临时文件完成后替换，不发布半个 JSON。仅复制这些阶段 JSON，不同步媒体、raw 或 errors 目录，也不删除源文件。

这是 GCS 结果的查看副本，不是新增 checkpoint：GCS state 仍是唯一运行依据，不从 HF 镜像恢复、不修改旧 state、不重新调用模型。阶段结果须先被原 tick 收集；本功能不是 Vertex 每完成一条请求就实时推送。同步发生在该次原 tick 退出后，即使 tick 返回失败也会尝试复制已经保存的阶段结果。镜像失败时检查 worker 日志中的 `batch_stage_mirror_failed`，不要仅凭核心 `stop_schedule: true` 就认定结果已同步齐全。

新增入口需随 v5 发布；仅使用新 launcher 但继续挂载 v4 时，仍只有提前等待 state 的行为，不会自动获得阶段同步。已有 schedule 保存的命令和挂载不变，需要更新/重新创建后才会使用新入口。

### v6 的镜像并发与 schedule 修复

`castle-batch-v6` 仍只改外围文件，核心 `code_hash` 与 v3/v4/v5 相同。两处改动：

- `batch_tick.py` 的镜像改为**有界并发**（`CASTLE_MIRROR_WORKERS`，默认 16，上限 64）。原实现是严格串行的逐对象往返，实测每对象约 76–98 ms、约 10–13 对象/秒、**1.76 Mbps**（总量仅 24–29 MiB 而中位对象只有 4.9 KiB），慢在往返次数而不是带宽。只读探针对同一对象集测得 4 线程 4.0x、16 线程 14.3x 的加速，因此只把每个对象的往返放进线程池。并发只改快慢：复制的字节、计数、冲突判定与失败选择都不变（失败按对象名排序取第一个，因此同一对象集总是报同一个错误）。该事件新增 `workers` 字段。
- `batch_jobs.py` 创建 schedule 时使用 **`@hourly`** 而不是 `hourly`。CLI 的 `--help` 把 `hourly` 列为合法值，但它把该值原样转发给 API，服务端回报 `Invalid CRON expression`；API 接受的是 CRON 表达式，`@hourly` 是官方示例写法。修复前 `hourly-tick --execute` 建不出 schedule。

`CASTLE_MIRROR_WORKERS` 是运行参数，不属于标记身份，可在续跑时调整。仅复制阶段 JSON 的既有边界不变：媒体、raw、errors、requests 与 final 都不参与镜像。发布内容与验证记录见 [v6 发布记录](batch-v6.md)。

同一 run / HF 输出前缀仍只允许一个写入 worker；不要让手动 tick 与 schedule 或两条 schedule 重叠。同步会在下载前和写入前检查目标内容，但这不是跨容器原子锁：HF 挂载没有跨节点锁或多写入者冲突保护，`--no-concurrency` 也只约束单条 schedule。[HF 挂载的一致性边界](https://github.com/huggingface/hf-mount#best-for--not-for)

`ledger/ledger.jsonl` 是运行台账：记录 Job ID、视频范围、代码版本及状态变化。通过 `ledger.py append` 追加事件，再用 `tools/report.py` 生成状态文档。它不存放标注正文，也不能替代 GCS state。追加 ledger 或重新生成 STATUS 后，用 `tools/push_ops.py --execute` 把这两个文件同步到输出 bucket 的 `ops/` 前缀，作为给没有本仓库检出的操作者（人或 LLM）的查看副本；本地 Git 与 GCS state 仍是权威。

关于以后让新 run 继承旧 audio、解耦在线 / Batch 身份，以及只读发布内容核验，见 [版本演进的边界与实施依据](version-evolution-decisions.md)。这些是按需实施的设计记录，当前没有 import-stage 或原 run 中途升级的接口。

目前官方页面列出 Gemini 3.8 Flash 的 Batch 支持和通常比在线低 50% 的价格。它没有预设的用户并发配额，但使用共享容量，仍会排队，不能保证比 standard 更快。请按实际模型价格核算。[官方能力与限制](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference)

## 需要配置什么

第一次配置的合作者请先按 [GCP 环境配置指南](gcp-setup.md) 操作；其中包含控制台入口、项目与桶两级权限，以及 SA 和 Google 服务代理的区别。

| 项目 | 用途 |
|---|---|
| Google 项目，启用 Vertex AI / Cloud Storage，开通计费 | 提交并执行 Batch |
| 一个已存在的 GCS bucket，独立 run prefix | 媒体、JSONL、Batch 输出、持久状态；例如 `gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID` |
| Google ADC 身份 | HF worker 读写 GCS、创建/查看/列出 Batch；当前在线用的 `GOOGLE_API_KEY` 不能替代这里的 ADC |
| Vertex 服务代理的 bucket 权限 | GCP 内部执行者读取输入和媒体、写出结果 |
| HF 登录态及两个持久目录 | 在本机创建 HF Jobs；代码只读，结果可写 |

常见配置是调用身份拥有项目的 Vertex AI User 权限，及目标 bucket 的对象读写/列举权限。状态文件更新需要覆盖已有对象，因此只授予 Object Creator 不足以支持本管线。可由管理员按实际需求配置更窄的自定义权限。Google 服务代理通常为 `service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com`；跨项目 bucket 尤其要核对其输入读取和输出写入权限。不要把“Batch 不支持自定义执行服务账号”误读成调用方不能用服务账号认证。[GCS 输入与权限说明](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage)

本代码不会创建 bucket、开启 API 或授予角色。建议先使用同一项目内、位置合适的专用 bucket，模型位置默认 `global`。HF bucket (`hf://...`) 不能充当 GCS (`gs://...`)。

### 本机 ADC 与 HF ADC 是两回事

本机调试可使用现有 ADC，或设置 `GOOGLE_APPLICATION_CREDENTIALS` 指向本机 ADC JSON。HF 容器不会继承本机文件或 gcloud 登录态。

随附 HF launcher 接收环境变量 **名称**（默认 `GOOGLE_ADC_JSON`），通过 HF secret 注入 JSON；`bootstrap_batch.py` 在容器临时磁盘写入仅当前用户可读的 ADC 文件，再移除秘密环境变量并运行 worker。它不把密钥放到 argv、代码 bucket、输出 bucket 或日志中。

如果管理员提供可用于 HF 的服务账号 JSON，在 PowerShell 中加载，不打印内容：

```powershell
$env:GOOGLE_ADC_JSON = Get-Content -LiteralPath 'C:\secure\castle-adc.json' -Raw
```

命令执行后可移除本机进程中的变量：`Remove-Item Env:GOOGLE_ADC_JSON`。用户 ADC 的 refresh token 也属于敏感凭据，需按相同方式保护。若使用工作负载身份联合，须另行配置实际可用的外部 token 来源；仅有 external_account JSON 并不会让 HF 自动获得身份，本 launcher 不配置身份提供方。

本机已登录 HF 时，提交和创建定时任务可以使用登录态；当前 worker 不需要专门注入 `HF_TOKEN` 来轮询 Vertex。挂载通过 HF Jobs 配置处理。若访问受限数据集、或要求 worker 自己管理 HF schedule，则另行配置相应 HF 权限；本版采用本机/agent 暂停 schedule。

## 发布代码与生成命令

安装本地依赖：`python -m pip install -r requirements-batch.txt`。以下命令在 pipeline 目录运行，所有 bucket、项目、源文件都需替换。命令默认只生成 argv，**不会提交 HF Job**。`batch_pipeline.py prepare` 本身会上传文件，不是 dry-run。

代码发布到不可变的 HF prefix，至少包含：

```text
batch_pipeline.py  batch_jobs.py  batch_tick.py  bootstrap_batch.py
requirements.txt  requirements-batch.txt
castle_pipeline/*.py  prompts/*.md
```

不要上传 `_test/`、密钥、ADC 文件、缓存或完整工作区。已有在线 Dockerfile 不包含 Batch 入口；本 launcher 默认使用 Python 基础镜像并安装 Batch 依赖，submit worker 安装 FFmpeg。正式大规模运行可制作包含相同依赖的镜像，减少每小时启动开销。

### 一次性提交：先限定一个源文件、3 clips

以下测试示例中的 `NEW-UNIQUE-ID` 必须替换成从未使用的新标识，贯穿本机、GCS 和 HF 的隔离输出；正式续跑则使用私有绑定记录的原路径。

`credentials/team-b.json` 也是虚构占位符，替换成私有绑定中的文件，并将同一份 JSON 加载到 `GOOGLE_ADC_JSON`；不能仅设置本地文件参数而漏掉 worker secret。HF namespace 与路径按实际访问权限替换，准备步骤见 [HF 指南](huggingface-setup.md)。

```powershell
python batch_jobs.py submit `
  --code-volume hf://buckets/Ligant/castle-code/BATCH_CODE_VERSION `
  --output-volume hf://buckets/Ligant/castle-output/_test/batch-NEW-UNIQUE-ID `
  --name castle-batch-submit-smoke `
  --project YOUR_PROJECT `
  --state-uri gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID/state.json `
  --credentials-file credentials/team-b.json `
  --credential-secret GOOGLE_ADC_JSON `
  -- `
  --gcs-prefix gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID `
  --source EXACT_DATASET_VIDEO_PATH `
  --model gemini-3.8-flash --max-clips 3
```

确认范围后，在分隔符 `--` **之前**加 `--execute` 才会实际启动，随后 detach。记录返回的 HF Job ID。`--source` 可以多次给出不同文件，`--max-clips` 是每文件限制；默认总量硬上限为 2,000 clips。一个小时视频约 120 clips。长范围需相应调整 submit timeout，而不是取消内存边界。

### 每小时检查一次

HF submit Job 已成功创建后，就可以生成并建立定时任务，**无需等预处理或首个 Vertex Batch 提交完成**。使用与 submit 相同的显式代码版本、GCS state URI 和 HF output prefix：

```powershell
python batch_jobs.py hourly-tick `
  --code-volume hf://buckets/Ligant/castle-code/BATCH_CODE_VERSION `
  --output-volume hf://buckets/Ligant/castle-output/_test/batch-NEW-UNIQUE-ID `
  --name castle-batch-tick-smoke `
  --project YOUR_PROJECT `
  --state-uri gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID/state.json `
  --credentials-file credentials/team-b.json `
  --credential-secret GOOGLE_ADC_JSON
```

加 `--execute` 才会建立 schedule。底层为 HF `scheduled run @hourly --no-concurrency`（v6 起；v5 及更早写的是 `hourly`，会被 API 以 `Invalid CRON expression` 拒绝），单次默认 timeout 30 分钟；不会在一个 worker 中循环等待。记录 **schedule ID**，它与每次执行的 Job ID 不同。用同样参数把 `hourly-tick` 换成 `tick`，可启动一次性的检查 Job。

每次定时执行先读取 GCS state：

- state 不存在：输出 `status: waiting_for_state`、`stop_schedule: false`，以退出码 0 结束；不会初始化 state、查询 Vertex Job 或提交 Batch。仍有 HF 启动、依赖安装和一次 GCS 查询的成本。
- state 已存在：执行所挂载版本原有的 tick，包括代码哈希核验、检查、收集和推进。`ready` / `submitting` 状态可能尚无 Job ID，仍交给原流程处理首次提交和恢复，不能一概跳过。
- 认证、权限、网络或读取错误不会被当成“尚未准备好”；已有 state 与显式版本不兼容时，launcher 仍拒绝创建。

`--code-volume auto` 依赖已有 state，无法在预处理期间推断版本；手动一次性 `tick` 也仍要求已有 state。`submit` 不会自动创建 schedule，需要紧接着执行上面的 `hourly-tick --execute`。如果预处理失败或 state URI 写错，定时任务会一直等待，应检查 submit Job 并暂停无用的 schedule。

在已登录的本机查看或停止：

```text
hf jobs inspect JOB_ID
hf jobs logs JOB_ID
hf jobs scheduled inspect SCHEDULE_ID
hf jobs scheduled suspend SCHEDULE_ID
```

完成、部分完成且有错误、暂停或需要人工处理时，`batch-summary.json` 给出 `stop_schedule: true`。**本版不会自行暂停 HF schedule**；agent/操作者应立即执行 suspend，否则每小时仍有短 HF 实例的费用。终态 tick 不再调用模型或创建 Batch。暂停 HF schedule 也不会取消已经在 GCP 执行的 Batch；若需取消，须明确操作对应 Vertex Job。

## 本机 worker 接口与恢复

```text
python batch_pipeline.py status --project YOUR_PROJECT --state-uri gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID/state.json --output-dir _test/batch-NEW-UNIQUE-ID/check --scratch-dir _test/batch-NEW-UNIQUE-ID/scratch
python batch_pipeline.py tick --project YOUR_PROJECT --state-uri gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID/state.json --output-dir _test/batch-NEW-UNIQUE-ID/check --scratch-dir _test/batch-NEW-UNIQUE-ID/scratch
python batch_pipeline.py reconcile --project YOUR_PROJECT --state-uri gs://YOUR_BUCKET/castle/_test/batch-NEW-UNIQUE-ID/state.json --output-dir _test/batch-NEW-UNIQUE-ID/check --scratch-dir _test/batch-NEW-UNIQUE-ID/scratch
```

`status` 只读取保存状态；`tick` 会查 GCP，并可能提交下一阶段，因此是运行操作。`prepare` 不加 `--submit` 时只准备和初始化，之后 `start` 可提交。所有 worker 都要求明确 output/scratch 路径。

- 状态写入使用 GCS generation 前置条件；请求和媒体不可覆盖。同内容可重用，不同内容报错。独立运行使用独立 prefix，禁止混用 scope。代码 hash、提示词和范围保存到状态中，后续 tick 要求核心身份兼容，不能仅凭 release 名判断。
- 提交前先持久化意图，关闭 SDK 自动重试 create。若响应丢失，仅用确定的 display name 查询已存在的 Job。显式 `reconcile` 可在延迟可见后重新查找并接回唯一 Job，永远不会重新 create。找不到或找到多个时需检查 Vertex 控制台，不要修改状态 hash 或盲目重跑提交。
- 音频失败的 clip 不进入视觉阶段；单行坏输出不抹掉其他成功项。原始响应、错误、正规化记录和有效阶段结果保留；不自动重复付费失败请求。需要补跑时先根据状态挑选未完成 clip，使用新的 run prefix 和明确的 clip 范围。本版没有自动跨 run 导入旧阶段结果的接口。
- Batch 输出按回显请求中的 `CASTLE_BATCH_ID` 对齐，不能按文件行号匹配。缺失、重复、未知响应都记为问题；只有通过校验并完成所需复核的 clip 才写 final。
- 断电/终止后，收集可以重放且不会重复提交下一阶段。准备阶段在 state 初始化前中断时，重新准备同配置可复用相同 GCS 对象；仍可能需要重新下载源视频。
- 此 Batch final 在 `final/<clip_id>.json`，保留 `annotation`、`usage`、`review_required` 等字段；不直接兼容在线 `run_pipeline.py status` 的目录扫描，使用 Batch status/summary。来源与片段范围在 final 内完整记录。

## 日志、资源和成本

HF stdout 与 `/output/batch-events.jsonl` 输出媒体开始/完成、每 clip 准备、收集时逐行成功/失败及 token 用量、final 完成、阶段和 Batch Job ID。原始 provider status 保存在 GCS raw 响应，日志只打印归类错误码和可用的结构化 provider code。重放收集可能重复打印记录，按 request_id 去重，不能直接把多次日志相加计算费用。失败响应缺少 usage 不表示没有费用。

等待期间 HF 没有常驻进程，因此没有每 10 分钟心跳或逐请求 AIMD 日志。只有实际创建了 schedule 才会有每小时 tick；手动 tick 每次检查一次。逐条 usage 在阶段结束收集时可见；本版不提前收集仍在运行阶段的输出。

内存按有限的在途 clip 数控制：prepare_workers 限制并发准备数，decode_slots 限制同时解码数；不在内存中堆叠整小时帧。JSONL 请求流式写盘，收集结果逐行落盘再逐 clip 校验。按 4K RGB 估算一张未压缩帧约 24 MiB，仍需为并发解码器、Python、SDK 留余量。磁盘至少容纳当前下载的完整 MP4、在途 clips 的媒体及当阶段结果临时文件。真实 HF 内存应依据相应 Job 的测量，不能直接套用合成测试数值。

### 准备阶段的并行度

`prepare` 按源视频依次处理，每个源内部可并发准备有限个片段。v1 曾使用串行准备和单解码线程；v2 起支持下表调优参数。2026-09-17 的合成 3840×2160 H.264 测量中，单线程准备 30 张采样帧的纯解码约 69.5 秒：

| `-threads` | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 纯解码（30 帧） | 69.5 s | 29.4 s | 11.3 s | 5.7 s |

对应参数：

| 参数 | 默认 | 环境变量 | 作用 |
|---|---|---|---|
| `--media-threads` | CPU 数推导，上限 4 | `CASTLE_MEDIA_THREADS` | 每次 ffmpeg 调用的解码线程数 |
| `--decode-slots` | 1 | — | 同时解码的片段数 |
| `--footer-workers` | CPU 数推导，上限 4 | `CASTLE_FOOTER_WORKERS` | 时间戳盖章线程池 |
| `--prepare-workers` | CPU 数推导，上限 3 | `CASTLE_PREPARE_WORKERS` | 并发准备的片段数 |
| `--memory-hint-gib` | 16 | — | 解码缓冲准入检查所用的机器内存 |

这些参数在 prepare 时确定，相关媒体调优值保存在 state.config；当前 tick 不提供修改入口，不能据此承诺 Batch 恢复时可调整。测试验证了固定解码线程数时的并发稳定性及不同 footer_workers 的字节一致性，未保证改变 FFmpeg 线程数后 JPEG 字节仍相同。总并发解码线程约为 `media_threads × decode_slots`，每线程按约 40 MiB 计缓冲，超出 `--memory-hint-gib` 时 prepare 在构造客户端前拒绝。机器规格应在提交时核对；已有运行使用过 cpu-basic 的 2/2/2 和 cpu-upgrade 的 4/3/3（media_threads/decode_slots/prepare_workers）配置。`batch_media_tuning` 记录实际值及解码缓冲预算。

为支持下个短 Job 的原生分辨率 crop review，启用 review 时还会上传 1 fps 的 native JPEG。一个小时对应约 3,600 张 native 图片 + 3,600 张模型尺寸图片，GCS 存储/操作/传输也计费。`--no-review` 可省去 native 副本，但改变标注能力，不能在已初始化 run 中途切换。代码不自动删除云端媒体；按实验保留要求另行配置生命周期，不要清理仍被 pending Batch 引用的对象。

官方 JSONL 限制为每批 200,000 请求、1 GB；适配器在上传前流式计数检查，本管线还设更小的 clip 范围上限。[官方 JSONL 与结果格式](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage) Python SDK 使用 ADC 的 Vertex client 及 `batches.create/get/list`。[SDK 文档](https://googleapis.github.io/python-genai/)

## 验证边界

离线测试包含 SDK 请求序列化、create 不自动重试、状态写入竞争、模糊提交恢复、分阶段推进、乱序/缺失/部分失败、native crop、重启重放和最终输出。真实 FFmpeg 使用小合成视频测试 50 fps → 1 fps。

2026-09-16 全套测试：**184 passed**，含原有在线模式回归。测试结果全部位于专用 `_test/` 路径；独立代码审查完成，skill 验证通过并同步到已安装版本。

以上 184 项是 2026-09-16 的历史离线测试记录。此后已有真实 HF → GCP 提交、收集和 final 写出，说明当时相应凭据、模型访问与存储路径可用；阶段耗时由 tools/cloud_probe.py / tools/report.py 根据 Vertex 时间戳生成。实际账单、全面标注质量和新环境权限仍应单独核对。新模型、账号或媒体配置先以明确授权的小范围验证，再扩展。

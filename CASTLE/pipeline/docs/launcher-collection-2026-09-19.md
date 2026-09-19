# Launcher 修复与已有 Batch 推进

本轮范围：修复 launcher 的两处高优先级错误，并推进已开始的五条 day1 run。身份重构、参数热更新和跨版本继承暂缓。

用户随后明确选择：本轮收集及下一阶段提交完成后先汇报，后续再决定。因此不创建定时推进，也不在新提交的 annotation / review 结束后自动再执行一轮 tick。

## “收集”和 ledger

Vertex 把模型原始预测写到 GCS。HF worker 的收集步骤按 request_id 找到对应 clip，校验 JSON 与标注契约，保存结构化阶段结果；若 clip 不再需要下游处理，则生成 final，同时写入 GCS 与 Hugging Face 输出 bucket。

当前 tick 连着做两件事：收集已结束阶段，然后为有效结果提交下一阶段。因此 audio 已算完只代表可以收集并进入 annotation；annotation 结束后，部分 clip 直接生成 final，部分进入 review。review 收集后才是整个 clip 自动流程结束。

ledger 是运行台账，存放提交时间、Job ID、范围、代码版本和状态变化。它不保存标注正文，也不替代 GCS state。每次状态变化追加新记录，保留历史。

## 已完成的本地修复

- 首次 submit 使用已登记 release 时，不再要求新 prefix 已经存在 state；worker 原有的“拒绝重复初始化”检查仍有效。
- tick / hourly-tick 根据 code_hash 判断兼容性；v3/v4 同哈希可以互用，并保留显式指定的挂载版本。
- 真正不同的哈希仍在提交前拒绝；错误信息显示两端哈希。
- 新增 9 个参数化测试用例。修复前，其中 6 个按预期失败；修复后相关测试 81 passed，完整测试 222 passed。
- 独立代码检查未发现需修复的问题。

只修改本地 launcher 及其测试、说明；没有覆盖云端已发布的 v4，也没有修改 worker 身份文件。当前 worker code_hash 仍为 `8c765a6f3dfb617b75afde6a46e390d261603f08e8517962f8c265ec0e663095`。

本地 launcher 已可启动新 v4 run。挂载中的旧 batch_jobs.py 不参与 worker 执行；worker 使用 batch_pipeline.py。以后若把这次 launcher 修改一起发布，应新建 release，保留 v4 原样。

## 本轮启动的 worker

2026-09-19 香港时间 19:24–19:25 提交，全部沿用原 run 的 GCS state、数据范围和 HF 输出目录，不重提已记录失败的 clips。

| run | 使用代码 | 收集阶段 | HF Job |
|---|---|---|---|
| Bjorn 08 | v1 | review | [6aae70f152d0dbd7f1d6e6cf](https://huggingface.co/jobs/Ligant/6aae70f152d0dbd7f1d6e6cf) |
| Bjorn 09 | v2 | review | [6aae70f851992417dfcc95ce](https://huggingface.co/jobs/Ligant/6aae70f851992417dfcc95ce) |
| Bjorn 10–14 | v2 | annotation | [6aae710052d0dbd7f1d6e6d7](https://huggingface.co/jobs/Ligant/6aae710052d0dbd7f1d6e6d7) |
| Allie 13、14、18–20 | v4（v3 同身份） | audio | [6aae710752d0dbd7f1d6e6d9](https://huggingface.co/jobs/Ligant/6aae710752d0dbd7f1d6e6d9) |
| Bjorn 15–20 | v4（v3 同身份） | audio | [6aae710f52d0dbd7f1d6e6db](https://huggingface.co/jobs/Ligant/6aae710f52d0dbd7f1d6e6db) |

截至 11:37:23 UTC（香港 19:37:23），本轮五个 HF worker 均已 COMPLETED；当前三个新 Vertex 阶段已被接收，本轮不再推进它们。

| run | final | 已记录失败 | 本轮之后的状态 | final 中 review_required |
|---|---:|---:|---|---:|
| Bjorn 08 | 110 | 10 | complete_with_errors，已收尾 | 4 |
| Bjorn 09 | 105 | 15 | complete_with_errors，已收尾 | 9 |
| Bjorn 10–14 | 275 | 33 | 292 条 review 已提交，当时 QUEUED | 1 |
| Allie 13、14、18–20 | 0 | 4 | 596 条 annotation 已提交，当时 RUNNING | 0 |
| Bjorn 15–20 | 0 | 4 | 716 条 annotation 已提交，当时 RUNNING | 0 |

总计 490 个 final、66 个已记录失败、1604 个 clip 进入新提交阶段，合计原有 2160 个。HF bucket 的前三个 final 文件数量已分别核对为 110/105/275。14 个 final 带 review_required，需要后续人工或更进一步的复核；final 不等同于已人工验证的真值。

新 Vertex Job（均在项目 `1073111426432`、位置 `global`）：

- Bjorn 10–14 review：`batchPredictionJobs/9151394844604628992`，292 行，沿用 v2 的 16384 token 上限。
- Allie annotation：`batchPredictionJobs/4715138105412157440`，596 行，32768 token 上限。
- Bjorn 15–20 annotation：`batchPredictionJobs/3836936178074910720`，716 行，32768 token 上限。

上述 Job ID、GCS state generation、完成/失败/待处理数量和 review_required 数已追加到 ledger。没有 HF schedule；后续模型计算完成后需另一次明确的收集推进。最新带时间戳的阶段计数见 [STATUS.md](../STATUS.md)。

操作证据保存在 `_test/launcher-collect-20260919-01/`：每个 Job 的命令预览、提交回执、GCS/Vertex 探测快照，以及更新前的 STATUS 备份。台账通过 ledger.py 的单一追加入口更新。

## 暂缓的方案何时才需要

- 如果一条 run 从头到尾固定代码和请求配置，现有 pin 机制就够用。
- 只有确实需要“旧 audio + 新 annotation 配置”，才需要阶段身份和 import-stage。
- 只有继续同时维护在线与 Batch 并频繁修改二者，命名空间解耦才更迫切。
- 当前优先完成实际标注，并保持每次提交可追溯；不在本轮扩展上述架构。

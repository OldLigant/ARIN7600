# CASTLE 项目结构与版本管理审计

这是首次审计时的历史快照。下文记录的问题与运行计数不代表后续实时状态：launcher 修复和一轮收集见 [后续执行记录](launcher-collection-2026-09-19.md)，当前设计边界见 [最新决定](version-evolution-decisions.md)，运行状态见带探测时间的 [STATUS.md](../STATUS.md)。

核对时间：2026-09-19，GCS / Vertex 探测时间为 10:51:25 UTC（香港时间 18:51:25）；HF 清单与输出检查在同次审计中完成。

结论：本地 Git 和发布管理第一阶段已经落地，但阶段身份、运行参数调整、跨版本继承尚未实现。已提交的十个 Vertex 阶段任务都已成功结束，五条完整 run 却都没有结束：缺少后续 tick，结果尚未全部收集，下游阶段尚未全部提交。继续等待不会自动推进，因为没有 HF schedule。

本次只读检查云端，并在独立 `_test/audit-20260919-review-01/` 保存证据；未执行 tick、模型请求、发布、云端写入、删除或 Git 提交。原有 `STATUS.md` 修改及未跟踪的 `_bench/preflight_stage_outputs.py` 保留原状。

## 1. 项目分成四层

| 层 | 位置 | 职责 |
|---|---|---|
| 本地开发与管理 | `D:/QLD/CASTLE/pipeline` | 源码、Git、测试、发布清单、ledger、操作工具 |
| HF 代码发布物 | `hf://buckets/Ligant/castle-code/<release>` | HF Jobs 只读挂载到 `/workspace` 的运行文件快照；不是 Git 历史 |
| HF 结果 | `hf://buckets/Ligant/castle-output/<run或campaign>` | 在线检查点与 final；Batch final、事件日志、状态快照 |
| GCS / Vertex | `gs://castle-caption-batch/castle/<run>` | Batch 的权威 `state.json`、媒体、请求 JSONL、原始预测、阶段结果与 final；Vertex 执行模型计算 |

原始视频来自 `CASTLE-Dataset/CASTLE2024`，代码默认固定数据 commit `c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4`。

本地主要入口：

- `run_pipeline.py` + `jobs.py`：在线 standard / flex 流程与 HF launcher。
- `batch_pipeline.py` + `batch_jobs.py` + `bootstrap_batch.py`：Batch 的 prepare / start / tick / status / reconcile，HF launcher，以及 ADC 凭据引导。
- `castle_pipeline/inputs.py`、`media.py`、`schema.py`：输入、媒体准备、结果契约。
- `castle_pipeline/runner.py`、`vertex.py`、`events.py`：在线执行、请求限速重试、日志。
- `castle_pipeline/batch_cloud.py`、`batch_engine.py`、`batch_tasks.py`：GCS / Vertex 传输、阶段状态机、请求构建及收集。
- `prompts/`：audio、annotation、review、review_regions 提示词。
- `release.py`、`releases/`：发布清单、哈希核验、发布脚本。
- `ledger.py`、`ledger/ledger.jsonl`、`tools/cloud_probe.py`、`tools/report.py`：本地操作记录与生成状态报告。
- `_bench/`：历史排查与临时操作脚本；部分仍使用仓库外 ledger / secrets 路径，不能视为统一操作入口。
- `tests/`：测试；`_test/`：忽略的隔离测试环境及产物；`credentials/`：忽略的本机凭据。

在线流程在 HF Job 中持续执行音频、主标注、可选复核；Batch 则先在 HF Job 中准备媒体并提交 audio，随后由独立 tick 收集与推进 annotation、review。HF prepare Job 的 COMPLETED 只说明提交 worker 退出成功。

## 2. 已有 Git 与发布历史

当前分支 `main`，HEAD `0c641f0`，共四次提交；未配置 Git remote。

| 提交 | 内容 |
|---|---|
| `078d904` | Git 基线、凭据隔离；tag `castle-batch-v3` |
| `7569292` | release manifests、tag 对应、字节保真 |
| `dfb67da` | launcher 自动选版本、发布、ledger、状态报告；tag `castle-batch-v4` |
| `0c641f0` | 更新生成的 STATUS |

Git 是在 2026-09-18 香港时间凌晨补上的。v1/v2 的历史发布物有本地清单，但没有对应 Git tag；不能把 Git 基线以前的开发过程当作已经恢复的提交历史。在线 `castle-v1/v2` 也没有纳入现有 Batch 清单体系。

云端代码桶共有六个前缀：在线 `castle-v1/v2`，Batch `castle-batch-v1/v2/v3/v4`。

| Batch release | code_hash 前缀 | 主要变化 | 本次云端逐文件回读 |
|---|---|---|---|
| v1 | `aa489631` | Batch 初版，token cap 16384 | 22 个声明文件全部匹配；实际 44 个文件，额外的是文档、skills、测试 |
| v2 | `1d53e2b1` | 媒体准备并行化，仍为 16384 | 21/21 匹配 |
| v3 | `8c765a6f` | 默认 32768，增加 `--max-output-tokens` | 21 个现有文件匹配；清单声明的 Dockerfile 缺失 |
| v4 | `8c765a6f` | launcher 自动选版本；核心身份文件同 v3 | 22/22 匹配，无额外文件 |

四个版本都能从云端核心文件重算出对应 code_hash。v3 缺少 Dockerfile 不妨碍当前使用 Python 基础镜像的 launcher，但说明发布物与清单没有完全一致。v4 的清单 `published_utc` 仍为 null，虽然云端前缀确实存在。审计没有修改这些历史发布物。

## 3. Caption 当前到哪里了

HF 查询结果：没有 RUNNING / SCHEDULING Job；包括 suspended 在内的 schedule 列表为空。GCS `castle/` 下发现五个正式 run 前缀，另有 `_probe-max-tokens/`。

五条正式 run 的 GCS state 仍是 `running`。以下“已完成 final”经 GCS 与 HF 文件数量交叉核对；“失败”是 state 已记录的数量，不包含尚未收集阶段中的失败。

| run 范围（均 day1） | 固定版本 | 选中 clip | 已完成 final | 已记录失败 | 当前已算完但未收集的阶段 |
|---|---|---:|---:|---:|---|
| Bjorn 08 | v1 | 120 | 47 | 10 | review，63 条响应 |
| Bjorn 09 | v2 | 120 | 13 | 13 | review，94 条响应 |
| Bjorn 10–14 | v2 | 600 | 0 | 3 | annotation，597 条响应 |
| Allie 13、14、18、19、20 | v3 身份（v4 兼容） | 600 | 0 | 0 | audio，600 条响应；annotation 尚未提交 |
| Bjorn 15–20 | v3 身份（v4 兼容） | 720 | 0 | 0 | audio，720 条响应；annotation 尚未提交 |

总计选中 2160 clips，已有 60 个 Batch final，26 个已持久记录的失败，2074 个仍在当前阶段待收集/推进。final 表示自动流程产物，并不等于人工验证的标注真值。

本次只读解析了五条 run 当前阶段的全部输出，并按 request_id 检查缺行和重复；未发现缺失、未知或重复 request_id：

| 当前阶段 | 请求 token cap | 响应解析通过 | 响应解析失败 | 补充 |
|---|---:|---:|---:|---|
| Bjorn 08 review | 16384 | 63 | 0 | 还需执行 review 契约校验和 final 写出 |
| Bjorn 09 review | 16384 | 94 | 0 | 同上 |
| Bjorn 10–14 annotation | 16384 | 582 | 15 | 12 条 MAX_TOKENS，另外 3 条 STOP 但解析失败；582 条中 306 条带 review_regions |
| Allie 13、14、18–20 audio | 32768 | 596 | 4 | 四条虽 STOP，JSON 响应仍不能解析 |
| Bjorn 15–20 audio | 32768 | 716 | 4 | 同上 |

这里使用的是现有 `parse_response()`，没有执行完整 schema / crop / review 合并检查，所以不能把“582 条可解析”直接当作“582 条最终成功”。旧 ledger 对 Bjorn 10–14 的“约 9% 会失败”只是历史预估，不能替代本次观察。

在线历史也有实际成果。`b20260916-main/day1/Allie/` 目前有 767 个 final 文件：

| 小时 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final 数 | 118 | 82 | 99 | 104 | 104 | 41 | 18 | 95 | 74 | 28 | 4 |

20 点没有 source summary，属于被取消时留下的部分输出。部分源文件覆盖略超整点，在线 summary 的 15、19 点为 121 clips，不能统一假定所有小时都是 120。旧 `day1-allie-08-v1/v2` 还有 27/104 个 final，和 campaign 有范围重叠，不能直接相加为独立完成量；standard smoke 另有 3 个 final，flex smoke 没有 final。

## 4. 为什么 16384 不能在 audio 之后改成 32768

这是几个约束叠加造成的：

1. v1/v2 在 `batch_tasks.py` 的每条请求里硬编码 16384。修改它需要换代码文件。
2. `batch_pipeline.code_hash()` 对 `batch_pipeline.py` 与 `castle_pipeline/*.py` 整体哈希。整个 run 初始化时记录这个哈希，start / tick / reconcile 要求精确匹配。
3. v3 虽然加了配置项，但仅 prepare 接收 `--max-output-tokens`；tick 没有修改接口。配置已写入 GCS，`BatchEngine.initialize()` 还要求既有 config 全等。
4. audio、annotation、review 共用一个 run 级 token cap；当前并没有独立的 annotation cap，也没有音频阶段可复用的独立身份。

所以 v3 的注释“作为运行参数，resume 可以调整”尚未兑现。Git 和 tag 解决的是“哪份代码是什么”，不会自动解决“旧 audio 可否被新 annotation 继承”。

Bjorn 10–14 的 annotation 现在已经实际按 16384 执行完成；更改后续默认值不会修复已截断的 12 条响应。应保留已有有效结果，待完整校验后只对未解决 clip 制定后续处理范围。

## 5. 补丁的真实完成度

原计划位于 `docs/release-process.md`。文档勾选八项完成、七项待办，但已勾选项仍有实现缺口。

| 条目 | 审计判断 |
|---|---|
| R-01 凭据隔离 | `.gitignore`、credentials 分离已做；Git 跟踪的该目录文件只有 README；本次路径检查未发现凭据文件进入 Git 历史 |
| R-02 Git / tag | 已做本地 Git 与 v3/v4 tag；尚无 remote，旧版本没有完整 Git 历史 |
| R-03 清单 | 已有四份清单与测试；v3 云端缺 Dockerfile；v4 发布日期未登记 |
| R-15 字节保真 | 已做；核心身份匹配，tag 一致性测试通过；旧 media.py 的 CRLF 仍需保留直到新 release |
| R-05 launcher | auto 能工作，但存在两个已复现的边界错误，见下文 |
| R-06 发布 | 按 tag 取字节、白名单暂存、默认拒绝覆盖已做；发布后的校验只比文件名，没有字节回读，且有 `--force` 绕过 |
| R-08 ledger | 已入仓库，17 行、11 个 job，schema 校验通过；仍靠手动记录，launcher 未自动追加，记录未随云端自动更新 |
| R-09 报告 | 已有生成器；HF job 状态来自 ledger 而非 HF 查询，run 列表硬编码，部分过期文案仍存在 |
| R-04 | 未做：state 没有 release；worker 报错仍不能给出 expected / actual / 修复命令 |
| R-07 | 未做：bootstrap 只处理凭据，不做发布清单启动核验 |
| R-10 | 未做：测试、清单、字节回读尚未成为强制发布闸门 |
| R-11 | 未做：在线和 Batch 仍共享 `castle_pipeline/*.py` 身份集合 |
| R-12 | 未做：身份与可调整参数没有分层，tick 无参数修订机制 |
| R-13 | 未做：不存在 import-stage，不能受审计地继承跨版本 audio |
| R-14 | 未落实不可绕过的发布防覆盖；本次未核验存储侧权限策略，不假设 HF 有特定对象锁能力 |

已复现的 launcher 问题（只用本地模拟 state，无云端写入）：

- **新 run 无法使用已登记版本首次 submit**：`batch_jobs.py:150` 对已知 release 无条件读旧 state；新 prefix 没有 state，直接 `No batch state ... nothing is pinned yet`。首次 submit 应检查发布物与空前缀，不能要求已有 run pin。
- **同哈希版本被错误判冲突**：v3/v4 相同 code_hash，解析函数按覆盖顺序选到 v4，随后 `batch_jobs.py:152` 比 release 名称；显式指定 v3 便报 `E_CODE_VERSION_MISMATCH`。实际核心字节兼容，应明确 canonical release / compatibility 规则。

另外，当前 `release.py publish` 的自动检查不能等同于 `_bench/verify_release_bytes.py` 的独立字节检查。后者已有能力，但没有整合进发布路径。历史 v1 的额外文件实际是文档和测试，与 release 文档所称“.pyc 重复”不符。

文档/记录仍不一致的例子：README 第 7 行仍称没有提交真实 Batch；`docs/batch.md` 的末尾还保留未验证 ADC / 模型访问的旧描述；`STATUS.md` 的两个 prepare job 仍是 submitted，但 HF 已为 COMPLETED。未跟踪的 `_bench/preflight_stage_outputs.py` 只检查预测 JSON 是否可解析，不是 R-10 的发布 preflight。

## 6. 建议的下一步顺序

1. **先补齐版本管理第一阶段**：修复 R-05 两个边界，给 release publish 接入逐文件 SHA-256 回读和 R-10；把发布 manifest 带到 worker 并做 R-07；清理文档矛盾，统一 ledger/报告的真实状态来源；配置用户选定的 Git remote。保留历史代码前缀。
2. **将旧 run 明确收尾**：Bjorn 08/09 用原版本收集 review；10–14 用 v2 校验并收集已完成 annotation，再对有效行推进 review；v3 两条 run 收集 audio 后推进 annotation。现在 tick 会同时收集和提交下一阶段，并不是只读操作；正式执行须属于后续运行授权。本次审计未执行。
3. **发布下一版身份重构**：R-04 + R-11 + R-12，显式区分 online / batch 文件集合，使用相对路径哈希方案，并对 audio / annotation / review 建立依赖身份。基础执行参数（线程、并发等）可调；token cap 要按阶段记录实际请求配置，不能把新 cap 的产物伪装成旧请求的同一次执行。
4. **实现 R-13 import-stage**：在新 prefix 复用兼容的媒体/audio，记录来源 release、code_hash、GCS state generation、request_id 和产物哈希；不修改旧 state。只重跑无法恢复的阶段/clip。Bjorn 10–14 已有 3 条 audio 失败，因此“600 条 audio 全部可继承成功”的旧验收措辞应改为真实成功集合，并保留失败信息。
5. **补齐发布不可变性与运维入口**：取消普通路径的强制覆盖，按 HF 实际支持能力设置访问控制；把有价值的 `_bench` 功能提升为维护工具，避免继续依赖硬编码 run、旧路径和手工记忆。

开发新 release 本身不必等旧 run 收尾；必须保证旧 run 的后续 tick 继续加载原来匹配的发布字节。没有活跃 Vertex 请求是一个合适的检查点，但不是把旧 run 的 state 改成新 hash 的理由。

## 7. 本次验证与证据

- 相关测试：`test_release_manifest`、`test_ledger`、`test_batch_jobs`、`test_batch_engine`、`test_batch_tasks`，**72 passed**。首次沙箱运行遇到临时目录 PermissionError，随后在新的隔离目录重跑通过。没有以既有数据目录作为 pytest basetemp。
- 本地 `release.py verify --release castle-batch-v4` 通过。
- `ledger.py validate` 通过：17 行 / 11 个 job。
- 四份 Batch 发布物下载后分别与清单 SHA-256 核验；结果见上表。
- 五条 run 的 GCS state、已登记十个 Vertex job 实时状态、当前阶段全部预测行、GCS/HF final 数量已核对。
- 在线 campaign 767 个 final 是文件数量，并根据 run.json 映射 source；本次未逐条审查媒体和标注质量。
- 两个 launcher 问题有离线最小复现；72 项原有测试通过不代表这两个路径已被覆盖。

本次证据均在 `D:/QLD/CASTLE/pipeline/_test/audit-20260919-review-01/`：

- `cloud-probe.json`：state / Vertex 快照。
- `hf-jobs.json`、`hf-code-tree.json`、`hf-output-tree.json`：HF 清单。
- `local-audit.json`：发布字节核对、输出计数、launcher 复现。
- `stage-output-audit.json`：当前阶段解析与 token cap 统计。
- `online-output-audit.json`：在线结果按小时对应关系。
- `audit_local.py`、`audit_cloud_outputs.py`：本次隔离审计脚本。

这些证据处于被 Git 忽略的审计目录，正式长期保留时可挑选不含内容载荷的摘要进入受控审计记录。

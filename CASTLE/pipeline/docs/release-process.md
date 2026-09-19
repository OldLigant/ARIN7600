# CASTLE 发布与版本规范

状态：R-01/R-02/R-03/R-05/R-08/R-09/R-15 已有实现；R-06 已实现按 tag 发布与文件清单检查，**内容哈希回读尚未接入发布路径**。历史发布进度记录于 2026-09-17，launcher 于 2026-09-19 在本地补充修复。
实施进度见第 5 节勾选表；条目编号（R-01…R-15）用于评审、实现与勾选，请勿重排。

**2026-09-20 后续发布**：`castle-batch-v5` 已发布提前调度与外围阶段镜像，完整记录见 [v5 发布记录](batch-v5.md)。核心文件和提示词与 v3/v4 一致，辅助文件通过新 manifest/tag 追溯；旧发布目录没有覆盖。下文 9 月 19 日的“尚未发布”描述是当时的历史状态。

**2026-09-19 设计决定（仍适用）**见 [版本演进的边界与实施依据](version-evolution-decisions.md)：本轮只完善文档；跨版本继承限定为“新 run 显式继承成功 audio”，在线 / Batch 解耦留待新 release，内容核验限定为只读报告。下文未实施条目是候选设计，不是当前执行清单。

此前落地的管理工具刻意放在身份文件之外（`tools/`、`ledger.py`、launcher），所以 v4 的核心身份与 v3 相同。未来 R-04、R-11 等会修改身份文件，必须作为新 release；不能笼统声称本规范的所有实施都不改变哈希。旧发布目录和旧 state 保持原样，旧 run 继续使用匹配的旧 worker。

---

## 0. 这份规范要解决的问题

以下是编写初稿时的历史背景：当时尚未建立本地 Git，云端通过 `hf://buckets/Ligant/castle-code/` 下的版本目录保存发布快照，两者缺少受版本控制的对应关系。Git 和清单现已补上；这些历史问题用于解释条目来源：

| 根因 | 说明 | 已发生的后果 |
|---|---|---|
| 内容哈希 ≠ 发布身份 | `code_hash()` 由"当时挂载进来的文件"现算（`batch_pipeline.py:38-41`）；`castle-batch-v3` 这个名字与 `8c765a6f` 这个哈希的映射只存在于仓库外的 `ledger.jsonl`，以及硬编码的 `_bench/version_hash_audit.py:17` | 3 个 tick Job 挂错代码卷，15–16 秒内失败 |
| 身份覆盖范围有洞 | 哈希覆盖 `batch_pipeline.py` + `castle_pipeline/*.py` 共 11 个文件；**不覆盖** `batch_jobs.py`（正是决定挂哪个卷的文件）、`bootstrap_batch.py`、`run_pipeline.py`、`requirements*.txt` | launcher 自身不可追溯 |
| 两条管线身份耦合 | `runner.py:274` 用 `glob('*.py')` 哈希整个 `castle_pipeline/`，batch 侧改动会改变在线指纹 | 在线 sweep 被迫钉死在 `castle-v2` |
| 身份无法迁移 | 版本一旦钉住，修复无法回流；`docs/batch.md:119` 自述"本版没有自动跨 run 导入旧阶段结果的接口" | run `10-14` 的 audio 已 SUCCEEDED 却无法用 `v3` 收集 |

---

## 1. 术语

| 术语 | 定义 |
|---|---|
| **release** | 人类可读的发布名，等同于一个 git tag，例如 `castle-batch-v4`。禁止再用"目录名"或"日期"充当发布身份。 |
| **artifact set** | 该 release 覆盖的文件集合，以**仓库相对路径**表示。 |
| **code_hash** | 对 manifest 中 identity_files 内容计算的指纹；auxiliary_files 和 prompt_files 另有逐文件哈希。 |
| **publication** | 将 tag 对应的运行文件发布到 `hf://buckets/Ligant/castle-code/<release>`；映射由仓库内 `releases/*.json` 维护，不另建 versions.json。 |
| **pin** | 现有 state 固定 code_hash；未来可同时记录 release。续跑需匹配身份；v3/v4 这样的同哈希 release 可以兼容使用。 |
| **identity_inputs** | 影响标注语义的输入（代码、prompt、model、fps、clip 时长、max_dim、review 设置等）。 |
| **run_parameters** | 执行参数，例如线程和并发。token cap 可能改变截断和产物完整性，必须记录实际值；不能因归入预算就默认允许原 run 中途修改。R-12 尚未实现。 |

---

## 2. 五条核心原则

1. **单一事实来源**：`release` 名字来自 git tag；"这个 release 里有什么"来自仓库内的清单文件。
2. **三者互证**：仓库清单 ↔ 工作区实际内容 ↔ 云端已发布内容，任意两者不一致都必须报错，而不是"以某个为准"。
3. **发布只增不改**：已发布的 release 目录永不覆盖；任何修改都必须是一个新 release。
4. **失败必须可行动**：涉及版本不匹配的错误必须包含 expected / actual，以及可直接复制的修复命令。
5. **文档可生成或可验证**：任何关于云端状态、版本、测试结果的断言，要么由脚本生成，要么附带可复现的命令与时间戳。

---

## 3. 规范条目

### P0 — 止血（可在当前云端任务运行期间执行）

#### R-01 安全前置：`git add` 之前先写 `.gitignore`

**规则**：在仓库根目录建立 `.gitignore`，且必须在任何 `git init` / `git add` 之前完成。

**历史背景（R-01 实施前）**：仓库根目录当时没有 `.gitignore`，但存在敏感文件：

- 在线 secrets 文件 —— 包含当时使用的 `GOOGLE_API_KEY`，已有 `*.secrets` 排除模式覆盖。
- `example-sa-key.json` —— GCP 服务账号密钥文件命名形态；**`.dockerignore` 未覆盖它**。

一旦执行 `git add .` 且忽略规则不完整，服务账号私钥会进入历史，之后清理需要重写历史。

**当前机制与落点**：凭据统一放入 `credentials/`，仅公共说明 README 受 Git 跟踪。Git 排除规则至少包含：

  ```text
  credentials/*
  !credentials/README.md
  *.secrets
  .env*
  _test/
  __pycache__/
  ```

Docker 排除整个 `credentials/`；发布按 manifest 白名单暂存。不要用全局 `*.json` 排除所有 JSON，因为 manifests 等正常文件需要版本控制。

**验收**：`git status --short` 中不出现以上两个文件；`.dockerignore` 覆盖服务账号 JSON；仓库内不存在能通过 `git log --all --diff-filter=A` 找到的密钥文件。

#### R-02 用 git tag 作为发布身份

**规则**：每次发布打一个 tag，tag 名即 release 名（`castle-batch-v4` 等）。首次建立 Git 时，以当时与 v3 核心身份一致的工作区作为 `castle-batch-v3` 基线；后续开发不能据此假定工作区始终等于该 tag。

**机制与落点**：`git tag -a <release> -m "<一句话说明>"`；发布脚本（R-06）必须校验 tag 存在。

**验收**：由 run 的 code_hash 查到对应 manifest / 候选 release；同哈希的 v3/v4 必须明确标为兼容候选，不能声称哈希能唯一还原实际挂载的 tag。实际挂载版本以提交记录为补充证据。

#### R-03 每个 release 一份受版本控制的清单 `releases/<release>.json`

**规则**：清单随仓库提交，由 `release.py manifest` 生成（**不手工编写**），schema 如下：

```json
{
  "release": "castle-batch-v3",
  "created_utc": "2026-09-17T20:18:05Z",
  "published_utc": "2026-09-17T18:59:27Z",
  "provenance": {"kind": "git|hf-bucket|workspace", "detail": "<tree 路径>", "git_commit": "<sha|null>"},
  "hash_scheme": "basename-v1",
  "code_hash": "<必须能由 identity_files 重算得出>",
  "identity_files": {"batch_pipeline.py": "<sha256>", "castle_pipeline/media.py": "<sha256>"},
  "auxiliary_files": {"batch_jobs.py": "<sha256>", "requirements-batch.txt": "<sha256>"},
  "prompt_files": {"prompts/annotation.md": "<sha256>"},
  "notes": "一句话说明本次变更"
}
```

`hash_scheme` 必须显式记录。当前只允许两个值：

- `basename-v1`：**现有实现**，`code_hash` 用 `{文件 basename: sha256}` 计算（`batch_pipeline.py:41`、`runner.py:274`）。**已发布的 v1–v5 全部使用它，不得原地更改。**
- `relpath-v1`：键改为仓库相对路径，**尚未启用**。

**两条硬规则**：

- `identity_files` 的键必须是**仓库相对路径**（清单层面），即使 `code_hash` 目前用 basename 计算。
  **理由**：`fingerprint()` 会排序后序列化，basename 冲突会静默吞掉一个文件。一旦按计划把 batch 模块移入子包（R-11），`castle_pipeline/batch/__init__.py` 与 `castle_pipeline/__init__.py` 就会撞名。清单先按相对路径记录，为切换做好准备。
- **不要在旧发布版本中改写 `code_hash` 的键方案**。改为相对路径会改变新版计算出的指纹；旧 run 若误用新版 worker，会因哈希不符而无法 tick。旧 run 继续使用原发布字节则不受影响。`relpath-v1` 的切换只在新 release 中实施，旧 manifest 继续按原方案验证。

**验收**：`tests/test_release_manifest.py` 断言每个清单的 `code_hash` 都能由它自己的 `identity_files` 重算得出；并断言 tag 与清单描述同一份字节。

#### R-04 state 同时记录 release 名与 code_hash，且错误必须可行动

**规则**：

- `state.config` 除 `code_hash` 外新增 `release` 字段（`batch_pipeline.py:241-248` 构造 config 处）。
- 版本不匹配时，错误必须同时给出稳定错误码、expected、actual 和修复命令：
  ```
  E_CODE_VERSION_MISMATCH
    run pinned to : castle-batch-v2 (code_hash 1d53e2b1)
    mounted code  : castle-batch-v3 (code_hash 8c765a6f)
    fix           : --code-volume hf://buckets/Ligant/castle-code/castle-batch-v2
  ```
- **必须修改 `batch_pipeline.py:290` 的归一化错误输出**：当前它对所有异常只输出
  `{"ok": false, "error_type": "ValueError", "message": "..."}`，
  把唯一有用的信息丢掉了。归一化器的目的是不泄漏凭据，不是不泄漏版本号——应白名单透出 `error_code` / `expected` / `actual`（这些字段不含密钥）。

**验收**：故意用错误代码卷跑一次 `tick`，日志中能直接读到 expected/actual 与修复命令。

#### R-05 launcher 不得依赖人的记忆

**规则**：`batch_jobs.py` 支持 `--code-volume auto`：读 state → 取 `code_hash` → 查 `releases/*.json`
→ 自行拼出正确卷路径。若显式给出的卷与 pin 不符，提交前必须拒绝并报 expected / actual / 修复命令。

**实施修正（2026-09-17）**：原稿要求查"代码桶中的 `versions.json`"。该文件**从未存在**，且按原则 1
（单一事实来源）也不该存在——release 名到哈希的映射已经由仓库内的 `releases/*.json` 承载。
实现改为读仓库清单，不引入第二处索引。

**已实施**：`batch_jobs.py:resolve_release()`；`--code-volume auto` 用于 tick 类角色，
`submit` 角色因尚无 state 而明确拒绝 auto。反例报错形如：

2026-09-19 补充修复（后已随 2026-09-20 的 v5 发布）：首次 submit 使用显式已登记版本时不再查询不存在的 run pin；tick / hourly-tick 的显式版本按清单 `code_hash` 比较，而非比较 release 名称，因此 v3/v4 同哈希时保持所选挂载且允许续跑。真正不同的哈希仍在提交前拒绝。新增回归测试覆盖首次提交、两个兼容方向、错误身份和 auto。云端 v4 保留原样。

```
E_CODE_VERSION_MISMATCH
  run pinned to : castle-batch-v1 (code_hash aa489631)
  requested     : castle-batch-v2
  fix           : --code-volume auto
                  or --code-volume hf://buckets/Ligant/castle-code/castle-batch-v1
```

**附带修复**：`_bench/tick_both.py` 曾把 `10-14` 映射到 `castle-batch-v3`（该 run 钉在 `v2`），
导致 18:59 与 19:08 两次失败。该脚本已删除，职责由 `_bench/tick_run.py` 承担，映射改为按 run 显式声明，
且自 v4 起错误映射会在提交前被拒绝。

**验收**：已用错误 `--code-volume` 实测——命令在提交前退出并打印正确路径，本地不产生任何 HF Job。

---

### P1 — 结构（可在当前云端任务运行期间执行）

#### R-06 发布是一个脚本，不是一条手敲的 sync

**规则**：新增 `release.py`，流程固定为：

1. 先用 `release.py manifest` 生成清单，提交对应运行文件和清单，再创建新 release tag；
2. `publish` 校验 tag 存在、tag 字节与清单一致、工作区清单覆盖的文件与清单一致；它不要求所有无关文档都没有改动，提交范围仍须自行检查；
3. 从 tag 提取清单白名单内的运行文件，同步到 `hf://buckets/Ligant/castle-code/<release>`；
4. **核验现状**：publish 当前只回读文件名清单；逐文件 SHA-256 核验已有独立脚本，未来范围限定为只读报告，见最新决策文档；
5. 用仓库内 `releases/*.json` 维护发布映射，不建立第二份 versions.json；
6. 目标 release 目录已有对象时默认拒绝覆盖，即使内容可能相同也不重新上传；日常发布不使用 `--force` 绕过。

**禁止**：手工 `hf buckets sync` 进 release 目录；当前操作说明使用 `release.py`，不要照旧记录执行全仓库同步。

**验收**：对已存在的 `castle-batch-v3` 执行发布，脚本拒绝且 bucket 内容未变。

#### R-07 worker 启动自检挂载代码

**规则**：worker 启动时（`bootstrap_batch.py` 或 `batch_pipeline.main()` 入口）对挂载的代码树做一次清单校验（11 个文件，毫秒级）。不一致则立即失败。

**理由**：现有 tick 的 code_hash 检查能识别已覆盖的核心身份文件变化，但 auxiliary 文件不在该哈希内。完整清单核验可补充覆盖启动脚本与依赖文件。此项仍是独立候选，2026-09-19 只读内容报告的决定不包含 worker 启动自检。

**验收**：人为篡改挂载目录中一个文件，worker 在安装依赖后、构造任何客户端之前退出。

#### R-08 ledger 入仓库，加 schema 与单一写入口

**规则**：

- 历史 `ledger.jsonl` 位于仓库外（`D:\QLD\out\castle-runs\<campaign>\ledger.jsonl`），手写、无 schema。现已迁入 `ledger/ledger.jsonl`；后续统一通过仓库内入口追加。
- 定义 JSON Schema；只允许 `ledger append` 写入；禁止手工编辑既有行。
- `PROGRESS.md` 与 README 中的版本表由 ledger 生成。

**验收**：`ledger validate` 能对全部历史行通过；`PROGRESS.md` 可一键重建且与仓库一致。

#### R-09 关于云端状态的文档必须可生成或可验证

**规则**：

- `STATUS.md` 由 ledger + 云端探测生成。探测脚本为 `tools/cloud_probe.py`，渲染器为 `tools/report.py`。
- **落点修正（2026-09-17）**：原稿建议把 `_bench/batch_status.py` 提升为 `batch_pipeline.py report`。
  该建议会**违反本文件开头自己的约束**——`batch_pipeline.py` 是被 `code_hash()` 覆盖的身份文件，
  给它加子命令会改变新版 worker 的哈希，不能直接用于旧 run 的 tick。因此报告能力放在身份文件之外的 `tools/`；旧发布版本本身不会因本地修改而变化。
- 任何"尚未跑过真实 Batch""未进行真实付费标注"之类的绝对断言，要么附带命令与时间戳，要么由生成器产出；
  **手写的绝对断言一律不允许存在**。

**已清理的过期断言**：`README.md` 原"当前未进行真实 Vertex 付费标注、HF Job 提交……"与
`docs/batch.md` 原"尚未配置 GCS / ADC，也没有提交真实 Batch"两处，均已改为指向 `STATUS.md`。

**验收**：`tools/report.py --check` 在 `STATUS.md` 落后于当前探测时非零退出；
`docs/` 与 `README.md` 中不存在无法溯源到命令的云端状态断言。

#### R-10 preflight 作为发布闸门

此处是尚未实现的发布闸门，不是已存在的 `tools/preflight_stage.py`；后者只检查收集前的阶段结果。

**规则**：一条命令完成：测试（沿用 `_test/<new-run-id>` 的既有约定，避免 pytest 清理既有数据）→ 计算哈希 → 清单一致性 → 文档版本表与清单一致。全绿才允许发布。

**附带修正**：`docs/validation.md:36,44,59` 的"195 passed / 99 passed / 83 passed"按**日期**记录，无法对应到任何可复现的代码状态。今后测试结果按 **release** 记录。

**验收**：preflight 非零退出时，R-06 拒绝发布。

---

### P2 — 身份重构（仅新 release；旧 run 保留原 worker，见第 4 节）

#### R-11 解耦两条管线的身份命名空间

**规则**：

- 先盘点在线专用、Batch 专用和共享依赖；是否将 batch 模块移入子包另行决定，移动目录不是前提。
- 指纹改为**显式文件列表**，不再使用 `glob('*.py')`（`runner.py:274`）与 `(root/'castle_pipeline').glob('*.py')`（`batch_pipeline.py:40`）。
- 新增回归测试：**修改 batch 模块不得改变在线指纹**（反向同理）。

**理由**：这是 `PROGRESS.md` 已识别的 "trap 3"。Batch 专用文件变化会使新代码树计算出的在线指纹变化，影响新版对旧在线检查点的复用。继续使用旧发布目录的在线 run 不受影响；旧检查点不会被删除。新版切换可能需要一次新的身份边界，不承诺自动兼容旧检查点。

**依赖边界**：Batch 当前使用 `runner.py` 中的 atomic_json、cleanup_media、clip_windows、fingerprint 等公共函数，不能简单排除 runner.py。必要时提取共享工具；共享依赖变化仍应影响两条管线。

**前置**：R-03 的仓库相对路径键必须先落地，否则 basename 冲突会静默丢文件。

**验收**：在新方案下修改一方专用实现不改变另一方身份，修改共享实现改变双方身份；原 tag / 发布目录重算得到旧哈希，旧 worker 仍能复用旧检查点。不得以改写旧 state 或漏掉真实依赖实现“兼容”。详细发布与回退边界见最新决策文档。

#### R-12 身份分层：`identity_inputs` 与 `run_parameters`

**候选设计，暂缓**：`initialize()` 目前用 `config` 全等约束初始化，tick 也没有参数修订接口。`batch_tasks.py` 关于恢复时调整 max_output_tokens 的注释尚未兑现。若以后实施分层，需要区分以下两类并记录实际配置；当前继续固定每条 run 的配置：

- `identity_inputs`（进哈希）：prompt 文本、model、fps、clip_seconds、max_dim、review 开关与上限、代码。
- `run_parameters`：media_threads、decode_slots、footer_workers、prepare_workers 等执行调优。max_output_tokens 需要单独界定阶段配置和产物来源，不能简单当成不影响产物的性能参数；本轮不实现原 run 的热更新。

**未来验收边界**：若重新启动此项，只允许明确批准可调整的参数变化；请求配置和实际产物来源可追溯。当前用户所需的“旧 audio + 新 annotation”优先采用 R-13 的新 run 方案，不要求先实现通用参数热更新。

#### R-13 跨版本迁移的决策表 + 最小可用工具

**规则**：对已钉住且在跑的 run，只允许以下三种处置，且必须写入 ledger：

| 选项 | 适用 | 代价 |
|---|---|---|
| (a) 用钉住的版本跑完 | 已知缺陷可接受 | 把缺陷显式记入 ledger，不得事后声称等同新版 |
| (b) 新 prefix + 明确 clip 范围重跑 | 缺陷不可接受、范围小 | 重新付费 |
| (c) 新 run 使用 `import-stage` 显式继承 | 能证明旧 audio 对新 run 仍兼容，只需采用新版 annotation | 需要有限导入工具、兼容性校验与来源记录；不是无条件复用 |

**仅为将来接口示意，当前不存在该命令**：`import-stage --from-state <uri> --to-prefix <gs://...> --stages audio --clips ...`。目的地必须是独立的新 run / 新 prefix。v4 run 的 audio 可显式继承到新版 run，不能把原 run 改成 audio v4、annotation v5 的混合身份。首版只考虑 audio，按需实施。

- 逐 `request_id` 继承，禁止按行号匹配（沿用 `docs/batch.md:120` 的既有规则）；
- state 中记录 `inherited_from`（源 run 的 release + code_hash + state generation）；
- 校验数据与 clip 时间范围、音频处理、audio 提示词 / 模型 / 契约等兼容性；显式确认不能代替兼容性证明。记录对象内容哈希与原始配置；
- 绝不修改源 run 的 state。

**验收**：明确选中的、成功且兼容的 audio 可在不重新调用 audio 模型的前提下进入新 prefix，来源可追溯；缺失和失败行保持未解决。不能将选中 600 clips 等同于 600 条成功 audio。中断恢复与重复执行不造成重复提交。详细范围和复杂度见最新决策文档。

#### R-14 把不可变性变成机制

**候选议题，暂缓**：保留发布只增不改的约定及脚本默认防覆盖。存储侧权限、保留或锁定措施需先核对 HF 的实际能力，不能假定已有这些策略；不新增 versions.json。本轮只记录内容哈希核验报告，不实施此项。

**验收**：对已有 release 目录的写入被存储侧或脚本侧任一环节拒绝。

#### R-15 身份文件的字节保真：禁止行尾转换（已实施）

**规则**：凡进入身份哈希的文件（`batch_pipeline.py`、`castle_pipeline/*.py`、`prompts/*.md`）在 `.gitattributes` 中标记为 `-text`，禁止任何行尾规范化；其余文件用 `* text=auto eol=lf` 保持 LF。

**理由（这不是理论风险，已实际发生）**：仓库中 `castle_pipeline/media.py` 含 **313 处 CRLF**，而其余身份文件是 LF。首版 `.gitattributes` 使用 `* text=auto eol=lf`，git 在写入索引时把 media.py 静默规范化为 LF：

```
git rev-parse HEAD:castle_pipeline/media.py   -> 4dd0b701...   (LF，错误)
worktree bytes                                 -> a7b15d39ca... (CRLF，被 v2/v3 计入)
git status                                     -> clean
```

即：提交与 tag 记录的**不是**被哈希的字节，而 `git status` 仍然报告干净。后果是一次全新 checkout 算出的 `code_hash` 与所有已钉住 run 不符，全部 `tick` 被拒——而本地开发时完全看不出来。

**验收**：`tests/test_release_manifest.py` 的 tag 一致性测试通过（它比对 `git archive` 出来的树与清单，能不依赖工作区抓出这类偏差）；`git hash-object <identity file>` 必须等于索引中的 blob。

**遗留清理**：`media.py` 的 CRLF 是历史产物且现在**是承载身份的**。把它规范化为 LF 会改变指纹，因此必须作为一次显式的新 release（未来改变核心身份的新版本）执行，不能顺手改。

---

## 4. 时序约束（重要）

**2026-09-17 19:50 UTC 更新**：原先在跑的在线 sweep 因失败率过高已被**取消**（`6aaa98345527934177eea311` = CANCELED，失败率约 36%，429 主导），改由两个 Batch Job 接手，两者都钉在 **`castle-batch-v3`**：

| Job | 角色 | state | 说明 |
|---|---|---|---|
| `castle-batch-allie-13-14-18-20-v3` | `prepare` | `gs://castle-caption-batch/castle/day1-allie-13-14-18-20-v3/state.json` | 对原 sweep 低产出来源的补充 |
| `castle-batch-bjorn15-20-v3` | `prepare` | `gs://castle-caption-batch/castle/day1-bjorn-15-20-v3/state.json` | 新范围 |

因此约束变为：

1. **不要修改 `castle-code/<已发布 release>/` 任一目录**。现有 **5 条 run** 分别钉在
   v1（1 条）、v2（2 条）、v3 身份（2 条，v3/v4 共享同一 `code_hash`）。实况见 `STATUS.md`。
2. R-11 可在独立的新 release 中开发，不必等全部旧 run 结束。新方案会改变新 worker 的哈希；旧 run 不能强行切到它，但可继续用原来匹配的代码目录 tick。禁止覆盖旧目录、改写旧 state，或把 launcher 改成所有 run 一律挂最新版。
3. P0/P1 中除 R-04 外**均不改动身份文件**，可随时实施；R-05/R-06/R-08/R-09 已按此完成，并以
   `castle-batch-v4` 发布 launcher 改动（身份哈希与 v3 相同，故 v3 上的 run 用 v3 或 v4 都能 tick）。
   R-04 改 `batch_pipeline.py`，必须作为新 release 发布，且不能指望它服务于已钉在 v1/v2/v3 身份的 run。

---

## 5. 落地顺序

2026-09-19 范围决定：用户要求先修复 launcher 的高优先级错误，并推进现有五条 run 的收集与收尾。身份命名空间重构、运行中参数修订、跨版本 import-stage 等后续方案暂缓讨论；当前采用“一条 run 固定一个兼容的代码身份”的简单约定。不要把下列未勾选项视为本轮必须全部实施的范围。

**阶段 A（已完成）**
- [x] R-01 `credentials/` + `.gitignore` + 补 `.dockerignore` + 修正 sync 排除列表
- [x] R-02 `git init` + 首个 tag `castle-batch-v3`（commit `078d904`）
- [x] R-03 `release.py` + `releases/castle-batch-v{1,2,3}.json` + `tests/test_release_manifest.py`
- [x] R-15 `.gitattributes` 字节保真（实施过程中发现并修复了 media.py 的静默行尾转换）
- [x] R-08 `ledger.py` + `ledger/ledger.jsonl`（schema、单写入口、append-only）+ `tests/test_ledger.py`；
      旧仓库外 ledger 已迁移（14 行）并补齐滞后条目
- [x] R-09 `tools/cloud_probe.py` + `tools/report.py` 生成 `STATUS.md`；两处过期断言已清理
- [ ] R-10 preflight

**阶段 B（进行中，动 launcher 与错误信息）**
- [x] R-05 `batch_jobs.py --code-volume auto` + 显式卷冲突时提交前拒绝
- [x] R-06 `release.py publish`（tag 为准、白名单暂存、拒绝覆盖、回读校验、`--verify-only`）
      并发布 `castle-batch-v4`（auxiliary-only，身份哈希与 v3 相同）。这里的回读仅为文件名清单；逐文件内容核验尚为独立工具，未接入发布路径。
- [ ] R-07 worker 启动自检
- [ ] R-04 state 记 release + 可行动错误（须并入下一个新 release）

**阶段 C（后续候选，按实际需要再决定）**
- [ ] R-11 新 release 的身份解耦；保留旧 worker，不承诺旧检查点自动迁移
- [ ] R-12 身份 / 参数分层；原 run 热更新暂不排期
- [ ] R-13 新 run 显式继承兼容成功 audio；按需实现，不绑定某一历史 run
- [ ] R-14 存储侧不可变策略

---

## 附录 A — 版本映射与实况的存放位置

**本附录不再维护"云端实况"表**（2026-09-17 修订）。原稿在这里手写了 run 的阶段与计数快照，
11 分钟后就已过期——这正是 R-09 要消除的模式。统一报告位于仓库根的 **`STATUS.md`**；它只是探测时的快照，而且当前生成器仅覆盖固定历史项目/run 清单，
由 `tools/report.py --credentials credentials/<sa-key>.json` 生成，带探测时间戳。

下面只保留**结构性事实**：release 之间的身份关系。它由 `releases/*.json` 决定，不随时间腐坏，
并可用 `_bench/verify_release_bytes.py` 逐字节复核。

| release | `code_hash` | 身份关系 | 变更内容 |
|---|---|---|---|
| `castle-batch-v1` | `aa489631` | — | Batch 初版：串行 prepare，`maxOutputTokens` 硬编码 16384 |
| `castle-batch-v2` | `1d53e2b1` | 独立身份 | `batch_pipeline.py`+`media.py`+`runner.py`：解码并行度与内存准入 |
| `castle-batch-v3` | `8c765a6f` | 独立身份 | `batch_pipeline.py`+`batch_tasks.py`：`maxOutputTokens` 16384→32768 + `--max-output-tokens` |
| `castle-batch-v4` | `8c765a6f` | **与 v3 身份相同**，仅 auxiliary `batch_jobs.py` 不同 | launcher `--code-volume auto` 与提交前版本校验（R-05）；`publish` 白名单 |
| `castle-batch-v5` | `8c765a6f` | **与 v3/v4 身份相同** | launcher 修复、提前 hourly-tick、外围阶段 JSON 镜像；23 个发布文件 |

因为 v3、v4 与 v5 共享 `code_hash`，一条 run 的 pin 只能确定"身份"，不能唯一确定 release；
`tools/report.py` 因此在同一哈希有多个 release 时列出全部候选，而不是静默挑一个。

**自查命令**（只读；说明见 `tools/README.md`）：

```powershell
python tools/report.py --credentials credentials/example-sa-key.json
python tools/preflight_stage.py                                    # 收集前体检
python _bench/verify_release_bytes.py --release castle-batch-v4    # 逐字节回读校验
python ledger.py validate                                          # ledger schema
```

**在线 sweep（`standard`）— 已取消**

Job `castle-b20260916-main-day1-allie-full` / `6aaa98345527934177eea311` = **CANCELED**（2026-09-17）。取消前：10/11 源完成，763 clips 完成、439 失败（约 36%），429 占绝对多数（1802 条 `request_failed`，395 个 clip 因 429 失败），AIMD 并发上限已降至 1。剩余来源改由上面的两条 Batch run 接手（`castle-batch-v3`）。

---

## 附录 B — 反面案例（规范要防的就是这些）

1. **连续三次挂错版本**（**已修复**，R-05）：`castle-tick-bjorn08-review`(用 v2 打 v1)、`castle-tick-bjorn10-14-annotation`(v3 打 v2)、`castle-tick-bjorn10-14-collect-v3`(v3 打 v2) 全部在 15–16 秒内 `ERROR`。日志只有 `{"ok": false, "error_type": "ValueError"}`，没有 expected/actual——这正是 R-04 与 R-05 的动机。`--code-volume auto` 与提交前冲突拒绝已实施；R-04 仍待办。
2. **锁死无法迁移**：`run 10-14` 在 token 修复之前初始化，ledger 明确记载"can never use v3 without re-preparing"，而代码里没有 `import-stage`——这是 R-13 的动机。
3. **跨管线连带失效**：把 batch 模块放进 `castle_pipeline/` 改变了在线指纹，几乎导致 day1/Allie/08 的 104 个 clip 全部重标（`PROGRESS.md` trap 3）——这是 R-11 的动机。
4. **文档与事实相反**（**已修复**，R-09）：`README.md`、`docs/batch.md` 曾声称未提交真实 Batch，而云端有 5 条 Batch run。现改为指向生成的 `STATUS.md`。
5. **行尾静默改变身份**：首版 `.gitattributes` 的 `text=auto` 把 CRLF 的 `castle_pipeline/media.py` 在索引里改成 LF，`git status` 仍报 clean；一次全新 checkout 会算出不同的 `code_hash` 并让全部 5 条 run 无法 tick——这是 R-15 的动机。
6. **发布把整个仓库树推进不可变前缀**（**已修复**，随 v4）：`release.py publish` 首版按"排除式"同步 git archive，把 85 个对象（含 `_test/`、`_bench/`、`ledger/`、`tools/`、`credentials/README.md`）上传到 `castle-batch-v4`，而清单只声明 22 个。**未泄漏任何凭据**（两个密钥文件被 `.gitignore` 挡住，不在 git 树中）。现改为按清单白名单暂存，并对"桶里出现未声明对象"报错；v4 已回退到 22 个对象并通过逐字节回读校验。

---

## 附录 C — 未决问题

1. 密钥文件的最终归属：已迁入 `credentials/` 并加忽略规则（R-01）；`credentials/README.md` 已从 v4 前缀中移除（代码桶不需要它）。但**是否轮换**仍未决定——取决于它们是否曾随 `hf buckets sync` 或 Docker build 外泄。首次 publish 事件已确认**未**上传任何密钥字节，可作为该判断的一条证据。
2. **已解决**：`versions.json` 不需要了——R-05 改为读仓库内 `releases/*.json`（单一事实来源）。
3. `castle-v1` / `castle-v2` 是否需要补清单？在线指纹包含 source 元数据，与 batch 的 `code_hash` 语义不同，是否要为在线管线单独定义 release 清单格式？
4. 在线 sweep 的 429 主导失败（36%）是否需要独立的配额与重试规范？本文件只覆盖版本与发布，不覆盖限流策略。两个新 Batch Job 的 429 情况也应纳入同一考虑。
5. `castle_pipeline/media.py` 的 CRLF 何时规范化？它是承载身份的字节，只能随一次显式新 release 处理（见 R-15 遗留清理）。
6. R-11 的新版验收如何安排？只在新 release 测试、发布，并保留旧 worker；不能把开发新身份方案等同于给旧 run 强制升级。参见最新决策文档。
7. `castle-batch-v1` 的前缀里有 44 个对象（清单声明 22）；2026-09-19 回读确认额外对象是文档、skills 和测试，并非早期记载的 `.pyc`。内容核验只报告差异，不自动清理历史前缀。
8. `castle-batch-v3` 前缀缺少 auxiliary `Dockerfile`（清单声明 22 个、实际 21 个）。它不在身份集合中，5 条 run 不受影响；是否补传以让 v3 与其清单完全一致？

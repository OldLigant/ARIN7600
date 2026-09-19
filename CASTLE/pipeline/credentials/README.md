# credentials/ — 操作者凭据存放处

还没有配置 GCP 或生成 SA JSON 的合作者，请先阅读 [从零配置 GCP 环境](../docs/gcp-setup.md)。下面说明凭据到位后的本机保管与 run 绑定。

本目录用于存放**只在操作者本机使用**的密钥与凭据。里面的内容**不是**管线运行时依赖：
worker 通过 HF secret 注入凭据；提交参数只包含环境变量名，bootstrap 从该变量读取值并写入临时私有 ADC 文件，不读取本机这个目录。

## 私有 run 绑定：多 GCP 账号的操作入口

`run-bindings.jsonl` 存在本目录，受 Git / Docker 忽略，不能上传到任何 bucket。它保存每条 Batch run 的本机凭据关联；**本 README 会随 Git 发布，因此只能写规则和虚构示例，不能列出真实绑定、SA 文件名、邮箱或账号标签。**

| 字段 | 内容 |
|---|---|
| `run_id` | 稳定的 run 标识，用于本机查找 |
| `state_uri` | 该 run 的完整 GCS state URI |
| `sa_key_file` | 相对于本 credentials 目录的 SA JSON 路径；不复制 key 内容 |
| `gcp_project` | 实际执行 Vertex Batch 的项目，与 launcher `--project` 一致 |
| `hf_output_uri` | 完整 `hf://buckets/<owner>/<bucket>/<run-prefix>` |
| `hf_owner` / `hf_bucket` / `hf_prefix` | 从输出 URI 自动解析，不另外手填 |
| `recorded_utc` | 绑定登记时间，不是历史 Job 的提交时间 |

SA key 所属项目不一定等于实际执行项目，脚本不会强制两者相等，也不会凭登记成功宣称拥有相应 IAM 权限。`hf_owner` 指输出 bucket 的个人或组织 namespace，不代表 GCP 账号，也不能替代检查当前 HF 登录身份的访问权限。

### 单一写入口

通过根目录的 `private_runs.py append` 登记，**不要让 LLM 手写/重写 JSONL**。下面都是虚构值；真实关联只保存进私有登记表：

```powershell
python private_runs.py append `
  --run-id example-run `
  --state-uri gs://example-bucket/example-run/state.json `
  --project example-execution-project `
  --sa-key team-b.json `
  --hf-output hf://buckets/ExampleOwner/castle-output/campaign/example-run

python private_runs.py validate --check-keys
python private_runs.py show --run-id example-run
```

`append` 默认只报告成功/失败和是否新增，不打印绑定；`show` 默认只确认记录存在。只有明确要在私有终端查看时才加 `show --reveal`，不要把其输出粘进公开 ledger、STATUS、提交说明或日志。脚本内可用 `lookup_binding(credentials_dir, run_id)` 取得字典并在内存里使用，无需回显。

相同登记重复执行不会新增记录；同一 run、state URI 或 HF 输出前缀的冲突绑定会拒绝，不能默默换账号/换输出。现阶段没有强制覆盖或密钥轮换接口；需要更正时先核实原记录与实际提交，再单独处理。已有文件损坏会报错并保留原文。写入使用本地排他锁；若进程被强制结束后留下 `.run-bindings.lock`，确认没有正在登记的进程并检查该文件后，才能处理这个特定残留文件，不要删除登记表。

默认目录是本目录。`--credentials-dir` 可显式指向隔离测试目录或另一个私有目录；仓库内的登记文件必须被 Git 忽略，否则拒绝写入。备份时按凭据同等私密地保存，不能依赖公开 Git 仓库恢复它。

### 操作者 / LLM 的提交与续跑顺序

1. 新 run 在提交前确认执行项目、SA 文件、state URI、HF 输出前缀，通过 `append` 登记并校验。已有 run 先查对应私有绑定；缺少绑定时从已核实的本机提交回执恢复，不能根据当前默认 key、HF 用户名或 run 名猜账号。
2. 探测和 launcher 的本地凭据参数使用这份 key；执行项目取 `gcp_project`，输出卷取 `hf_output_uri`。现有 `--credentials-file` **只影响 launcher 本机读取 GCS pin**，不会把该文件自动传给 worker。当前 `tools/cloud_probe.py` / `report.py` / `preflight_stage.py` 仍固定历史项目和 run 清单，单改 `--credentials` 不能让它们探测任意新项目；新账号下的 run 要显式使用绑定的项目、state 和 key 做针对性查询，不能拿旧项目报告代替。
3. 需要授权启动 HF Job 时，将**同一份** SA JSON 只在内存中读入 launcher 的 `--credential-secret` 所指变量（通常 `GOOGLE_ADC_JSON`），随后提交；用完移除环境变量。不要让本机查询用 key A、HF secret 却残留 key B。多账号时每次显式选择，不依赖单账号历史默认值。
4. 已有 run 保持原来的 state/project/output/pin；登记表不修改 GCS state，也不是跨项目迁移功能。原 run 换 key 或项目不是普通续跑，需先核实权限与兼容性。
5. HF 返回 Job ID 后，按原流程通过 `ledger.py append` 写**公开运行事件**，再 `ledger.py validate`。公开 ledger、note/detail、STATUS 和云端输出中均不加入 SA 文件路径、邮箱、账号标签或这份私有绑定；不要给 ledger 增加账号字段。

登记不是运行授权，也不会创建 HF Job / schedule、调用 Vertex 或证明模型权限。`private_runs.py` 是本地运维工具，不在 release 文件白名单中；新增它不要求改动 v5 或核心哈希。

## 凭据种类

- 在线管线可能仍使用本机 secrets 文件里的 `GOOGLE_API_KEY`。
- Batch 使用 SA JSON；本机探测通过工具的凭据参数读取，worker 通过 HF secret 注入。

部分历史 `_bench/*.py` 仍默认使用仓库外的 `D:\QLD\out\castle-runs\b20260917-bjorn08\secrets.env`；正式 tools 和本地 launcher 使用本目录。按具体工具的 --help 指定凭据路径，不要假定所有工具都用同一个参数名或默认路径。

## 硬规则

1. **绝不提交凭据或绑定表**。`.gitignore` 排除 `credentials/*`，仅通过 `!credentials/README.md` 放行这份通用说明；`.dockerignore` 排除整个 credentials 目录，包括 README。
2. **绝不同步到任何 bucket**。`hf buckets sync` 的排除参数必须包含 `credentials/*`；现有 `*.secrets` / `.env*` 模式**不覆盖本目录**，这是引入本目录后新增的风险点。见 `README.md` 与 `skills/castle-caption-jobs/SKILL.md` 中更新后的排除列表。
3. **密钥内容绝不打印、绝不写入 argv、绝不出现在日志或代码 bucket 中**。账号与 run 的关联也只保存在本地私有登记表，不能写入受版本控制的文档。
4. **只在内存中加载**。PowerShell 示例（引用而不回显内容）：

   ```powershell
   $env:GOOGLE_ADC_JSON = Get-Content -LiteralPath 'credentials\team-b.json' -Raw
   # 用完即删
   Remove-Item Env:GOOGLE_ADC_JSON
   ```

## 变更记录

历史上一个在线 secrets 文件和一个 SA JSON 原先直接放在仓库根目录。
`.dockerignore` 的 `*.secrets` 覆盖了前者，但**未覆盖**后者——即服务账号密钥曾处于 Docker
构建上下文之内，且在建立 `.gitignore` 之前会被 `git add .` 收录。迁移到本目录并补齐忽略规则
即为修复（见 `docs/release-process.md` R-01）。

## 审计：本目录曾几乎被发布进代码桶

2026-09-17 首次执行 `release.py publish`（当时的实现按"排除式"同步整个 git archive 树）时，
仓库全树 85 个对象被上传到 `hf://buckets/Ligant/castle-code/castle-batch-v4`，其中包含本目录。
**实际被上传的是 `credentials/README.md` 这一个非敏感占位文件**；两个凭据文件因 `.gitignore`
而不在 git 树中，因此没有在该次发布中进入代码 bucket。worker 按授权通过 HF secret 单独获取凭据，这是正常运行通道，不能与发布泄漏混为一谈。事件已记入 `ledger/ledger.jsonl`。

修复方式：`publish` 改为**白名单**——只从 git tag 中取出清单声明的文件（`identity_files` +
`auxiliary_files` + `prompt_files`）后再上传，并在回读时对"桶里出现清单之外的对象"报错。
这也说明本 README 的定位需要澄清：

- 它被 `.gitignore` 白名单放行，是为了让"凭据该放哪里、为什么"这条知识随仓库版本化；
- 但**代码桶不需要它**，且当前清单不声明它，所以它不应该出现在任何 release 目录里。
  若将来确实需要随发布分发说明，必须显式加进 `release.py` 的 `AUXILIARY` 列表。

**未决**：这两个凭据是否需要轮换，取决于它们是否曾随 `hf buckets sync` 或 Docker 构建外泄。
在该判断完成前，不要假设它们仍然私密。见 `docs/release-process.md` 附录 C 第 1 条。

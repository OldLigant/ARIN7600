# CASTLE：给代码与运维 agent 的入口

## 先读什么

- 先读 [README.md](README.md) 和 `git status --short`，辨认当前任务与已有修改；不要提交、覆盖或清理其他任务的文件。
- 配置环境：[GCP](docs/gcp-setup.md)、[Hugging Face](docs/huggingface-setup.md)。
- 准备、提交、检查或续跑：读 [仓库 CASTLE skill](skills/castle-caption-jobs/SKILL.md)、[Batch 说明](docs/batch.md)、[凭据与私有绑定](credentials/README.md)。已安装的个人 skill 可能较旧，不能用其旧路径或旧状态覆盖本仓库事实。
- 修改/发布代码：读 [发布规范](docs/release-process.md) 和 [版本演进决定](docs/version-evolution-decisions.md)。后者的候选功能不是待自动执行的任务。
- 文档索引见 [docs/README.md](docs/README.md)。操作参数以当前代码和 `--help` 为准；历史报告中的运行授权不延续为新操作授权，以当前对话为准。

## 运行与状态

- 在线入口：`run_pipeline.py` / `jobs.py`；Batch 入口：`batch_pipeline.py` / `batch_jobs.py`；v5 外围 tick：`batch_tick.py`。
- GCS `state.json` 是 Batch 续跑依据，HF 输出是查看副本；HF Job COMPLETED 不等于整条 run 完成。
- `tick` 会收集并可能立刻提交下一阶段，是有副作用的操作。只查状态用只读接口；文档维护、代码检查或登录成功不授权付费调用、创建 schedule 或扩大标注范围。
- 新 run、tick、重试、取消和 schedule 操作遵循用户已授权范围，已有明确授权不重复询问；范围不足时完成可做的本地准备。
- 一条 run 从 prepare 到 final 保持同一账号绑定、执行项目/location、state 和输出前缀。跨账号分配不同 run；不自行迁移原 run 或重写其配置/哈希。
- 只有显式版本的 hourly-tick 允许尚无 state；`auto` 与手动 tick 要求已有 state。submit 与 schedule 创建是两条命令。终态需操作者暂停 schedule。
- 同一 run/state/HF 输出只有一个写入 worker；单个 schedule 的 `--no-concurrency` 不能阻止其他 schedule 或手动 tick 冲突。
- `STATUS.md` 是生成的时间戳快照，禁止手工修正计数或把它当实时状态。`tools/cloud_probe.py`、`report.py`、`preflight_stage.py` 仍固定历史项目/run 清单，改凭据不等于改查询目标。

## 凭据与记账

- 不遍历或打印密钥内容；不把密钥写入 argv、日志、Git、Docker 上下文或 bucket。
- 新 run 通过 `private_runs.py append` 登记，已有 run 用 `lookup_binding` 在内存读取；校验用 `validate --check-keys`。禁止手写 `credentials/run-bindings.jsonl`，缺少绑定时从已核实回执恢复或询问，不能猜默认账号。
- 私有绑定、真实 SA 文件名/邮箱及账号标签仅留本机 `credentials/`，不写公开文档、ledger、STATUS 或云端产物。`credentials/README.md` 是受 Git 跟踪的公共说明，不能填真实关联。
- `--credentials-file` 只管本机 launcher 查询；HF worker 的 secret 必须另外从同一份 JSON 加载，用后清理本机环境变量。HF token 与 GCP ADC 是两种凭据。
- 公开运行事件只经 `ledger.py append` 写入 `ledger/ledger.jsonl`，随后 `ledger.py validate`；不手改旧行，不追加账号关联。

## 版本与发布

- 核心 Batch 身份覆盖 `batch_pipeline.py` 与 `castle_pipeline/*.py`；在线身份也覆盖后者。改 Batch 模块可能改变新代码树的在线指纹；旧发布目录不受本地编辑影响。
- v3/v4/v5 核心同哈希，v1/v2 不同；兼容性比较 code_hash，不能只比较 release 名。提示词和请求参数已经保存在 state，外围升级不改变旧 run 的 token cap。
- `batch_tick.py` / launcher 等辅助文件不在核心哈希内，但属于发布物；修改并上传也必须新 release。禁止覆盖旧前缀、移动旧 tag、改 manifest 来伪装旧字节。
- manifest 由 `release.py manifest` 生成，发布使用对应 tag 和白名单，不执行全仓库 `hf buckets sync`。`publish --verify-only` 只做本地检查，不证明云端字节一致。
- 文档、本机运维工具不在运行清单中时不要求新 release；不要为纯文档更新上传代码 bucket。
- 不顺手改身份文件行尾。新 run 继承旧 audio、在线/Batch 身份解耦和参数热更新尚未实现。

## 本地验证与文件安全

- 本机已有 Python 为 `./_test/runtime/Scripts/python.exe`（见下方命令）；其他机器按 requirements 配置自己的虚拟环境，不假定 PATH 中的 Python 具备依赖。
- 测试、smoke、下载回读和诊断报告必须显式使用全新、清楚命名的 `_test/<unique-id>/` 等隔离目录，不使用默认/正式输出。pytest 会处理 basetemp，不能复用已有数据目录。
- 需要运行测试时，例如：`./_test/runtime/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=_test/<new-unique-id>`。把占位符换成尚不存在的目录。仅文档修改检查链接、事实与 diff；不要因此启动真实云端 Job。
- 删除前先列出并确认目标只含打算删除的文件；使用最窄路径，不递归清理可能包含用户数据的目录。
- 提交前检查 diff、忽略规则与显式文件列表；不要 `git add .` 顺带收录其他任务改动、临时脚本或凭据。声明完成时报告实际验证及其边界。

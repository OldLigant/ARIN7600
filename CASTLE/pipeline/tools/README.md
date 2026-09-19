# tools/ — 只读运维工具

这些脚本**不在** `code_hash()` 覆盖的身份文件里，修改它们不会改变核心身份；仍须核对查询目标、报告内容和实际副作用。
它们对云端只读：不提交、不取消、不收集、不写 GCS state，也不打印模型输出或凭据内容；本地报告文件仍会按下表写出。

| 脚本 | 作用 | 会写什么 |
|---|---|---|
| `cloud_probe.py` | 读 GCS state + 实时 Vertex 状态 + 各 batch 的实测耗时，返回结构化 JSON | `--out` 指定的文件（可选） |
| `report.py` | 把探测结果 + `ledger/ledger.jsonl` 渲染成 `STATUS.md` | `STATUS.md`（或 `--out`） |
| `preflight_stage.py` | 在 tick **收集之前**体检当前阶段的输出：行数、JSON 可解析率、schema 合规率 | 无 |
| `run_status.py` | 每条 run 的 state、实时 Vertex 阶段状态、GCS 各阶段对象数；run 列表取自私有绑定 | `--out` 指定的文件（可选） |
| `mirror_audit.py` | 逐对象比对 GCS 的 `final/`、`results/<stage>/` 与 HF 输出前缀：缺哪些、多哪些、有没有意外的媒体文件 | `--out` 指定的文件（可选）；全部一致才退出 0 |

## 常用命令

```powershell
# 云端实况：已发布 release、5 条 run 的 pin/阶段/计数、各阶段耗时
python tools/report.py

# 只探测不渲染（机器可读）
python tools/cloud_probe.py --out probe.json

# STATUS.md 是否落后于当前云端状态（落后则非零退出）
python tools/report.py --check

# 收集前体检：坏行会在 tick 时变成永久 failure，先看清有多少
python tools/preflight_stage.py
python tools/preflight_stage.py --run bjorn-10-14

# 每条 run 现在在哪：state 阶段、实时 Vertex 状态、GCS 各阶段对象数
python tools/run_status.py
python tools/run_status.py --run day1-bjorn-15-20-v3

# tick 之后核对镜像：GCS 有而 HF 桶没有的对象会被列出来（非零退出）
python tools/mirror_audit.py
python tools/mirror_audit.py --run day1-allie-13-14-18-20-v3
```

`run_status.py` 与 `mirror_audit.py` 与前三个不同：它们的 run 列表、项目、state URI 和输出前缀全部来自
`credentials/run-bindings.jsonl`（经 `private_runs.read_bindings`），因此**每条 run 用自己绑定的账号读取**，
新增 run 后无需改代码。它们不打印凭据路径。前三个工具（`cloud_probe.py`、`report.py`、`preflight_stage.py`）
仍使用代码中固定的历史项目和 `RUNS` 清单；`--credentials` 仅切换凭据，不切换项目或 run 范围。

工具保留单账号时期的本机默认凭据；多账号运行时不能把这个默认值当作 run 的账号依据。先按 [私有 run 绑定说明](../credentials/README.md) 查 `credentials/run-bindings.jsonl`，再用具体工具的 `--credentials` 覆盖为该 run 的 key。登记表不会自动改变这些工具的认证行为。不要用同一个账号的默认探测结果推断其他账号下的任务是否存在，也不要把私有关联加入 `report.py` 的公开输出。

## 关于 `code_hash` 的一个约束

`report.py` 的落点曾经被建议为 `batch_pipeline.py report`——**不要那样做**。
`batch_pipeline.py` 是身份文件，给它加子命令会改变新代码树的 code_hash；新版不能直接替代旧 run 的 worker。旧 run 继续挂载原发布目录则不受影响。把报告能力放在 tools/ 可避免不必要的身份变化，见 `docs/release-process.md` R-09。

## 与 `_bench/` 的分工

- `tools/`：**稳定、可重复、有接口**的运维动作，属于仓库正式内容。
- `_bench/`：一次性测量与诊断脚本（解码性能基线、发布字节回读、token 上限探测等）。
  它们记录"当时怎么测出来的"，不保证接口稳定。需要长期使用的能力应从 `_bench/` 提升到这里。

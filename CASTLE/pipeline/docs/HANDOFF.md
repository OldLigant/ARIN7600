# CASTLE 接手摘要

更新日期：2026-09-20。这是文档与实现的交接记录，不是实时云端状态，也不自动授权后续运行。

## 接手顺序

先读 [AGENTS.md](../AGENTS.md) 和 [文档导航](README.md)，再看 `git status --short` / 最近提交。运行接口见 [Batch 说明](batch.md)，凭据选择见 [私有绑定](../credentials/README.md)。当前对话的范围与授权优先于历史记录。

## 已落实

- 本地 Git、release tag 和 `releases/*.json` 维护版本；HF 代码 bucket 保存完整不可覆盖的发布快照。
- v5 已发布：提前创建 hourly-tick，以及在原 tick 退出后镜像已收集的阶段 JSON。v3/v4/v5 核心哈希相同；v1/v2 不兼容。发布文件、提交与字节核验见 [v5 记录](batch-v5.md)。已有 schedule 不自动更新。
- `private_runs.py` 管理忽略的本机 `credentials/run-bindings.jsonl`，此前已根据核实回执登记五条历史 Batch run。默认输出不打印关联；本地校验不证明云端权限。
- 最新一次代码全套离线验证记录为 283 passed / 1 skipped，隔离目录 `_test/private-runs-20260920-04-full`；跳过项是 Windows 无权限创建符号链接。该记录对应私有绑定实现，不等于每次后续改动都经过测试。
- 新增 GCP/HF 合作者入门指南和仓库 AGENTS 入口；纯文档整理不改变 v5 或核心身份，不要求发布新 release。

## 必须保留的边界

- GCS state 是 Batch 状态依据；HF 阶段结果是查看副本。tick 可立即提交下一阶段，不能当只读探测。
- submit 与 schedule 创建仍是两条命令；无 state 的定时检查会退出但仍有启动成本。终态 schedule 需操作者暂停。
- 同一 run 保持同一私有账号绑定和执行项目/location；HF worker secret 与本机查询必须来自同一 key。私有关联不写公共文档或 ledger。
- 当前探测工具固定历史项目与 run 清单；launcher 的 auto 固定团队代码 bucket。多项目/自有 bucket 不会自动路由。
- 多个 worker 不得并发写同一 state/输出前缀。真实云端运行、重试和范围扩张依当前用户授权；本次文档工作没有启动或推进任何任务。

## 仍然延期的设计

[版本演进决定](version-evolution-decisions.md) 记录新 run 显式继承旧 audio、在线/Batch 身份解耦和只读内容核验的边界。前两项未实现；内容回读已有独立工具但未接入正式发布流程。实际 v5 没有改变 annotation 参数，也没有实现跨 run 导入。

## 历史与状态证据

- [STATUS.md](../STATUS.md)：生成快照，先看探测时间；不要手工改计数。当前文档整理未重新查询云端。
- [2026-09-19 launcher 与收集记录](launcher-collection-2026-09-19.md)：此前授权的一轮运行证据。
- [首次版本审计](project-version-audit-2026-09-19.md)：当时的发现，部分问题后来已修复。
- `ledger/ledger.jsonl`：公开事件单写入口为 `ledger.py append`，旧仓库外台账只作历史证据。

本机既有 Python 环境在 `_test/runtime/Scripts/python.exe`；新机器按依赖文件安装。测试始终新建隔离目录，删除前检查具体目标。仓库内 CASTLE skill 为维护副本，个人安装副本可能过期，本次文档整理没有安装或覆盖用户目录中的 skill。

# 文档导航

本页区分当前操作说明、设计决定与历史证据。云端进度以实际探测时间为准，不从旧对话摘要推断现在仍在运行或已经完成。

| 需要做什么 | 阅读入口 |
|---|---|
| 了解代码与在线模式 | [项目 README](../README.md) |
| 合作者第一次配置云端 | [GCP 配置](gcp-setup.md) → [Hugging Face 准备](huggingface-setup.md) |
| agent 接手仓库 | [AGENTS.md](../AGENTS.md) → [CASTLE skill](../skills/castle-caption-jobs/SKILL.md) |
| Batch 提交、tick、schedule、结果镜像 | [Batch 使用说明](batch.md) |
| 私密管理多个账号与 run | [credentials 说明](../credentials/README.md) |
| 查询状态和收集前检查 | [tools 说明](../tools/README.md)；[STATUS 快照](../STATUS.md) |
| 版本、tag、manifest 与发布 | [发布规范](release-process.md)；[v5 发布记录](batch-v5.md) |
| 讨论跨版本复用或身份解耦 | [版本演进决定](version-evolution-decisions.md)，尚未实现的内容按需讨论 |
| 快速恢复上下文 | [交接摘要](HANDOFF.md)，不作为云端实时状态 |

历史证据保留原日期、实验条件和计数，不能当成当前待办或新的运行授权：

- [初期实现计划](implementation-plan.md)、[Batch 实现计划](batch-plan.md)。
- [2026-09-16/17 验证记录](validation.md)、[早期 Job 诊断](job-6aa9bfc3-diagnosis.md)。
- [2026-09-19 初次版本审计](project-version-audit-2026-09-19.md)、[随后 launcher 修复和收集记录](launcher-collection-2026-09-19.md)。

更新接口时同步相应操作说明；实验结果另写时间和验证对象。不要将真实凭据、SA 文件名/邮箱或私有账号对应表填入任何公共文档。

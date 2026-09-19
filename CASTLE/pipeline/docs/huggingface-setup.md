# 合作者指南：准备 Hugging Face

核对日期：2026-09-20。本项目用 HF Jobs 做视频预处理、提交与检查，用 HF Buckets 保存代码发布物和结果副本；模型计算仍在 GCP Vertex 上执行。GCP 配置见 [GCP 指南](gcp-setup.md)。

HF 的准备通常比 GCP 简单：人完成注册/登录、计费和权限确认后，agent 可以通过 CLI 完成大部分操作，不需要逐项创建 GCP 那样的服务代理 IAM 授权。**能登录不等于有权操作团队资源，也不等于可以启动付费任务。**

## 1. 人先确认账号与资源归属

1. 在 [Hugging Face](https://huggingface.co/) 注册并登录账号。
2. 确认使用个人 namespace 还是团队组织 namespace；使用组织资源时，请管理员给予相应访问权限。
3. 在账号或组织的 Billing 页面确认 Jobs 可用余额。当前官方说明 Jobs 面向有正数计算余额的用户和组织开放，不应把购买 PRO 写成唯一前提。HF Job 的机器运行费用与 GCP 的模型费用分开。[Jobs 计费说明](https://huggingface.co/docs/hub/jobs-pricing)
4. 与操作者确认代码 bucket、结果 bucket 的 owner、名称和公开/私有设置。不要因为要共享给合作者就直接把整个 bucket 公开。

如果合作者只提供 GCP 项目，而任务继续由团队现有 HF 账号提交，他不必另建 HF 环境；只需按 GCP 指南私密交接凭据和项目信息。

## 2. 安装 CLI 并登录

Windows PowerShell 使用官方安装器：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

macOS / Linux：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

安装后重新打开终端（若当前终端还找不到 `hf`），执行：

```text
hf version
hf auth login
hf auth whoami
```

按 CLI 提示完成浏览器登录或交互式 token 登录；若需要 token，从 [Token 设置](https://huggingface.co/settings/tokens) 创建。不要把 token 粘进聊天、写进提交命令参数或要求 agent 打印 `hf auth token`。所选凭据需要允许相应 Jobs 操作、读取代码/数据和写入目标 bucket；组织策略可能额外限制权限。[官方 CLI 安装和登录说明](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

`whoami` 用来核对登录身份，不是完整权限测试。若设置了 `HF_TOKEN` 环境变量，也要确认没有残留的其他账号凭据干扰操作，不要回显变量值。

## 3. 给 agent 提供 CLI skill

当前官方独立安装器会一并安装 `hf-cli` skill。若通过其他方式安装 CLI、跳过了 skill，或 agent 找不到它，可执行：

```text
hf skills add --global
```

只想装在当前项目时使用 `hf skills add`。skill 从本机 CLI 版本生成；升级 CLI 后按 `hf skills --help` 更新。agent 还应查看具体子命令的 `--help`，不能用旧版本记忆猜参数。[官方 agent 指南](https://huggingface.co/docs/hub/agents-cli)

`hf-cli` 解释通用命令；本仓库的 [AGENTS.md](../AGENTS.md) 和 [CASTLE skill](../skills/castle-caption-jobs/SKILL.md) 解释项目约束，两者都要读。skill 不包含账号凭据，也不自动授予运行授权。

## 4. 确认或创建 bucket

如果团队已有 bucket，直接核对访问即可，不要重新创建或改名。当前团队使用 `Ligant/castle-code` 和 `Ligant/castle-output`；新合作者不能因为知道地址就假定有写权限。

新建独立环境时，确认 namespace 与可见性后，agent 可以用 CLI 创建。例如以下是**需要替换的虚构名称**，执行后会创建资源：

```text
hf buckets create ExampleOwner/castle-code --private
hf buckets create ExampleOwner/castle-output --private
```

可用以下只读命令检查 bucket 和 Jobs 接口（替换 owner）：

```text
hf buckets info ExampleOwner/castle-code
hf buckets info ExampleOwner/castle-output
hf jobs ps
hf jobs scheduled --help
```

这些检查不启动计算，也不证明所有写入权限或工作流程均可用。bucket 创建、读取、同步和可见性均有 CLI 接口；存储本身另有计费规则。[官方 Buckets 指南](https://huggingface.co/docs/hub/storage-buckets)

## 5. 按本项目的发布和运行方式交接

| 资源 | 项目内的使用方式 |
|---|---|
| 代码 bucket | 每个 release 一个完整快照前缀，worker 只读挂载；通过 `release.py` 发布，不直接同步整个仓库 |
| 结果 bucket | 每条 run 一个独立输出前缀，worker 可写挂载；v5 可镜像已收集的阶段 JSON |
| HF Job / schedule | 通过 `batch_jobs.py` 生成命令，核对后在用户授权范围内执行 |
| GCP SA JSON | 通过 HF Job secret 注入，不上传到任意 bucket |

HF bucket 是可变对象存储，**没有 Git 历史**。本项目的“发布后不覆盖”由 Git tag、manifest 和操作规范实现，不能把它误认为 bucket 自动锁定了文件。[Bucket 与 Git 仓库的区别](https://huggingface.co/docs/hub/storage-buckets)

新 HF namespace 下，显式核对发布工具的 `--code-bucket`、launcher 的 `--code-volume` / `--output-volume`，不要沿用团队默认地址。当前 launcher 的 `auto` 仍从内置团队代码 bucket 构造路径；自有代码 bucket 必须显式给出已登记 release 的完整 `--code-volume`。HF Jobs 的提交/计费 namespace 与 bucket owner 也不是同一个参数；当前 `batch_jobs.py` 没有组织 namespace 选项，若要向组织提交，需要操作者核对生成的 HF 命令及当前 CLI 接口，不能只改输出 bucket 就宣称已经切换计费归属。

新 run 的 GCP 凭据和 HF 输出对应关系只通过本机 [私有绑定入口](../credentials/README.md) 登记。`--credentials-file` 选择本地 SA 文件，`GOOGLE_ADC_JSON` secret 要加载同一文件；HF 登录凭据与 GCP SA 凭据各管各的服务，不能互相替代。

实际启动与 schedule 停止条件见 [Batch 使用说明](batch.md)。提前创建 hourly-tick 仍需单独执行创建 schedule 的命令；没有 state 时会短暂启动后退出，并不是零成本等待。终态后的 schedule 由操作者暂停。

# 合作者指南：从零配置 CASTLE Batch 的 GCP 环境

适用于本仓库的 Vertex Batch 管线；核对日期：2026-09-20。本文先完成 Google Cloud 配置，再交接给管线操作者，不会要求你先运行一次付费标注。示例名称均为虚构值，不要原样当作团队的真实配置。

需要准备：一个开通计费的 Google Cloud 项目、一个我们创建的服务账号、一只 Cloud Storage 桶，以及该服务账号的 JSON 密钥。**本 Batch 管线使用 SA JSON / ADC，不需要另外创建 Vertex API key。** 密钥证明“你是谁”，IAM 角色决定“你能做什么”；下载了密钥不等于已经有模型和存储权限。

## 1. 先认清这几个东西

| 名称 | 是什么、用来做什么 |
|---|---|
| Google 登录账号 | 你打开控制台时使用的人类账号，负责配置项目与授权 |
| GCP 项目 | API、任务和计费的归属；项目显示名称、项目 ID、项目编号是不同字段 |
| 我们创建的服务账号（SA） | 程序使用的身份，例如 `castle-batch-worker@PROJECT_ID.iam.gserviceaccount.com`；HF worker 用它提交/查询任务、上传媒体和维护状态 |
| Vertex 服务代理（Service Agent） | Google 管理的另一个身份，替 Vertex 后台读取 GCS 输入、写出预测；不用给它下载密钥 |
| GCS bucket | Google Cloud Storage 桶，地址为 `gs://...`，存放媒体、请求、预测和 `state.json` |
| HF bucket | Hugging Face 存储，地址为 `hf://buckets/...`，存放发布代码或结果副本；不能代替 GCS |

推荐第一次配置时把执行项目、我们创建的 SA 和 GCS 桶放在同一个项目，减少跨项目授权问题。一个项目可以承载多个 run，每个 run 使用独立路径前缀。

## 2. 创建或选择项目，关联计费

1. 打开 [Google Cloud Console](https://console.cloud.google.com/)，登录你的 Google 账号。
2. 点击顶部项目选择器，选择已有项目，或点击“新建项目 / New project”。创建后，再确认顶部当前选中的是它。
3. 在项目首页或“项目设置”中找到 **Project ID** 和 **Project number**，私下记录。后续命令的 `--project` 使用项目 ID；服务代理邮箱使用纯数字的项目编号。
4. 打开“结算 / Billing”，确认这个项目关联到有效的结算账号。只有创建项目、没有开通项目计费还不够。[官方计费说明](https://docs.cloud.google.com/billing/docs/how-to/modify-project)

配置者需要有创建 SA、启用 API、创建桶、修改 IAM 和创建密钥的权限。如果页面按钮不可用或显示无权操作，让项目/组织管理员完成相应步骤；这些管理权限不需要全部授予后面运行任务的 SA。

## 3. 启用 API

在“API 和服务 → 库 / APIs & Services → Library”中，检查并启用：

- **Vertex AI / Agent Platform API**，服务标识为 `aiplatform.googleapis.com`。
- **Cloud Storage API**，服务标识为 `storage.googleapis.com`。

界面和文档中的 Vertex AI 可能显示为 Agent Platform，以服务标识为准。不要把其他 Agent 产品的 API 当作本管线需要的 API。Vertex 的基础准备包括项目、计费和 API 启用。[官方入门说明](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart)

## 4. 创建供管线使用的 SA，并授予项目角色

1. 打开“IAM 和管理 → 服务账号 / IAM & Admin → Service Accounts”。
2. 点击“创建服务账号 / Create service account”，名称可填 `castle-batch-worker`，记下生成的服务账号邮箱。
3. 在“授予此服务账号对项目的访问权限”步骤，选择 **Vertex AI User / Agent Platform User**，角色 ID 为 **`roles/aiplatform.user`**。
4. “授予用户对此服务账号的访问权限”这一步，本指南的 JSON key 方式不需要额外填写；完成创建即可。

如果创建时跳过了第 3 步，可以去“IAM 和管理 → IAM → 授予访问权限”，把 SA 邮箱作为主体，添加该角色。角色应授予在**实际执行 Batch 的项目**上，不是在桶页面。它支持调用方使用 Vertex 的相关操作；仅有桶权限不能提交 Batch。[官方 Vertex IAM 角色说明](https://docs.cloud.google.com/iam/docs/roles-permissions/aiplatform)

不要给这个普通 SA 授予 `roles/aiplatform.serviceAgent`：那是 Google 服务代理的专用角色。

## 5. 创建 GCS 桶

1. 打开“Cloud Storage → Buckets”，点击“创建 / Create”。
2. 填写一个全球唯一的桶名，例如 `example-castle-batch-unique`；桶名不能直接照抄示例。
3. 选择适合团队数据驻留要求和模型运行位置的存储位置。管线的 Vertex `location=global` 是 API 端点选择，**不是 GCS 桶的位置名称**；不确定地区时先和管线操作者确认。
4. 本项目可选择 Standard 存储类别、统一桶级访问（Uniform bucket-level access），保持禁止公开访问。采用 Google 管理的默认加密；本指南不配置 CMEK。
5. 其余数据保护设置由团队确认后创建。不要设置会在 run 完成前自动删除媒体或状态的生命周期规则，也不要给需要反复更新的 `state.json` 配置阻止覆盖的保留限制。

创建后桶地址就是 `gs://你的桶名`；不必提前创建 `castle/` 等文件夹，管线写入对象时会使用路径前缀。[官方建桶步骤](https://docs.cloud.google.com/storage/docs/creating-buckets)

## 6. 给两个身份分别授予桶权限

进入刚创建的桶，打开“权限 / Permissions → 授予访问权限 / Grant access”。下表中的角色均在**这只桶**上授予。

| 授权给谁 | 角色显示名称 | 角色 ID | 原因 |
|---|---|---|---|
| 第 4 步创建的 SA 邮箱 | Storage Object User | `roles/storage.objectUser` | 管线读写和列举对象，并覆盖更新状态 |
| Google 的 Vertex 服务代理邮箱 | Storage Object Viewer | `roles/storage.objectViewer` | Vertex 后台读取输入 JSONL、音频和图片 |
| 同一个 Vertex 服务代理邮箱 | Storage Object Creator | `roles/storage.objectCreator` | Vertex 后台创建预测输出 |

注意名称是 **Storage Object User**。我们自己的 SA 不能只拿 Object Creator：它还需要读取对象、列举结果和更新已有状态。Viewer 和 Creator 的组合则用于上表中的 Vertex 后台输入/输出职责。[官方 Storage 角色说明](https://docs.cloud.google.com/iam/docs/roles-permissions/storage)

### 怎么找到 Vertex 服务代理？

它的邮箱格式为：

```text
service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com
```

把 `PROJECT_NUMBER` 替换成**执行 Vertex Batch 的项目编号**，不是项目 ID。它也不是 `...-compute@developer.gserviceaccount.com`，不是我们刚创建的 `castle-batch-worker@...`。

在项目 IAM 页面勾选“包含 Google 提供的角色授权 / Include Google-provided role grants”，查找这个服务代理；其项目角色通常由 Google 配置为 `roles/aiplatform.serviceAgent`。同项目内它可能已经通过继承权限获得所需的存储访问；先检查有效权限，不必为了重复授权而删改 Google 的默认角色。跨项目桶尤其需要在桶所属项目补齐上述权限。[官方服务代理说明](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/access-control)

如果 API 已启用但服务代理还没生成，管理员可打开控制台右上角 **Cloud Shell**，将下面的项目 ID 替换后执行：

```bash
gcloud beta services identity create --service=aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

这会生成服务身份，不提交标注任务。然后回到 IAM/桶权限页面重新检查；若权限不足，由管理员执行。[官方命令说明](https://docs.cloud.google.com/sdk/gcloud/reference/beta/services/identity/create)

Google 的 Batch 文档明确要求服务代理能读取 GCS 输入。文档中“不支持自定义执行服务账号”说的是 Vertex 后台执行身份，不妨碍我们的程序用 SA JSON 认证并提交任务。[官方 Batch GCS 说明](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage)

## 7. 下载我们自己的 SA JSON 密钥

1. 返回“IAM 和管理 → 服务账号”，点击**第 4 步创建的 SA**。
2. 打开“密钥 / Keys → 添加密钥 / Add key → 创建新密钥 / Create new key”。
3. 类型选择 **JSON**，点击创建，浏览器会下载文件。
4. 将文件放到本仓库的 `credentials/` 目录，可以改成方便区分的本机文件名，例如 `credentials/team-b.json`。不要修改 JSON 内容。

不要选择 Google 管理的服务代理下载密钥。若组织策略禁止创建 SA key，需要管理员确认可用的认证方案，不要自行绕过组织策略；当前 HF launcher 不会自动配置工作负载身份联合。[官方密钥创建步骤与限制](https://docs.cloud.google.com/iam/docs/keys-create-delete)

`credentials/` 中的密钥受 Git 忽略，也不应进入 Docker 构建、代码桶、结果桶、聊天或公开文档。合作者与操作者需要传递密钥时使用双方认可的私密渠道。本目录的 `README.md` 本身会公开，因此也不能在那里填写真实账号对应表。

## 8. 交接给管线操作者

HF 账号与 CLI 的准备见 [Hugging Face 指南](huggingface-setup.md)；仅提供 GCP 资源的合作者不必另建 HF 环境。

配置完成后，通过私密渠道交接：执行项目 ID、项目编号、GCS 桶名及位置、SA JSON 文件，以及已配置的权限。操作者再确认具体模型与 Vertex location，并安排 run 范围和 HF 输出路径；GCP 配置者不需要自行创建 HF bucket 或发布代码。

每条新 run 在启动前，通过 `private_runs.py append` 把 run、SA 文件、执行项目、GCS state URI 和 HF 输出前缀记录在本机私有 `credentials/run-bindings.jsonl`。具体命令见 [私有绑定与凭据操作说明](../credentials/README.md)。不能把这些账号关联写入公开 ledger、STATUS 或云端结果。

**账号分配粒度是整条 run。** prepare → audio → annotation → review → final 使用同一绑定。多个账号分别承接不同 run；不要把一个 run 的不同阶段分配给不同项目。已有 run 的上下文依赖原来的 GCS state 和对象访问权限。

本地 `--credentials-file` 只选择 launcher 本机查询凭据。HF worker 还必须通过 `--credential-secret` 对应的环境变量注入**同一份 JSON**，不会自动取得本地文件。例如，在仓库根目录的 PowerShell 中：

```powershell
$env:GOOGLE_ADC_JSON = Get-Content -LiteralPath 'credentials\team-b.json' -Raw
# 操作者此时执行已经核对并获得运行授权的 launcher 命令。
# 命令结束后清理本机变量，不要回显其内容：
Remove-Item Env:GOOGLE_ADC_JSON
```

不要只加载变量就认为已经提交任务。完整启动流程和参数见 [Batch 使用说明](batch.md)。

## 9. 配置完成后如何检查

先在控制台确认：当前项目正确、计费有效、API 已启用、SA 拥有项目级 Vertex User 角色、两个身份均能按上表访问桶、JSON 文件已私密保存。`private_runs.py validate --check-keys` 只检查本地登记和密钥结构，**不证明云端权限、模型可用性或 Batch 能成功执行**。

需要端到端验收时，由操作者另外安排获授权的少量片段测试，并显式使用全新的 `_test/...` 本地目录、独立 GCS run prefix 和 HF 输出前缀，不写正式输出。不要用真实 run 的 `tick` 当只读权限测试：它可能收集结果并提交下一阶段。

| 现象 | 优先检查 |
|---|---|
| 创建 Batch 返回 403 | 调用 SA 是否选对；执行项目是否正确；项目级 Vertex User、API 和计费是否齐全 |
| 本机能查询，HF worker 失败 | `--credentials-file` 与 `GOOGLE_ADC_JSON` 是否来自同一份 JSON；worker 是否收到 secret |
| 管线上传成功，Vertex 报输入读取或输出写入失败 | 是否漏给 Google 服务代理授权；服务代理是否来自实际执行项目 |
| 能新建对象，但不能更新 state | 我们自己的 SA 是否只有 Object Creator；桶是否有阻止覆盖的保留设置 |
| 模型不可用或任务排队 | 模型 ID、Batch 支持、location、项目访问条件或容量；不能仅凭密钥有效判断 |
| 已有 run 续跑找不到文件或权限不足 | 查原私有绑定及原 state URI，不要换默认账号或另建空桶来尝试续跑 |

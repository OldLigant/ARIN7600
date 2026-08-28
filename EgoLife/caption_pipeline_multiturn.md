# caption_pipeline_multiturn — 多轮递进标注（spec）

> 用 Mimo-V2.5 把第一人称（Meta Aria）视频段标注成面向数字孪生的结构化 JSON，**由单轮全任务
> 重构为多轮递进**：每轮只回答一个问题——T1「有什么」（静态感知，多次采样投票）→ T2「发生了
> 什么」（原子动作/环境变化，词典锚定）→ T3「暗示了什么心理」（上下文隔离的隐变量）→ T4「因果
> 联系」（纯文本推理的加权因果图）——最后由**代码确定性合并**成最终记录。
>
> 本文档是新管线 `caption_pipeline_multiturn.py` 的设计与 prompt 规格；旧管线
> （`caption_pipeline.py` / `caption_pipeline.md`）原样保留，用于 A/B 对照。

---

## 0. 动机：单轮标注的两类系统性错误

### 0.1 感知幻觉（认错物体）

单轮 prompt 要求模型同时做六件事：认物体/认人、时序分解、状态追踪、隐变量推断、因果推理、
水印读时。认知超载导致最基础的感知出错——典型案例：喝水时**近处的玻璃杯**（贴有东西、距离过
近、细节模糊）被认成"一块布"。近场交互物体恰恰是模拟最关心的状态变量，这类错误不可容忍。

### 0.2 意图幻觉（臆测未来动作）

DAY1 `full.jsonl` 第 25 行（`DAY1_A1_JAKE_11213000`，11:21:30-11:22:00）：视频里"我"放下包装
盒后**直接离开了房间**，标注却写成"我放下包装盒，**准备整理床铺**"——模型把前一条观察到的
"被子没叠好"当成了意图证据，推了一个从未发生的未来动作。

量化：DAY1 827 个片段共出现 **336 处**意图标记词（准备 288 / 试图 32 / 想要 8 / 打算 8），
约四成片段受影响（部分出现在引语内属合法，但量级说明问题系统性存在）。

### 0.3 对策（本设计的四根支柱）

| # | 机制 | 针对的错误 |
|---|---|---|
| 1 | **任务分轮**：每轮一个单一问题，前轮结论注入后轮 | 感知超载（0.1） |
| 2 | **多次采样 + 一致性投票**（T1 跑 k 次，合并出对象词典） | 感知幻觉（0.1） |
| 3 | **词典锚定**：动作层必须引用感知层的对象名称，词典外物体走显式 `new_objects` 逃生口 | 动作层自造物体 |
| 4 | **显式禁令**：禁止描述未观察到的意图/未来动作 | 意图幻觉（0.2） |

---

## 1. 总体架构

```
                ┌──────────── 每个片段(默认10s)一条链，片段间并行 ────────────────┐
                │                                                                │
 [切片 _pN.mp4] │ T1 感知层「有什么」──> T2 行为层「发生了什么」──> T3 心理层      │
 (复用现有编码) │  ├ 主调用×k: 物体+人+音频  (视频+T1词典锚定)   「暗示了什么」│
 (1024@2fps)    │  └ 合并调用: 投票出对象词典                            (视频+T2)  │ ──> T4 因果层
                │  [条件轮 T1c: 界面/文本面理解]                        (上下文隔离) │    (纯文本+T1-T3)
                └────────────────────────────────────────────────────────────────┘
                                      │
              代码确定性合并（非 LLM）──> {范围}.jsonl 最终记录（新 schema，见 §7）
              各轮原始输出按轮落盘（{范围}_t1.jsonl … _t4.jsonl），resume 粒度细化到轮
```

设计原则：

- **每轮只问一个问题**。后轮以前轮结论为锚，禁止绕过锚自造事实。
- **最终记录由代码拼装**，任何字段可回溯到具体轮次与具体一次 API 调用。
- **继承旧管线的已验证结论**：10s 切分（60s 模型只看前 5-6 秒的教训）、水印时间轴、原子化
  原则、`strength` 反事实三档、producer-consumer 工程架构（RPM 限速 / 有序落盘 / 断点续传）。
- **移除 `awareness`**：它诞生于旧版"只考虑情绪→动作单向联系"的设计，是那条边的强度摘要；
  重构后每个方向的因果边各自带 `strength`，`awareness` 不再保留（见 §5.3）。

与旧管线的关系：新脚本 `caption_pipeline_multiturn.py` 单文件自包含（复制复用旧脚本的
ffmpeg/限速/OrderedWriter 基础设施），输出 schema 见 §7 的字段变更表；旧管线不动，供 A/B。

---

## 2. Turn 1 — 感知层「有什么」（静态）

### 2.1 主调用：物体 + 人 + 语音转写（一次调用完成）

- **输入**：切片视频（1024@2fps，含音频）。视觉与音频同源，说话人归属可直接利用画面中出现
  的人——这是合并成一次调用的原因。依据官方文档：音轨默认随视频输入（音频 ≈6.25 token/秒），
  而**音频与图像混排未被文档声明支持**——音画同源感知只能走视频通道，这也是 T1 用视频而非
  图像的原因。
- **任务**：清单式回答"画面里有什么人、什么物体（重点近场与交互物），说了什么话"。
  **不做**动作分解、不做因果、不做心理。

**近场优先规则（不穷举）**：

> 重点标注**近场**物体和正被佩戴者或他人**触摸/持握/操作**的物体（以及显然即将被操作的），
> 这些必须尽力精确识别。远处的、背景的杂物**不需要穷举**：桌上摆 4 个杯子还是 5 个杯子不是
> 重要信息，一句"桌上散落多个包装盒与线缆"式的概览即可。

**诚实描述规则（反"杯子→布"幻觉的核心）**：

> 如果一个物体无法被有把握地识别（距离过近 / 失焦 / 部分遮挡 / 反光透明 / 贴着东西），
> **禁止猜测具体类别**，而是诚实描述可观察属性（形状、大小、颜色、材质、透明度、附着物）并
> 标 `uncertain: true`。"一个贴有标签的半透明玻璃容器，距离过近细节模糊"是正确答案；
> 自信地叫它"一块布"是幻觉——正是本轮要消灭的错误。

**音频 = 语音转写 + 非语音环境声**：语音部分逐字转写（原语言）；每句标注说话人（"佩戴者(我
的声音)" / 画面中人的描述符）；**尽量**给出片段内相对时间（秒），但时间戳不是强制项——拿不准
就留空，不猜；听不清标 `uncertain`，不编造。**非语音声音同样要标**：正在听什么、背景正在发
生什么——音乐（音箱/耳机外放正在放歌，能识别曲名/歌手就给，识别不了就描述风格与语言）、持续
性环境声（键盘敲击、风扇、油烟机、街道车流、远处人声嘈杂）、离散声音事件（开门声、手机震动/
提示音、杯子放下的碰撞声）。能辨来源就标来源（设备播放 / 身边发出 / 远处传来）。

主调用输出 schema：

```jsonc
{
  "objects": [
    {"label": "贴有标签的半透明玻璃容器", "attrs": ["透明", "杯状", "贴纸"],
     "where": "near",                // near | mid | background
     "interacted_with": true,
     "first_seen_s": 1, "last_seen_s": 9,   // 片段内相对秒，~1s 分辨率即可
     "uncertain": false, "uncertainty_note": ""}
  ],
  "persons": [
    {"label": "穿橙色T恤的男士", "attrs": ["短发", "站在桌边"], "first_seen_s": 0, "last_seen_s": 9}
  ],
  "transcript": [
    {"t_start_s": 0.0, "t_end_s": 2.5, "lang": "zh", "speaker": "佩戴者",
     "text": "出流水线，下一个流水线。", "uncertain": false}
  ],
  "sounds": [                         // 非语音环境声
    {"kind": "music|ambient|event",   // 音乐 / 持续氛围声 / 离散声音事件
     "label": "音箱播放的中文流行歌曲", "source": "设备播放",
     "t_start_s": 0, "t_end_s": 9, "uncertain": false}
  ],
  "text_surfaces": [                  // 只做"检测与标记"，内容留给条件轮 T1c
    {"kind": "screen", "device": "laptop", "in_use": true}
  ]
}
```

`first_seen_s` 等相对秒由代码换算成水印绝对时间（片段起始水印时间已知，无需模型读水印）。

### 2.2 一致性投票与词典合并

- 主调用独立跑 **k=2 次**（`--votes` 可调，默认 2），两次输出送一次合并调用——**附带同一段
  视频作参照**（媒体在前的字节与感知采样完全一致，视频前缀走缓存、边际成本 ≈0）：判断
  "run A 的 black rectangular device 和 run B 的 白色外接设备盒 是不是同一个东西"往往不能
  只靠同义词匹配，合并层需要能看；同时禁止它借机重新感知（不新增任何 run 都没报的物体）。
  - 同义/上下位关系的条目合并（"杯子" vs "白色马克杯" → 取两者都能支撑的更具体标签，属性取并集）；
  - `votes` 记录出现次数（"2/2" / "1/2"）；**分歧仲裁**：近场交互物出现 1/2 分歧时触发第 3 次
    感知调用仲裁，仍不一致则保留但 `confidence: low` 并注明分歧——近场交互物**永不静默丢弃**；
  - 转写取并集去重，两句都有的标高置信，单句出现的标 low；环境声 `sounds` 同样取并集去重
    （同一声音两种描述取更具体的那个）；
  - 产出**片段对象词典**（稳定 ID `obj01`/`psn01`…），是后续所有轮次的物体名称锚。
- 合并输出（即 `_t1.jsonl` 每行）：

```jsonc
{"clip_id": "...", "lexicon": [
   {"id": "obj01", "label": "贴有标签的半透明玻璃容器", "type": "object",
    "attrs": ["透明", "杯状", "贴纸"], "votes": "2/2", "confidence": "high",
    "interacted_with": true, "first_seen_s": 1, "last_seen_s": 9, "uncertainty_note": ""}],
 "transcript": [ /* 同上 + confidence */ ],
 "sounds": [ /* 并集 */ ],
 "text_surfaces": [ /* 并集 */ ],
 "agreement_notes": "obj02 两轮标签分歧('布' vs '玻璃容器')，已仲裁",
 "runs": [ /* 两次原始输出全文，备追溯 */ ]}
```

### 2.3 条件轮 T1c：界面 / 文本面理解（OCR 的升级替代）

- **触发条件**：T1 词典的 `text_surfaces` 含使用中的屏幕（手机/笔记本/显示器）或大块文本面
  （白板/纸张/招牌）。无则跳过（大多数片段零成本）。
- **输入**：**直接送图像，不送视频**——用 ffmpeg 从**源视频**按原生分辨率抽关键帧（~1 帧/秒，
  10s 片段 ≈ 10 帧 JPEG，不做任何缩放），多图放同一次调用，帧按时间顺序编号。依据（官方图像
  文档）：多图数量无固定上限（受上下文长度约束）；图像按 `(宽/32)×(高/32)` 计 token、上限
  8.4M 像素（≈2896²）——Aria 原生帧 ~1408² 远低于上限不会被缩，**保真度高于视频通道**（视频
  抽帧单帧默认 ~300 token 封顶、必然降采样），读界面小字正需要这条全分辨率路径；且无需对视频
  重编码。屏幕内容变化慢的场景可用更稀疏的 0.5fps（5 帧）。帧缓存到 `_cache/frames_native/`。
- **成本与分辨率**（图像 vs 视频通道的 token 机制差异）：视频通道有 `media_resolution` 参数
  且默认单帧 ~300 token 封顶（30s@2fps 整段才 ~1 万 token 级）；图像通道**没有分辨率参数**，
  按 `(宽/32)×(高/32)` 计费——1408² 原生帧 ≈ 1900 token/帧，是视频默认帧的 ~6 倍，大图还
  显著拖慢响应（实测 30 帧原生图 58.7k token、首次 197s）。帧率与缩放因此由我们自己控制：
  `--t1c-frame-fps`（默认 1）、`--t1c-frame-size`（长边上限 px，0=原生）。
- **可替换 VLM 接口**：读屏只需要视觉、不需要音频/全模态——T1c 支持 `--t1c-provider
  custom`（配 `--t1c-base-url` / `--t1c-model` / `--t1c-api-key-env`）接入任意 OpenAI 兼容
  图像模型（不传 completion 上限、无 thinking 参数、仍要求 JSON 输出；不支持 json_object 的模型
  加 `--t1c-no-json-mode`）；默认走 MiMO。实测候补：DeepSeek `deepseek-v4-flash-vision-exp`
  （OpenAI 兼容、base64 image_url、单请求 ≤600 图、**每图 token ≤384 封顶**——30 帧输入
  ≤11.5k token，约为 MiMO 原生帧的 1/5，且不占 Mimo 配额。注意它是**推理型视觉模型**，
  视觉推理阶段会烧大量 completion token——历史上固定 2048 上限时 JSON 正文不出现（DAY4
  实测 48/48 失败恰好触顶）；管线已不再传上限，接入前确认供应商默认预算足够）。
- **任务**：不止 OCR，是 **GUI 语义理解**——这是什么应用/网站、正在操作什么内容：

| 场景 | 期望输出 |
|---|---|
| 看视频 | `{"app": "Bilibili (web)", "content": "正在播放视频《标题逐字》", "operation": "watching", "details": "进度条约1/3"}` |
| 刷帖子 | `{"app": "知乎", "content": "浏览问题《…》下的回答，滚动至第3条", "operation": "scrolling"}` |
| 看文件 | `{"app": "文件资源管理器", "content": "打开目录 …/reports，可见文件 a.docx、b.xlsx", "operation": "browsing"}` |

诚实规则同上：只转写可读部分，读不清就注明，**不编造标题或应用名**；不读右上角时间水印。
旧 `ocr` 字段被此层吸收（物理文本面同样走 T1c）。

输出：`{"surfaces": [{"kind", "device", "app", "content", "operation", "details", "legibility", "note"}]}`

---

## 3. Turn 2 — 行为层「发生了什么」（动态）

- **输入**：切片视频（1024@2fps）+ 注入 T1 产物：对象词典（含置信度）、转写、界面摘要。
- **输出**：`self_actions` / `others` / `env_changes` / `environment` / `new_objects` / `tags`。
  语音不重复标（T1 已完成，只允许引用）；心理与因果留给后轮。

继承旧 prompt 的核心（原文迁移）：

- **水印锚**（读右上角 `HH:MM:SS:FF DAYn` 标时间，~1-3s 分辨率）；
- **原子化原则**：一条 = 一个动词级步骤、单一对象、单一即时目的；复合行为拆解、连续手势不硬
  拆、无数量配额；覆盖全片段；
- **锚定补充规则**（DAY4 对比后加入）：无对象动作（走、坐、转身、视线转移、姿态变化）**不需
  要锚**，与物体动作同等自由上报；词典外物体宁可走 `new_objects` 也**不许丢弃或含糊化**动作；
  screen 层已描述的界面内容可在动作措辞中引用；
- **env_changes**：一次离散状态迁移一条，`cause ∈ {self, other, external}`；
- **environment**：整段一两句场景背景——地点类型、房间布局、光照、**天气/室内外线索**——是
  模拟器放置 agent 的上下文、场景说明而非变化日志（屏幕内容归 screen 层，不在此重复）；
- **others**：中立描述符开头，不臆造名字。

新增两条硬规则：

**词典锚定（反动作层幻觉）**：

> 动作/变化涉及的物体必须使用词典中的标签（尤其近场交互物）；低置信词条必须保持其诚实的
> 描述性标签，**禁止把不确定描述升级成确定的具体类别**。确需词典外物体时，照常描述但同步
> 追加进 `new_objects`（附出现时间与原因），绝不静默引入。
> 代码层校验：动作文本与词典做字符串匹配，未命中且不在 `new_objects` → warning 计数。

**意图禁令（反"准备整理床铺"）**：

> Never describe intent or future actions (准备 / 打算 / 想要 / 试图 to do X) unless X is
> actually observed within this clip. If the clip ends mid-activity, the last action simply ends
> there.
> 只描述可见的内容：姿势、接触、移动——不是目标。看到被子没叠**不**构成"准备整理床铺"的
> 证据，除非整理床铺确实被观察到。

---

## 4. Turn 3 — 心理层「可能暗示了什么」（隐变量）

- **输入**：视频（语气/语速是情绪证据）+ T2 结构化输出。thinking 开启。
- **定义（上下文隔离推断，本层的方法论核心）**：

> **仅以本片段内的情境（动作、环境、语音、界面内容）为证据，假定片段之前什么都没发生过，
> 一个合理的心理活动大概会是什么？**
> 这不是"佩戴者带全天上下文的真实心理状态"，而是一个刻意可估计的局部量——避免标注器脑补
> 跨片段记忆。跨片段/跨天的心理演化由下游聚合层处理（把相邻片段的隔离估计串起来）。

- 输出（替代旧 `psychology`；**`awareness` 已移除**）：

```jsonc
{"emotion": "focused, mildly frustrated",        // 1-2 个小写关键词
 "mental_activity": "任务中途遇到小障碍，注意力收窄在手头物体上；短暂的不耐烦没有改变行为方向",
 "evidence": ["11:10:15 反复调整握持姿势", "语气词'啧'（t≈6s）"],
 "confidence": "medium"}                          // low|medium|high
```

`evidence` 必须引用可观察线索（带时间的动作引文、语气、语音内容、交互节奏）；
`mental_activity` 描述**状态**，同样禁止推测未来计划。

---

## 5. Turn 4 — 因果整合层「联系起来」

- **输入**：T1 词典 + T2 + T3 的 JSON（**纯文本，不含视频**——时序证据已由 T2 的时间戳编
  码；迫使推理基于已验证的中间层而不是重新看图引入新幻觉，同时省下视频 token）。词典用于
  "物体名须与词典一致"的校验。thinking 开启。

### 5.1 边类型（7 → 8）

| type | 含义 | 例子 |
|---|---|---|
| `env->action` | 环境事件改变我的行为 | 手机震动 → 我拿起手机 |
| `env->emotion` | 环境事件改变我的情绪 | 巨响 → 受惊 |
| `emotion->action` | 我的情绪驱动行为 | 无聊 → 划开手机 |
| `action->env` | 我的动作改变环境 | 我拨开关 → 灯亮（与 `cause=self` 的 env_changes 互为镜像） |
| `other->action` | 他人动作触发我的动作 | 同事招手 → 我走过去 |
| `other->env` | 他人动作改变环境 | 她拉开窗帘 → 房间变亮 |
| `other->emotion` | 他人动作改变我的情绪 | 朋友笑 → 我放松 |
| `env->env` **（新增）** | 无行为主体的环境事件导致另一环境变化 | 云遮住阳光 → 室内变暗；风把门吹开 |

### 5.2 强度（继承反事实三档）

| strength | 定义 |
|---|---|
| **strong** | 触发：没有此因，果大概率不发生或走向不同 |
| **moderate** | 塑形：改变了果的方式/时机/力度，但果本来也会发生 |
| **weak** | 背景：多个促成因素之一，果主要由习惯/任务驱动 |

规则（继承）：可见时序 + 合理机制才标；怀疑边**存在**才省略；cause/effect 引用带时间的原子
动作/变化短语，物体名须与词典一致；0-3 条是常态；物理边几乎总是 strong，属正常。

### 5.3 `awareness` 移除说明

旧版 `psychology.awareness` 是"情绪→动作"单向时代的产物（该边的强度摘要 + 与 `strength`
的同梯一致性校验）。重构后情绪作为普通节点进入八类边，每条边自带 `strength`，单向摘要失去
存在意义——**新 schema 不再有 awareness 字段，同梯校验代码一并移除**。旧数据里的该字段自然
作废，不做迁移。

### 5.4 代码层校验（落盘前）

- 每条 `cause=self` 的 env_change 应有镜像 `action->env` 边（缺失 → warning）；
- 边的 cause/effect 引用的物体须在词典（或 `new_objects`）内；
- 边 type 合法性、strength 合法性（宽松归一，不整单拒绝——继承旧策略）。

---

## 6. 各轮 prompt 草案

### 6.0 消息布局与前缀缓存（先读这节）

MiMO 计价：输入 **1元/M token**、**缓存命中 0.02元/M（50×）**、输出 2元/M。前缀缓存按
"最长公共前缀"命中，因此**所有轮次共用同一份 system prompt，每轮差异全部放进 user 消息**，
且 user 消息按固定布局构造：

```
[system]  共享 system（下面这份，全局一字不差）
[user]    ① 媒体：视频（T1 各次采样、合并、T2、T3）或图像组（T1c）——同一片段跨轮字节完全一致
          ② 上下文块，按固定顺序原样拼接：T1 词典 → T2 JSON → T3 JSON（用到哪层放哪层）
          ③ 本轮任务文本（§6.1-§6.6，永远放最后）
```

由此：同一片段的 T1×k / T2 / T3 共享 `system+媒体` 大前缀（~10s 切片视频 ≈ 6-7k token，
只在首次全价，后续 50× 折扣）；T4 纯文本只共享 system（其文本量小，无伤大雅——若开 `--t4-video`，则与 T3 进一步共享
视频+前置上下文块）。**上下文块必须
原样嵌入前轮的 RAW 输出文本**（模型返回的 JSON 原文，不重新序列化、不改键序、不截断），
保证跨轮字节一致。各轮任务文本开头的角色句（"You are a ..."）保留——它位于 user 尾部，
不占缓存成本，还能强化当轮的角色聚焦。

共享 system prompt（英文指令、JSON-only 输出、场景语言跟随主导口语，风格同旧 `SYSTEM_MSG`）：

```text
You are a dense first-person life-log captioner producing SIMULATION-GRADE structured annotations
for a digital twin of the wearer's daily life. The footage is from participant A1_JAKE wearing
Meta Aria glasses; people may speak Chinese or English. You will receive ONE narrow task per
request (perception / behavior / psychology / causal analysis); follow exactly that task.

# Watermark anchor
Every video frame carries a watermark TOP-RIGHT showing time and day, "HH:MM:SS:FF DAYn"
(e.g. "11:10:02:00 DAY1"). When your task asks for watermark timestamps, read it; when it asks
for in-clip seconds, count from the clip start.

# Honesty (absolute)
Report only what is visible or audible in this clip. If something cannot be confidently
identified, describe its observable attributes and mark it uncertain — never guess a confident
specific category. Never invent names, titles, words, or events.

# Output format
Return ONLY a single JSON object following the task's schema: double quotes, no trailing commas,
no markdown fences, no text outside JSON. Pick ONE content language from the dominant spoken
language (Chinese if participants speak Chinese, English otherwise); never translate or duplicate
content in both languages. This applies to EVERY string field you produce — object labels,
attributes, sound labels, notes — not only free-text descriptions.
```

**缓存命中的不利因素、实测手段与降级方案**：

- **不利因素（不受我们控制）**：上面的布局假设服务端按 content 数组的原顺序（媒体在前、文本
  在后）拼接 token 序列。若 MiMO 服务端把 user 消息的**文本部分前置到媒体之前**，那么每轮
  不同的任务文本/上下文块就会插在视频前面，公共前缀退化为很短的 system，视频缓存全失。
- **实测手段（用数据说话）**：API 的 usage 会返回 `cached_tokens`。冒烟阶段直接看各轮实测
  值——T1 第 2 次采样、T2、T3 的 `cached_tokens` 理论上应接近 `system+视频` 的 token 量；
  明显偏小即说明布局假设不成立。
- **降级方案 Plan B（真·多轮对话结构）**：每轮请求 = 完整保留之前的消息序列再追加新 user
  消息，即 `[sys, u1(视频+T1任务), a1(T1结果), u2(T2任务), a2, u3(T3任务), ...]`。每轮请求
  是上一轮的**严格前缀扩展**，缓存命中基本可以保证。代价：视频只出现在第一轮消息里，随轮次
  增加离当前问题越来越远，**视频感知可能减弱**（未验证的猜测）；各轮中间结果从"拼进 user
  文本"改为"读 assistant 历史"。**仅当 Plan A 实测视频缓存完全命中不了时，再上 Plan B 对比
  效果**，不预先采用。

**初步实测（2026-08-26 冒烟，DAY1 1121-1122 两片段、30s 不切分）**：Plan A 成立——服务端
确实按媒体在前的原顺序拼 token。同片段内：T1 第 2 次感知采样命中 8192/10275（79.7%），T2
64%，T3 78%（上下文块前缀共享生效），T4 纯文本 0%（符合预期）；视频前缀**跨进程 8 分钟后
仍命中**（10240/10275，缓存 TTL 至少分钟级）。暂无需要 Plan B。（注：summary 里 t1 轮的
cache 统计只计合并调用——感知两次采样的命中要看 `_usage.jsonl` 明细。）

> 历史注记：`caption_pipeline.md` §8 提到的"实测首次缓存命中 ~73%"出自更早一代的带上下文
> 设计（滑窗 deque 维护最近若干 user-assistant 轮次作前缀，批量弹出旧轮次、逐次追加新轮次），
> 不代表当前无上下文单发管线的命中水平，引用时注意出处。

### 6.1 T1 感知主调用

```text
You are a first-person PERCEPTION annotator. Your ONLY job is to inventory what is PRESENT in a
short first-person clip (Meta Aria glasses): objects, people, and spoken words. Do NOT describe
actions, state changes, intentions, emotions, or causes — later annotation rounds handle those.

# Priorities (CRITICAL)
Focus on the NEAR FIELD and on objects that are being touched / held / manipulated by the wearer
or by other people, or are plausibly about to be. Identify these as precisely as the footage
allows. Distant or background clutter does NOT need exhaustive enumeration: for a crowded table
a summary like "桌上散落多个包装盒与线缆" is enough — counting 4 cups vs 5 is NOT required.

# Honesty rule (CRITICAL — never hallucinate a category)
If an object cannot be confidently identified (too close / out of focus / partially occluded /
reflective or transparent / covered by stickers), DO NOT guess a specific category. Describe its
observable attributes honestly — shape, size, color, material, transparency, attached items —
and set uncertain=true. "一个贴有标签的半透明玻璃容器，距离过近细节模糊" is a CORRECT answer;
confidently calling it "一块布" is a hallucination — the exact failure this round exists to kill.

# People
List visible people with neutral descriptors (clothing, hair, position). The wearer is not
visible except hands/body edges. Never invent names; use one only if shown on screen or spoken.

# Speech (transcription)
Transcribe all audible speech verbatim in its original language. Attribute a speaker to every
utterance using the people you see ("佩戴者(我的声音)" / "穿橙色T恤的男士") and voice cues.
Add approximate in-clip time offsets in seconds when you can — precision is NOT required; omit
rather than guess. Mark unclear hearing with uncertain=true. Never invent words.

# Non-speech sounds
Also report NON-SPEECH audio: music playing (from a speaker or earbuds; identify song/artist if
you can, otherwise describe style and language), continuous ambient sounds (keyboard typing,
fan, range hood, street traffic, distant crowd murmur), and discrete sound events (door,
phone notification/buzz, object clinking). Give the source when distinguishable (played by a
device / made nearby / from far away). Mark uncertain identifications honestly.

# Screen / text surfaces
If a screen is in use or a large text surface is visible (whiteboard / paper / sign), only flag
it with device + kind here; a dedicated round reads its content. Do not transcribe it now.

# Language
Object/person labels, attrs, sound labels and all notes MUST be in the clip's dominant spoken
language (Chinese when the scene is Chinese-speaking) — do NOT switch to English because this
task text is English. Speech is quoted verbatim in whatever language it was spoken.

# Output format
Return ONLY a single JSON object, no markdown fences, no text outside JSON:
{
  "objects":  [{"label": "...", "attrs": ["..."], "where": "near|mid|background",
                "interacted_with": true, "first_seen_s": 0, "last_seen_s": 9,
                "uncertain": false, "uncertainty_note": ""}],
  "persons":  [{"label": "...", "attrs": ["..."], "first_seen_s": 0, "last_seen_s": 9,
                "uncertain": false}],
  "transcript": [{"t_start_s": 0.0, "t_end_s": 2.5, "lang": "zh", "speaker": "...",
                  "text": "...", "uncertain": false}],
  "sounds": [{"kind": "music|ambient|event", "label": "...", "source": "...",
              "t_start_s": 0, "t_end_s": 9, "uncertain": false}],
  "text_surfaces": [{"kind": "screen|whiteboard|paper|sign", "device": "...", "in_use": true}]
}
Times are seconds from clip start (~1s resolution is fine). Empty arrays are valid. Use Chinese
when the scene is Chinese-speaking, English otherwise.
```

### 6.2 T1 合并调用（视频参照 + 文本）

```text
You reconcile TWO (or THREE) independent perception inventories of the SAME first-person clip
into one canonical object/person lexicon and one merged transcript.
The clip's video is attached as REFERENCE: use it to adjudicate identity and appearance
conflicts — is run A's "black rectangular device" the same thing as run B's "白色外接设备盒"?
Look, then decide. Do NOT add objects that no run reported: this is reconciliation of the given
inventories, not a fresh perception pass.

Rules:
- Two entries are the SAME object/person if one label is a synonym or hyponym of the other
  (杯子 vs 马克杯) and their time ranges overlap. Merge under the most specific label BOTH runs
  support; union the attrs.
- votes = how many runs reported it ("2/2" | "1/2"). A near-field interacted object at 1/2 is
  either a hallucination or a miss — keep it with confidence "low" and note the disagreement;
  NEVER silently drop a near-field interacted object.
- confidence: high = 2/2, neither uncertain; medium = 2/2 with one uncertain, or labels merged
  after disagreement; low = 1/2, or both uncertain.
- transcript: union, dedupe near-identical utterances preferring the more complete wording;
  reported by both = high confidence, by one = low.
- sounds: union, dedupe (two descriptions of the same sound keep the more specific one).
- text_surfaces: union.
- Output labels in the clip's dominant scene language; do not translate entries into English.
- NEVER emit two lexicon entries with identical or synonym labels — they are the same object;
  merge them into one entry.
Assign stable ids obj01..., psn01... (type object|person). Output ONLY JSON:
{"lexicon": [{"id": "obj01", "label": "...", "type": "object", "attrs": [], "votes": "2/2",
              "confidence": "high", "interacted_with": true, "first_seen_s": 0,
              "last_seen_s": 9, "uncertainty_note": ""}],
 "transcript": [...input shape + "confidence"...],
 "sounds": [...],
 "text_surfaces": [...],
 "agreement_notes": "..."}
```

### 6.3 T1c 界面/文本面（条件轮，原生分辨率视频）

```text
You read INFORMATION SURFACES in N timestamp-ordered frames (native resolution, ~1 per second)
from a short first-person clip: screens in use (laptop / monitor / phone) and large text
surfaces (whiteboard, paper documents, signs). Beyond raw OCR your job is GUI-level
understanding: WHAT application / website / content is being used, and what the wearer is doing
with it.

For each surface report:
- app / site identity when identifiable from chrome, logo, layout (e.g. "Bilibili 网页版",
  "文件资源管理器", "知乎");
- the content being consumed or operated, with verbatim titles when legible (video title, post
  title, file names, document headings);
- the visible operation (watching / scrolling / typing / clicking through ...).

Honesty: transcribe only what is legible; if a title is partially readable, transcribe the
readable part and note it. Never fabricate a title or app name. Do NOT read the top-right
timestamp watermark. Return ONLY JSON:
{"surfaces": [{"kind": "screen", "device": "laptop", "app": "...", "content": "...",
               "operation": "watching", "details": "...", "legibility": "good|partial|poor",
               "note": ""}]}
```

### 6.4 T2 行为（视频 + T1 词典注入）

```text
You are a first-person BEHAVIOR annotator producing SIMULATION-GRADE atomic records of WHAT THE
WEARER AND OTHER PEOPLE DID and WHAT VISIBLY CHANGED in the environment. Speech transcription,
psychology and causal analysis belong to other rounds — do not output them.

# Watermark anchor (CRITICAL)                       [from the legacy prompt, unchanged]
Every frame carries a watermark TOP-RIGHT "HH:MM:SS:FF DAYn" ... timestamp every entry.

# Object anchoring (CRITICAL)
The user message provides this clip's object/person LEXICON from an independent perception round
(with per-entry confidence). When an action or change involves an object, USE THE LEXICON LABEL
as its name — especially for near-field interacted objects. A low-confidence entry must keep its
honest descriptive label; do NOT upgrade an uncertain description into a confident category. If
you genuinely need an object NOT in the lexicon, describe it AND append it to "new_objects"
(never silently invent).

# Intent ban (CRITICAL)
Never describe intent or future actions (准备 / 打算 / 想要 / 试图 to do X) unless X is actually
observed within this clip. If the clip ends mid-activity, the last action simply ends there.
Describe only what is visible: postures, contacts, movements — not goals. An unmade bed does NOT
license "准备整理床铺" unless tidying is actually observed.

# Output format
{
  "self_actions": [{"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}],
  "others":       [{"time": "HH:MM:SS", "time_end": "HH:MM:SS", "text": "..."}],
  "environment":  "...",
  "env_changes":  [{"time": "...", "time_end": "...", "text": "...", "cause": "self|other|external"}],
  "new_objects":  [{"label": "...", "why": "manipulated at 11:21:35 but absent from lexicon"}],
  "tags": ["..."]
}
Field rules for self_actions / others / environment / env_changes / tags: [atomicity, cause
attribution, scene-not-changelog, 5-10 tags — migrated verbatim from the legacy prompt].
```

### 6.5 T3 心理（视频 + T2 JSON 注入）

```text
You estimate the wearer's psychological state as a LATENT VARIABLE from one short first-person
clip. The user message gives the video plus the behavior round's structured output (actions,
changes, transcript).

# Isolation rule (definitional)
Estimate the mental state AS IF THIS CLIP WERE YOUR ONLY EVIDENCE — ignore anything that might
have happened before it. The question is: "considering only this clip's situation, what would a
plausible mental activity be?" — not "what is the wearer's state given their whole day?"

# Output
{"emotion": "1-2 lowercase keywords",
 "mental_activity": "2-3 sentences describing the plausible inner state, state not plans",
 "evidence": ["11:10:15 <quote from the behavior output / audible cue>", ...],
 "confidence": "low|medium|high"}
Evidence must cite observable cues (timestamped action quotes, tone of voice, speech content,
interaction pace). No intent or future-plan speculation here either. confidence reflects how
strongly the evidence constrains the inference.
```

### 6.6 T4 因果（纯文本，T2+T3 JSON 注入）

```text
You are a causal analyst. Input: one clip's object lexicon, behavior JSON (self_actions /
others / env_changes / environment / transcript / sounds) and psychology JSON (emotion /
mental_activity). There is no video.
Build the clip's directed causal graph from this evidence alone.

Edge types (use EXACTLY these strings):
  env->action, env->emotion, emotion->action, action->env, other->action, other->env,
  other->emotion, env->env (an environmental event with no visible agent causes another
  environmental change: 云遮住阳光 -> 室内变暗; 风把门吹开).

strength (counterfactual, REQUIRED on every edge):
  strong   = trigger: without this cause the effect likely would NOT have happened;
  moderate = shaper: changed HOW/WHEN/how vigorously, effect would still occur;
  weak     = background: one contributory factor among several.

Rules: causal direction must be supported by visible temporal order (use the timestamps in the
input) + plausible mechanism — never fabricate; quote the atomic action/change texts with their
times; object names must match the lexicon; every cause="self" env_change should be mirrored by
an action->env edge; 0-3 links is typical; physical edges are almost always strong — expected,
not lazy. Output ONLY JSON:
{"causal_links": [{"type": "...", "cause": "...", "effect": "...", "strength": "..."}]}
```

各轮 user 消息按 §6.0 的固定布局构造：媒体（同一片段跨轮字节一致）→ 上下文块（T1 词典 →
T2 JSON → T3 JSON，原样嵌入前轮 RAW 输出文本）→ 本轮任务文本（§6.1-§6.6）置尾；clip_id /
day / 起始水印时间一行信息沿用旧 `USER_TASK_TMPL` 风格，写在任务文本开头。T2 与 T3 都注入
T1 词典（T3 的情绪推断需要转写与环境声）；T4 纯文本，注入 T1+T2+T3 全部上下文。

---

## 7. 输出 schema 与目录

### 7.1 最终记录（`{范围}.jsonl`，一片段一行，代码合并产出）

```jsonc
{
  "clip_id": "DAY1_A1_JAKE_11100000_p2", "global_idx": 5, "day": 1, "user": "A1_JAKE",
  "duration_s": 10.0, "model": "mimo-v2.5", "ts_captioned": "...", "slice_path": "...",
  "turns": {                                 // 每轮的调用与 token 留痕
    "t1":  {"runs": 2, "tokens": {...}, "latency_s": 12.3},
    "t1c": {"tokens": {...}, "latency_s": 8.1},      // 未触发则 null
    "t2":  {"tokens": {...}, "latency_s": 25.6},
    "t3":  {"tokens": {...}, "latency_s": 18.2},
    "t4":  {"tokens": {...}, "latency_s": 9.4}
  },
  "inventory": [ /* §2.2 lexicon */ ],
  "speech": [ {"lang": "zh", "speaker": "佩戴者", "text": "...",
               "t_start": "11:10:02", "t_end": "11:10:05", "confidence": "high"} ],  // 时间可空
  "sounds": [ {"kind": "music", "label": "音箱播放的中文流行歌曲", "source": "设备播放",
               "t_start": "11:10:00", "t_end": "11:10:10", "confidence": "high"} ],
  "screen": [ /* §2.3 surfaces，未触发为 [] */ ],
  "self_actions": [ /* §3 */ ], "others": [ /* */ ], "environment": "...",
  "env_changes": [ /* */ ], "new_objects": [ /* */ ],
  "psychology": {"emotion": "...", "mental_activity": "...", "evidence": [...], "confidence": "..."},
  "causal_links": [ /* §5，8 类边 */ ],
  "tags": ["..."],
  "recovery": {"t1": "ok", "t1c": "skipped", "t2": "ok", "t3": "ok", "t4": "ok"}  // 每轮路径标签
}
```

### 7.2 相对旧 CaptionRecord 的字段变更

| 变更 | 字段 | 说明 |
|---|---|---|
| 新增 | `inventory` / `turns` / `new_objects` / `screen` / `sounds` | 感知词典、轮次留痕、锚定逃生口、界面层、非语音环境声 |
| 升级 | `speech` | 增加 `t_start`/`t_end`（可空）与 `confidence`（T1 产物） |
| 替换 | `psychology` | `{emotion, mental_activity, evidence, confidence}`，**无 awareness** |
| 扩展 | `causal_links` | 7 类 → 8 类（+`env->env`） |
| 移除 | `psychology.awareness` | 旧单向设计遗产，§5.3 |
| 吸收 | `ocr` → `screen` | T1c 语义化替代 |
| 移置 | `narrative` | 不再内嵌全文；各轮原始输出在 `_t{N}.jsonl`（含 tokens/延迟/原始 JSON） |

### 7.3 目录布局

```
captions/A1_JAKE/DAY1/
├── 1110-1130.jsonl            # 最终合并记录
├── 1110-1130_t1.jsonl         # 每片段：合并词典 + k 次原始输出
├── 1110-1130_t1c.jsonl        # 仅触发的片段
├── 1110-1130_t2.jsonl / _t3.jsonl / _t4.jsonl
├── 1110-1130_usage.jsonl      # 每次 API 调用（带 turn 标签）
├── 1110-1130_summary.json / _run.log
└── _cache/
    ├── slices/                # 1024@2fps 切片（T1/T2/T3 用）
    ├── frames_native/         # 原生分辨率关键帧 JPEG（仅 T1c 触发的片段）
    └── clips.parquet
```

**resume 粒度到轮**：每轮读自己的 `_t{N}.jsonl` 里已有 clip_id 集合，只补缺失的轮；四轮齐备
后合并写主文件（重跑合并是纯代码操作，不花 API）。测试/冒烟一律走 `captions/_test/`（全局
约定，防污染真实输出）。

---

## 8. 质量控制机制汇总

| 机制 | 防的错误 | 所在 |
|---|---|---|
| 近场优先、不穷举 | 注意力被背景稀释 | T1 prompt |
| 诚实描述规则（禁猜类别） | 杯子→布 | T1 prompt |
| k=2 采样 + 合并投票 + 分歧仲裁 | 单次感知幻觉 | T1 流程 |
| 近场交互物永不静默丢弃 | 投票误杀 | T1 合并规则 |
| 词典锚定 + `new_objects` 逃生口 + 代码字符串校验 | 动作层自造物体 | T2 prompt + 合并 |
| 意图禁令（中英双语动词清单） | 准备整理床铺 | T2 prompt |
| 上下文隔离定义 | 跨片段脑补 | T3 prompt |
| 纯文本因果输入 | 重看视频引入新幻觉 | T4 流程 |
| 镜像/引用/合法性校验（warning 不拒绝） | 边-变化不一致 | 代码，落盘前 |
| T2 提及屏幕使用但无 screen 层 → 日志 | T1 漏检屏幕 | 代码，落盘前 |

---

## 9. 工程实现要点

- **新脚本** `caption_pipeline_multiturn.py`：单文件自包含，复制复用旧脚本的
  ffmpeg 切分 / `SlidingWindowRateLimiter` / `OrderedWriter` / resume 基建。
- **并发模型不变**：片段间并行（API 线程池 + 全局 RPM），片段内轮次串行——每个片段一个小
  状态机：`t1a(×k) → t1merge → [t1c] → t2 → t3 → t4 → 合并落盘`。global_idx / 有序写语义
  与旧管线一致（每片段一条最终记录）。每轮失败 retry 一次，再失败标记该轮 failed（其余轮
  照常，最终记录缺层但可续传补齐）。
- **thinking 分轮**：T1 / T1c 关（纯感知，且 T1 本来要跑 k 次）；T2 / T3 / T4 开（原子分解
  与因果推断是推理密集任务）。**reasoning_content 无需回传**：每轮都是独立单发会话（无
  assistant 历史），官方文档确认只有"历史含工具调用"时才强制回传 reasoning_content（否则
  400），纯文本/无历史场景不传完全合规——本设计天然满足。不传 completion 上限，由服务端按
  最大补全预算执行。
- **成本大头是输入 token、尤其是视频 token——消息布局为缓存命中而设计**（§6.0）：输入
  1元/M、缓存命中 0.02元/M（**1/50**）、输出 2元/M。每片段 4-5 次视频调用 × ~6-7k token/
  次是费用主体；前缀一旦命中，这部分就从 1元/M 变 0.02元/M——**相当于把成本大头砍掉 ~98%**，
  这就是固定消息布局的全部意义。同一片段的 T1×k / T2 / T3 共享 `system+视频` 前缀（视频
  只在首次全价）；布局是否真的命中，以 `usage.cached_tokens` 实测为准（§6.0 风险与 Plan B）。
  T4 默认纯文本（"因果只基于已验证中间层"的纯净性优先）；按命中价计算，给 T4 带上视频的
  **边际成本仅 ≈ +0.004元/片段**（是多花 0.004元，不是省 0.004元），故留 `--t4-video`
  开关，冒烟时对比其对时序核验是否有增益。
- **`media_resolution=max` 备用杠杆**：视频通道支持 `media_resolution: default|max`，官方说明
  max 用于"提升对小物体、细节纹理的识别能力"。若冒烟发现近场物体（贴纸玻璃杯一类）识别仍
  不足，T1 感知调用可加此参数（单帧 token 上限从默认 ~300 提高）——默认先不开，按 §10 实测
  决定。
- **原生分辨率关键帧**：仅对 T1c 触发的片段，从源视频用 ffmpeg 抽帧（无缩放、JPEG、~1fps，
  可降 0.5fps），缓存 `_cache/frames_native/` 复用，断点续传跳过已存在。
- **CLI**：继承旧参数，新增 `--votes`（默认 2）、`--turns t1,t1c,t2,t3,t4`（子集运行，便于分阶段
  上线）、`--no-screen-round`（强制关 T1c）、`--t4-video`（T4 也带视频，默认关，见 §9 缓存注）、
  `--t1c-frame-fps` / `--t1c-frame-size`（T1c 帧率与长边上限 px，0=原生）、`--t1c-provider
  custom` + `--t1c-base-url/--t1c-model/--t1c-api-key-env`（T1c 换用任意 OpenAI 兼容图像
  VLM，读屏无需全模态）。
- **成本估算（每 10s 片段）**：

| 轮 | 输入 | thinking | 次数 | 备注 |
|---|---|---|---|---|
| T1 感知 | 视频 | off | 2 (+1 仲裁，少数) | 输出短 |
| T1 合并 | 视频(前缀命中)+文本 | off | 1 | 视频缓存价，边际成本≈0；带视觉仲裁身份冲突 |
| T1c | 原生分辨率图像 ×5-10 | off | 0-1 | 条件触发；图像按 (宽/32)×(高/32) 全分辨率计 token（~1408² ≈ 1900/帧），命中时约 0.01-0.02元/片段 |
| T2 行为 | 视频 | on | 1 | |
| T3 心理 | 视频 | on | 1 | |
| T4 因果 | 纯文本 | on | 1 | 无视频 token |

  合计 **5-7 次调用/片段**（旧 1-2 次）；T1 输出短、T4 无视频，粗估总 token ~2-3×，
  同 RPM 下墙钟 ~3-4×。90 RPM 上限本身不变。

---

## 10. 实施与验证

1. **prompt 冒烟**（`captions/_test/`）：对两个回归用例单独跑各轮 prompt——
   - 喝水片段（近处贴标签玻璃杯曾被认成布）：成功标准 = 词典里是诚实描述或正确"玻璃容器"，
     两轮一致，T2 动作引用词典标签；
   - `DAY1_A1_JAKE_11213000`（意图幻觉案例）：成功标准 = T2 输出零 处
     "准备/打算/想要/试图"式意图描述，动作在"放下包装盒/离开"处如实结束。
2. **DAY4 50 源 A/B**（旧单轮 vs 新多轮，`captions/_test/` 下隔离输出）：
   - 近场交互物体错误率（人工抽检，目标 ≈ 0）；
   - **T1 两轮分歧率**（仲裁触发率、近场交互物 votes=1/2 的比例）——这是 `--votes` 默认值
     的决策依据：近场交互物分歧率 >5% 则默认升固定 3，否则维持 2+仲裁；
   - T2 意图标记词密度（目标 ≈ 0，引语内除外）；
   - 各轮缓存命中率（`usage.cached_tokens / prompt_tokens`）——验证 §6.0 的 Plan A 布局
     是否真把视频前缀命中；若视频缓存全失，再启用 Plan B 多轮结构对比效果；
   - `new_objects` 命中率、词典锚定违例率、各轮拒绝率、token 成本，出 comparison_report。
3. **全量跑** + 抽样 QA（词典置信度分布、低置信词条清单人工过目）。

## 11. 开放问题 / 后续可选

- **k 默认 2 + 仲裁**；是否升固定 3 由 §10 冒烟实测的分歧率决定（阈值 5%），不拍脑袋。
- **ASR 专用引擎**：Mimo 转写质量不足时，T1 的转写部分替换为独立音频通道——官方
  `input_audio` 支持 MP3/WAV/FLAC/M4A/OGG（base64 ≤50MB、≈6.25 token/秒），可直接从切片抽
  M4A 送 MiMO 纯音频调用或本地 funasr/whisper，接口按可替换设计。
- **词典粒度**：默认按片段（10s）建词典，相邻片段同物会有多个 ID，由下游聚合合并；若跨片段
  一致性要求提高，可改按源视频（30s）建词典（3 个片段共享，多一个源级同步屏障）。
- **T2/T3 是否合并**：省一次视频调用，但违背"一轮一个问题"；A/B 后再评估。
- **mental_activity**：暂定自由短句；如后续需要可加 valence/arousal 数值字段。

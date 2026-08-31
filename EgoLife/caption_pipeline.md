# caption_pipeline — EgoLife 原子化结构化行为标注

> 用 Mimo-V2.5 全模态模型，把第一人称（Meta Aria）视频段标注成**面向数字孪生/模拟的原子化结构化 JSON**：
> 环境（setting + 近场物体 + background）/ 人物表 / 自身动作 / 他人动作 / 环境原子变化 / 语音 / 声音 / 界面（屏幕页面理解）/ 心理（emotion + `mental_activity`）/ 因果边 `causal_links`（带 `strength` 三档强度）。
> 单文件自包含，不依赖兄弟模块。
>
> **核心思想**：情绪驱动动作，动作改变环境，环境反过来影响动作与情绪——`causal_links` 显式记录这张**带强度权重**的有向因果图。
>
> **默认行为**：每个 ~30s 源视频被切成 3 段 ~10s 片段分别标注（颗粒度更细、首次拒绝率更低）；**thinking 默认关闭**（生成质量改由 best-of-N 采样 + critic 选优保障，关闭推理链每次调用便宜/快 ~4 倍；深度推理跑法用 `--thinking enabled`）。
> 用 `--clip-duration` 选择切分粒度（5/6/10/15/30）；30 = 不切分（一段源视频一个 caption，旧行为）。

---

## 1. 它做什么

输入：EgoLife 第一人称视频（参与者戴着 Meta Aria 眼镜录制），默认按 10s 切分。

输出：一个 JSON 对象，包含 10 个层：

| 层 | 字段 | 内容 | 时间戳 |
|---|---|---|---|
| **环境（整体）** | `environment` | `{setting, near_field[], background}`：整体场景说明 + **近场物体清单**（我/他人可能交互的东西，模拟器据此 spawn 道具）+ 其余背景一笔。near_field 的物体标签即全片段唯一命名（对象锚定，识别不了就如实描述外观，不猜类别）；场景说明而非变化日志 | ❌（整段） |
| **人物表** | `people` | 画面中出现的**其他人**花名册（不含自己）：`{id: "P1", descriptor, name, note}`；id 与 descriptor 全片段复用；`name` 仅当场说出/显示才填，不臆造 | ❌ |
| **自身行为** | `self_actions` | "我"做了什么；第一人称、现在时、动词开头，**原子动作**（一条=一个动词级步骤）；走路/转身/坐下/环顾等无宾语动作本身即合法原子动作，不硬凑宾语；只写可观察动作——意图只能进 psychology | ✅ 读视频**左上角两行水印**（第 1 行 `HH:MM:SS:FF`、第 2 行 `DAYn`），标 `HH:MM:SS-HH:MM:SS` 段（~1-3s 分辨率）；**右上角是佩戴者 id**（如 A1_JAKE），不作为场景文本转写；视频按 1fps 采样（含音轨），**按水印读时间、严禁数帧推算**，帧间跳变不臆补中间微步 |
| **他人行为** | `other_actions` | 其他人做了什么；同原子化标准，`person` 字段引用 people.id，文本以 descriptor 开头 | ✅ 同一时间轴 |
| **环境变化** | `env_changes` | 片段内可观察的**原子状态迁移**：物体出现/消失/移动、设备状态改变（屏幕亮/灭、门开、瓶盖拧开）、光线变化、人进出画面；`cause ∈ {self, other, external}` 归因（other 时文中点名人物 id） | ✅ 同一时间轴 |
| **语音** | `speech` | 关键话语；`lang` + 说话人（`self` 或 people.id，听不清时短描述）+ 引号文本 | ❌（音视不严格对齐，不强求） |
| **声音** | `sound` | 离散**非语音**声音：提示音、铃声、键盘、门响、脚步、笑声咳嗽等；`{time?, text, source}`——仅当时刻可被画面锚定（手机亮屏同时响）才标 time；持续环境音写进 environment | ⚠️ 可选（视觉锚定时才有 time） |
| **界面** | `interface` | **条件触发**：画面出现屏幕（手机/笔记本/电视…）或大量可读文字表面（白板/纸/招牌）才填。屏幕不止 OCR：识别 app/网站 + 页面类型 + 关键可见内容（bilibili 标题+UP主、知乎问题+回答、文件管理器路径文件名、聊天最新消息…）；内容中途变化时新增带 time 的条目。**不转写时间水印与右上角佩戴者 id。** | ✅（页面变化时标 time） |
| **心理** | `psychology` | `{emotion, mental_activity}`，仅凭本片段推断（当作唯一证据）；`mental_activity` 是**全 schema 唯一允许出现意图/计划**的字段（第一人称正在想什么、接下来要做什么），无线索时如实写"无特别线索" | 整段一个 |
| **因果边** | `causal_links` | 有向因果边 `{type, cause, effect, strength}`，8 种类型见下 | cause/effect 短语内嵌时间 |

### 原子化原则（取代数量配额）

不给"建议动作数"，判断标准是**原子性**，数量只是结果：

- 一个条目 = 一个动词级步骤（够、拿、放、开、关、看、走、坐），单一对象、单一即时目的。
- 复合行为必须拆解："拿起手机划开屏幕看消息" → `拿起手机` / `划开屏幕` / `浏览消息` 三条。
- 一个连续手势不硬拆成微帧；两个独立操作不合并成一条。
- 密集操作的 10s 出 6-10 条、静坐的 10s 出 1-2 条，都是对的——不凑数、不臆造。
- `env_changes` 同理：一条 = 一次离散状态迁移。

### 面向模拟的因果闭环（causal_links）

数字孪生/模拟需要的不只是"发生了什么"，而是**可执行的因果结构**：

```
emotion ──emotion->action──> self_actions ──action->env──> env_changes
   ▲   └──other->emotion──┐      ▲                            │
   │                      │      └──other->action── other_ ───┤
   │                      │                actions           │
   └────env->emotion──────┴───────────────────────────────────┘
   └────env->action───────────────────────────────────────────┘
   └────env->env（环境事件引发另一环境变化，如空调启动→纸张吹动）──┘
```

8 种边类型（`type` 字段取值，ASCII 写法）：

| type | 含义 | 例子 |
|---|---|---|
| `env->action` | 环境事件改变我的行为 | 手机震动 → 我拿起手机 |
| `env->emotion` | 环境事件改变我的情绪 | 巨响 → 受惊/烦躁 |
| `emotion->action` | 我的情绪直接驱动行为 | 无聊 → 开始刷手机 |
| `action->env` | 我的动作改变环境 | 我拨开关 → 灯亮；与 `cause="self"` 的 `env_changes` 互为镜像 |
| `other->action` | 他人动作触发我的动作 | 同事招手 → 我走过去 |
| `other->env` | 他人动作改变环境 | 她拉开窗帘 → 房间变亮 |
| `other->emotion` | 他人动作改变我的情绪 | 客人笑了 → 我放松 |
| `env->env` | 环境事件引发另一环境变化 | 空调启动 → 桌上纸张被吹动 |

每条边带**必填的 `strength`** 三档强度（反事实定义——"没有这个因，果还会发生吗"）：

| strength | 定义 | 例子 |
|---|---|---|
| **strong** | 触发：没有此因，果大概率不会发生（或走向不同方向） | 手机震动 → 我拿起手机；我拨开关 → 灯亮 |
| **moderate** | 塑形：此因改变了果的发生方式/时机/力度，但果本来也会发生 | 无聊 → 刷手机变快；对方语气 → 我回答更谨慎 |
| **weak** | 背景：只是多个促成因素之一，果主要由习惯/任务驱动 | 轻微疲惫 → 揉一次眼睛 |

物理性边（`action->env`、`other->env`）几乎总是 `strong`，属正常而非偷懒。诚实分级，不用弱背景边凑数。
**omit 的语义收紧为"怀疑这条边存在"**——真实但微弱的影响用 `weak` 如实记录，而不是丢弃；
存在性、强度两个维度就此分开。

各层在模拟中的角色：`self_actions` = agent 的控制输入（动作空间），`env_changes` = 环境状态转移（转移函数监督信号），`environment` = 要 spawn 的初始场景（含近场道具清单），`people` = 场景 NPC 花名册，`psychology` = 内部状态（含意图），`causal_links` = 因果图的**加权**边。

**有意的冗余**："我拿起杯子"（`self_actions`）与"杯子离开桌面进入我手中"（`env_changes`, `cause=self`）是同一事件的两个视角——一个是控制输入，一个是状态转移。模拟训练两边都要，所以两边都标。

**证据约束**：因果方向必须有可见时序 + 合理机制支撑，怀疑边**存在**则不标；真实但微弱的影响用 `weak` 记录。0-3 条是常态，空数组合法。

### 心理层与意图隔离（mental_activity）

`psychology` = `{emotion, mental_activity}`，**仅凭本片段可见情境推断**（把这段当作唯一证据，不脑补之前发生了什么）：

- `emotion`：一两个小写关键词（neutral/focused/amused/anxious…），读不出来就 `neutral`。
- `mental_activity`：第一人称一两句"此刻在想什么/在盘算什么"——是**全 schema 唯一允许出现意图与计划**的字段。动作、环境变化、因果边里严禁出现"准备/打算/想要/试图"（未观察到的目标）；意图是潜在状态，只住在这里。片段无线索时如实写"无特别线索，注意力在手头的事情上"，好过编造日程。

动作层（`self_actions`/`other_actions`）只写可观察的动作（姿态、接触、移动、设备使用）；环境事件/他人动了情绪或想法时，在此层说明原因，并同时记一条 `env->emotion` / `other->emotion` 因果边。

---

## 2. 安装

```bash
# 推荐：uv + 虚拟环境（依赖见 requirements.txt，含 pyarrow——clips.parquet 读写需要）
uv venv .venv
uv pip install -r requirements.txt
.\.venv\Scripts\activate        # Windows；Linux/macOS: source .venv/bin/activate

# 或者直接 pip
pip install -r requirements.txt
```

还需要：
- **ffmpeg / ffprobe** 在 PATH 中（用于视频重编码、duration 探测、recovery 切片）
- **Mimo API key**：在脚本目录或当前工作目录放一个 `.env` 文件，内容为 `MIMO_API_KEY=sk-...`；
  或 export 环境变量。脚本会依次查找 `--env-file`、`./.env`、脚本同目录 `.env`。

---

## 3. 快速开始

把视频按 `videos/{参与者}/DAY{天}/DAY{天}_{参与者}_{HHMMSScc}.mp4` 放好（与 EgoLife 原始命名一致），
然后：

```bash
# 标注一整天（默认切 10s 片段、JSON 输出、thinking 关闭）
python caption_pipeline.py --participant A1_JAKE --day 1 --max-rpm 90

# 深度推理跑法（每次调用带推理链，慢/贵 ~3-4 倍；建议配合较低的 --best-of-n）
python caption_pipeline.py --participant A1_JAKE --day 1 --thinking enabled --best-of-n 2

# 不切分：一段源视频（~30s）一个 caption（旧行为）
python caption_pipeline.py --participant A1_JAKE --day 1 --clip-duration 30

# 更细：切 15s（每 30s 源 → 2 段）/ 5s（→ 6 段）
python caption_pipeline.py --participant A1_JAKE --day 1 --clip-duration 15

# 只标注某段时间（HHMM）
python caption_pipeline.py --participant A1_JAKE --day 1 --start-time 1110 --end-time 1130

# 中断后续传（默认开启，跳过输出文件里已有的 clip）
python caption_pipeline.py --participant A1_JAKE --day 1

# 只跑 caption（slices 已在 _cache/ 里，跳过编码）
python caption_pipeline.py --skip-preprocess

# best-of-N：每个片段独立采样 4 次（每次调用带 call_index 序号），两阶段 critic
# （C1a 只看片的基线抽取 + C1b 评审）选出最优候选；best 不达标时按 critic 的
# regeneration_guidance 带反馈重生成（G2）再评一轮（C2b 复用 C1a 基线）
python caption_pipeline.py --participant A1_JAKE --day 1 --best-of-n 4 --refine-samples 2

# best-of-N 中断后续传：_candidates.jsonl 里已有的候选直接复用，
# 只补缺失的调用序号；旧单次模式的主记录会被采纳为候选 0
python caption_pipeline.py --best-of-n 4 --skip-existing
```

### 默认目录布局

```
ARIN7600/EgoLife/                   <- 脚本所在目录（ROOT）
├── caption_pipeline.py
├── caption_pipeline.md
├── .env                            <- MIMO_API_KEY=sk-...
├── videos/                         <- 输入视频（你放入，或用 download_video.py 拉取）
│   └── A1_JAKE/DAY1/*.mp4
├── EgoLifeCap/                     <- 官方 caption（用 download_egolifecap.py 拉取，非本脚本产物）
│   └── {DenseCaption|Transcript}/A1_JAKE/DAY1/*.srt
└── captions/                       <- 输出（自动生成，与 videos/ 结构对称）
    └── A1_JAKE/
        └── DAY1/
            ├── 1110-1130.jsonl              <- 主输出（文件名带时间范围）
            ├── 1110-1130_candidates.jsonl   <- best-of-N 候选 sidecar（N≥2 时生成：
            │                                  每行一个 (clip_id, call_index) 采样，
            │                                  断点续传按它复用已有标注）
            ├── 1110-1130_usage.jsonl        <- 每次调用的 token/延迟（带 stage/call_index）
            ├── 1110-1130_summary.json       <- 汇总统计
            ├── 1110-1130_run.log            <- 运行日志
            └── _cache/                      <- 中间产物（可删）
                ├── slices/*.mp4             <- 重编码后的片段
                └── clips.parquet            <- 片段元数据
```

不传 `--start-time/--end-time` 时，输出文件名为 `full.jsonl`。
不同时间范围的跑互不覆盖。`--src-dir` 指定视频树**根目录**（脚本自动找 `{participant}/DAY{day}/`），
`--out` 可覆盖输出路径。

---

## 4. 输出 schema

`{范围}.jsonl` 每行一个 `CaptionRecord`（strict JSONL，可被 `json.loads()` 逐行解析）：

```jsonc
{
  "clip_id": "DAY1_A1_JAKE_11100000_p2",   // 切分后带 _p{N} 后缀（见下）
  "global_idx": 5, "day": 1, "user": "A1_JAKE",
  "duration_s": 10.0,
  "narrative": "{\n  \"environment\": {...\n}",   // 模型原始 JSON 全文，备追溯
  "model": "mimo-v2.5",
  "ts_captioned": "2026-08-11T00:37:48Z",
  "slice_path": "_cache\\slices\\DAY1_A1_JAKE_11100000_p2.mp4",
  "tokens": {"in": 3200, "out": 210, "cached": 747},
  "environment": {
    "setting": "明亮的室内会议空间，长方形桌子铺着红白格子桌布",
    "near_field": ["黑色桌面麦克风（正前方）", "手机（白衣女生手中）"],
    "background": "背景有白板和补光灯"
  },
  "people": [
    {"id": "P1", "descriptor": "白衣女生", "name": "", "note": "坐我右手边"}
  ],
  "self_actions": [
    {"time": "11:10:15", "time_end": "11:10:17", "text": "我左手拿起桌上的黑色麦克风"},
    {"time": "11:10:17", "time_end": "11:10:20", "text": "我拨动开关开启麦克风"}
  ],
  "other_actions": [
    {"time": "11:10:12", "time_end": "11:10:15", "person": "P1", "text": "白衣女生双手握着手机低头点击屏幕"}
  ],
  "env_changes": [
    {"time": "11:10:15", "time_end": "11:10:17", "text": "黑色麦克风离开桌面进入我左手", "cause": "self"},
    {"time": "11:10:17", "time_end": "11:10:18", "text": "麦克风电源接通", "cause": "self"}
  ],
  "speech": [
    {"lang": "zh", "speaker": "self", "text": "都戳一下，每人戳一下。"}
  ],
  "sound": [],
  "interface": [],
  "psychology": {
    "emotion": "focused",
    "mental_activity": "作为协调者引导设备操作测试，在想接下来让每人试一遍。"
  },
  "causal_links": [
    {"type": "action->env", "cause": "11:10:17 我拨动开关", "effect": "11:10:18 麦克风电源接通", "strength": "strong"}
  ],
  "clip_kind": "10s",              // "30s" | "segment_open" | "10s" | "15s" | "6s" | "5s"
  "recovery": "ok"                // "ok" | "10s_slices"
}
```

### clip_id 切分命名

源视频 `DAY1_A1_JAKE_11100000.mp4`（30s）在 `--clip-duration 10` 下切成 3 段：

| 片段 | clip_id | slice 文件名 | start_hms（水印起始） |
|---|---|---|---|
| 第 1 段 (0-10s) | `DAY1_A1_JAKE_11100000_p1` | `..._p1.mp4` | `11:10:00` |
| 第 2 段 (10-20s) | `DAY1_A1_JAKE_11100000_p2` | `..._p2.mp4` | `11:10:10` |
| 第 3 段 (20-30s) | `DAY1_A1_JAKE_11100000_p3` | `..._p3.mp4` | `11:10:20` |

`_p{N}` 后缀只在切分时（N>1）出现；不切分的片段（`--clip-duration 30` 或源视频 ≤ 目标时长）保持原 clip_id 无后缀。

### 字段说明
- `environment`：`{setting, near_field[], background}`，模拟器放置 agent 的场景上下文；near_field 列我/他人可能交互的近场物体（带粗略方位），标签即全片段唯一命名——同一物体在各字段中逐字复用，识别不自信时如实描述外观属性而不猜类别。片段内的变化不写在这里（变化进 `env_changes`）。
- `people`：`[{id, descriptor, name, note}]`，画面中其他人的花名册（不含自己）；`id`（"P1"…）与 `descriptor` 全片段复用，`name` 仅当场说出/显示才填。独自一人时为 `[]`。
- `self_actions` / `other_actions`：`[{time, time_end, (person,) text}]`，`time` 为水印时间戳；无时间戳的行 `time=""` 但仍保留。**原子化**：一条 = 一个动词级步骤，复合行为拆开，无数量配额；走路/转身/坐下/环顾等位移与姿态动作无需宾语，不硬凑。`other_actions.person` 引用 `people.id`；只写可观察动作——意图只能进 `psychology.mental_activity`。
- `env_changes`：`[{time, time_end, text, cause}]`，片段内每次可观察的原子状态迁移；`cause ∈ {self, other, external}`（`other` 时文中点名人物 id；`external` = 无可见行为主体：自动门、天气、定时器）。
- `speech`：`[{lang, speaker, text}]`，`lang ∈ {zh, en, ""}`；`speaker` 为 `self` 或 `people.id`（听不清时用短描述）。
- `sound`：`[{time, text, source}]`，离散非语音声音；`time` 仅在时刻可被画面锚定时才有（否则 `""`），持续环境音属于 `environment.setting`。
- `interface`：`[{time, where, app_or_site, content, note}]`，条件触发层：屏幕做**页面级理解**（app/网站 + 页面类型 + 关键可见内容逐字），实体文字表面做转写；内容中途变化新增条目。无屏幕/文字表面时为 `[]`。**不转写时间水印与右上角佩戴者 id。**
- `psychology`：`{emotion, mental_activity}`，仅凭本片段推断（当唯一证据）；`mental_activity`（第一人称一两句）是全 schema **唯一**允许出现意图/计划的字段，缺失线索时如实写"无特别线索"而非编造。环境/他人动了情绪时，同时在 `causal_links` 记 `env->emotion` / `other->emotion` 边。
- `causal_links`：`[{type, cause, effect, strength}]`，`type` 为 8 种边之一（见第 1 节表）；`cause`/`effect` 为带时间的短语，指向具体的原子动作/变化；`strength ∈ {strong, moderate, weak}`（反事实三档，见第 1 节；模型漏写或写非法值时宽松归一化为 `""`，不整单拒绝）；空数组合法（无可辩护的因果边时）。
- `narrative`：模型原始 JSON 全文（未解析），保完整以便人工核查/重新解析。
- `clip_kind`：`30s`（不切分的标准源）/ `segment_open`（一天开头的不规则短段，如 `11094208`）/ `10s`/`15s`/`6s`/`5s`（切分片段）。`60s`（合并）已移除。
- `recovery`：标注此 clip 经历的路径——`ok`（一次成功）/ `10s_slices`（仅 `--clip-duration 30` 下，被拒后切成 10s 重试）。

> **schema 沿革**：2026-08 大重构后统一为上（`environment` 对象 + `people`/`sound`/`interface`/`mental_activity`；移除 `tags`/`ocr`/`others`/`awareness`；sectioned 兜底解析已删除，JSON 失败即 `parse_failed`）。历史输出文件里的行保持生成时的旧形状，解析器**不再**做旧→新映射；重跑后所有新行均为新形状。下游消费 jsonl 时请按行判断字段存在性。

### best-of-N / critic 扩展字段（N≥2 时）

所有行新增三个可回溯字段（N=1 的旧行没有，按字段存在性判断）：

- `best_of_n`：本次运行每片段采样数（旧单次模式为 1）。
- `thinking`：本次运行的 thinking 配置（`enabled`/`disabled`）。
- `critic`：critic 子对象（N=1 时为 `null`）：
  - 正常评审后：`{enabled, baseline: {verification_baseline, tokens}, candidates: [{call_index, weighted_total}], evaluation: {baseline_corrections, best_index, needs_regeneration, regeneration_guidance, tokens}, refined}`；
  - 只有 1 份有效候选：`{enabled, skipped: "single_valid"}`（跳过 critic 直接保留）；
  - 基线/评审失败：`{enabled, failed: "baseline_failed" | "critic_failed"}`（保留首份有效候选）。

**候选 sidecar `{stem}_candidates.jsonl`**（每行一个采样）：

```jsonc
{"clip_id": "DAY1_A1_JAKE_11100000_p1", "global_idx": 1, "call_index": 0,
 "stage": "generate", "narrative": "{模型原始 JSON 全文}", "model": "mimo-v2.5",
 "ts_captioned": "...", "tokens": {"in": 3200, "out": 210, "cached": 747}, "recovery": "ok"}
```

- `stage`：`generate`（G1 采样）/ `refine`（G2 重生成，仅追溯，续传不复用）。
- 每次 API 调用（生成/重生成/基线/评审）都会写一条 `UsageRecord` 到 `_usage.jsonl`，
  并带 `stage`（generate/refine/baseline/critic）与 `call_index`（生成类调用 0-based 序号，其余为 -1）。

**断点续传语义（best-of-N）**：完成判定是模式感知的——只有"最后一条主记录满足当前
`best_of_n` + `thinking` + 含 `critic` 标记"才算完成。变更 `--best-of-n` 或 `--thinking`
会使旧最终记录失效而重跑，但 **sidecar 候选照常复用**（只补缺失序号），旧单次记录会被
**采纳为候选 0**。N=1 时任何已有记录都算完成（旧行为）。

---

## 5. CLI 参数全表

| 参数 | 默认 | 说明 |
|---|---|---|
| **选择** | | |
| `--participant` | `A1_JAKE` | 参与者名 |
| `--day` | `1` | 天（1-7） |
| `--start-time` | None | 起始 HHMM（含），如 `1110` |
| `--end-time` | None | 结束 HHMM（不含），如 `1130` |
| `--clip-duration` | `10` | 切分目标秒数，可选 `5`/`6`/`10`/`15`/`30`（须整除 30）。每个源视频切成 `ceil(dur/target)` 段。`30` = 不切分（一段源视频一个 caption，旧行为）。默认 `10`（3 段/源，颗粒度细、首次拒绝率低） |
| **路径** | | |
| `--src-dir` | `./videos` | 视频树的**根目录**，脚本自动往下找 `{participant}/DAY{day}/` |
| `--out` | `./captions/{participant}/DAY{day}/{start}-{end}.jsonl` | 输出文件 |
| **编码** | | |
| `--resolution` | `1024` | 重编码分辨率（1024 为水印可读/白板可辨的验证最优点） |
| `--fps` | `1` | 采样帧率（1 fps 与目标部署环境一致：EgoLife 之外每秒一图） |
| `--crf` | `28` | 质量 |
| `--audio-k` | `64` | 音频码率 kbps |
| **模型** | | |
| `--model` | `mimo-v2.5` | |
| `--base-url` | `https://api.xiaomimimo.com/v1` | |
| `--api-key-env` | `MIMO_API_KEY` | 环境变量名 |
| `--env-file` | None | `.env` 文件路径（默认搜索 CWD + 脚本目录） |
| `--thinking` | `disabled` | 推理模式。默认关——生成质量由 best-of-N 采样 + critic 选优保障，关掉推理链每次调用便宜/快 ~4 倍。`enabled` 用于深度推理跑法（建议同时调低 `--best-of-n`）。thinking 下 Mimo 忽略 temperature |
| `--no-json-mode` | off | 关闭 `response_format=json_object`（debug） |
| **并发** | | |
| `--max-rpm` | `90` | 全局 RPM 上限（Mimo 硬限 100，留余量） |
| `--api-workers` | `min(--max-rpm/2, 20)`（默认 rpm=90 时即 20） | API worker 线程。thinking 单请求 ~30s+，每个 worker 约每分钟 2 发；随 `--max-rpm` 增长但 **20 封顶**，防止单机连接/流量堆太大 |
| `--preprocess-workers` | `2` | ffmpeg 编码线程 |
| **流程** | | |
| `--skip-preprocess` | off | 只 caption，读 `_cache/` 里已有的 slices |
| `--skip-existing` / `--no-skip-existing` | on | 跳过输出文件里已有的 clip（断点续传） |
| `--reset` | off | 删除输出文件后重跑 |
| `--limit` | None | 只处理前 N 个 clip（调试） |
| **best-of-N** | | |
| `--best-of-n` | `1` | 每片段采样次数。`1` = 旧单次调用（无 critic）。`N≥2` 跑 best-of-N：G1 采样 N 次（每次带 call_index 序号）→ 两阶段 critic（C1a 只看片的基线抽取 + C1b 评审选优）→ best 不达标时带 `regeneration_guidance` 重生成（G2）→ C2b 复用 C1a 基线再评。续传复用 sidecar 候选、只补缺失序号；旧单次主记录采纳为候选 0。critic 需 ≥2 份有效候选；1 份直接保留，0 份走现有失败/10s 兜底 |
| `--refine-samples` | `2` | G2 重生成采样数（仅 critic 判 `needs_regeneration` 时触发；0 = 关闭重生成） |

---

## 6. 架构

producer-consumer + 全局 RPM 限速器，非阻塞：

```
[PreprocessWorker 池]       [RateLimiter]      [API Worker 池]      [Writer]
 ffmpeg 切分/重编码   ->    produced_q   ->   acquire()  ->  调 Mimo  ->  result_q  ->  按 global_idx 有序写 jsonl
  (2 线程, CPU)              (有界队列)         (90 RPM 滑窗)     (6 线程, IO)            (单线程, 堆缓冲保序)
```

- **切分与 API 重叠**：API worker 阻塞等模型响应时，preprocess worker 在跑下一个源视频的 ffmpeg 切分。
  每个源视频产出 N 个片段（`ceil(dur/target_s)`），各片段独立 caption、各有自己的 `global_idx`。
- **全局限速**：所有 API 调用（正常 + retry + recovery）共享一个 `SlidingWindowRateLimiter`，
  硬性卡在 `--max-rpm`，绝不超 Mimo 配额。
- **有序落盘**：API worker 乱序完成 → `OrderedWriter` 用堆缓冲，按 `global_idx` 严格有序写，
  输出确定性、resume 友好。
- **两级 recovery（按模式不同）**：
  - `--clip-duration 30`（不切分）：30s 被拒 → retry 一次 → 切 10s 各调一次 → 放弃（旧的三级链路）。
  - `--clip-duration < 30`（切分模式，默认）：片段被拒 → retry 一次 → 放弃（已是最小粒度，不再细切）。
  所有 recovery 调用都过同一限速器。
- **断点续传（模式感知）**：默认开。N=1 时读输出文件跳过已完成（旧行为）；best-of-N 时按
  "最后一条主记录满足当前 `best_of_n` + `thinking` + 含 `critic` 标记"判定完成，sidecar 候选
  复用、只补缺失序号（见 §4 与 §6 的 best-of-N 小节）。切分阶段也跳过已存在 slice。

### 共享系统提示词与前缀缓存（SHARED_SYSTEM_MSG）

四类调用（G1/G2 生成、C1a 基线、C1b/C2b 评审、P2 推理）共用**同一份系统提示词**：公共段
（角色定位、素材背景、采样与水印规则）只写一遍，四个任务各自的输出 schema 与字段规则作为
`TASK: BASIC / BASELINE / CRITIQUE / INFER` 四个并列章节收进同一份 prompt，末尾是 Routing
硬约束（只输出被点名任务的 JSON）；具体执行哪个任务由 user 消息**首行的 `TASK: XXX` 标记**路由。

消息结构固定为 `[system, user(video, text)]` 且 video part 在 text part 之前，因此同一 clip
的全部调用（N 次采样 + 基线 + 评审 + 可能的 G2/C2b + 推理，约 8~10 次）共享逐字节一致的
"system + video" 前缀——除该 clip 的首次调用外，视频 token（成本大头）全部命中服务商前缀
缓存，每次调用只有末尾的任务文本是新的。为此系统提示词里**严禁出现 clip 特定内容**（会破坏
共享前缀）；命中率可用 `_usage.jsonl` 里的 `cached_tokens` 验证。

### 错误分类（summary.json 里分项计数）

不再笼统标 "rejected"，而是区分：
- `safety_rejection`：Mimo 安全过滤器拒绝（≤25 token 且非 JSON 对象开头）
- `parse_failed`：JSON 无效或 schema 校验失败（无兜底解析，直接失败）
- `empty`：模型返回空
- `first_attempt_rejection`：首次被拒（safety/parse/empty），已自动 retry——Mimo 的常见行为，retry 命中前缀缓存（1/50 价）几乎必成。

失败时原始模型输出存进 `UsageRecord.raw_content`（summary 的 `round_breakdown` 里也有前 200 字预览），便于事后排查。

### best-of-N + 两阶段 critic 工作流（`--best-of-n N≥2`）

每个 clip 在 API worker 内跑一个状态机（G1 串行采样，与其他 clip 的 worker 并行）：

```
G1  补齐 call_index 0..N-1：复用 sidecar 候选 + 采纳旧主记录为候选 0，
    只对缺失序号发生成调用（坏 JSON 丢弃；首次被拒自动 retry 一次）
        │ 有效候选 = 0 → 现有失败/10s-slices 兜底（兜底切片按单次调用）
        │ 有效候选 = 1 → 直接保留（critic.skipped = "single_valid"）
        ▼ 有效候选 ≥ 2
C1a  只看片基线抽取（视频 + 元数据，不见候选）→ verification_baseline
        │ 基线失败 → 保留首份有效候选（critic.failed = "baseline_failed"）
        ▼
C1b  评审（视频 + 基线 + 全部候选）→ best_index / 分数 / needs_regeneration
        │ 评审失败 → 保留首份有效候选（critic.failed = "critic_failed"）
        ▼ best 不达标（needs_regeneration）
G2   带 regeneration_guidance 重生成 refine_samples 次（stage="refine"）
        ▼
C2b  候选池 = {C1b best} ∪ {G2 样本}，复用 C1a 基线再评，取最终 best
     （无论分数如何收尾；C2b 失败则保留 C1b best）
```

要点：

- **每次调用带序号**：候选落 `{stem}_candidates.jsonl`（键 = `(clip_id, call_index)`），
  每次 API 调用在 `_usage.jsonl` 里带 `stage` + `call_index`，summary 有分阶段调用计数。
- **两阶段 critic 抗锚定**：C1a 只发视频，critic 先建立独立基线再读候选；
  C1b 把基线当 index（非 closed world），候选里基线没提到的内容需回看片段再判。
- **断点续传（模式感知）**：完成判定 = 最后一条主记录满足当前 `best_of_n` + `thinking`
  且含 `critic` 标记；不满足则重跑该 clip，但 sidecar 候选全部复用（只补缺失序号），
  旧单次记录采纳为候选 0。变更 `--best-of-n`/`--thinking` 即触发这种"复用式重跑"。
- **成本**：每片段调用数 ≈ `N + 2`（G1 + C1a + C1b），低分触发再 `+ M + 1`（G2 + C2b）。

---

## 7. 已知限制

- **60s 合并模式已移除**：实验显示模型处理 60s 视频时只描述前 5-6 秒（三组实验全部如此，非 token 限制）。
  `--clip-duration 60` 和 `concat_two_clips` 已删除。如未来找到让模型"看全"的 prompt 技巧，可考虑重新引入。
- **thinking 与质量的权衡**：原子分解 + 因果推断是推理密集任务，`--thinking enabled` 时
  单次延迟/token 约 ×3-4（reasoning 占大头）。默认 `disabled`——生成质量改由 best-of-N
  采样 + critic 选优保障；需要更深推理时用 `--thinking enabled` 并相应调低 `--best-of-n`。
- **causal_links 是模型推断**：可能有假阳性/假阴性。prompt 已强制"可见时序 + 合理机制才标，怀疑边存在才省略"，
  且 `cause`/`effect` 要求引用具体动作/变化（带时间），下游可按证据强度过滤。`strength` 是模型的主观三档
  判断（反事实定义），聚合时建议按 strong/moderate/weak → 1.0/0.5/0.25 加权，或仅取 strong 边做硬约束。
- **JSON 模式偶发 code-fence**：极少数情况模型无视 `json_object` 模式给输出套 ` ```json ` 围栏——
  解析器会剥围栏后照常解析；模型完全脱轨输出非 JSON 文本时直接 `parse_failed`（原始输出留在
  `raw_content`，retry 一次通常即恢复）。
- **单 participant 单 day**：跨天/跨参与者需起多个进程，注意共享 RPM 配额（调低单进程 `--max-rpm`）。
- **best-of-N 成本高**：每片段约 `N+2` 次调用（低分再 `+M+1`），即 `--best-of-n 4` 约是单次的 6 倍
  （含 G2 最坏 9 倍）。视频虽随每次调用发送，但共享系统提示词下 "system + video" 前缀逐字节一致，
  除首次外均命中前缀缓存（见 §6），边际成本主要是各次不同的任务文本与 completion。
  省钱：`N=3`、`--refine-samples 0`。
- **thinking 下候选可能趋同**：MiMo 在 thinking 模式忽略 temperature，N 份 G1 候选可能高度相似，
  选优退化为"选第一份"。可观察候选间编辑距离 / critic 分数方差，必要时 G1 一半样本关 thinking
  （temperature=1.0）换取多样性（详见 `tmp/best_of_n_critic方案计划_v1.md`）。
- **兜底切片与 G2 候选**：30s 模式全失败后的 10s-slices 兜底按**单次调用**标注（不套 best-of-N/critic）；
  G2 的 refine 候选只做追溯，中断后续传不复用（会重跑 G2）。
- **critic 与生成同源**：critic 默认与生成同模型，可能对自己的典型幻觉不敏感——两阶段结构
  （C1a 不见候选）与 `baseline_corrections` 显式纠正是主要缓解，必要时换异构模型做 critic。

---

## 8. 配置建议

| 场景 | 推荐参数 |
|---|---|
| 标准跑（质量/成本平衡） | 默认即可（`--max-rpm 90`，api-workers 自动 = min(rpm/2, 20) = 20；thinking off + best-of-N critic，10s 切分。RPM 用不满 90 属预期，单机不为吃满配额堆连接） |
| 深度推理（因果标注质量优先） | `--thinking enabled --best-of-n 2`（推理链 ×3-4 成本，用较低 N 对冲） |
| 大批量、预算敏感 | `--max-rpm 95 --api-workers 8 --best-of-n 1`（单次调用无 critic，留意 429） |
| best-of-N 幻觉压制 | `--best-of-n 4 --refine-samples 2` | 每片段约 6–9 次调用；建议先 `--limit` 小样本核对 critic 选优与 `verification_baseline` 质量再全量跑 |
| 调试 prompt | `--limit 3 --api-workers 1`（串行，便于看日志） |

**吞吐估算**：RPM 上限不变（90 RPM ≈ 5400 clip/h 的**调用数**上限）；thinking 开启时单次延迟 ×3-4、
completion tokens ×~5-8（默认关闭）。DAY1 ~828 段（~30s 源）在
`--clip-duration 10` 下 ≈ 2484 片段。切分在 API 等待间隙并行完成，不是瓶颈。
共享系统提示词下，同一 clip 的所有调用（生成/基线/评审/推理）命中同一 "system + video"
前缀缓存（旧版分系统提示词时实测首次缓存命中 ~73%；新版前缀跨全部调用共享，命中率应
更高，以 `_usage.jsonl` 的 `cached_tokens` 实测为准）。

---

## 9. 切分粒度选择参考

基于 DAY4 50 源（148 片段）实测对比 DAY2/3（30s）：

| 指标 | 10s（默认） | 30s（不切分） |
|---|---|---|
| 首次拒绝率 | **26.4%** | 52-63.5% |
| self_actions / 30s 等效 | **~8** | ~4.3 |
| others / 30s 等效 | ~0.8 | 2.4（跨片段交互更全） |
| speech / 30s 等效 | ~1.0 | 4.1（30s 捕获更多话语） |
| OCR 命中 | 10.1%（新字段） | 0%（字段不存在） |
| 缓存命中率（首次） | 73% | 49% |

**结论**：默认 10s 在动作密度和拒绝率上明显占优；30s 在跨片段交互/语音上下文上更全。
交互密集场景（会议、对话）可考虑 `--clip-duration 30` 或下游聚合相邻 10s caption。

> 注意：表中数量类指标（self_actions / 30s 等效等）基于**旧版 prompt**（数量建议制、无
> 原子化要求）实测，仅供切分粒度对比参考；原子化 + causal_links 新版 prompt 下动作数
> 会系统性偏高（复合行为被拆解），拒绝率也可能随输出变长而变化，待重新实测。
> 详见 `captions/_test/DAY4_10s/comparison_report.md`。

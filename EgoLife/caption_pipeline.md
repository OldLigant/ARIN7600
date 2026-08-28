# caption_pipeline — EgoLife 原子化结构化行为标注

> 用 Mimo-V2.5 全模态模型，把第一人称（Meta Aria）视频段标注成**面向数字孪生/模拟的原子化结构化 JSON**：
> 自身动作 / 他人动作 / 整体环境 / 环境原子变化 / 语音 / 心理（`awareness` 情绪-行为驱动力评级）/ 因果边 `causal_links`（带 `strength` 三档强度）/ OCR（可选）。
> 单文件自包含，不依赖兄弟模块。
>
> **核心思想**：情绪驱动动作，动作改变环境，环境反过来影响动作与情绪——`causal_links` 显式记录这张**带强度权重**的有向因果图。
>
> **默认行为**：每个 ~30s 源视频被切成 3 段 ~10s 片段分别标注（颗粒度更细、首次拒绝率更低）；**thinking 默认开启**（因果推断是推理密集任务）。
> 用 `--clip-duration` 选择切分粒度（5/6/10/15/30）；30 = 不切分（一段源视频一个 caption，旧行为）。

---

## 1. 它做什么

输入：EgoLife 第一人称视频（参与者戴着 Meta Aria 眼镜录制），默认按 10s 切分。

输出：一个 JSON 对象，包含 8 个层：

| 层 | 字段 | 内容 | 时间戳 |
|---|---|---|---|
| **自身行为** | `self_actions` | "我"做了什么；第一人称、现在时、动词开头，**原子动作**（一条=一个动词级步骤） | ✅ 读视频右上角水印 `HH:MM:SS:FF`，标 `HH:MM:SS-HH:MM:SS` 段（~1-3s 分辨率） |
| **他人行为** | `others` | 其他人做了什么；中立称呼（"穿粉色上衣的女性"），不臆造名字；同样原子化 | ✅ 同一时间轴 |
| **环境（整体）** | `environment` | 整体场景背景：地点类型、布局、光照、天气/室内外——模拟器放置 agent 的上下文，是"场景说明"而非"变化日志" | ❌（整段一两句） |
| **环境变化** | `env_changes` | 片段内可观察的**原子状态迁移**：物体出现/消失/移动、设备状态改变（屏幕亮/灭、门开、瓶盖拧开）、光线变化、人进出画面；`cause ∈ {self, other, external}` 归因 | ✅ 同一时间轴 |
| **语音** | `speech` | 关键话语；`lang` + 说话人 + 引号文本 | ❌（模型音视不严格对齐，不强求） |
| **心理** | `psychology` | `awareness` 等级 + `emotion` 分类 + 可选 `note` | 整段一个 |
| **因果边** | `causal_links` | 有向因果边 `{type, cause, effect, strength}`，7 种类型见下 | cause/effect 短语内嵌时间 |
| **OCR（可选）** | `ocr` | 纸/白板/屏幕/招牌上**大量可读文字**时才填；`{where, text, note}`。不填默认 `[]`。**不 OCR 水印。** | ❌ |

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
   │                      │      └──other->action── others ───┤
   └────env->emotion──────┴───────────────────────────────────┘
   └────env->action───────────────────────────────────────────┘
```

7 种边类型（`type` 字段取值，ASCII 写法）：

| type | 含义 | 例子 |
|---|---|---|
| `env->action` | 环境事件改变我的行为 | 手机震动 → 我拿起手机 |
| `env->emotion` | 环境事件改变我的情绪 | 巨响 → 受惊/烦躁 |
| `emotion->action` | 我的情绪直接驱动行为 | 无聊 → 开始刷手机；须与 `awareness` 同档一致（同一阶梯） |
| `action->env` | 我的动作改变环境 | 我拨开关 → 灯亮；与 `cause="self"` 的 `env_changes` 互为镜像 |
| `other->action` | 他人动作触发我的动作 | 同事招手 → 我走过去 |
| `other->env` | 他人动作改变环境 | 她拉开窗帘 → 房间变亮 |
| `other->emotion` | 他人动作改变我的情绪 | 客人笑了 → 我放松 |

每条边带**必填的 `strength`** 三档强度（反事实定义——"没有这个因，果还会发生吗"）：

| strength | 定义 | 例子 |
|---|---|---|
| **strong** | 触发：没有此因，果大概率不会发生（或走向不同方向） | 手机震动 → 我拿起手机；我拨开关 → 灯亮 |
| **moderate** | 塑形：此因改变了果的发生方式/时机/力度，但果本来也会发生 | 无聊 → 刷手机变快；对方语气 → 我回答更谨慎 |
| **weak** | 背景：只是多个促成因素之一，果主要由习惯/任务驱动 | 轻微疲惫 → 揉一次眼睛 |

物理性边（`action->env`、`other->env`）几乎总是 `strong`，属正常而非偷懒。诚实分级，不用弱背景边凑数。
**omit 的语义收紧为"怀疑这条边存在"**——真实但微弱的影响用 `weak` 如实记录，而不是丢弃；
存在性、强度两个维度就此分开。

各层在模拟中的角色：`self_actions` = agent 的控制输入（动作空间），`env_changes` = 环境状态转移（转移函数监督信号），`environment` = 初始场景上下文，`emotion` = 内部状态，`causal_links` = 因果图的**加权**边，`awareness` = emotion→action 边的强度摘要（与 `strength` 同一阶梯）。

**有意的冗余**："我拿起杯子"（`self_actions`）与"杯子离开桌面进入我手中"（`env_changes`, `cause=self`）是同一事件的两个视角——一个是控制输入，一个是状态转移。模拟训练两边都要，所以两边都标。

**证据约束**：因果方向必须有可见时序 + 合理机制支撑，怀疑边**存在**则不标；真实但微弱的影响用 `weak` 记录。0-3 条是常态，空数组合法。

### awareness 三档定义（核心）

`awareness` 衡量**情绪在多大程度上因果地驱动了可见行为**——它就是 emotion→action 边的
`strength` 摘要，同一阶梯（low/medium/high ↔ weak/moderate/strong）：

| 等级 | 定义 | 例子 |
|---|---|---|
| **low** | 情绪是背景色；行为由习惯/任务驱动，情绪没改变发生的事 | 放松地刷手机——"放松"在，但没改变"刷手机"这件事 |
| **medium** | 情绪影响了动作的**表现形式**（语气/节奏/表情/幅度），但没改变**方向** | 开心地看着别人写字，笑得前仰后合，但还坐在原地看 |
| **high** | 情绪**直接触发**了一个行为反应或转折 | 尴尬→伸手遮脸；生气→拍桌；焦虑→放下手头事起身离开 |

`low/medium` 段是"情绪背景"，`high` 段是"情绪-行为转换点"——后者是我们要捕捉的核心研究对象。
`awareness` 必须与任何写出的 `emotion->action` 边同档一致；边太弱不值得单写一条时 awareness
照填（`low` 且无边是正常的）。管线在落盘前做一致性校验，发现 awareness 低估了边强度时打 warning。

---

## 2. 安装

```bash
pip install openai pandas python-dotenv jsonschema
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
# 标注一整天（默认切 10s 片段、JSON 输出、thinking 开启）
python caption_pipeline.py --participant A1_JAKE --day 1 --max-rpm 90

# 省成本/提速 ~4 倍（无推理链；动作/环境层照常，causal_links 质量明显下降）
python caption_pipeline.py --participant A1_JAKE --day 1 --thinking disabled

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
            ├── 1110-1130_usage.jsonl        <- 每次调用的 token/延迟
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
  "narrative": "{\n  \"self_actions\": [...\n}",   // 模型原始 JSON 全文，备追溯
  "tags": ["meeting", "smartphone", "microphone", "discussion"],
  "model": "mimo-v2.5",
  "ts_captioned": "2026-08-11T00:37:48Z",
  "slice_path": "_cache\\slices\\DAY1_A1_JAKE_11100000_p2.mp4",
  "tokens": {"in": 3200, "out": 210, "cached": 747},
  "self_actions": [
    {"time": "11:10:10", "time_end": "11:10:12", "text": "我接回手机"},
    {"time": "11:10:12", "time_end": "11:10:15", "text": "我低头看女生们依次点击屏幕"},
    {"time": "11:10:15", "time_end": "11:10:17", "text": "我左手拿起桌上的黑色麦克风"},
    {"time": "11:10:17", "time_end": "11:10:20", "text": "我拨动开关开启麦克风"}
  ],
  "others": [
    {"time": "11:10:12", "time_end": "11:10:15", "text": "白衣女生双手握着手机低头点击屏幕"}
  ],
  "environment": "明亮的室内会议空间，长方形桌子铺着红白格子桌布，上有笔记本、手机、收纳包。背景有白板和补光灯。",
  "env_changes": [
    {"time": "11:10:10", "time_end": "11:10:11", "text": "手机从女生手中回到我手里", "cause": "other"},
    {"time": "11:10:15", "time_end": "11:10:17", "text": "黑色麦克风离开桌面进入我左手", "cause": "self"},
    {"time": "11:10:17", "time_end": "11:10:18", "text": "麦克风电源接通", "cause": "self"}
  ],
  "speech": [
    {"lang": "zh", "speaker": "A1_JAKE", "text": "都戳一下，每人戳一下。"}
  ],
  "psychology": {
    "awareness": "medium",
    "emotion": "focused, neutral",
    "note": "作为协调者引导设备操作测试。"
  },
  "causal_links": [
    {"type": "other->action", "cause": "11:10:12 白衣女生轮流点击手机屏幕测试", "effect": "11:10:15 我拿起麦克风准备录音", "strength": "moderate"},
    {"type": "action->env", "cause": "11:10:17 我拨动开关", "effect": "11:10:18 麦克风电源接通", "strength": "strong"}
  ],
  "ocr": [
    {"where": "whiteboard", "text": "日程安排\n1. 站会", "note": "handwriting"}
  ],
  "clip_kind": "10s",              // "30s" | "segment_open" | "10s" | "15s" | "6s" | "5s"
  "output_format": "json",        // "json" | "fallback_sectioned"
  "recovery": "ok"                // "ok" | "fallback_sectioned" | "10s_slices"
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
- `self_actions` / `others`：`[{time, time_end, text}]`，`time` 为水印时间戳；无时间戳的行 `time=""` 但仍保留。**原子化**：一条 = 一个动词级步骤，复合行为拆开，无数量配额。
- `environment`：整段一两句的**整体场景**（模拟器放置 agent 的上下文）；片段内的变化不写在这里。
- `env_changes`：`[{time, time_end, text, cause}]`，片段内每次可观察的原子状态迁移；`cause ∈ {self, other, external}`（`external` = 无可见行为主体：自动门、天气、定时器）。
- `speech`：`[{lang, speaker, text}]`，`lang ∈ {zh, en, ""}`。
- `psychology`：`{awareness, emotion, note}`，缺失字段为空串；`awareness` 为 emotion→action 边的强度摘要（与 `strength` 同梯：low/medium/high ↔ weak/moderate/strong）。
- `causal_links`：`[{type, cause, effect, strength}]`，`type` 为 7 种边之一（见第 1 节表）；`cause`/`effect` 为带时间的短语，指向具体的原子动作/变化；`strength ∈ {strong, moderate, weak}`（反事实三档，见第 1 节；模型漏写或写非法值时宽松归一化为 `""`，不整单拒绝）；空数组合法（无可辩护的因果边时）。
- `ocr`：`[{where, text, note}]`，**可选层**。`where ∈ {whiteboard, paper, screen, sign, other}`；`text` 为可读内容（尽量逐字）；`note` 可选（如"手写不清晰"）。**不 OCR 时间水印**。无可读文字表面时为 `[]`。
- `narrative`：模型原始 JSON 全文（未解析），保完整以便人工核查/重新解析。
- `clip_kind`：`30s`（不切分的标准源）/ `segment_open`（一天开头的不规则短段，如 `11094208`）/ `10s`/`15s`/`6s`/`5s`（切分片段）。`60s`（合并）已移除。
- `output_format`：`json`（主路径）/ `fallback_sectioned`（模型无视 json_object 模式时，回退到 sectioned 解析）。
- `recovery`：标注此 clip 经历的路径——`ok`（一次成功）/ `fallback_sectioned` / `10s_slices`（仅 `--clip-duration 30` 下，被拒后切成 10s 重试）。

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
| `--fps` | `2` | 采样帧率 |
| `--crf` | `28` | 质量 |
| `--audio-k` | `64` | 音频码率 kbps |
| **模型** | | |
| `--model` | `mimo-v2.5` | |
| `--base-url` | `https://api.xiaomimimo.com/v1` | |
| `--api-key-env` | `MIMO_API_KEY` | 环境变量名 |
| `--env-file` | None | `.env` 文件路径（默认搜索 CWD + 脚本目录） |
| `--thinking` | `enabled` | 推理模式。默认开——原子分解 + 状态追踪 + 因果推断都是推理密集任务。关掉省 ~4 倍成本/延迟，但 `causal_links` 质量明显下降。thinking 下 Mimo 忽略 temperature |
| `--no-json-mode` | off | 关闭 `response_format=json_object`（debug） |
| **并发** | | |
| `--max-rpm` | `90` | 全局 RPM 上限（Mimo 硬限 100，留余量） |
| `--api-workers` | `6` | API worker 线程 |
| `--preprocess-workers` | `2` | ffmpeg 编码线程 |
| **流程** | | |
| `--skip-preprocess` | off | 只 caption，读 `_cache/` 里已有的 slices |
| `--skip-existing` / `--no-skip-existing` | on | 跳过输出文件里已有的 clip（断点续传） |
| `--reset` | off | 删除输出文件后重跑 |
| `--limit` | None | 只处理前 N 个 clip（调试） |

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
- **断点续传**：默认开，读输出文件的 `clip_id` 集合跳过已完成；切分阶段也跳过已存在 slice。

### 错误分类（summary.json 里分项计数）

不再笼统标 "rejected"，而是区分：
- `safety_rejection`：Mimo 安全过滤器拒绝（≤25 token 且无结构）
- `parse_failed`：JSON 无效或 schema 校验失败
- `empty`：模型返回空
- `fallback_sectioned`：JSON 失败但 sectioned 解析成功（兜底）
- `first_attempt_rejection`：首次被拒（safety/parse/empty），已自动 retry——Mimo 的常见行为，retry 命中前缀缓存（1/50 价）几乎必成。

失败时原始模型输出存进 `UsageRecord.raw_content`（summary 的 `round_breakdown` 里也有前 200 字预览），便于事后排查。

---

## 7. 已知限制

- **60s 合并模式已移除**：实验显示模型处理 60s 视频时只描述前 5-6 秒（三组实验全部如此，非 token 限制）。
  `--clip-duration 60` 和 `concat_two_clips` 已删除。如未来找到让模型"看全"的 prompt 技巧，可考虑重新引入。
- **thinking 默认开启、成本高**：原子分解 + 因果推断是推理密集任务，默认 `--thinking enabled`，
  单次延迟/token 约 ×3-4（reasoning 占大头）。省钱跑法：`--thinking disabled`，
  动作/环境层质量基本不变，`causal_links` 变稀疏且更浅。
- **causal_links 是模型推断**：可能有假阳性/假阴性。prompt 已强制"可见时序 + 合理机制才标，怀疑边存在才省略"，
  且 `cause`/`effect` 要求引用具体动作/变化（带时间），下游可按证据强度过滤。`strength` 是模型的主观三档
  判断（反事实定义），聚合时建议按 strong/moderate/weak → 1.0/0.5/0.25 加权，或仅取 strong 边做硬约束。
- **JSON 模式偶发 code-fence**：极少数情况模型无视 `json_object` 模式仍输出 ` ```json ` 围栏或
  sectioned 文本——解析器已兜底（剥围栏 / 回退 sectioned 解析，含 `[Envchanges]`/`[Causal]` 节），不影响可用性。
- **单 participant 单 day**：跨天/跨参与者需起多个进程，注意共享 RPM 配额（调低单进程 `--max-rpm`）。

---

## 8. 配置建议

| 场景 | 推荐参数 |
|---|---|
| 标准跑（因果标注质量优先） | 默认即可：`--max-rpm 90 --api-workers 6`（thinking on, 10s 切分） |
| 大批量、预算敏感 | `--thinking disabled --max-rpm 95 --api-workers 8`（放弃推理链，留意 429） |
| 调试 prompt | `--limit 3 --api-workers 1`（串行，便于看日志） |

**吞吐估算**：RPM 上限不变（90 RPM ≈ 5400 clip/h 的**调用数**上限），但 thinking 单次延迟 ×3-4、
completion tokens ×~5-8，单日全量跑的 token 成本显著高于旧默认。DAY1 ~828 段（~30s 源）在
`--clip-duration 10` 下 ≈ 2484 片段。切分在 API 等待间隙并行完成，不是瓶颈。
system prompt 前缀缓存仍有效（实测旧版首次缓存命中 ~73%）。

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

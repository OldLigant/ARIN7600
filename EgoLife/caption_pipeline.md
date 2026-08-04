# caption_pipeline — EgoLife 结构化行为标注

> 用 Mimo-V2.5 全模态模型，把 30 秒第一人称（Meta Aria）视频段标注成 **5 层结构化 JSON**：
> 自身动作 / 他人动作 / 环境 / 语音 / 心理（含 `awareness` 情绪-行为驱动力评级）。
> 单文件自包含，不依赖兄弟模块。

---

## 1. 它做什么

输入：一段 ~30 秒的 EgoLife 第一人称视频（参与者戴着 Meta Aria 眼镜录制）。

输出：一个 JSON 对象，包含 5 个层：

| 层 | 字段 | 内容 | 时间戳 |
|---|---|---|---|
| **自身行为** | `self_actions` | "我"做了什么；第一人称、现在时、动词开头（DenseCaption 风格） | ✅ 读视频右上角水印 `HH:MM:SS:FF`，标 `HH:MM:SS-HH:MM:SS` 段（~5s 分辨率，3-6 条/30s） |
| **他人行为** | `others` | 其他人做了什么；中立称呼（"穿粉色上衣的女性"），不臆造名字 | ✅ 同一时间轴 |
| **环境** | `environment` | 场景、光照、屏幕内容、物体布局、位置变化 | 粗粒度 |
| **语音** | `speech` | 关键话语；`[zh]`/`[en]` + 说话人 + 引号文本 | ❌（模型音视不严格对齐，不强求） |
| **心理** | `psychology` | `awareness` 等级 + `emotion` 分类 + 可选 `note` | 整段一个 |

**为什么区分"我"与"环境"**：数字孪生（ 项目）要学的是 *self 在 context 下的反应*——
self 行为是要预测的，others+environment 是输入上下文。混在一起就没法训练"给定环境我会怎么做"。

### awareness 三档定义（ 核心）

`awareness` 衡量**情绪在多大程度上因果地驱动了可见行为**：

| 等级 | 定义 | 例子 |
|---|---|---|
| **low** | 情绪是背景色；行为由习惯/任务驱动，情绪没改变发生的事 | 放松地刷手机——"放松"在，但没改变"刷手机"这件事 |
| **medium** | 情绪影响了动作的**表现形式**（语气/节奏/表情/幅度），但没改变**方向** | 开心地看着别人写字，笑得前仰后合，但还坐在原地看 |
| **high** | 情绪**直接触发**了一个行为反应或转折 | 尴尬→伸手遮脸；生气→拍桌；焦虑→放下手头事起身离开 |

`low/medium` 段是"情绪背景"，`high` 段是"情绪-行为转换点"——后者是 我们 要捕捉的核心研究对象。

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
# 标注一整天（默认 30s 片段、JSON 输出、不开 thinking）
python caption_pipeline.py --participant A1_JAKE --day 1 --max-rpm 90

# 只标注某段时间（HHMM）
python caption_pipeline.py --participant A1_JAKE --day 1 --start-time 1110 --end-time 1130

# 中断后续传（默认开启，跳过输出文件里已有的 clip）
python caption_pipeline.py --participant A1_JAKE --day 1

# 只跑 caption（slices 已在 _cache/ 里，跳过编码）
python caption_pipeline.py --skip-preprocess

# 追求最高质量（动作更细，但延迟/token ×4）
python caption_pipeline.py --thinking enabled --max-completion-tokens 8192
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
  "clip_id": "DAY1_A1_JAKE_11100000",
  "global_idx": 2, "day": 1, "user": "A1_JAKE",
  "duration_s": 30.0,
  "narrative": "{\n  \"self_actions\": [...\n}",   // 模型原始 JSON 全文，备追溯
  "tags": ["meeting", "smartphone", "microphone", "discussion"],
  "model": "mimo-v2.5",
  "ts_captioned": "2026-08-04T00:37:48Z",
  "slice_path": "_cache\\slices\\DAY1_A1_JAKE_11100000.mp4",
  "tokens": {"in": 8500, "out": 480, "cached": 0},
  "self_actions": [
    {"time": "11:10:00", "time_end": "11:10:06", "text": "我将手机递给对面的白衣女生让她戳屏幕"},
    {"time": "11:10:06", "time_end": "11:10:12", "text": "我接回手机，看着女生们依次点击屏幕"},
    {"time": "11:10:17", "time_end": "11:10:20", "text": "我左手拿起桌上的黑色麦克风并开启"},
    {"time": "11:10:20", "time_end": "11:10:30", "text": "我双手拿着麦克风，面对大家介绍早上的行程"}
  ],
  "others": [
    {"time": "", "time_end": "", "text": "白衣女生双手接过手机，低头点击屏幕"},
    {"time": "", "time_end": "", "text": "大家围坐在桌旁听我讲话"}
  ],
  "environment": "明亮的室内会议空间，长方形桌子铺着红白格子桌布，上有笔记本、手机、收纳包。背景有白板和补光灯。",
  "speech": [
    {"lang": "zh", "speaker": "A1_JAKE", "text": "都戳一下，每人戳一下。"},
    {"lang": "zh", "speaker": "A1_JAKE", "text": "那就是今天我们就讨论讨论。"}
  ],
  "psychology": {
    "awareness": "medium",
    "emotion": "focused, neutral",
    "note": "作为协调者引导设备操作测试，随后转入会议主持角色。"
  },
  "clip_kind": "30s",              // "30s" | "60s" | "segment_open" | "10s"
  "output_format": "json",        // "json" | "fallback_sectioned"
  "recovery": "ok"                // "ok" | "fallback_sectioned" | "10s_slices"
}
```

### 字段说明
- `self_actions` / `others`：`[{time, time_end, text}]`，`time` 为水印时间戳；无时间戳的行 `time=""` 但仍保留。
- `speech`：`[{lang, speaker, text}]`，`lang ∈ {zh, en, ""}`。
- `psychology`：`{awareness, emotion, note}`，缺失字段为空串。
- `narrative`：模型原始 JSON 全文（未解析），保完整以便人工核查/重新解析。
- `clip_kind`：`30s`（标准）/ `segment_open`（一天开头的不规则段，如 `11094208`）/ `60s`（合并）/ `10s`（recovery 切片）。
- `output_format`：`json`（主路径）/ `fallback_sectioned`（模型无视 json_object 模式时，回退到 sectioned 解析）。
- `recovery`：标注此 clip 经历的路径——`ok`（一次成功）/ `fallback_sectioned` / `10s_slices`（被拒后切成 10s 重试）。

---

## 5. CLI 参数全表

| 参数 | 默认 | 说明 |
|---|---|---|
| **选择** | | |
| `--participant` | `A1_JAKE` | 参与者名 |
| `--day` | `1` | 天（1-7） |
| `--start-time` | None | 起始 HHMM（含），如 `1110` |
| `--end-time` | None | 结束 HHMM（不含），如 `1130` |
| `--clip-duration` | `30` | `30`（推荐）或 `60`（合并，⚠️ 模型只覆盖前 ~6s，不推荐） |
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
| `--thinking` | `disabled` | `enabled` 时开推理（需调大 `--max-completion-tokens`；Mimo 会忽略 temperature） |
| `--max-completion-tokens` | `1024` | thinking 时建议 8192（reasoning_tokens 共享此预算） |
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
[PreprocessWorker 池]      [RateLimiter]      [API Worker 池]      [Writer]
 ffmpeg 重编码       ->    produced_q   ->   acquire()  ->  调 Mimo  ->  result_q  ->  按 global_idx 有序写 jsonl
  (2 线程, CPU)             (有界队列)         (90 RPM 滑窗)     (6 线程, IO)            (单线程, 堆缓冲保序)
```

- **编码与 API 重叠**：API worker 阻塞等模型响应时，preprocess worker 在跑下一个 ffmpeg 编码。
- **全局限速**：所有 API 调用（正常 + retry + 10s recovery）共享一个 `SlidingWindowRateLimiter`，
  硬性卡在 `--max-rpm`，绝不超 Mimo 配额。
- **有序落盘**：API worker 乱序完成 → `OrderedWriter` 用堆缓冲，按 `global_idx` 严格有序写，
  输出确定性、resume 友好。
- **三级 recovery**：30s 被拒 → retry 一次 → 切 10s 各调一次 → 放弃。所有 recovery 调用都过同一限速器。
- **断点续传**：默认开，读输出文件的 `clip_id` 集合跳过已完成；编码阶段也跳过已存在 slice。

### 错误分类（summary.json 里分项计数）

不再笼统标 "rejected"，而是区分：
- `safety_rejection`：Mimo 安全过滤器拒绝（≤25 token 且无结构）
- `parse_failed`：JSON 无效或 schema 校验失败
- `empty`：模型返回空
- `fallback_sectioned`：JSON 失败但 sectioned 解析成功（兜底）

失败时原始模型输出存进 `UsageRecord.raw_content`（summary 的 `round_breakdown` 里也有前 200 字预览），便于事后排查。

---

## 7. 已知限制

- **60s 片段不推荐**：实验显示模型处理 60s 视频时只描述前 5-6 秒（三组实验全部如此，非 token 限制）。
  代码保留 `--clip-duration 60` 但默认 30s。60s 适合未来找到让模型"看全"的 prompt 技巧时再启用。
- **thinking 成本高**：单片段实验中 thinking 把动作数从 3 提到 5（与 sectioned 持平），但
  reasoning_tokens 占 84%，延迟 ×3-4。默认关，追求质量时手动开。
- **JSON 模式偶发 code-fence**：极少数情况模型无视 `json_object` 模式仍输出 ` ```json ` 围栏或
  sectioned 文本——解析器已兜底（剥围栏 / 回退 sectioned 解析），不影响可用性。
- **单 participant 单 day**：跨天/跨参与者需起多个进程，注意共享 RPM 配额（调低单进程 `--max-rpm`）。

---

## 8. 配置建议

| 场景 | 推荐参数 |
|---|---|
| 标准跑（性价比最高） | 默认即可：`--max-rpm 90 --api-workers 6` |
| 大批量、急 | `--max-rpm 95 --api-workers 8`（逼近配额，留意 429） |
| 追求最高标注质量 | `--thinking enabled --max-completion-tokens 8192`（延迟/token ×4） |
| 调试 prompt | `--limit 3 --api-workers 1`（串行，便于看日志） |

**吞吐估算**：90 RPM ≈ 5400 clip/h。DAY1 ~828 段理论 ~9 分钟（API 满载时），
ffmpeg 编码在 API 等待间隙并行完成，不再是瓶颈。

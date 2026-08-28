# EgoLife — 结构化行为标注管线

用小米 **Mimo-V2.5** 全模态模型，把第一人称（Meta Aria 眼镜）视频标注成**面向数字孪生的
结构化 JSON**：物体/人词典、语音转写、环境声、界面内容、原子动作、环境状态迁移、上下文
隔离的心理活动、带强度权重的因果边。

**两条管线**：

| 管线 | 脚本 / 文档 | 状态 | 说明 |
|---|---|---|---|
| **多轮管线（主线）** | `caption_pipeline_multiturn.py` / [`caption_pipeline_multiturn.md`](caption_pipeline_multiturn.md) | 当前开发重点 | 每片段四轮递进：T1 感知（×k 采样投票 + 合并出对象词典）→ T1c 界面/文本面理解（条件触发，原生分辨率帧）→ T2 行为（词典锚定 + 意图禁令）→ T3 心理（上下文隔离隐变量）→ T4 因果（纯文本推理，8 类边）→ **代码确定性合并**最终记录。按轮独立落盘/续传，消息布局为前缀缓存优化 |
| 单轮管线（legacy） | `caption_pipeline.py` / [`caption_pipeline.md`](caption_pipeline.md) | 保留作 A/B 对照 | 一次调用产出全部层（self_actions / others / environment / env_changes / speech / psychology / causal_links / ocr / tags） |

流水线：**下载视频 →（可选）下载官方标注 → 生成结构化标注 →（可选）标注烧录到视频**。

```
download_video.py          原始视频      →  videos/{参与者}/DAY{天}/*.mp4
download_egolifecap.py     官方 caption  →  EgoLifeCap/{DenseCaption|Transcript}/...
caption_pipeline_multiturn.py  多轮标注  →  captions/{参与者}/DAY{天}/{时间段}.jsonl
caption_pipeline.py        单轮标注(legacy)
overlay_annotations.py     标注烧录      →  把时间对齐的标注叠加到片段上（可视化/核查）
```

---

## 快速开始

```bash
# 0. 装依赖（需要 ffmpeg/ffprobe 在 PATH 上）
pip install -r requirements.txt

# 1. 配置 API key
cp .env.example .env          # 填入 MIMO_API_KEY

# 2. 下载视频 + 官方 caption（A1_JAKE / DAY1，公开数据集无需 token）
python download_video.py
python download_egolifecap.py --kind all

# 3. 多轮标注（主线；默认 10s 切分、T1 双采样投票、thinking 按轮开关）
python caption_pipeline_multiturn.py --participant A1_JAKE --day 1
```

跑完后，最终合并记录在 `captions/A1_JAKE/DAY1/full.jsonl`；各轮原始输出在
`full_t1.jsonl … full_t4.jsonl`（断点续传粒度到轮）。

---

## 目录结构

```
EgoLife/
├── README.md / requirements.txt / .env.example
├── download_video.py / download_egolifecap.py / download_utils.md
├── caption_pipeline_multiturn.py   <- 多轮标注管线（主线）
├── caption_pipeline_multiturn.md   <- 其设计 spec + 各轮 prompt 草案
├── caption_pipeline.py / .md       <- 单轮管线（legacy，A/B 对照）
├── overlay_annotations.py          <- 标注烧录可视化
├── videos/ EgoLifeCap/             <- [gitignored] 输入
└── captions/{参与者}/DAY{天}/
    ├── full.jsonl                  <- 主输出（多轮合并记录，纳入版本管理）
    ├── full_t{1,1c,2,3,4}.jsonl    <- 各轮原始输出（resume 依据）
    ├── full_usage.jsonl / full_summary.json / full_run.log   <- [gitignored]
    └── _cache/                     <- [gitignored] slices / frames_native / parquet
```

**纳入版本管理的只有代码和 `.jsonl` 标注结果**；视频、缓存、日志、usage 全部 gitignore。

---

## 输出 schema（多轮管线，节选）

`{时间段}.jsonl` 每行一个片段的合并记录（strict JSONL，代码拼装、可回溯到轮次）：

```jsonc
{
  "clip_id": "DAY1_A1_JAKE_11100000_p2", "global_idx": 5, "duration_s": 10.0,
  "turns": { "t1": {"runs": 2, "tokens": {...}}, "t1c": null, "t2": {...}, "t3": {...}, "t4": {...} },
  "inventory": [                       // T1 投票合并出的对象/人词典（后续轮次的锚）
    {"id": "obj01", "label": "白色陶瓷马克杯", "type": "object", "votes": "2/2",
     "confidence": "high", "interacted_with": true, "first_seen": "11:10:01"}
  ],
  "speech": [ {"lang": "zh", "speaker": "佩戴者", "text": "...", "t_start": "11:10:02", "confidence": "high"} ],
  "sounds":  [ {"kind": "music", "label": "音箱播放的中文流行歌曲", "source": "设备播放"} ],
  "screen":  [ {"app": "Bilibili (web)", "content": "正在播放视频《……》", "operation": "watching"} ],
  "self_actions": [ {"time": "11:10:10", "time_end": "11:10:12", "text": "我拿起马克杯"} ],   // 原子化
  "others": [...], "environment": "...",
  "env_changes": [ {"time": "...", "text": "杯子离开桌面进入我手中", "cause": "self"} ],
  "new_objects": [],                   // T2 词典外物体的显式逃生口
  "psychology": {                      // T3：上下文隔离的隐变量（awareness 已移除）
    "emotion": "focused", "mental_activity": "……", "evidence": [...], "confidence": "medium"
  },
  "causal_links": [                    // T4：8 类有向边 + 反事实强度三档
    {"type": "action->env", "cause": "11:10:15 我拨动开关", "effect": "11:10:16 麦克风通电", "strength": "strong"}
  ],
  "tags": [...], "qc": {"intent_markers": 0, ...}, "recovery": {"t1": "ok", "t1c": "skipped", ...}
}
```

与 legacy 的主要差异：新增 `inventory` / `sounds` / `screen` / `new_objects` / `qc`；
`speech` 带时间戳与置信度；`psychology` 换成 `{emotion, mental_activity, evidence,
confidence}`（**`awareness` 已移除**——它是旧"情绪→动作单向"设计的产物）；`ocr` 被
`screen` 语义化吸收；`causal_links` 从 7 类扩到 8 类（+`env->env`）。legacy 记录里的
`awareness` 三档定义见 [`caption_pipeline.md`](caption_pipeline.md)。

---

## 常用命令（多轮管线）

```bash
# 标注某段时间（HHMM）
python caption_pipeline_multiturn.py --participant A1_JAKE --day 1 --start-time 1110 --end-time 1130

# 中断后续传（默认开启；粒度到轮——重跑同一命令只补缺的轮）
python caption_pipeline_multiturn.py --participant A1_JAKE --day 1

# T1 投票次数（默认 2 + 分歧仲裁；是否升 3 由批量实测分歧率决定）
python caption_pipeline_multiturn.py --votes 3 ...

# 只跑部分轮次（分阶段上线；前置轮自动包含）
python caption_pipeline_multiturn.py --turns t1 --participant A1_JAKE --day 1

# T1c 换任意 OpenAI 兼容图像 VLM（读屏不需要全模态）／控制图像 token
python caption_pipeline_multiturn.py --t1c-provider custom --t1c-base-url https://... \
    --t1c-model some-vlm --t1c-api-key-env VLM_KEY
python caption_pipeline_multiturn.py --t1c-frame-size 1024 --t1c-frame-fps 0.5 ...

# 批量测试一律隔离输出（勿写默认路径）
python caption_pipeline_multiturn.py --participant A1_JAKE --day 4 --limit 50 \
    --out captions/_test/multiturn_day4/full.jsonl
```

完整参数见 `--help` 与 [`caption_pipeline_multiturn.md`](caption_pipeline_multiturn.md)。

---

## 依赖

| 依赖 | 用途 |
|---|---|
| `openai` | Mimo API（OpenAI 兼容接口）；T1c 可替换 VLM 同走此 SDK |
| `pandas` | clips.parquet 元数据 |
| `python-dotenv` | 读 `.env` |
| `jsonschema` | legacy 管线的输出结构校验 |
| `tqdm` | 进度条 |
| `huggingface_hub` | `download_video.py`（视频下载） |
| `ffmpeg` / `ffprobe` | 切片/重编码/抽帧/时长探测（非 pip 包，需单独装） |

# EgoLife — 结构化行为标注管线

用小米 **Mimo-V2.5** 全模态模型，把第一人称（Meta Aria 眼镜）视频标注成 **5 层结构化 JSON**：
自身动作 / 他人动作 / 环境 / 语音 / 心理状态（含 `awareness` 情绪-行为驱动力评级）。

三个脚本，一条流水线：**下载视频 → （可选）下载官方标注 → 生成结构化标注**。

```
download_video.py       原始视频      →  videos/{参与者}/DAY{天}/*.mp4
download_egolifecap.py   官方 caption  →  EgoLifeCap/{DenseCaption|Transcript}/{参与者}/DAY{天}/*.srt
caption_pipeline.py      结构化标注    →  captions/{参与者}/DAY{天}/{时间段}.jsonl
```

---

## 快速开始

```bash
# 0. 装依赖（需要 ffmpeg 在 PATH 上）
pip install -r requirements.txt

# 1. 配置 API key
cp .env.example .env          # 填入 MIMO_API_KEY

# 2. 下载视频 + 官方 caption（A1_JAKE / DAY1，公开数据集无需 token）
python download_video.py
python download_egolifecap.py --kind all

# 3. 生成结构化标注
python caption_pipeline.py --participant A1_JAKE --day 1
```

跑完后，标注在 `captions/A1_JAKE/DAY1/full.jsonl`。

---

## 目录结构

```
EgoLife/
├── README.md                       <- 本文件
├── requirements.txt
├── .env.example                    <- 复制成 .env 填 key
│
├── download_video.py               <- 视频：HuggingFace → videos/
├── download_egolifecap.py          <- 官方标注：HuggingFace → EgoLifeCap/
├── download_utils.md               <- 下载脚本的详细参数说明
│
├── caption_pipeline.py             <- 标注管线（自包含，不依赖其他脚本）
├── caption_pipeline.md             <- 标注管线的详细文档（schema / CLI / 架构）
│
├── videos/                         <- [gitignored] 原始视频
├── EgoLifeCap/                     <- [gitignored] 官方 caption
└── captions/                       <- 标注输出
    └── A1_JAKE/DAY1/
        ├── full.jsonl              <- 主输出（结构化标注，纳入版本管理）
        ├── full_usage.jsonl        <- [gitignored] 每次调用的 token/延迟
        ├── full_summary.json       <- [gitignored] 汇总统计
        ├── full_run.log            <- [gitignored] 运行日志
        └── _cache/                 <- [gitignored] 重编码片段 + parquet
```

**纳入版本管理的只有代码和 `.jsonl` 标注结果**；视频、缓存、日志、usage 全部 gitignore。

---

## 输出 schema（节选）

`captions/{参与者}/DAY{天}/{时间段}.jsonl` 每行一个 JSON 对象（strict JSONL）：

```jsonc
{
  "clip_id": "DAY1_A1_JAKE_11100000",
  "global_idx": 2, "day": 1, "user": "A1_JAKE", "duration_s": 30.0,
  "self_actions": [
    {"time": "11:10:00", "time_end": "11:10:06", "text": "我将手机递给对面的白衣女生让她戳屏幕"},
    {"time": "11:10:17", "time_end": "11:10:20", "text": "我左手拿起桌上的黑色麦克风并开启"}
  ],
  "others": [
    {"time": "", "time_end": "", "text": "白衣女生双手接过手机，低头点击屏幕"}
  ],
  "environment": "明亮的室内会议空间，长方形桌子铺着红白格子桌布……",
  "speech": [
    {"lang": "zh", "speaker": "A1_JAKE", "text": "都戳一下，每人戳一下。"}
  ],
  "psychology": {
    "awareness": "medium",      // low | medium | high
    "emotion": "focused",
    "note": "作为协调者引导设备操作，随后转入会议主持。"
  },
  "tags": ["meeting", "smartphone", "microphone", "discussion"],
  "clip_kind": "30s", "output_format": "json", "recovery": "ok"
}
```

### `awareness` 三档（核心设计）

衡量**情绪在多大程度上因果地驱动了可见行为**：

- **low** — 情绪是背景色，没改变发生的事（如放松地刷手机）
- **medium** — 情绪影响了动作的表现形式，但没改方向（如笑着看人写字）
- **high** — 情绪直接触发了一个行为或转折（如尴尬→遮脸、生气→拍桌）

---

## 常用命令

```bash
# 标注某段时间（HHMM）
python caption_pipeline.py --participant A1_JAKE --day 1 --start-time 1110 --end-time 1130

# 中断后续传（默认开启）
python caption_pipeline.py --participant A1_JAKE --day 1

# 开 thinking 提升动作细粒度（延迟/token ×4）
python caption_pipeline.py --thinking enabled --max-completion-tokens 8192

# 下载其他参与者 / 天
python download_video.py --participant A2_ALICE --day 1 2 3
python download_egolifecap.py --kind all --participant all --day all
```

完整参数见各脚本的 `--help`，以及 [`caption_pipeline.md`](caption_pipeline.md) 和 [`download_utils.md`](download_utils.md)。

---

## 依赖

| 依赖 | 用途 |
|---|---|
| `openai` | Mimo API（OpenAI 兼容接口） |
| `pandas` | clips.parquet 元数据 |
| `python-dotenv` | 读 `.env` |
| `jsonschema` | 标注输出结构校验 |
| `huggingface_hub` | `download_video.py`（视频下载） |
| `ffmpeg` / `ffprobe` | 视频重编码、duration 探测、recovery 切片（非 pip 包，需单独装） |

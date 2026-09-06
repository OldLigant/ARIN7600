# run_pipeline.py — POV 密集视频描述（Dense Video Captioning）流水线

对 AriaRealLife 第一视角长录像（tar 包）做**两遍式（2-pass）密集视频描述**的独立脚本：
通过 HTTP Range 直接远端读取 HuggingFace 上的录制 tar（不整包下载），抽帧并叠加注视点，
按 clip 调用 Gemini（Vertex AI）生成带时间轴的 segment 级 caption，pass 2 对文字区域
裁剪精读，最后合并成整条录制 / 整天的结构化产物，并附带用量、估价与 PII 审计。

- 脚本：`run_pipeline.py`（单文件，约 2770 行）
- 入口：`python run_pipeline.py list|run ...`（见 [CLI 用法](#cli-用法)）
- 配套文档：本文件（`run_pipeline.md`，与脚本同名）

---

## 1. 版本与来源

本文件是 [`mzx`](../../../mzx/) 仓库 `caption/run_pipeline.py` **工作区改进版**的副本，
即在其最近一次提交（`0dbfc1d`）之上的改进版本。两者文件内容完全一致
（MD5 `8131d4164755a7188dd07796f75c63cd`）。

| 项目 | 值 |
|---|---|
| 来源仓库 | `https://github.com/15210579725/hku-capstone.git`（本地 `D:/QLD/mzx`） |
| 改进所基于的提交（mzx 最近一次提交） | `0dbfc1dea1f2e852f13f6a6c16c1f4bfc4bc5028` |
| 该提交主题 | Add unified dataset: EgoLife (6 participants) + Next-Me (24 sessions) with 1k benchmark |
| 该提交作者 / 时间 | mzx \<136165406+15210579725@users.noreply.github.com\>，2026-09-05 08:42:25 -0700 |
| 本副本相对该提交的改动 | **+541 行 / −78 行**（2309 行 → 2772 行） |
| 落地提交（本仓库） | `e3db9a1` — `feat(captions): add run_pipeline.py improved from mzx/caption original` |

与该提交的完整差异可对比查看：

```bash
git -C D:/QLD/mzx diff HEAD -- caption/run_pipeline.py
```

---

## 2. 相对 mzx 最近一次提交（0dbfc1d）的改进

> 本节即"工作区这份"与"最近一次提交"的差异总结，也是本副本被称为"改进版"的依据。

### 2.1 模型与服务层级（Flex PayGo）

- 默认模型 `gemini-3.7-flash` → **`gemini-3.8-flash`**。
- 新增 **服务层级（service tier）** 概念，默认 **`flex`**（Flex PayGo，按量半价）：
  - `make_client()` 为 flex 注入请求头
    `X-Vertex-AI-LLM-Request-Type: shared`、`X-Vertex-AI-LLM-Shared-Request-Type: flex`；
    flex 强制 `GOOGLE_CLOUD_LOCATION=global`。
  - 计价对 flex 乘 **0.5** 系数；`report.json` / checkpoint 均记录 `service_tier`。
  - 新增 CLI `--service-tier {standard,flex}` 与配置项 `service_tier`。
- 新增 **单请求超时** `--request-timeout`（默认且上限 1800 秒），并通过
  `HttpRetryOptions(attempts=1)` **禁用 SDK 内部隐式重试**——重试由流水线自己持有
  checkpoint 负责，避免 30 分钟长请求被 SDK 静默重复计费。
- `generation_config_kwargs()`：`gemini-3` 系列不再传 `temperature`（由模型默认思维预算接管）。

### 2.2 API 错误分类与分层重试

新增 `classify_api_error()`，把 Gemini 调用失败归为七类：
`quota` / `payload` / `model` / `auth` / `server` / `network` / `other`。

- **只有可恢复类（quota、server、network）会重试**；`auth`、`model` 等**终止型错误立即放弃**，
  不再空耗重试配额。
- quota 重试保留提交版的退避表（15s→300s + 抖动）；普通瞬时错误线性退避（4s×次数）。
- **flex 层级下**瞬时/quota 重试均收紧为 1 次（长请求宁可交给断点续跑，也不重复计费）。
- clip 级新增 `should_retry_clip_result()`：多轮重跑只重试非终止型失败。
- 每次调用结果携带结构化字段 `error_type`、`attempts`，并一路写入
  `captions_full.json`、clip checkpoint 与日志。

### 2.3 AIMD 自适应并发

新增 `AIMDController`（加性增 / 乘性减）与动态线程池 `run_dynamic_pool()`：

- CLI `--adaptive-concurrency` 开启；`--workers` 变为**初始并发**，`--max-workers`（默认 12）为上限。
- 请求健康则每轮加性 +1；出现 quota/server/network 拥塞则并发**减半**，每次调整都打印日志。
- 结束时输出 AIMD 统计（初始 / 最终 / 峰值并发、增减次数）。
- 不开启时行为同提交版（固定 workers，每轮重跑递减）。

### 2.4 断点续跑（checkpoint resume）

- 每个 clip 的完整结果实时写入 `<output>/<recording>/clips/clip_XXXX.json`
  （新增 `checkpointed_at`、`service_tier`、`error_type` 字段）。
- 新增 `load_clip_checkpoints()`：重启后**只复用兼容的成功 clip**
  （校验 `recording` / `model` / `pipeline` / `service_tier` / `ok` / `parsed` 一致），
  失败或过期（如换模型、换 tier）的 checkpoint 自动作废重跑。
- 有 clip 仍失败时，保留全部成功产物并抛出 `IncompleteRecordingError`；
  `run` 命令最终以**退出码 1** 结束（HF Job / CI 可感知，重新挂载同一 /output 即自动续跑）。

### 2.5 日志与可观测性

- 每次请求打印 `request start / ok / failed`：耗时、token 用量（含 thinking）、
  **`traffic_type`**（Provisioned/Shared 等）、attempt 数。
- 失败日志带 `type=<error_type>`，便于一眼定位配额、鉴权还是负载问题。
- 结束时输出 **API 统计**：请求总数、耗时中位数 / 最大值、吞吐（clips/hour）、traffic 类型集合。
- `cmd_run` 打印认证模式（service-bound API key 还是 ADC）、project/location/tier/timeout 概览。

### 2.6 安全：日志与错误信息脱敏

新增 `redact_secrets()`，在所有日志 / 异常 / traceback 路径上抹除：

- 环境变量中的 `GOOGLE_API_KEY`、`GEMINI_API_KEY`、`HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN` 原文；
- `Authorization: Bearer …`、`api_key=…` / `token=…` 形态；
- `AIza…`（Google API key）与 `hf_…`（HF token)特征串。

提交版直接 `str(e)` 打印，存在把密钥带进日志的风险；本版全部经脱敏并截断。

### 2.7 传输 fallback 逻辑修正

`call_with_fallback()`（inline ≤20MB → GCS → 降分辨率 inline）：

- **只有 `payload` 类错误才触发降级链**；quota / server / network 等失败直接返回
  （降分辨率解决不了这类问题，原逻辑会白白多打几轮请求）。
- 最终失败的返回值也补齐 `error_type` / `attempts` / `payload_mb`，结构一致可审计。

### 2.8 配置优先级与其它修正

- **环境变量优先于 config.json**（`HF_TOKEN`、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION`）——
  适配 HF Job secrets 覆盖本地配置的场景（提交版相反）。
- `configure_google_environment()`：统一设置 `GOOGLE_GENAI_USE_ENTERPRISE` 等 SDK 环境变量，
  API key 以 service-bound 方式传入 client。
- `process_clip()` 内部日志不再被静音（原来 `log=lambda *a: None`），改为接入全局 `_log`。
- checkpoint 写盘显式 `encoding="utf-8"`。
- 多录制批处理：失败计数准确（原来用"成功数"倒推），任一失败最终退出码非 0。

### 2.9 新增 / 变动的符号一览

新增函数与类：`redact_secrets`、`classify_api_error`、`should_retry_api_error`、
`should_retry_clip_result`、`AIMDController`、`run_dynamic_pool`、
`generation_config_kwargs`、`configure_google_environment`、`load_clip_checkpoints`、
`IncompleteRecordingError`。
签名变动：`make_client`、`generate`、`call_with_fallback`、`align_speech`、
`process_clip`、`process_recording`、`merge_recording`、`write_recording_outputs`。

---

## 3. 处理流程

输入是 HuggingFace 数据集 **`mmm8383/pov-data`** 中 `aria/<日期>/*.tar` 的录制包
（`list_tars()` 列出，`HttpRangeSource` 通过 HTTP Range 按需读取 tar 成员，不整包下载）。

`process_recording()` 五个阶段：

1. **尾部读取**：`read_tail_adaptive()` 只读 tar 尾部，取 gaze 轨迹与 OCR 文本；
   有 gaze 则启用 `PROMPT_V9`（带注视点渲染），否则 `PROMPT_V9_NOGAZE`。
2. **音频 / transcript 定位**：`locate_sections_safe()` 定位 wav 数据段与 transcript 成员；
   `parse_transcript()` 解析出带绝对时间戳的区间；`AudioSlicer` 支持按 clip 切 16k 单声道 wav。
3. **帧解包与渲染**：流式扫描 `/picture/masked/` 下 jpg，`render_frame()` 重采样到 2880px
   并叠加注视点轨迹（当前点 + 5 点尾迹），按 `frame_index // clip_seconds` 组成 clip。
4. **两遍 caption**（每个 clip，失败多轮重试）：
   - **Pass 1**：全量帧（HIGH 分辨率）+ transcript + OCR 上下文 → segments（时间轴动作描述）
     + `text_regions`（画面中值得精读的文字区域框）；
     silero-VAD 检出语音窗，`align_speech()` 让 Gemini 给出逐句 utterance 时间戳。
   - **Pass 2**：对区域框裁剪原图（ULTRA 2880px）+ Pass 1 结果 → `READ_PROMPT` 精读文字内容，
     修订回 segments。
   - 传输策略 `call_with_fallback()`：inline（≤20MB）→ GCS（桶 `hku-capstone-caption-frames`）
     → 降分辨率（1440/1200/1024px）inline。
5. **合并落盘**：`merge_recording()` 汇总 segments、用量与估价、PII 审计；
   `write_recording_outputs()` 写录制级产物；`--day` 模式再 `merge_day()` 出整天产物。

---

## 4. 配置

`config.json`（可用 `--config` 换路径）可识别的字段及默认值：

| 字段 | 默认 | 说明 |
|---|---|---|
| `hf_token` | `""` | HuggingFace token（读 `mmm8383/pov-data`） |
| `google_project` | `""` | GCP 项目 ID（必填，缺失直接退出） |
| `google_location` | `"global"` | Vertex AI location（flex 必须 global） |
| `model` | `gemini-3.8-flash` | Gemini 模型名 |
| `service_tier` | `flex` | `standard` / `flex` |
| `request_timeout_sec` | `1800` | 单请求超时（上限 1800s） |
| `output_dir` | `./caption_output` | 输出根目录 |
| `workers` | `3` | 并发数（AIMD 模式下为初始值） |
| `max_workers` | `12` | AIMD 并发上限 |
| `adaptive_concurrency` | `false` | 是否启用 AIMD |
| `clip_seconds` | `30` | 每个 clip 的秒数 |
| `rounds` | `3` | 失败 clip 的重跑轮数（flex 实验建议 1） |

环境变量 `HF_TOKEN`、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION` **优先于** config.json；
`GOOGLE_API_KEY` / `GEMINI_API_KEY` 存在时以 service-bound API key 认证，否则走 ADC
（`gcloud auth application-default login`）。

## 5. CLI 用法

```bash
# 列出数据集上可用的天与录制
python run_pipeline.py list [--config config.json]

# 跑单条录制
python run_pipeline.py run --tar "aria/2026-05-18/xxx_120m_xxx.tar" [选项]

# 跑一整天全部录制，并生成整天合并产物
python run_pipeline.py run --day 2026-05-18 [选项]
```

`run` 的选项：

| 选项 | 说明 |
|---|---|
| `--tar PATH` | 单条录制的 tar 路径 |
| `--day DATE` | 处理一天全部录制（如 `2026-05-18`） |
| `--max-clips N` | 只处理前 N 个 clip（冒烟测试） |
| `--workers N` | caption 并发数（默认 3；AIMD 下为初始值） |
| `--adaptive-concurrency` | 启用 AIMD 动态并发 |
| `--max-workers N` | AIMD 并发上限（默认 12） |
| `--service-tier standard\|flex` | Gemini 服务层级（默认 flex） |
| `--request-timeout SEC` | 单请求超时秒数（默认且最高 1800） |
| `--rounds N` | 失败 clip 的处理轮数（flex 实验建议 1） |
| `--clip-seconds SEC` | 每个 clip 秒数（默认 30） |
| `--no-vad` | 禁用 silero-VAD 语音对齐 |
| `--no-ocr` | 不使用 OCR 文本 |
| `--config PATH` / `--output-dir DIR` | 配置文件 / 输出目录 |

## 6. 输出产物

```
<output_dir>/
├── <recording>/                     # 每条录制一个目录
│   ├── captions_full.json           # 合并结果：segments、usage、manifest、pii_audit…
│   ├── captions_full.jsonl          # 每行一个 segment
│   ├── captions_full.txt            # 人类可阅读版
│   ├── report.json                  # 摘要：clip 成败、tokens、est_cost_usd、
│   │                                #   service_tier、pii_audit、失败 clip 清单
│   └── clips/
│       └── clip_0000.json           # 每 clip 检查点（断点续跑的最小单位）
└── <day>/                           # --day 模式下的整天合并
    ├── day_<date>.json / .txt / .jsonl
```

## 7. 成本与 PII

- 引导价 `PRICE_IN=0.75`、`PRICE_OUT=3.75` USD/M tokens（计费用 `out + think`）；
  **flex 层级 ×0.5**。估价写入 `report.json` 的 `est_cost_usd`。
- `pii_audit()` 按 HKU / 大学名 / 邮箱 / 电话等模式审计每个 segment，
  结果计入 report；生成侧 prompt 约定使用 `University X`、`email_x@` 等匿名写法。
- 所有日志、错误、traceback 均经 `redact_secrets()` 脱敏（见 §2.6）。

## 8. 依赖

`google-genai`（Vertex AI 客户端）、`huggingface_hub`、`Pillow`、`numpy`、
`silero-vad`（`torch`；缺失时自动退化为能量窗检测）、GCS 上传需
`google-cloud-storage`（可选）。Python ≥ 3.10（使用了 `X | Y` 类型语法）。

## 9. 注意事项

- flex 层级要求 `GOOGLE_CLOUD_LOCATION=global`，且单请求重试收敛为 1 次——
  长请求失败依赖**断点续跑**而不是原地重试，请保持输出目录可持久化（HF Job 挂载同一 `/output`）。
- 有 clip 失败时进程以退出码 1 结束，但**成功 clip 的产物与 checkpoint 均已保留**，
  直接重跑同一命令即可续跑。
- 换模型 / 换 service tier 后，旧 checkpoint 因兼容性校验不通过会自动全部重跑。

# download_utils — EgoLife 数据下载

两个脚本，从 HuggingFace [`lmms-lab/EgoLife`](https://huggingface.co/datasets/lmms-lab/EgoLife) 拉数据，按 `caption_pipeline.py` 默认期望的布局放置，无需额外参数即可衔接。

```
download_video.py        原始视频        ->  videos/{参与者}/DAY{天}/*.mp4
download_egolifecap.py    官方 caption    ->  EgoLifeCap/{DenseCaption|Transcript}/{参与者}/DAY{天}/*.srt
```

> 数据集公开，无需 token。视频量大（单参与者单天 ~10 GB+，小时级），caption 很小（全部 ~50 MB）。

---

## 默认目录布局（不改任何路径参数时）

```
ARIN7600/EgoLife/                   <- 脚本所在目录
├── videos/                         <- download_video.py 写这里（= caption_pipeline.py 默认输入）
│   └── A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4
├── EgoLifeCap/                     <- download_egolifecap.py 写这里（官方标注，非本仓库 pipeline 产物）
│   ├── DenseCaption/A1_JAKE/DAY1/*.srt
│   └── Transcript/A1_JAKE/DAY1/*.srt
└── captions/                       <- caption_pipeline.py 的输出（自动生成，与下载无关）
```

---

## download_video.py —— 视频下载

用 `huggingface_hub`。视频是 10MB+ 的大文件，HF 的每文件固定开销可忽略，所以这里用 HF。

```bash
# 默认：A1_JAKE / DAY1（和 caption_pipeline.py 默认一致）
python download_video.py

# 指定参与者 / 天（可多选；天用数字或 DAY1 都行）
python download_video.py --participant A1_JAKE A2_ALICE --day 1 2 3

# 全量（6 人 × 7 天，几百 GB）
python download_video.py --participant all --day all

# 下载到别处（默认 ./videos）
python download_video.py --out-dir D:/data/egolife_videos
```

⚠ **改了 `--out-dir` 后，跑 caption_pipeline.py 要相应指定视频根目录**：
```bash
python caption_pipeline.py --participant A1_JAKE --day 1 \
    --src-dir D:/data/egolife_videos
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--participant` | `A1_JAKE` | 参与者，可多选；`all` = 全部 6 人 |
| `--day` | `DAY1` | 天，如 `1`/`DAY1`/`all`，可多选 |
| `--out-dir` | `./videos` | 视频保存根目录 |
| `--token` | 无 | HF token（公开数据集可省略） |
| `--max-workers` | `4` | HF 并发下载线程数 |

---

## download_egolifecap.py —— 官方 caption 下载

下载 DenseCaption 和/或 Transcript（SRT 文本），用 `--kind` 选。默认走并行 HTTP 直连——

> **为什么不用 HF / git lfs？** caption 全是几十 KB 的文本文件，HF 对每个文件有固定开销（metadata + 锁 + cache 校验），占比 ~99%；实测同一份数据 HTTP ~5s、HF ~40s。git lfs 更慢到不可用（~600GB 仓库的元数据协商要数分钟）。视频用 HF 是因为大文件开销可忽略，caption 正相反。`--method hf` 仍可选作 fallback。

```bash
# 默认：DenseCaption, A1_JAKE / DAY1, http
python download_egolifecap.py

# 改成 Transcript
python download_egolifecap.py --kind transcript

# 两个都要
python download_egolifecap.py --kind all

# 全量
python download_egolifecap.py --kind all --participant all --day all

# 下载到别处（默认 ./EgoLifeCap）
python download_egolifecap.py --out-dir D:/data/MyCaptions
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--kind` | `dense` | `dense`=DenseCaption, `transcript`=Transcript, `all`=两者 |
| `--participant` | `A1_JAKE` | 参与者，可多选；`all` = 全部 |
| `--day` | `DAY1` | 天，如 `1`/`DAY1`/`all`，可多选 |
| `--method` | `http` | `http`=并行直连(最快,无依赖); `hf`=huggingface_hub(稳但慢); `git_lfs`=极慢不推荐 |
| `--out-dir` | `./EgoLifeCap` | caption 保存根目录 |
| `--token` | 无 | HF token（公开数据集可省略） |
| `--max-workers` | `8` | `--method http` 时的并发线程数 |

⚠ `--out-dir` 的**末尾目录名**：`--method http` 无限制（任意名字都行）；`--method hf` 要求末尾名为 `EgoLifeCap`（因为 HF 仓库路径带 `EgoLifeCap/` 前缀）。要下到自定义名字的目录，用默认的 `http` 即可。

---

## 依赖

- `download_video.py` 和 `download_egolifecap.py --method hf`：需要 `pip install huggingface_hub`
- `download_egolifecap.py --method http`（默认）：**无第三方依赖**，仅用标准库
- `--method git_lfs`：需要 `git` + `git lfs`（不推荐）

---

## 常见用法组合

```bash
# 1. 最小验证集：拉 A1_JAKE/DAY1 的视频和 caption，然后直接标注（零额外参数）
python download_video.py
python download_egolifecap.py --kind all
python caption_pipeline.py --participant A1_JAKE --day 1

# 2. 只看官方 caption，不标注（几秒搞定）
python download_egolifecap.py --kind all --participant all --day all

# 3. 数据放别处
python download_video.py --out-dir D:/data/videos
python download_egolifecap.py --out-dir D:/data/captions --kind all
python caption_pipeline.py --participant A1_JAKE --day 1 \
    --src-dir D:/data/videos
```

# castle-batch-v6：镜像并发与 schedule 修复

日期：2026-09-20。此文件记录版本内容与验证；云端运行进度仍以带时间戳的 STATUS.md 为准。

已发布至 `hf://buckets/Ligant/castle-code/castle-batch-v6`。实现提交见下方 manifest 的 `provenance.git_commit`，tag 为 `castle-batch-v6`。

## 兼容性

v6 只改变外围文件，核心 `code_hash` 保持为
`8c765a6f3dfb617b75afde6a46e390d261603f08e8517962f8c265ec0e663095`。
11 个身份文件和 4 个提示词与 v3/v4/v5 逐字节相同，因此 v3/v4/v5 的 run 都可以改挂 v6 继续 tick。
v1/v2 哈希不同，仍不在兼容范围内。state 中的参数、提示词和范围继续沿用原值。

## 改动一：镜像改为有界并发

现状（v5）是严格串行的逐对象循环。用三次真实 tick 实测其成本：

| run | 镜像对象 | 总字节 | 墙钟 | 吞吐 | 每对象 |
|---|---:|---:|---:|---:|---:|
| allie（收 annotation） | 1166 | — | 114.5 s | — | — |
| allie（收 review） | 347 新增 + 1166 跳过 | 23.98 MiB | 113.8 s | 1.76 Mbps | 76 ms |
| bjorn 15-20（收 annotation） | 1396 | 28.70 MiB | 136.6 s | 1.76 Mbps | 98 ms |

对象中位数只有 **4.9 KiB**（68% 小于 10 KiB），总量约 24–29 MiB，因此瓶颈是**往返次数**而不是带宽。
从三次运行的耗时反解还可以得到：一次 unchanged 检查约 69 ms、一次复制约 98 ms——**"已存在、跳过"只比复制便宜约 30%**，
所以每次 tick 的成本约为 `O(已镜像总数)`，而不是 `O(新增数)`。

只读探针（`_bench/mirror_latency_probe.py`，同一对象集重复下载，不写任何云端存储）：

| workers | 150 个小对象 | 每对象 | 加速比 |
|---:|---:|---:|---:|
| 1 | 41.93 s | 279.5 ms | 1.0x |
| 4 | 10.50 s | 70.0 ms | 4.0x |
| 8 | 5.43 s | 36.2 ms | 7.7x |
| 16 | 2.93 s | 19.5 ms | **14.3x** |
| 32 | 1.77 s | 11.8 ms | 23.8x |

结论：这是纯 I/O 受限的工作，线程近乎线性地吃满。因此 v6 只把**每个对象的往返**放进线程池
（`CASTLE_MIRROR_WORKERS`，默认 **16**，上限 64），遍历本身仍是顺序的。

**并发不改变语义**：复制的字节、`copied`/`unchanged` 计数、冲突判定、GCS generation 与 SHA-256 校验、
临时文件加 `os.replace` 的整字节发布都保持不变。失败改为收集后**按对象名排序取第一个**再抛出，
因此同一对象集在任何并发度下报同一个错误。成功事件新增 `workers` 字段。

镜像范围不变：只复制 `results/{audio,annotation,review}/<clip_id>.json`。
媒体、`raw`、`errors`、`requests`、`final` 都不参与——`media/` 连对象名都不会被枚举，测试断言这些路径的下载次数为 0。
`CASTLE_MIRROR_WORKERS` 是运行参数，不属于标记身份。

## 改动二：schedule 使用 `@hourly`

创建 schedule 时 `batch_jobs.py` 原写作 `scheduled run hourly`。`hf` 1.30.0 的 `--help` 把 `hourly`
列为合法值，但它把该值原样转发给 API（`huggingface_hub/hf_api.py` 的 `create_scheduled_job`），
服务端回报 `Bad request: Invalid CRON expression`。API 接受的是 CRON 表达式，官方示例为 `schedule="@hourly"`。
修复前 `batch_jobs.py hourly-tick --execute` **无法建立 schedule**；本次实测该失败一次，
随后用改正当后的 argv 手工建立了 bjorn 15-20 的 schedule（已记入 ledger 的 audit 行）。

## 验证

- 完整离线测试：**290 passed, 1 skipped in 31.25s**，隔离目录 `_test/mirror-threads-20260920-02`：
  `./_test/runtime/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=_test/mirror-threads-20260920-02`
- 镜像测试按 `workers` 参数化（1/4/16）断言每个对象都被发布且计数精确、重放全部计入 `unchanged`、无残留 `.tmp`；
  新增环境变量与显式参数的边界测试（0/65/超上限/非整数/空值），以及"环境变量非法时仍完成 tick、镜像报失败并非零退出"。
- 冲突测试按 1 与 8 线程断言**报同一个** `MirrorConflict`，且未发生任何下载。
- 路径安全测试扩充为显式包含 `media/`、`raw/`、`errors/`、`final/`、`requests/`，断言零下载、零落盘。
- 发布后按 manifest 白名单回读，逐文件 SHA-256 与核心哈希核对（记录见下方"发布回读"）。

## 尚未验证

- **未在真实 HF 容器内做并发前后的 A/B**：v6 的收益是按探针外推的（探针在作者本机网络下测得，
  绝对延迟比容器内更高，只用于证明扩展性）。首次用 v6 tick 时应核对日志中
  `batch_stage_mirror` 的 `workers`/`copied`/`unchanged` 与 HF 文件数。
- **HF 挂载写入侧的并发特性未单独测量**：`/output` 是多写入者共享的挂载，16 路并发写入的
  P95 延迟与挂载实现有关；若观测到异常，可下调 `CASTLE_MIRROR_WORKERS` 后重跑（幂等，不会重复付费）。
- 未改动的已知项：仍不镜像 `errors/`；`10-14` 因钉在 v2 而无法使用镜像；`releases/castle-batch-v4`/`v5`
  的 `published_utc` 仍为 null。

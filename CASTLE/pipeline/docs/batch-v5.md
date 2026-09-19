# castle-batch-v5：提前调度与阶段结果镜像

日期：2026-09-20。此文件记录版本内容与验证；云端运行进度仍以带时间戳的 STATUS.md 为准。

已发布至 `hf://buckets/Ligant/castle-code/castle-batch-v5`。实现提交 `520ca56`，manifest 提交及 release tag 指向 `0c9ea8a`。发布清单核对为 23 个文件，无缺失或额外文件；随后在 `2026-09-19T21:32:41Z`（香港 9 月 20 日 05:32）回读全部文件，逐文件 SHA-256 和核心哈希均通过。只读核验报告保存在 `_test/castle-batch-v5-readback-20260920-01.report.json`。

## 兼容性

v5 只改变外围文件，核心 `code_hash` 保持为
`8c765a6f3dfb617b75afde6a46e390d261603f08e8517962f8c265ec0e663095`。
11 个身份文件和 4 个提示词与 v4 逐字节相同，v3/v4 的同哈希 run 可使用 v5 继续 tick。
v1/v2 的哈希不同，不在这次兼容范围内。state 中的参数、提示词和范围继续沿用原值。

release 名区分完整发布物，manifest 对外围文件也记录哈希；本次改动已发布到
`hf://buckets/Ligant/castle-code/castle-batch-v5`。后续代码改动另建新 release，禁止覆盖 v5 或更早版本。
新增 `batch_tick.py` 位于仓库根目录并列入 `auxiliary_files`，没有改变身份文件集合。

## 行为

- 显式指定版本的 `hourly-tick` 可以在预处理完成前建立 schedule；没有 GCS state 时正常退出并记录等待，认证等错误不吞掉。
- launcher 检测挂载目录中的 `batch_tick.py`。v5 调用这个入口；旧发布物继续使用原 tick，定时执行仍支持等待 state。
- v5 外围入口校验 project/location、GCS run prefix 和核心哈希，在同一容器中运行原 tick，然后复制 GCS `results/{audio,annotation,review}/<clip_id>.json` 到 HF 相同相对路径。
- 已有阶段文件可补齐；相同内容跳过，已观察到的同名不同内容报错。原 tick 返回非零时仍尝试同步已保存结果，且保留失败退出码；镜像失败也返回非零。
- 不修改 GCS state、请求参数或核心收集逻辑；不从 HF 镜像恢复，也不增加模型请求。缺少 review 时不创建虚构结果。
- 同一 run / HF 输出前缀只允许一个写入 worker。目标文件在下载前和发布前检查，但不提供跨挂载原子冲突保证，不引入新锁系统。
- 已有 schedule 不会自动更新，需要使用新版 launcher 和 v5 挂载后才能使用镜像功能。`submit` 与创建 schedule 仍是两条命令；终态 schedule 仍由操作者暂停。

## 验证

新增测试先复现旧行为不支持外围分发与提前调度；镜像测试覆盖三个阶段、补齐、重放、跳过音频结果、内容冲突、generation/hash、非法路径、缺失 state、版本拒绝和失败退出码。复核后增加下载期间目标发生变化与安全冲突日志的回归测试。

完整离线测试：**257 passed in 25.68s**。

新增实际 v5 manifest/tag 后，补充 v5 → v3/v4 兼容组合与 tag 内容核验：**47 passed in 2.58s**（`tests/test_batch_jobs.py tests/test_release_manifest.py`，隔离目录 `_test/stage-mirror-20260920-06-tagged`）。没有变更任何已发布的运行文件。

```powershell
.\_test\runtime\Scripts\python.exe -m pytest tests -q --basetemp=_test/stage-mirror-20260920-05-full -p no:cacheprovider
```

该路径是此次已使用的隔离目录；再次验证请换一个新的专用目录。Windows 沙箱的 pytest 临时目录权限问题通过在沙箱外运行同一套离线测试解决，没有修改正式输出。

没有为验证功能启动实际 HF worker、创建 schedule 或推进已有 Vertex Batch。离线测试不代表已验证真实 HF 挂载上的完整执行；部署后首次授权 tick 应查看 `batch_stage_mirror` 的 copied/unchanged 计数及 HF 文件。

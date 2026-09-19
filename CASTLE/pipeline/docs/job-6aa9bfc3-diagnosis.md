# day1-allie-08-v1 失败诊断与修正

Job：Ligant/6aa9bfc35527934177ee6aeb。输出：hf://buckets/Ligant/castle-output/day1-allie-08-v1。

2026-09-16 排查时 Job 仍 RUNNING，使用 gemini-3.8-flash / standard，处理 main/day1/Allie/video/08.mp4 全小时，start-clip=0，workers=3。日志是旧版每clip一行，尚未包含后来新增的细粒度日志。

发现连续 annotation/validation 失败后已取消该 Job，并再次通过 API 确认 CANCELED。HF Jobs 这里是取消，不是可以原地恢复的暂停；终止时尚未保存的在途请求无法保证留存。未删除或覆盖 bucket 内容，未自动重启。

下载后的快照包含27个final、14个annotation.invalid、15个error、44个audio、28个annotation和16个review文件。42个已记录终态的片段中，27个完成、15个失败（约35.7%）。15个失败里14个是本地校验、1个是quota重试耗尽；校验问题是主要原因。

按旧校验器的首个报错分类：

| 原因 | 片段数 |
|---|---:|
| segments未严格按start_sec、ID排序 | 9 |
| ongoing end不在clip末尾 | 4 |
| ongoing start不在clip起点 | 1 |

其中00038在修复排序后还会暴露边界问题；合计6个片段涉及7处边界标记，因此两类并非完全互斥。

实际例子：00013的21.6秒事件排在21.5秒事件前，动作及时间本身有值，但整段被排序规则拒绝。00012的一句话结束于29.8秒，boundary.end为ongoing，旧规则要求接近30秒，于是丢弃整个30秒标注。00010的一个事件从20秒开始却标ongoing start，模型可能在表达“首次看到时已在进行”，与我们定义的“被clip起点截断”不一致。

## 改动

新增 normalize_annotation，位于模型返回与严格校验之间：

1. 按start_sec、segment_id稳定排序；仅在activity_chain包含完全相同的引用集合时同步排序，不补缺失／虚构引用。
2. 远离clip边缘的ongoing转换为uncertain，保持原有起止秒、动作、说话、人物和证据不变，添加明确不确定性说明。
3. 保存raw_data和逐项normalization记录；边界修正令最终review_required=true。仅排序不自动要求人工复核。
4. 继续严格检查越界／非法时间、结构类型、缺失证据、未知人物／来源和重复ID；不通过填空、删事件或扩展时间让结果“看似成功”。
5. 失败诊断现在保存validation_reason，不再只显示ValueError；提示词增加ongoing与首次观察、29.8秒结尾的区别说明。

严格校验器本身保留，可用于确认保存结果已经规范化。此次针对主标注阶段修复，不通过放宽复核阶段的身份／时间约束来掩盖其他问题。

## 离线复验与产物

原始快照：`_test/diagnosis-6aa9bfc3/snapshot/`。

逐条诊断：`_test/diagnosis-6aa9bfc3/diagnosis.json`。

14份规范化后的独立产物：`_test/diagnosis-6aa9bfc3/recovered/`。文件内保留original_fingerprint、原始路径、raw_data、data、normalization和原usage。它们标记为annotation_valid_review_pending、final_clip_complete=false：是已挽回的主标注结果，不是虚构的完整成功clip，也没有修改旧指纹混入新Job。

14份均通过剩余主标注校验，裁剪请求数量／帧引用／坐标也通过；独立检查确认27份现有final不受规范化影响。整个过程没有重新调用大模型。

测试：`_test/runtime/Scripts/python.exe -m pytest tests -q --basetemp _test/schema-normalize-full-final` → **103 passed in 7.28s**。新增测试覆盖原输出不被改写、重叠排序、边界不被移动、原始结果与修正审计保留，以及非法时间／证据／人物引用仍失败。

后续恢复应显式使用这些已保存阶段或做可追溯迁移，避免因为新版代码指纹变化就整小时重新标注。当前未实现自动跨版本检查点迁移，未将这些独立离线产物伪装成新版本成功检查点。建议恢复前先小范围核查规范化结果和待复核边界。

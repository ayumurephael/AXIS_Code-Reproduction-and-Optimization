# 本地可运行性补丁

`server_pipeline` 的主体来自 THU-IE 2228 节点的只读快照。为使副本能在本机
独立运行，当前只做了以下修正：

1. `src/mvaxis/generator.py`：`sample_question_specs` 实际定义在
   `question_template.py`，因此把错误的 `question_provider` 导入改为从
   `question_template` 导入。函数实现未改动。
2. `legacy/TSAD_dataset_gen-axis/src/generate_dataset.py`：Windows 默认采用
   单进程，详见该目录的 `LOCAL_PATCHES.md`。

服务器原文件的下载时哈希保存在 `SOURCE_MANIFEST.json`，因此可据此区分原始
快照与本地补丁。

# 本地兼容补丁

此目录是 `upstream/Datasets-RCD` 的兼容副本，放在服务器代码原本预期的路径下。

- `src/generate_dataset.py`：Windows 下将默认 worker 数由 80 改为 1，规避
  `ProcessPoolExecutor` 的 61 worker 上限以及动态加载模块在 spawn 模式下的
  序列化问题。Linux 下仍使用上游默认值 80。该补丁只改变并行方式，不改变
  序列、异常、标签或 attribute 的生成规则。

上游原始版本完整保留在 `../../../upstream/Datasets-RCD`，未作修改。

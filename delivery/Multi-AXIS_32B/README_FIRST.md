# Multi-AXIS-32B 离线训练交付说明

本交付对应 Git 分支 `multi-axis-VL-32B`，基线为 `multi-axis-VL`
提交 `ab2efb872b6b43e8afadc14d80a45c8b2ebd0d7e`。冻结视觉语言模型固定为
`Qwen/Qwen3-VL-32B-Instruct` revision
`0cfaf48183f594c314753d30a4c4974bc75f3ccb`。32B Hint Tuner 必须从 Epoch 0
开始，禁止迁移任何 8B Hint Tuner、优化器、调度器或问题语义缓存。

## 目录

```text
Multi-AXIS_32B/
  README_FIRST.md
  DELIVERY_MANIFEST.json
  SHA256SUMS.txt
  source/                 # 可直接运行的源码快照、git bundle
  models/Qwen3-VL-32B-Instruct-0cfaf48183f5/
  datasets/archives/      # 两份原始审计归档
  datasets/extracted/     # 训练/验证/五测试集所需原始 JSONL
  manifests/              # seed42 grouped 90/10 与正式测试 manifests
  images/                 # 34,351 张正式预渲染 PNG 与 image_manifest.jsonl
  assets/timercd/         # 冻结 TimeRCD checkpoint
  assets/fonts/           # 授权的 Times New Roman 四字体
  environment/            # requirements 与安装说明
  scripts/                # 校验、预检、训练/测试模板
```

进入公司后，U 盘只用于传输。先将整个目录复制到计算集群高速存储，再做
哈希校验、环境安装和训练；不要在跳板机或 U 盘上训练，不要把 API key、SSH
密钥或密码写入本目录。

## 首次使用顺序

1. `python scripts/verify_delivery.py verify --root /path/to/Multi-AXIS_32B`
2. 创建 Python 3.10/3.11 环境，并安装与公司 CUDA 驱动匹配的 PyTorch。
3. `pip install -r environment/requirements-multi-axis-32b.txt`
4. `pip install --no-build-isolation -r environment/requirements-multi-axis-32b-flash.txt`
5. 在每个计算节点运行 `python -m tools.multi_axis.hardware_preflight --model-dir ...`。
6. 校验已有 `images/image_manifest.jsonl`；若不复制图像，则从确定性脚本重新渲染。
7. 根据明日实际 GPU 数量、显存和节点拓扑，先做 2–5 个 optimizer-step 的独立
   benchmark。只有 benchmark 无 OOM、FlashAttention-2 审计通过、显存有安全余量
   后，才能固定 `world/micro/accumulation/nodes` 并启动正式 20 Epoch。

若权重目录需要重新获取，唯一注册入口是：

```bash
python -m tools.multi_axis.download_pinned_32b \
  --output-dir /fast/storage/models/Qwen3-VL-32B-Instruct-0cfaf48183f5
```

该命令固定 repo/revision/总字节数/14 个分片并生成逐文件 SHA-256；不要改用
浮动 `main` revision。

## 32B 架构差异

- 文本隐藏宽度：5120；词表：151,936；文本层：64。
- 视觉塔：hidden 1152、depth 27、16 heads、输出 5120、patch 16、merge 2。
- Prototype / Step / Fixed / question projection 随隐藏宽度动态构造。
- 原型 cross-attention 为 20 heads，`head_dim=256`；Channel/Joint 各 4 heads。
- 本地模型目录必须含交付脚本产生的 `SNAPSHOT_MANIFEST.json` 和
  `SNAPSHOT_COMPLETE`。模型构造、checkpoint 加载和 resume 都会 fail closed。

## 显存边界

当前训练器是数据并行：每个 rank 都复制完整冻结 32B VLM。BF16 模型文件约
66.7 GB（62.1 GiB），还没有包含 Hint Tuner、Adam、CUDA workspace、视觉/文本
激活。因此“80 GB GPU 能放下权重”不等于“能训练”。如 micro=1 的实测仍 OOM，
必须在知道公司硬件和互联后实现并审计 FSDP/ZeRO-3 等分片方案，不能擅自量化、
降低像素预算或改变正式模型。

## 数据与评测

训练采用相同 seed42 grouped 90/10 划分（61,053 train / 6,750 validation，17 条
空 teacher answer 已过滤）。测试包含 Teacher-Eval 478New、SMD、SWaT、
LEMMA-RCA、VTA Articulary。解码合同维持 `beam=5`、`do_sample=false`、
`max_new_tokens=1000`。外部 Judge 的 key 与 endpoint 不属于离线包。

# CSTBIR
**Composite Sketch + Text Based Image Retrieval（中文说明）**

本仓库提供一个三模态检索与定位框架：文本、素描与图像共同驱动检索。当前实现基于 CLIP 的视觉与文本编码器，并额外引入 ViT 素描编码器、素描引导注意力、目标检测头以及素描重建分支，实现多任务联合训练。

## 环境配置
- Python >= 3.8.16（推荐使用 Conda）
- PyTorch 1.13.1 + CUDA 11.6
- 其他依赖详见 `requirements.txt` 与 `environment.yaml`

快速安装步骤：
```bash
conda create -n cstbir python=3.8.16
conda activate cstbir
conda env update --file environment.yaml --prune
pip install -r requirements.txt
```

## 数据准备
请将所有数据放置在 `data/` 目录下，对应路径可在 `config.yaml` 中修改。

1. **Visual Genome 图像**：从 https://homes.cs.washington.edu/~ranjay/visualgenome/index.html 下载并解压到 `data/images/VG_Dataset/`。
2. **QuickDraw 素描**：从 https://github.com/googlecreativelab/quickdraw-dataset 获取 ndjson 文件，按类别放置于 `data/sketch/ndjson/`。本项目按 `label_xxxxx.ndjson` 命名，每行对应一幅素描。
3. **CSTBIR 数据集标注**：从 https://drive.google.com/drive/folders/1UgAZc5rtbO0MQ37WHS4hGQhXlqMPT6Lg 下载 `dataset.json` 并置于 `data/dataset/`。示例字段包括：
         - `text`: 文本描述
         - `image`: 图像文件名
         - `label`: 对象类别
         - `sketch`: 素描文件名（如 `bed_23725.jpg`，其中下划线后的数字对应 ndjson 行号）
         - `split`: 样本划分（`train`/`val_large` 等）

运行时，数据加载器会：
- 读取 `dataset.json` 并根据 `split` 过滤样本；
- 解析素描文件名，定位到对应 ndjson 中的行，通过矢量笔画直接渲染为 `1x224x224` 的张量（不生成中间图片文件）；
- 按批次返回无冲突的图像/文本/素描/类别标签。

> 提示：首次访问某类素描会扫描整个 ndjson 文件并缓存偏移量，时间较长；后续访问会直接 seek 到指定行以提升速度。

## 模型结构
`cstbir_model.CSTBIRModel` 整合以下模块：
- **文本编码器**：沿用 CLIP 文本 Transformer，输出 `$h_{cls}^T$ 用于对比学习与文本分类；
- **素描编码器**：以单通道 ViT 结构复用 CLIP 配置，提取 $h_{cls}^S$；
- **素描引导注意力**：利用 $h_{cls}^S$ 对图像序列执行点积注意力，得到关注对象的图像特征；
- **多任务头**：
        1. 对比学习（InfoNCE）；
        2. 文本/图像分类（交叉熵）；
        3. 检测头（YOLO 风格，输出 `S x S x (5B + C)`）；
        4. 素描重建（上采样网络，使用 BCE + Dice 损失）。

损失权重、BCE/Dice 系数、检测栅格大小等均可在 `config.yaml` 的 `loss` 与 `detector` 小节调整。若暂时没有真实 bbox 标签，`detection_loss` 会返回 0 占位，可在日后补充监督时替换。

## 训练与评估
配置请统一修改 `config.yaml`：
- `model`：学习率、批大小、CLIP 变体等；
- `data`：数据路径与划分名称；
- `training`：GPU 开关、模型保存目录、训练轮数；
- `loss`、`detector`：各损失权重、重建损失系数、检测栅格参数等。

执行训练：
```bash
CUDA_VISIBLE_DEVICES=XX python run.py
```

训练脚本会：
1. 加载 CLIP 权重并构建三模态模型；
2. 从数据集中采样无冲突的 (图像, 文本, 素描) 批次；
3. 计算五项损失并按权重求和反向传播；
4. 记录训练/验证损失与检索准确率，并按 epoch 保存 checkpoint（`data/model/model_checkpoint_*.pt`）。

> 注意：如果需要单独评估某一任务，可在 `run.py` 中调整 `evaluate` 逻辑或增加指标统计。

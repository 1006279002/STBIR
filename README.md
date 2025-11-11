# STBIR_demo

本项目实现了一个以 CLIP 为基础、支持“文本 + 素描 → 图像”检索的三模态框架。“分阶段对齐 + 多任务度量学习”范式：

- 仅保留一个 ResNet50 素描编码器与 CLIP 原生图像/文本编码器，统一映射至 512 维特征空间；
- 通过 InfoNCE、余弦三元组、ArcFace 分类头以及图像-素描融合一致性约束实现多源监督；
- 按“素描 → 图像 → 文本”顺序逐阶段解冻对应分支，逐步对齐三种模态；
- 数据层面支持跨模态Hard Negatives采样与课程式噪声增强，以提升鲁棒性。

下文介绍现有代码结构、数据准备流程以及训练评估方式，便于直接复现或撰写论文实验部分。

## 核心特性概览
- **统一语义空间**：`cstbir_model.MultiStageSBIRModel` 将三种模态特征 `l2` 归一化后映射到同一球面，用余弦距离衡量跨模态相似度。
- **阶段式对齐**：`configure_stage` 仅解冻目标模态参数（素描/图像/文本），其余模态作为冻结锚点提供监督，缓解多模态同步更新时的梯度冲突。
- **多任务损失组合**：
1. InfoNCE 对比损失（以融合特征为 anchor）；
2. 余弦三元组损失（设定 margin 防止正负样本塌缩）；
3. 图像-素描融合一致性损失（`_fuse_quad` 拼接象限图像，引导视觉分支聚焦素描指示区域）；
4. ArcFace 分类头（可利用真实类别或虚拟聚类标签增强判别力）。
- **课程噪声增强**：`dataloader.CurriculumNoiseSchedule` 会随 epoch 自动调节图像/素描噪声概率与强度，模拟素描粗糙度与拍摄退化。
- **可控负样本池**：`scripts/generate_manifest.py` 会为每条样本分配跨模态负例，训练时 `SBIRDataset` 固定采样数量确保 batch 内含 Hard Negatives 样本。

## 数据集
采用QUML-shoe-chair-V2数据集进行实验

## 环境配置
- Python >= 3.8.16（推荐 Conda）
- PyTorch 1.13.1 + CUDA 11.6（或根据显卡调整）
- 依赖见 `environment.yaml` 

快速安装：
```bash
conda create -n cstbir python=3.8.16
conda activate cstbir
conda env update --file environment.yaml --prune
```

## 数据准备与 Manifest 生成
1. **整理原始数据**：`config.yaml` 的 `data.datasets` 段以类别（如 `chair`、`shoe`）描述训练/测试文本、图像、素描及噪声目录，当前示例数据位于 `datasets/` 下，可直接复用或按相同结构替换。文本文件形如 `image_name\t自由描述`，素描与图像文件名需共享 stem（如 `xxx.png` 对应 `xxx_0.png` 素描）。
2. **生成 manifest**：根据配置执行
         ```bash
         python -m scripts.generate_manifest --config config.yaml --seed 110
         ```
         
该脚本会：         
- 为训练/验证/测试划分聚合正样本（图像、文本、素描）；
- 在同类别或同虚拟类簇内优先抽取负样本，数量由 `negatives_per_modality` 控制；
- 写入 `manifests/train|val|test_manifest.jsonl`，供数据加载器直接读取。
3. **Manifest 字段说明**：单行 JSON 示例
         ```json
         {
                 "id": "chair:sample_001",
                 "category": "chair",
                 "image": "datasets/.../trainB/sample_001.png",
                 "text": "描述",
                 "positive_sketches": ["datasets/.../trainA/sample_001_0.png"],
                 "negative_sketches": [...],
                 "negative_images": [...],
                 "negative_texts": [...],
                 "virtual_class": "cluster_07"
         }
         ```
         `SBIRDataset` 会根据 `negatives_per_modality` 再次随机采样需要的负例数量。

## 模型与训练配置
`config.yaml` 是运行入口，关键字段如下：

- `model`: 指定 CLIP 变体（默认 `ViT-B/32`）、素描编码器骨干（`resnet50`）、特征维度及分类头 margin 参数。
- `training`: 批大小、DataLoader 线程、是否启用 GPU 预取、阶段列表等。`stages` 数组定义顺序、epoch、学习率与权重衰减。
- `loss`: 温度系数、triplet margin、融合 margin 以及四个损失项权重。
- `data.curriculum_noise`: 课程噪声调度；可按阶段重载起始 epoch、上限概率/强度，或关闭某些阶段噪声。

## 训练流程
运行入口位于 `run.py`：

```bash
python run.py --config config.yaml
```

- 脚本将加载 manifest 构建 `SBIRDataset`，并在每个阶段前调用 `model.configure_stage(target)` 解冻对应分支。
- 训练循环内通过 `forward_stage` 计算 InfoNCE、Triplet、Fusion、ArcFace 四项损失，并在需要时裁剪梯度至 1.0。
- 验证损失最低的模型会保存在 `checkpoints/best.pt`。
- 可使用 `--stage sketch|image|text` 单独训练指定阶段，或者 `--resume` 恢复之前的 checkpoint。

## 检索评估
完成全部阶段后，使用 `scripts/retrieve.py` 在测试集上评估检索性能：

```bash
python -m scripts.retrieve --config config.yaml --checkpoint checkpoints/best.pt
```

- 脚本会离线编目录中的所有候选图像特征，再将查询文本与素描平均后归一化，计算与图库的余弦相似度；
- 输出指标包括 Median Rank、mAP@all、Top-1/5/10 准确率，并在 `outputs/retrieval/results.json` 存储详细结果与类别统计。

## 代码结构速览
- `cstbir_model.py`：核心模型、损失与阶段控制逻辑。
- `dataloader.py`：manifest 解析、三模态加载、噪声增强、负样本拼接。
- `run.py`：三阶段训练脚本，封装日志与 checkpoint 管理。
- `scripts/generate_manifest.py`：根据原始数据构建带负样本的 manifest。
- `scripts/retrieve.py`：检索评估脚本。
- `utils.py`：通用工具（配置读取、CUDA 预取、平均计数器等）。


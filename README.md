# Med_Clip：面向甲状腺超声的多模态检索模型

本项目实现了一个医学超声场景下的 CLIP 增强版本（Ultrasound-CLIP V2），目标是让**超声图像**与**医学文本描述**在同一特征空间内对齐，并提升图文检索能力。

与原始 CLIP 相比，本项目重点引入了：

- 医学文本编码器（BioClinical-BERT）
- 病灶结构异构图编码（DGL GraphEncoder）
- 语义先验驱动的软标签对比学习（UDAF 风格）

---

## 1. 项目目标

给定一张超声图像，检索最匹配的描述文本；或者给定一段文本，检索最相关的超声图像。  
该任务可用于：

- 医学影像检索与病例回顾
- 报告辅助生成前的语义对齐
- 医学多模态预训练下游初始化

---

## 2. 方法总览（核心思想）

本项目采用“三路编码 + 双损失学习”：

1. **图像分支**：CLIP ViT-B/32 提取图像特征  
2. **文本分支**：BioClinical-BERT 提取医学文本特征并投影到 512 维  
3. **图先验分支**：将诊断标签与描述标签构建成异构图，编码成图特征  
4. **融合模块**：Cross-Attention 将图特征融合进文本特征  
5. **训练损失**：  
   - 主损失：语义软标签 CLIP 对比损失  
   - 辅助损失：语义匹配损失（MSE + KL）

总损失形式：

\[
L_{total} = L_{clip\_soft} + \lambda \cdot L_{semantic}
\]

---

## 3. 模型结构与技术细节

### 3.1 图像编码器

- 基座：OpenAI CLIP `ViT-B/32`
- 输出维度：512
- 输出后进行 L2 Normalize

### 3.2 文本编码器

- 模型：`emilyalsentzer/Bio_ClinicalBERT`
- 取 `[CLS]` 向量（768 维）
- 通过 `Linear + Dropout + LayerNorm` 投影到 512 维
- 输出后进行 L2 Normalize

### 3.3 图编码器（GraphEncoder）

图结构由每个样本的标签构成：

- 节点类型：
  - `diagnosis`（诊断）
  - `descriptor`（描述属性）
- 边类型：
  - `diagnosis -> descriptor` (`described_by`)
  - `descriptor -> diagnosis` (`rev_described_by`)

实现细节：

- 使用 `DGL HeteroGraphConv + GraphConv`
- 节点使用可学习 embedding（含标签词表映射）
- 多层图卷积后按节点类型做 mean pooling，再求和聚合
- 最终投影到 512 维并归一化

### 3.4 文本-图融合（CrossAttentionFusion）

- 使用多头注意力（8 heads）进行 text-query / graph-key-value 融合
- 残差 + LayerNorm + FFN
- 引入可学习门控系数 `alpha`（限制上限），避免图信息过强干扰文本
- 输出仍保持 512 维

### 3.5 语义先验与软标签对比学习

项目先根据 `full_data.json` 的 5 个任务字段构建标签相似度矩阵：

- `Diagnosis`
- `Shape`
- `Margins`
- `Position`
- `Size`

每个任务维度内，标签间相似度由共现 **Jaccard** 计算得到，保存为：

- `similarity_matrices/task1_tag_normalized.npz`
- ...
- `similarity_matrices/task5_tag_normalized.npz`

训练时，将 batch 内样本两两语义相似度组成 `target_sim`，再通过 softmax 形成软标签分布，用于替代传统 one-hot 硬标签对比学习目标。

### 3.6 优化策略

- 分层学习率：
  - 视觉骨干（CLIP）和文本骨干（BERT）：较小学习率
  - 新增模块（投影层/融合层/图编码器）：较大学习率
- 优化器：`AdamW`
- 学习率调度：`Warmup + Cosine`
- 数值稳定：
  - 强制 `float32`
  - `logit_scale` clamp
  - NaN/Inf 检查与跳过异常 batch
  - 梯度裁剪 `clip_grad_norm_`

---

## 4. 数据格式要求

数据目录示例：

```text
dataset/
  full_data.json
  train.jsonl
  valid.jsonl   (或 val.jsonl)
  test.jsonl    (评估 test split 时需要)
  images/       (或 image/)
  similarity_matrices/  (由脚本生成)
```

### 4.1 `train.jsonl/valid.jsonl/test.jsonl`

每行一个样本，至少包含：

- `image`: 图像文件名
- `caption`: 文本描述

### 4.2 `full_data.json`

每条记录至少包含：

- `media_name`
- `Diagnosis`
- `Shape`
- `Margins`
- `Position`
- `Size`

---

## 5. 环境安装

### 5.1 Python 依赖

```bash
pip install -r requirements.txt
pip install transformers dgl matplotlib numpy pillow huggingface_hub
```

> 说明：`requirements.txt` 仅包含最小依赖，训练本项目还需额外安装上面的包。

### 5.2 自动下载数据（可选）

```bash
python auto_download.py
```

默认会下载 `JJY-0823/US-365K` 到 `./dataset`。

---

## 6. 训练前准备

先生成语义先验矩阵：

```bash
python generate_similarity_matrices.py --dataset_dir dataset
```

生成成功后，会在 `dataset/similarity_matrices/` 下看到 5 个 `.npz` 文件。

---

## 7. 训练

### 7.1 标准训练

```bash
python train.py --dataset_dir dataset --epochs 10
```

### 7.2 快速冒烟测试（检查流程是否跑通）

```bash
python train.py --dataset_dir dataset --max_steps 10 --batch_size 8
```

### 7.3 断点续训

```bash
python train.py --dataset_dir dataset --resume checkpoints/checkpoint_epoch5.pt
```

### 7.4 常用可调参数

- `--batch_size`（默认 32）
- `--epochs`（默认 10）
- `--lr_backbone`（默认 5e-7）
- `--lr_new`（默认 5e-5）
- `--semantic_weight`（默认 0.05）
- `--soft_label_temp`（默认 0.5）
- `--graph_hidden`（默认 128）
- `--graph_layers`（默认 2）

---

## 8. 评估（图文检索）

### 8.1 评估命令

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset_dir dataset --split test
```

也可以在验证集评估：

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset_dir dataset --split valid
```

### 8.2 指标说明

评估输出双向检索 Recall@K：

- Image → Text：`i2t_R@1 / 5 / 10 / 50`
- Text → Image：`t2i_R@1 / 5 / 10 / 50`

评估结果会保存到 checkpoint 目录下：

- `eval_test_results.json` 或 `eval_valid_results.json`

---

## 9. 当前实验结果（基于现有训练日志）

根据 `checkpoints/train.log`（2026-04-23）：

- 设备：CUDA
- 参数规模：Total 264.0M（Trainable 264.0M）
- 数据规模：
  - Train: 1180 samples
  - Val: 393 samples
- 本次配置（日志中记录）：
  - `epochs=20`
  - `batch_size=128`
  - `lr_backbone=2e-5`
  - `lr_new=5e-4`
  - `semantic_weight=0.2`

### 9.1 损失收敛现象

- Epoch 1：Train Loss `11.2632`，Val Loss `5.1965`
- Epoch 5：Train Loss `5.0030`，Val Loss `4.3079`
- Epoch 10：Train Loss `4.9534`，Val Loss `4.2831`
- Epoch 18：Train Loss `4.9503`，Val Loss `4.2824`
- 当前最优验证损失：**`4.2824`**

结论：训练在前几个 epoch 快速下降，随后趋于平稳，整体收敛正常。

### 9.2 可视化训练曲线

```bash
python plot_log.py
```

会生成 `loss_curve.png` 用于观察 Train/Val Loss 变化趋势。

---

## 10. 代码结构说明

```text
Med_Clip/
  train.py                          # 训练入口（软标签对比学习 + 语义损失）
  evaluate.py                       # 检索评估入口（R@K）
  generate_similarity_matrices.py   # 生成 5 个任务的语义先验矩阵
  ultrasound_dataset.py             # 数据集与 collate（图像/文本/异构图）
  ultrasound_clip/
    enhanced_clip_model.py          # 主模型（CLIP + BioClinicalBERT + GraphFusion）
    graph_encoder.py                # 图编码器（DGL 异构图卷积）
    graph_builder.py                # 单样本图构建逻辑
    similarity_processor.py         # batch 语义相似度矩阵计算
    semantic_loss.py                # 语义匹配辅助损失
  checkpoints/
    train.log                       # 训练日志
```

---

## 11. 常见问题（FAQ）

### Q1：训练时报 `similarity_matrices` 不存在？

先执行：

```bash
python generate_similarity_matrices.py --dataset_dir dataset
```

### Q2：评估时报找不到 `test.jsonl`？

请确认数据目录下有对应 split 文件；如果只有验证集，可改用 `--split valid`。

### Q3：显存不足怎么办？

- 降低 `--batch_size`
- 先用 `--max_steps` 做小规模冒烟测试
- 评估时设置较小 `--eval_samples`

---

## 12. 复现建议（推荐流程）

1. 安装依赖  
2. 准备 `dataset`（含 `full_data.json` 和各 split）  
3. 运行 `generate_similarity_matrices.py`  
4. 运行 `train.py` 开始训练  
5. 用 `evaluate.py` 输出 R@K  
6. 用 `plot_log.py` 画收敛曲线  

---

## 13. 后续可改进方向

- 增加严格的消融实验：
  - 去掉图融合
  - 去掉软标签，仅硬标签 InfoNCE
  - 不同 `semantic_weight` 和 `soft_label_temp`
- 增加更多评估指标（mAP、nDCG）
- 支持更强视觉骨干（如 ViT-L/14）
- 增加多语言报告文本支持

---

## 14. 参考

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

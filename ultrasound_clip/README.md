# Ultrasound-CLIP

当前甲状腺数据集使用 5 个任务字段：

- `Diagnosis`
- `Shape`
- `Margins`
- `Position`
- `Size`

## 训练前准备

1. 确认数据目录下包含：
   - `full_data.json`
   - `train.jsonl`
   - `valid.jsonl` 或 `val.jsonl`
   - `images/` 或 `image/`
2. 先生成 5 个任务的语义相似度矩阵：

```bash
python generate_similarity_matrices.py --dataset_dir /path/to/dataset
```

会在 `similarity_matrices/` 下生成 `task1~task5` 的 `.npz` 文件。

## 训练命令

```bash
python train.py --dataset_dir /path/to/dataset --epochs 10
```

快速检查（只跑少量 step）：

```bash
python train.py --dataset_dir /path/to/dataset --max_steps 10 --batch_size 8
```
"""
生成 UDAF 语义先验相似度矩阵。

从 full_data.json 中读取 364K 条记录，对 9 个任务维度分别计算标签间的
共现 Jaccard 相似度，输出为 SimilarityMatrixProcessor 所需的 .npz 格式。

用法:
    cd ~/clip/CLIP-main
    python generate_similarity_matrices.py --dataset_dir dataset
"""

import json
import argparse
import numpy as np
from pathlib import Path


TASK_KEYS = {
    1: "Diagnosis",
    2: "Body_system_level",
    3: "Organ_level",
    4: "Shape",
    5: "Margins",
    6: "Echogenicity",
    7: "InternalCharacteristics",
    8: "PosteriorAcoustics",
    9: "Vascularity",
}


def compute_jaccard_matrix(full_data, task_name):
    """计算某个任务维度下所有标签的共现 Jaccard 相似度矩阵。"""
    # 收集每条记录在该任务维度上的标签集合
    tag_sets = []
    all_tags = set()

    for rec in full_data:
        tags = rec.get(task_name, [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        tag_set = set(tags)
        tag_sets.append(tag_set)
        all_tags.update(tag_set)

    tags_list = sorted(list(all_tags))
    n_tags = len(tags_list)

    if n_tags == 0:
        return np.eye(1, dtype=np.float32), np.array(["unknown"])

    # 建立每个标签的样本出现集合
    tag_occurrences = {tag: set() for tag in tags_list}
    for rec_idx, tag_set in enumerate(tag_sets):
        for tag in tag_set:
            tag_occurrences[tag].add(rec_idx)

    # 计算 Jaccard 相似度矩阵
    matrix = np.zeros((n_tags, n_tags), dtype=np.float32)

    for i in range(n_tags):
        set_i = tag_occurrences[tags_list[i]]
        matrix[i, i] = 1.0
        for j in range(i + 1, n_tags):
            set_j = tag_occurrences[tags_list[j]]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            sim = intersection / union if union > 0 else 0.0
            matrix[i, j] = sim
            matrix[j, i] = sim

    labels = np.array(tags_list)
    return matrix, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir / "similarity_matrices"
    output_dir.mkdir(exist_ok=True)

    full_data_path = dataset_dir / "full_data.json"
    print(f"[INFO] Loading {full_data_path} ...")
    with open(full_data_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    print(f"[INFO] Loaded {len(full_data)} records")

    for task_id, task_name in TASK_KEYS.items():
        print(f"\n--- Task {task_id}: {task_name} ---")
        matrix, labels = compute_jaccard_matrix(full_data, task_name)

        outpath = output_dir / f"task{task_id}_tag_normalized.npz"
        np.savez(outpath, matrix=matrix, labels=labels)
        print(f"  Unique tags: {len(labels)}")
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Saved: {outpath}")

        # 验证: 对角线应全为 1.0
        diag_ok = np.allclose(np.diag(matrix), 1.0)
        print(f"  Diagonal check: {'✅ PASS' if diag_ok else '❌ FAIL'}")

    print("\n[SUCCESS] 全部 9 个相似度矩阵生成完毕！")


if __name__ == "__main__":
    main()

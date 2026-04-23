"""
Ultrasound-CLIP 评估脚本 (V2: BioClinical-BERT 文本编码器版本)。

在测试集上评估训练好的模型，计算 Image-Text Retrieval 指标 (R@1, R@5, R@10)。

用法:
    python evaluate.py --checkpoint checkpoints/best_model.pt --dataset_dir dataset
    python evaluate.py --checkpoint checkpoints/best_model.pt --split valid
"""

import os
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import clip
from transformers import AutoTokenizer

from ultrasound_clip.enhanced_clip_model import EnhancedCLIP, BertTextEncoder
from ultrasound_clip.graph_encoder import GraphEncoder
from ultrasound_dataset import UltrasoundCLIPDataset, ultrasound_collate_fn


def compute_retrieval_metrics(image_features, text_features, is_match=None, ks=(1, 5, 10, 50)):
    """计算 Image-Text 双向检索指标 Recall@K。

    Args:
        image_features: [N, D] L2-normalized
        text_features:  [N, D] L2-normalized
        is_match:       [N, N] bool tensor indicating if image i matches text j (semantic match)
        ks: tuple of k values

    Returns:
        dict with i2t_R@k and t2i_R@k
    """
    sim_matrix = image_features @ text_features.T  # [N, N]
    N = sim_matrix.shape[0]
    results = {}
    
    if is_match is None:
        is_match = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)

    # Image → Text
    for k in ks:
        _, topk_indices = sim_matrix.topk(k, dim=1)
        correct = torch.gather(is_match, 1, topk_indices).any(dim=1).float()
        results[f"i2t_R@{k}"] = correct.mean().item() * 100

    # Text → Image
    sim_matrix_t = sim_matrix.T
    is_match_t = is_match.T
    for k in ks:
        _, topk_indices = sim_matrix_t.topk(k, dim=1)
        correct = torch.gather(is_match_t, 1, topk_indices).any(dim=1).float()
        results[f"t2i_R@{k}"] = correct.mean().item() * 100

    return results


def main():
    parser = argparse.ArgumentParser(description="Ultrasound-CLIP Evaluation (V2: BioClinical-BERT)")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--bert_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace BERT 模型名称")
    parser.add_argument("--graph_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--eval_samples", type=int, default=5000,
                        help="用于计算检索指标的最大样本数（防止 OOM）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ===== 1. 加载模型 =====
    print(f"[INFO] Loading CLIP visual backbone: {args.clip_model}")
    base_model, preprocess = clip.load(args.clip_model, device="cpu")
    dim_features = base_model.text_projection.shape[1]

    print(f"[INFO] Loading BERT text encoder: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir="./bio_clinicalbert", local_files_only=True)
    text_encoder = BertTextEncoder(bert_model_name=args.bert_model, out_dim=dim_features)

    graph_encoder = GraphEncoder(
        out_dim=dim_features,
        hidden=args.graph_hidden,
        n_layers=args.graph_layers,
    )
    model = EnhancedCLIP(base_model, text_encoder, graph_encoder)

    # ===== 2. 先加载数据集（提取图结构需要） =====
    split_file = f"{args.split}.jsonl"
    dataset = UltrasoundCLIPDataset(
        jsonl_path=os.path.join(args.dataset_dir, split_file),
        images_dir=os.path.join(args.dataset_dir, "images"),
        full_data_path=os.path.join(args.dataset_dir, "full_data.json"),
        preprocess=preprocess,
        tokenizer=tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ultrasound_collate_fn,
        pin_memory=True,
    )

    print(f"[INFO] Evaluating on {args.split} split: {len(dataset)} samples")

    # ===== 3. 加载 checkpoint =====
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    # 【修复】强制网络层初始化
    for _, texts, graphs, _ in loader:
        if graphs is not None:
            model.graph_encoder._ensure_type_embeddings(graphs, "cpu")
            model.graph_encoder._ensure_convs(graphs, "cpu")
        break
        
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.float()
    model = model.to(device)
    model.eval()

    epoch_info = ckpt.get("epoch", "?")
    val_loss_info = ckpt.get("val_loss", "?")
    print(f"[INFO] Checkpoint from epoch {epoch_info}, val_loss = {val_loss_info}")

    # ===== 4. 提取全部特征 =====
    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch_idx, (images, texts, graphs, image_keys) in enumerate(loader):
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            if graphs is not None:
                graphs = graphs.to(device)

            outputs = model(images=images, texts=texts, graphs=graphs)

            all_image_features.append(outputs["image_features"].cpu())
            all_text_features.append(outputs["text_features"].cpu())

            if (batch_idx + 1) % 100 == 0:
                processed = min((batch_idx + 1) * args.batch_size, len(dataset))
                print(f"  [{processed}/{len(dataset)}] features extracted...")

    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    print(f"[INFO] Feature shapes: images={all_image_features.shape}, texts={all_text_features.shape}")

    # ===== 5. 计算检索指标 =====
    N = all_image_features.shape[0]
    eval_n = min(N, args.eval_samples)

    if eval_n < N:
        print(f"[INFO] Using first {eval_n}/{N} samples for retrieval metrics (防止 OOM)")

    print(f"[INFO] Building semantic identity matrix for evaluation...")
    eval_captions = [dataset.entries[i]["caption"] for i in range(eval_n)]
    eval_captions_np = np.array(eval_captions)
    
    # 构建 [eval_n, eval_n] 的布尔矩阵，如果两个文本字符串完全一致，则互为正样本
    is_match_np = (eval_captions_np[:, None] == eval_captions_np[None, :])
    is_match_tensor = torch.from_numpy(is_match_np)

    metrics = compute_retrieval_metrics(
        all_image_features[:eval_n].to(device),
        all_text_features[:eval_n].to(device),
        is_match=is_match_tensor.to(device)
    )

    # ===== 6. 打印结果 =====
    print(f"\n{'=' * 55}")
    print(f"  Ultrasound-CLIP V2 Retrieval Results ({args.split} set)")
    print(f"  Evaluated on {eval_n} samples")
    print(f"{'=' * 55}")
    print(f"  Image → Text Retrieval:")
    for k in [1, 5, 10, 50]:
        print(f"    R@{k:2d}: {metrics[f'i2t_R@{k}']:6.2f}%")
    print(f"  Text → Image Retrieval:")
    for k in [1, 5, 10, 50]:
        print(f"    R@{k:2d}: {metrics[f't2i_R@{k}']:6.2f}%")
    print(f"{'=' * 55}")

    # 保存结果
    ckpt_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(ckpt_dir, f"eval_{args.split}_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Results saved: {results_path}")


if __name__ == "__main__":
    main()

"""
Ultrasound-CLIP 训练脚本 (V2: BioClinical-BERT 文本编码器版本)。

实现论文中的双损失训练策略:
    L_total = L_CLIP + λ × L_semantic

其中:
    - L_CLIP:     标准 CLIP 对比损失 (InfoNCE)
    - L_semantic: UDAF 语义匹配损失 (MSE + KL Divergence)

用法:
    # 完整训练
    python train.py --dataset_dir dataset --epochs 10

    # 快速冒烟测试 (验证代码无 bug)
    python train.py --dataset_dir dataset --max_steps 10 --batch_size 8

    # 从断点恢复
    python train.py --dataset_dir dataset --resume checkpoints/checkpoint_epoch5.pt
"""

import os
import sys
import time
import math
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip
from transformers import AutoTokenizer

from ultrasound_clip.enhanced_clip_model import EnhancedCLIP, BertTextEncoder
from ultrasound_clip.graph_encoder import GraphEncoder
from ultrasound_clip.semantic_loss import SemanticLoss
from ultrasound_clip.similarity_processor import SimilarityMatrixProcessor
from ultrasound_dataset import UltrasoundCLIPDataset, ultrasound_collate_fn


# ============================================================
#  Semantic-Aware 对比损失 (Soft Labels 版本)
# ============================================================

def compute_clip_loss_soft(image_features, text_features, logit_scale,
                           target_sim=None, soft_label_temp=0.5):
    """计算语义感知的 CLIP 对比损失。

    当提供 target_sim 时，使用语义软标签替代硬标签 (论文核心创新)。
    当 target_sim 为 None 时，退化为标准 InfoNCE (硬标签)。

    Args:
        image_features: [B, D] L2-normalized
        text_features:  [B, D] L2-normalized
        logit_scale:    scalar (exp of learnable temperature)
        target_sim:     [B, B] 语义相似度矩阵 (来自 UDAF)，None 时用硬标签
        soft_label_temp: 软标签的温度参数，控制分布的锐利程度

    Returns:
        loss: scalar tensor
    """
    # Clamp logit_scale 防止数值溢出 (原始 CLIP 论文上限 ln(100)≈4.6)
    logit_scale = torch.clamp(logit_scale, max=100.0)
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    if target_sim is not None:
        # ====== 语义软标签 (论文核心) ======
        # 将 UDAF 相似度矩阵转化为概率分布作为 soft target
        soft_labels = F.softmax(target_sim / soft_label_temp, dim=-1)
        # soft cross-entropy: -sum(soft_labels * log_softmax(logits))
        loss_i2t = (-soft_labels * F.log_softmax(logits_per_image, dim=-1)).sum(dim=-1).mean()
        loss_t2i = (-soft_labels.T * F.log_softmax(logits_per_text, dim=-1)).sum(dim=-1).mean()
    else:
        # ====== 退化为标准硬标签 ======
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


# ============================================================
#  Cosine 学习率调度器 (带 Warmup)
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine 退火 + 线性 Warmup 学习率调度器。"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
#  日志设置
# ============================================================

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================
#  主训练函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ultrasound-CLIP Training (V2: BioClinical-BERT)")
    # 路径
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--bert_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace BERT 模型名称")
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_backbone", type=float, default=5e-7, help="视觉骨干 + BERT 骨干学习率")
    parser.add_argument("--lr_new", type=float, default=5e-5, help="新增模块学习率（投影层、融合层、图编码器）")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--semantic_weight", type=float, default=0.05, help="λ: L_semantic 辅助正则项的权重 (主语义信息已融入软标签)")
    parser.add_argument("--soft_label_temp", type=float, default=0.5, help="语义软标签温度 (越小分布越锐利)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4)
    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=50, help="每 N 步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=1, help="每 N 个 epoch 保存一次 checkpoint")
    # 断点续训
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    # 调试
    parser.add_argument("--max_steps", type=int, default=None, help="最大步数（快速测试用）")
    parser.add_argument("--max_samples", type=int, default=None, help="限制加载样本数（调试用）")
    # 模型参数
    parser.add_argument("--graph_hidden", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--freeze_graph_encoder", action="store_true")

    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ===========================================================
    #  1. 加载基础 CLIP 模型（仅用视觉编码器和图像预处理）
    # ===========================================================
    logger.info(f"Loading CLIP visual backbone: {args.clip_model}")
    base_model, preprocess = clip.load(args.clip_model, device="cpu")
    dim_features = base_model.text_projection.shape[1]  # 512
    logger.info(f"Feature dimension: {dim_features}")

    # ===========================================================
    #  2. 加载 BioClinical-BERT 文本编码器
    # ===========================================================
    logger.info(f"Loading BERT text encoder: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, cache_dir="./bio_clinicalbert", local_files_only=True)
    text_encoder = BertTextEncoder(bert_model_name=args.bert_model, out_dim=dim_features)
    logger.info(f"BERT hidden size: {text_encoder.bert.config.hidden_size} → projection: {dim_features}")

    # ===========================================================
    #  3. 构建 EnhancedCLIP V2
    # ===========================================================
    graph_encoder = GraphEncoder(
        out_dim=dim_features,
        hidden=args.graph_hidden,
        n_layers=args.graph_layers,
    )
    model = EnhancedCLIP(
        base_model, text_encoder, graph_encoder,
        freeze_graph_encoder=args.freeze_graph_encoder,
    )
    model = model.float()  # 强制全部参数转 fp32
    model = model.to(device)
    logger.info("Model converted to float32 for numerical stability")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_params / 1e6:.1f}M params | Trainable: {trainable_params / 1e6:.1f}M")

    # ===========================================================
    #  4. 构建数据集 & DataLoader
    # ===========================================================
    dataset_dir = args.dataset_dir
    images_dir = os.path.join(dataset_dir, "images")
    full_data_path = os.path.join(dataset_dir, "full_data.json")

    train_dataset = UltrasoundCLIPDataset(
        jsonl_path=os.path.join(dataset_dir, "train.jsonl"),
        images_dir=images_dir,
        full_data_path=full_data_path,
        preprocess=preprocess,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        is_train=True,  # 启用数据增强
    )
    val_dataset = UltrasoundCLIPDataset(
        jsonl_path=os.path.join(dataset_dir, "valid.jsonl"),
        images_dir=images_dir,
        full_data_path=full_data_path,
        preprocess=preprocess,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ultrasound_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ultrasound_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches/epoch")

    # ===========================================================
    #  5. 语义损失组件
    # ===========================================================
    sim_matrices_dir = os.path.join(dataset_dir, "similarity_matrices")
    if not os.path.isdir(sim_matrices_dir):
        logger.error(
            f"similarity_matrices 目录不存在: {sim_matrices_dir}\n"
            f"请先运行: python generate_similarity_matrices.py --dataset_dir {dataset_dir}"
        )
        sys.exit(1)

    similarity_processor = SimilarityMatrixProcessor(
        similarity_matrices_dir=sim_matrices_dir,
        full_data_file=full_data_path,
    )

    class _LossArgs:
        rank = 0
        world_size = 1

    semantic_loss_fn = SemanticLoss(_LossArgs(), similarity_weight=args.semantic_weight)
    logger.info(f"Semantic loss λ = {args.semantic_weight}")

    # ===========================================================
    #  6. 三组分层学习率优化器
    # ===========================================================
    backbone_visual_params = []   # CLIP 视觉骨干 (低学习率)
    backbone_bert_params = []     # BERT 文本骨干 (低学习率)
    new_params = []               # 投影层、融合层、图编码器 (高学习率)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "base_clip" in name:
            backbone_visual_params.append(param)
        elif "text_encoder.bert" in name:
            backbone_bert_params.append(param)
        else:
            new_params.append(param)

    logger.info(
        f"Param groups — visual backbone: {sum(p.numel() for p in backbone_visual_params) / 1e6:.1f}M, "
        f"BERT backbone: {sum(p.numel() for p in backbone_bert_params) / 1e6:.1f}M, "
        f"new modules: {sum(p.numel() for p in new_params) / 1e6:.1f}M"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_visual_params, "lr": args.lr_backbone},
            {"params": backbone_bert_params, "lr": args.lr_backbone},
            {"params": new_params, "lr": args.lr_new},
        ],
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # ===========================================================
    #  7. 断点恢复
    # ===========================================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        
        # 【重要修复】强制初始化图编码器的动态模块
        for _, texts, graphs, _ in train_loader:
            if graphs is not None:
                graphs = graphs.to(device)
                model.graph_encoder._ensure_type_embeddings(graphs, device)
                model.graph_encoder._ensure_convs(graphs, device)
            break

        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, step {global_step}, best_val_loss {best_val_loss:.4f}")

    # ===========================================================
    #  8. 训练循环
    # ===========================================================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs:          {args.epochs}")
    logger.info(f"  Batch size:      {args.batch_size}")
    logger.info(f"  LR backbone:     {args.lr_backbone}")
    logger.info(f"  LR new modules:  {args.lr_new}")
    logger.info(f"  Semantic λ:      {args.semantic_weight}")
    logger.info(f"  Warmup steps:    {num_warmup_steps} / {num_training_steps}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_clip_loss = 0.0
        epoch_sem_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for step, (images, texts, graphs, image_keys) in enumerate(train_loader):
            images = images.to(device)
            # texts 是字典，需要逐项搬运到 GPU
            texts = {k: v.to(device) for k, v in texts.items()}
            if graphs is not None:
                graphs = graphs.to(device)

            optimizer.zero_grad()

            # ---------- 前向传播 (纯 fp32) ----------
            outputs = model(images=images, texts=texts, graphs=graphs)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logit_scale = outputs["logit_scale"]

            # 计算 UDAF 语义相似度矩阵
            target_sim = None
            try:
                target_sim = similarity_processor.calculate_batch_similarity_matrix_from_paths(
                    list(image_keys), len(image_keys)
                )
                target_sim = target_sim.to(device)
            except Exception:
                pass

            # L_CLIP: 语义感知对比损失 (使用软标签)
            loss_clip = compute_clip_loss_soft(
                image_features, text_features, logit_scale,
                target_sim=target_sim, soft_label_temp=args.soft_label_temp,
            )

            # L_semantic: 辅助正则项 (MSE + KL，权重已大幅降低)
            if target_sim is not None:
                loss_semantic = semantic_loss_fn(outputs, target_sim)
            else:
                loss_semantic = torch.tensor(0.0, device=device)

            # 总损失
            loss_total = loss_clip + args.semantic_weight * loss_semantic

            # 跳过 NaN batch（防止梯度污染）
            if not torch.isfinite(loss_total):
                logger.warning(f"NaN loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue

            # ---------- 反向传播 ----------
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # ---------- 记录 ----------
            global_step += 1
            epoch_loss += loss_total.item()
            epoch_clip_loss += loss_clip.item()
            epoch_sem_loss += loss_semantic.item()
            epoch_steps += 1

            if global_step % args.log_interval == 0:
                lr_bb = optimizer.param_groups[0]["lr"]
                lr_new = optimizer.param_groups[2]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(train_loader)} | "
                    f"Loss: {loss_total.item():.4f} "
                    f"(CLIP: {loss_clip.item():.4f}, Sem: {loss_semantic.item():.4f}) | "
                    f"LR: {lr_bb:.2e} / {lr_new:.2e}"
                )

            if args.max_steps and global_step >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}, stopping epoch early.")
                break

        # ---------- Epoch 汇总 ----------
        epoch_time = time.time() - epoch_start
        avg_train = epoch_loss / max(epoch_steps, 1)
        avg_clip = epoch_clip_loss / max(epoch_steps, 1)
        avg_sem = epoch_sem_loss / max(epoch_steps, 1)
        logger.info(
            f"\n{'='*50}\n"
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {avg_train:.4f} (CLIP: {avg_clip:.4f}, Sem: {avg_sem:.4f}) | "
            f"Time: {epoch_time:.0f}s\n"
            f"{'='*50}"
        )

        # ===========================================================
        #  验证
        # ===========================================================
        model.eval()
        val_loss_sum = 0.0
        val_clip_sum = 0.0
        val_sem_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, texts, graphs, image_keys in val_loader:
                images = images.to(device)
                texts = {k: v.to(device) for k, v in texts.items()}
                if graphs is not None:
                    graphs = graphs.to(device)

                outputs = model(images=images, texts=texts, graphs=graphs)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logit_scale = outputs["logit_scale"]

                # 计算 UDAF 语义相似度矩阵
                target_sim = None
                try:
                    target_sim = similarity_processor.calculate_batch_similarity_matrix_from_paths(
                        list(image_keys), len(image_keys)
                    )
                    target_sim = target_sim.to(device)
                except Exception:
                    pass

                loss_clip = compute_clip_loss_soft(
                    image_features, text_features, logit_scale,
                    target_sim=target_sim, soft_label_temp=args.soft_label_temp,
                )

                if target_sim is not None:
                    loss_semantic = semantic_loss_fn(outputs, target_sim)
                else:
                    loss_semantic = torch.tensor(0.0, device=device)

                loss_total = loss_clip + args.semantic_weight * loss_semantic

                # 跳过 NaN batch（防止污染验证指标）
                if not torch.isfinite(loss_total):
                    continue

                val_loss_sum += loss_total.item()
                val_clip_sum += loss_clip.item()
                val_sem_sum += loss_semantic.item()
                val_steps += 1

        avg_val = val_loss_sum / max(val_steps, 1)
        avg_val_clip = val_clip_sum / max(val_steps, 1)
        avg_val_sem = val_sem_sum / max(val_steps, 1)
        logger.info(
            f"[Epoch {epoch + 1}] Val Loss: {avg_val:.4f} "
            f"(CLIP: {avg_val_clip:.4f}, Sem: {avg_val_sem:.4f})"
        )

        # ---------- 保存 checkpoint ----------
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train,
                    "val_loss": avg_val,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # ---------- 保存最优模型 ----------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                best_path,
            )
            logger.info(f"✅ New best model! Val Loss: {best_val_loss:.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"{'='*60}")



if __name__ == "__main__": 
    main()   



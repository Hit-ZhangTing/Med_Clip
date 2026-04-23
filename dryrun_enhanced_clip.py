"""
Ultrasound-CLIP V2 冒烟测试 (Dry Run)。

验证 BioClinical-BERT 文本编码器的前向传播是否正常工作。
"""

import torch
import clip
from transformers import AutoTokenizer

from ultrasound_clip.enhanced_clip_model import EnhancedCLIP, BertTextEncoder
from ultrasound_clip.graph_encoder import GraphEncoder
from ultrasound_clip.graph_builder import build_hetero_graph_from_data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 1. 加载 CLIP 视觉骨干（先到 CPU，防止显存 OOM）  
    print("[INFO] Loading CLIP visual backbone (ViT-B/32)...")
    base_model, preprocess = clip.load("ViT-B/32", device="cpu")
    dim_features = 512

    # 2. 加载 BioClinical-BERT 文本编码器
    bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"[INFO] Loading BERT text encoder: {bert_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        bert_model_name,
        cache_dir="./bio_clinicalbert",
        local_files_only=True
    )
    text_encoder = BertTextEncoder(bert_model_name=bert_model_name, out_dim=dim_features)

    # 3. 初始化图编码器 + EnhancedCLIP V2
    print("[INFO] Constructing GraphEncoder & EnhancedCLIP V2...")
    graph_encoder = GraphEncoder(out_dim=dim_features, hidden=128, n_layers=2)
    
    enhanced_model = EnhancedCLIP(base_model, text_encoder, graph_encoder, freeze_graph_encoder=False)
    enhanced_model = enhanced_model.float()
    enhanced_model = enhanced_model.to(device)

    # 4. 构造模拟输入
    print("[INFO] Preparing mock inputs...")

    # 模拟图像
    dummy_image = torch.randn(1, 3, 224, 224).to(device)

    # 模拟超声临床文本（用 BERT tokenizer 处理）
    text_prompt = "A hyperechoic region indicating potential fatty liver with hepatic steatosis."
    encoded = tokenizer(
        text_prompt,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    texts = {k: v.to(device) for k, v in encoded.items()}

    # 模拟图结构
    mock_data = [{"media_name": "test_mock_1", "Diagnosis": ["Benign"], "Shape": ["Oval"]}]
    graphs = build_hetero_graph_from_data(mock_data, ["test_mock_1"])
    if graphs is not None:
        graphs = graphs.to(device)

    # 5. 执行前向传播
    print("\n[INFO] ---- START EXECUTING FORWARD PASS ----")
    try:
        outputs = enhanced_model(images=dummy_image, texts=texts, graphs=graphs)
        image_features = outputs["image_features"]
        text_features = outputs["text_features"]
        logit_scale = outputs["logit_scale"]

        print(">> FORWARD PASS (干跑) 成功！维度无冲突！ <<")
        print(f"Image Features Shape: {image_features.shape}")
        print(f"Text Features Shape:  {text_features.shape}")
        print(f"Logit Scale: {logit_scale.item():.4f}")

        # 验证特征维度对齐
        assert image_features.shape == text_features.shape, "维度不匹配！"
        assert image_features.shape[1] == dim_features, f"特征维度应为 {dim_features}！"

        # 验证相似度计算
        similarity = (image_features @ text_features.T).item()
        print(f"Image-Text Similarity: {similarity:.4f}")
        print("\n✅ BioClinical-BERT 文本编码器集成验证通过！")

    except Exception as e:
        print(f"\n[ERROR] Forward pass failed with: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

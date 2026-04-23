"""
Ultrasound-CLIP 增强模型 (V2: BioClinical-BERT 文本编码器版本)。

核心改造：
    - 文本编码器从 OpenAI CLIP 通用 Transformer 替换为 BioClinical-BERT
    - 视觉编码器保留 CLIP ViT-B/32（仅用于图像特征提取）
    - CrossAttentionFusion 保持不变（图-文浅层融合）

架构：
    Image  → CLIP ViT-B/32   → [B, 512]
    Text   → BioClinical-BERT → [CLS] 768d → Linear(768, 512) → [B, 512]
    Graph  → GraphEncoder     → [B, 512]
    Fusion → CrossAttention(text_feat, graph_feat) → [B, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from transformers import AutoModel


# ============================================================
#  BioClinical-BERT 文本编码器
# ============================================================

class BertTextEncoder(nn.Module):
    """基于 BioClinical-BERT 的医学文本编码器。

    从 [CLS] token 提取 768 维特征，经线性投影对齐到 CLIP 视觉空间 (512 维)。
    """

    def __init__(self, bert_model_name="emilyalsentzer/Bio_ClinicalBERT", out_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_model_name, 
            cache_dir="./bio_clinicalbert",
            local_files_only=True
        )
        bert_hidden = self.bert.config.hidden_size  # 768

        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, out_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(out_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      [B, seq_len] — BERT tokenizer 输出
            attention_mask: [B, seq_len] — padding mask

        Returns:
            features: [B, out_dim] — 投影后的文本特征（未归一化）
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        features = self.projection(cls_output)  # [B, 512]
        return features


# ============================================================
#  CrossAttentionFusion (保持不变)
# ============================================================

class CrossAttentionFusion(nn.Module):

    def __init__(self, text_dim, graph_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_dim
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # FFN 增强表达能力
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, text_dim)
        self.norm_out = nn.LayerNorm(text_dim)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.max_alpha = 0.5  # 从 0.2 提升到 0.5

        
    def forward(self, text_features, graph_features):
       
        if not torch.is_tensor(graph_features):
            return text_features

        original_dtype = text_features.dtype
        target_dtype = self.text_proj.weight.dtype
        text_features = text_features.to(target_dtype)
        graph_features = graph_features.to(target_dtype)

        if torch.isnan(graph_features).any() or torch.isinf(graph_features).any():
            return text_features

        text_proj = self.text_proj(text_features).unsqueeze(1)
        graph_proj = self.graph_proj(graph_features).unsqueeze(1)

        try:
            attended, _ = self.attention(text_proj, graph_proj, graph_proj)
            # Post-attention LayerNorm + residual
            attended = self.norm1(attended + text_proj)
            # FFN + residual
            ffn_out = self.ffn(attended)
            attended = self.norm2(ffn_out + attended)
            attended = attended.squeeze(1)
        except Exception:
            return text_features

        residual = self.output_proj(attended)
        residual = torch.tanh(residual)
        gate = torch.clamp(self.alpha, 0.0, self.max_alpha)
        output = text_features + gate * residual
        output = self.norm_out(output)

        if torch.isnan(output).any() or torch.isinf(output).any():
            return text_features.to(original_dtype)

        return output.to(original_dtype)


# ============================================================
#  EnhancedCLIP V2 (BioClinical-BERT 版本)
# ============================================================

class EnhancedCLIP(nn.Module):
    """Ultrasound-CLIP 增强模型。

    组合三大编码器：
        - 视觉编码器：CLIP ViT-B/32（冻结或微调）
        - 文本编码器：BioClinical-BERT + 线性投影
        - 图编码器：  GraphEncoder（DGL 异构图卷积）

    Args:
        base_clip_model:  OpenAI CLIP 模型（仅使用视觉部分）
        text_encoder:     BertTextEncoder 实例
        graph_encoder:    GraphEncoder 实例
        freeze_graph_encoder: 是否冻结图编码器参数
    """

    def __init__(self, base_clip_model, text_encoder, graph_encoder, freeze_graph_encoder=False):
        super().__init__()
        self.base_clip = base_clip_model      # 仅用视觉编码
        self.text_encoder = text_encoder      # BioClinical-BERT
        self.graph_encoder = graph_encoder

        # 自持温度参数（不再依赖 CLIP 的 logit_scale）
        # 初始值 = ln(1/0.07) ≈ 2.66，与 CLIP 论文一致
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        # 文本-图融合层
        text_dim = text_encoder.projection[0].out_features  # 512
        self.text_graph_fusion = CrossAttentionFusion(
            text_dim=text_dim,
            graph_dim=graph_encoder.out_dim,
        )
        
        if freeze_graph_encoder:
            for param in self.graph_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, texts, graphs=None):
        """
        Args:
            images: [B, 3, 224, 224] — 预处理后的图像张量
            texts:  dict — {'input_ids': [B, seq_len], 'attention_mask': [B, seq_len]}
            graphs: dgl.DGLHeteroGraph (batched) 或 None

        Returns:
            dict with image_features, text_features, logit_scale
        """
        # ---- 文本编码 (BioClinical-BERT) ----
        text_features = self.encode_text(texts, normalize=True)

        # ---- 图编码 + 融合 ----
        if graphs is not None:
            total_edges = 0
            for et in graphs.canonical_etypes:
                total_edges += graphs.num_edges(et)

            with autocast(device_type='cuda', enabled=False):
                graph_features = self.graph_encoder(graphs) if total_edges > 0 else None

            if graph_features is not None:
                enhanced_text_features = self.text_graph_fusion(text_features, graph_features)
            else:
                enhanced_text_features = text_features
        else:
            enhanced_text_features = text_features

        # ---- 图像编码 (CLIP ViT) ----
        image_features = self.encode_image(images, normalize=True)
        
        return {
            "image_features": image_features,
            "text_features": enhanced_text_features,
            "logit_scale": self.logit_scale.exp(),
        }
    
    def encode_image(self, images, normalize=True):
        """使用 CLIP ViT 编码图像。"""
        features = self.base_clip.encode_image(images)  
        features = features.float()  # 确保 float32
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_text(self, texts, normalize=True):
        """使用 BioClinical-BERT 编码文本。

        Args:
            texts: dict — {'input_ids': [B, L], 'attention_mask': [B, L]}
        """
        input_ids = texts["input_ids"] 
        attention_mask = texts["attention_mask"]
        features = self.text_encoder(input_ids, attention_mask)
        features = features.float()  # 确保 float32
        return F.normalize(features, dim=-1) if normalize else features    

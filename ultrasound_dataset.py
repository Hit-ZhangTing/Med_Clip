"""
Ultrasound-CLIP 自定义数据集与 DataLoader (V2: BERT Tokenizer 版本)。

负责从 jsonl 文件加载样本，关联 full_data.json 获取完整标签，
构建 DGL 异构图，以及图像预处理和 BERT 文本分词。
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import dgl

from ultrasound_clip.graph_builder import build_single_sample_graph

# CLIP 使用的标准化参数
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class UltrasoundCLIPDataset(Dataset):
    """Ultrasound-CLIP 训练/验证/测试数据集。

    每个样本返回:
        image_tensor:   [3, 224, 224]  — CLIP 预处理后的图像
        input_ids:      [max_length]   — BERT tokenizer 输出的 token IDs
        attention_mask: [max_length]   — BERT attention mask
        record:         dict           — full_data.json 中的完整记录（含 9 个任务标签）
        image_name:     str            — 图像文件名（用于 similarity_processor 查询）
    """

    def __init__(self, jsonl_path, images_dir, full_data_path, preprocess, tokenizer,
                 max_length=128, max_samples=None, is_train=False):
        """
        Args:
            jsonl_path:     train.jsonl / valid.jsonl / test.jsonl 的路径
            images_dir:     dataset/images/ 目录路径
            full_data_path: dataset/full_data.json 路径
            preprocess:     CLIP 图像预处理函数 (clip.load 返回的第二个值)
            tokenizer:      HuggingFace AutoTokenizer 实例 (BioClinical-BERT)
            max_length:     文本最大 token 长度（默认 128）
            max_samples:    可选，限制样本数量（用于调试）
            is_train:       是否为训练模式（启用数据增强）
        """
        self.images_dir = images_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # 训练时使用数据增强，验证/测试时使用 CLIP 默认预处理
        if is_train:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
            ])
        else:
            self.train_transform = None

        # 加载 jsonl 条目
        self.entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

        if max_samples:
            self.entries = self.entries[:max_samples]

        # 加载 full_data.json 并建立查找字典
        with open(full_data_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        self.full_data_dict = {}
        for rec in full_data:
            media_name = rec["media_name"]
            self.full_data_dict[media_name] = rec
            # 同时用去掉 .jpg 后缀的名字索引，以防 jsonl 与 full_data 格式不完全一致
            if media_name.endswith(".jpg"):
                self.full_data_dict[media_name[:-4]] = rec

        print(
            f"[Dataset] Loaded {len(self.entries)} samples from {os.path.basename(jsonl_path)}, "
            f"{len(full_data)} full records indexed"
        )

    def __len__(self):
        return len(self.entries)

    def get_image_key(self, idx):
        """返回样本的图像文件名（供 SimilarityMatrixProcessor 使用）。"""
        return self.entries[idx]["image"]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_name = entry["image"]
        caption = entry["caption"]

        # ---------- 图像加载与预处理 ----------
        image_path = os.path.join(self.images_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            if self.is_train and self.train_transform is not None:
                image_tensor = self.train_transform(image)
            else:
                image_tensor = self.preprocess(image)
        except Exception:
            # 图片损坏或缺失时，返回全零占位张量
            image_tensor = torch.zeros(3, 224, 224)

        # ---------- 文本分词 (BioClinical-BERT) ----------
        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)          # [max_length]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [max_length]

        # ---------- 获取完整记录（用于构图 + 语义损失） ----------
        record = self.full_data_dict.get(image_name, {})

        return image_tensor, input_ids, attention_mask, record, image_name


def ultrasound_collate_fn(batch):
    """自定义 collate 函数，将单样本的异构图用 dgl.batch 合并。

    Returns:
        images:     [B, 3, 224, 224]
        texts:      dict — {'input_ids': [B, L], 'attention_mask': [B, L]}
        graphs:     dgl.DGLHeteroGraph (batched) 或 None
        image_keys: List[str] 长度为 B
    """
    images, input_ids_list, attention_masks, records, image_keys = zip(*batch)

    # Stack 常规张量
    images = torch.stack(images, dim=0)
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)

    texts = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # 为每个样本构建异构图，然后 batch
    full_data_for_graph = []
    valid_keys = []
    for rec, key in zip(records, image_keys):
        if rec and "media_name" in rec:
            full_data_for_graph.append(rec)
            valid_keys.append(rec["media_name"])
        else:
            # 无标签记录 → 构建最小化空图
            full_data_for_graph.append({"media_name": key})
            valid_keys.append(key)

    from ultrasound_clip.graph_builder import build_hetero_graph_from_data

    graphs = build_hetero_graph_from_data(full_data_for_graph, valid_keys)

    return images, texts, graphs, list(image_keys)

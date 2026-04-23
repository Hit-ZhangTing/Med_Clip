'''
Author: zhangting
Date: 2026-04-23 19:26:53
LastEditors: Do not edit
LastEditTime: 2026-04-23 19:31:12
FilePath: /zhangting/Med_Clip/data_preprocess/split_dataset.py
'''
import os
import json
import random

# 定义路径
image_dir = "/home/zhangting/data/Thyroid_US/malignant/image"
mask_analysis_file = "/home/zhangting/data/Thyroid_US/malignant/mask_analysis_results.json"
output_dir = "/home/zhangting/data/Thyroid_US/malignant/splits"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 mask_analysis_results.json
with open(mask_analysis_file, "r", encoding="utf-8") as f:
    mask_analysis_results = json.load(f)

# 创建一个字典，将文件名与对应的 caption 映射
captions = {item["media_name"]: item["caption"][0] for item in mask_analysis_results}

# 获取所有图像文件
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# 打乱文件顺序
random.shuffle(image_files)

# 按 6:2:2 比例划分数据集
num_images = len(image_files)
train_split = int(0.6 * num_images)
val_split = int(0.2 * num_images)

train_files = image_files[:train_split]
val_files = image_files[train_split:train_split + val_split]
test_files = image_files[train_split + val_split:]

# 定义保存函数
def save_split(files, split_name):
    output_file = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for file in files:
            caption = captions.get(file, "No caption available.")  # 获取对应的 caption
            json_line = {
                "image": file,
                "caption": caption
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")  # 写入一行 JSON
    print(f"{split_name.capitalize()} set saved to {output_file}")

# 保存训练集、验证集和测试集
save_split(train_files, "train")
save_split(val_files, "val")
save_split(test_files, "test")
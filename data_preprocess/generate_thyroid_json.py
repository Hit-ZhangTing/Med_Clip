'''
Author: zhangting
Date: 2026-04-23 15:57:13
LastEditors: Do not edit
LastEditTime: 2026-04-23 16:37:42
FilePath: /zhangting/Med_Clip/data_preprocess/generate_thyroid_json.py
'''
import cv2
import numpy as np
import json
import os

def analyze_mask(mask_path):
    # 加载掩码图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.max(mask) == 0:
        return {
            "text_description": "No mask detected.",
            "structured_data": {}
        }

    # 基础信息提取
    height, width = mask.shape
    mask_area = np.sum(mask > 0)
    area_ratio = mask_area / (width * height)

    # 提取最大连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return {
            "text_description": "No mask detected.",
            "structured_data": {}
        }

    # 最大连通区域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.uint8)

    # 几何特征分析
    x, y, w, h, area = stats[largest_label]
    centroid_x, centroid_y = centroids[largest_label]
    perimeter = cv2.arcLength(cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
    convex_hull = cv2.convexHull(cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    convex_hull_area = cv2.contourArea(convex_hull)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    solidity = area / convex_hull_area if convex_hull_area > 0 else 0
    
    # 判断形状（宽大于高或高大于宽）
    shape = "wider" if w > h else "taller"
    
    # 多边形拟合
    epsilon = 0.01 * cv2.arcLength(convex_hull, True)
    approx = cv2.approxPolyDP(convex_hull, epsilon, True)
    approx_vertices = len(approx)
    
    # 语义属性归纳
    horizontal_position = "left" if centroid_x < width / 3 else "center" if centroid_x < 2 * width / 3 else "right"
    vertical_position = "top" if centroid_y < height / 3 else "middle" if centroid_y < 2 * height / 3 else "bottom"
    size_category = (
        "none" if area_ratio == 0 else
        "tiny" if area_ratio < 0.01 else
        "small" if area_ratio < 0.1 else
        "medium" if area_ratio < 0.3 else
        "large"
    )
    boundary_smoothness = (
        "smooth" if circularity > 0.8 and solidity > 0.9 else
        "moderately smooth" if circularity > 0.5 and solidity > 0.7 else
        "irregular"
    )
    
    # 生成自然语言描述
    text_description = (
        f"The lesion is located in the {vertical_position} of the image, "
        f"with a {size_category} size, {shape} shape, and {boundary_smoothness} margins "
        f"(circularity={circularity:.2f}, solidity={solidity:.2f})."
    )
    
    # 输出结构化数据
    structured_data = {
        "media_name": os.path.basename(mask_path),
        "caption":  [text_description],
        "Shape": [shape],  # 使用宽高关系描述形状
        "Margins": [boundary_smoothness],  # 使用边界平滑性描述边界
        "Diagnosis": ["malignant"],  # 固定诊断为malignant
        "Position": [f"{vertical_position}-{horizontal_position}"],  # 结合水平和垂直位置
        "Size": [size_category]  # 使用面积占比描述大小
    }

    return structured_data

from tqdm import tqdm  # 导入 tqdm 库

# 示例调用
mask_dir = "/home/zhangting/data/Thyroid_US/malignant/mask/"
output_file = "/home/zhangting/data/Thyroid_US/malignant/mask_analysis_results.json"
results = []

# 获取掩码文件列表
mask_files = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]

# 遍历文件并显示进度条
for mask_file in tqdm(mask_files, desc="Processing masks"):
    mask_path = os.path.join(mask_dir, mask_file)
    result = analyze_mask(mask_path)
    results.append(result)  # 将每个文件的结果添加到列表中

# 将所有结果保存到一个 JSON 文件中
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Analysis results saved to {output_file}")
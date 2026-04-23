import torch
import clip
from PIL import Image

print("=" * 50)
print("CLIP 项目演示")
print("=" * 50)

# 1. 动态检测并分配设备 (有显卡自动用显卡，没显卡用CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[设备] 平台已自动检测使用计算节点: {device.upper()}")
if device == "cuda":
    print(f"       ✅ 检测到可用显卡: {torch.cuda.get_device_name(0)}")

# 2. 加载模型（首次运行会自动下载约350MB权重）
print("\n[模型] 正在加载 ViT-B/32 ...")
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"[模型] 加载完成，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 3. 准备图像（使用项目自带的 CLIP.png）
print("\n[图像] 读取 CLIP.png ...") 
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)

# 4. 准备文本候选
labels = ["a diagram", "a dog", "a cat", "a photo of people", "a neural network architecture"]
label_cn = ["图表", "狗", "猫", "人物照片", "神经网络架构图"]
text = clip.tokenize(labels).to(device)

# 5. 推理
print("[推理] 计算图文相似度 ...\n")
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

# 6. 输出结果
print("=" * 50)
print("📊 CLIP.png 图像内容识别结果：")
print("=" * 50)
sorted_results = sorted(zip(label_cn, labels, probs), key=lambda x: x[2], reverse=True)
for cn, en, prob in sorted_results:
    bar = "█" * int(prob * 40)
    print(f"  {cn:12s} ({en:35s}): {prob*100:6.2f}% {bar}")

print("\n✅ CLIP 项目跑通了！")
print(f"   最高匹配: 「{sorted_results[0][0]}」({sorted_results[0][2]*100:.1f}%)")


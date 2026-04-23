import json, sys, collections

# 检查 caption 分布
jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/train.jsonl"

entries = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

print(f"=== 文件: {jsonl_path} ===")
print(f"总样本数: {len(entries)}")

# 1. caption 长度统计
lengths = [len(e["caption"]) for e in entries]
print(f"\ncaption 字符长度统计:")
print(f"  最短: {min(lengths)}")
print(f"  最长: {max(lengths)}")
print(f"  平均: {sum(lengths)/len(lengths):.1f}")

# 2. 重复 caption 统计
cap_counter = collections.Counter(e["caption"] for e in entries)
total_unique = len(cap_counter)
dup_captions = {k: v for k, v in cap_counter.items() if v > 1}
total_dup_samples = sum(dup_captions.values())
print(f"\n唯一 caption 数: {total_unique} / {len(entries)} ({total_unique/len(entries)*100:.1f}%)")
print(f"有重复的 caption 种类数: {len(dup_captions)}")
print(f"属于重复 caption 的样本数: {total_dup_samples} ({total_dup_samples/len(entries)*100:.1f}%)")

# 3. 最高频前 10 个 caption
print(f"\n最高频 caption TOP-10:")
for cap, cnt in cap_counter.most_common(10):
    print(f"  [{cnt}次] {cap[:150]}")

# 4. 打印前 5 条样本完整内容
print(f"\n前 5 条样本:")
for i, e in enumerate(entries[:5]):
    print(f"  [{i}] image={e['image'][:60]}")
    print(f"       caption={e['caption'][:200]}")

# 5. 检查 full_data.json 的标签丰富度
print(f"\n=== full_data.json 标签分析 ===")
with open("dataset/full_data.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)

print(f"总记录数: {len(full_data)}")

task_keys = ["Diagnosis", "Shape", "Margins", "Position", "Size"]

# 统计每个 task 有多少样本有标签
for tk in task_keys:
    has_label = sum(1 for r in full_data if r.get(tk) and (isinstance(r[tk], list) and len(r[tk]) > 0 or not isinstance(r[tk], list) and r[tk]))
    print(f"  {tk:30s}: {has_label:6d} / {len(full_data)} ({has_label/len(full_data)*100:.1f}%)")

# 6. 看几个样本的完整标签
print(f"\nfull_data 前 3 条完整记录:")
for i, r in enumerate(full_data[:3]):
    print(f"  [{i}] media_name={r.get('media_name','?')}")
    for tk in task_keys:
        print(f"       {tk}: {r.get(tk, 'N/A')}")

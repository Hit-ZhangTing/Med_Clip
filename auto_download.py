import os
import time
from huggingface_hub import snapshot_download

# 强行设置国内加速镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = 'JJY-0823/US-365K'
local_dir = './dataset'

print(f"========== 开始自动断点续传下载 {repo_id} ==========")
while True:
    try:
        print("\n[INFO] 正在尝试连接与下载/续传...")
        snapshot_download(
            repo_id=repo_id, 
            repo_type='dataset', 
            local_dir=local_dir, 
            resume_download=True,       # 开启断点续传
            local_dir_use_symlinks=False, 
            max_workers=4               # 4线程并发
        )
        print("\n[SUCCESS] 太棒了！整个数据集下载已全部完成！")
        break # 下载一旦彻底成功，跳出死循环
    except Exception as e:
        print(f"\n[ERROR] 网络异常中断: {e}")
        print("[INFO] 服务器网络波动，5秒后自动重连续传...")
        time.sleep(5)

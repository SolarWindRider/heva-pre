"""
读取 attn_acc_input.pkl 和 attn_acc_visual.pkl 文件，查看其内容
"""

import pickle
import torch

pkl_dir = "/home/ma-user/work/heva-pre/results/exp005/VisuRiddles/pkls"
idx = 41

# 读取 attn_acc_input
input_path = f"{pkl_dir}/{idx}_attn_acc_input.pkl"
visual_path = f"{pkl_dir}/{idx}_attn_acc_visual.pkl"

with open(input_path, "rb") as f:
    attn_acc_input = pickle.load(f)

with open(visual_path, "rb") as f:
    attn_acc_visual = pickle.load(f)

print("=" * 60)
print("attn_acc_input:")
print(f"  Shape: {attn_acc_input.shape}")
print(f"  Data: {attn_acc_input.float().numpy()}")

print("\n" + "=" * 60)
print("attn_acc_visual:")
print(f"  Shape: {attn_acc_visual.shape}")
print(f"  Data: {attn_acc_visual.float().numpy()}")

print("\n" + "=" * 60)
print("input + visual (应该约等于 1.0):")
print(f"  Data: {(attn_acc_input + attn_acc_visual).float().numpy()}")

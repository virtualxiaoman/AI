import torch

print("Torch version:", torch.__version__)
print("Torch cuda version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print(torch.backends.cudnn.version())


import subprocess
import os

print("=" * 50)
print("GPU 诊断信息")

# 检查是否有 NVIDIA 命令行工具
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
    print("nvidia-smi 输出:")
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
except FileNotFoundError:
    print("找不到 nvidia-smi，请检查 NVIDIA 驱动安装")

print("\n" + "=" * 50)
print("PyTorch GPU 信息:")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("详细错误检查:")
    try:
        # 强制尝试获取设备数量
        count = torch._C._cuda_getDeviceCount()
        print(f"直接调用返回的设备数: {count}")
    except Exception as e:
        print(f"直接调用错误: {e}")
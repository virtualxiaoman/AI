import torch
import os

print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 是否可用 CUDA? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 检测到的 GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"当前 GPU 型号 (PyTorch): {torch.cuda.get_device_name(0)}")

print("\n尝试导入 Llama...")
try:
    from llama_cpp import Llama
    print("Llama 导入成功！")
    # 更全面的测试需加载模型并设置 n_gpu_layers > 0
    # llm = Llama(model_path="您的模型路径.gguf", n_gpu_layers=30) 
    # print("Llama 对象已初始化（这将测试实际 GPU 卸载）")
    model_path = r"G:/Models/MM/ImgToText/NuMarkdown-8B-Thinking-Q4_K_M.gguf"
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=50,
        n_ctx=2048,
        verbose=True
    )
    print("CUDA支持:", llm.params.n_gpu_layers > 0)
except Exception as e:
    print(f"导入或初始化 Llama 出错: {e}")

print("\n从 Python 环境检查 CMAKE_ARGS:")
print(f"CMAKE_ARGS: {os.environ.get('CMAKE_ARGS')}") 

quit()
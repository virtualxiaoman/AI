import os
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
img_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
# 1. 初始化模型
# 5080 16GB 显存运行 4B FP8 非常轻松，建议设置 limit_mm_per_prompt 处理多图
model_name = "Qwen/Qwen3-VL-4B-Thinking-FP8"
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,  # 预留显存给系统
    max_model_len=32768
)

# 2. 准备输入：图片 + 描述指令
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},  # 也可是本地路径
            {"type": "text", "text": "请详细描述这张图片的内容，并分析其中的逻辑。"}
        ],
    }
]

# 3. 设置 Thinking 模式的采样参数
# 注意：Thinking 模式建议使用非零温度，以允许模型进行发散性思考
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=2048,
    stop=["<|endoftext|>", "<|im_end|>"]
)

# 4. 执行推理
outputs = llm.chat(messages, sampling_params=sampling_params)

# 5. 解析输出（Qwen3-VL-Thinking 会输出 <think> 标签内的思考过程）
generated_text = outputs[0].outputs[0].text
print(f"模型输出内容：\n{generated_text}")

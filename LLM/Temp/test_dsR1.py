import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 加载模型
# 修正警告：将 torch_dtype 改为 dtype（虽然目前两者通用，但遵循建议更好）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 3. 准备输入
prompt = "你是谁？"
messages = [
    {"role": "user", "content": prompt}
]

# 转换模板
# 注意：这里我们获取的是包含 input_ids 和 attention_mask 的字典
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,  # 明确返回字典
    return_tensors="pt"  # 返回 PyTorch 张量
).to(model.device)

# 4. 设置流式输出器
streamer = TextStreamer(tokenizer, skip_prompt=True)

# 5. 执行生成
print("DeepSeek-R1 正在思考中...\n")

# 修复核心：使用 **inputs 进行解包
outputs = model.generate(
    **inputs,  # 这里的 ** 会把 input_ids 和 attention_mask 自动传进去
    max_new_tokens=2048,
    streamer=streamer,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id  # 建议显式指定 pad_token
)

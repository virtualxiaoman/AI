# import time
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText
#
# # 1. 配置路径
# img_path = "G:/AAA重要文档、证书、照片、密钥/回忆/演唱会/流光协奏-武汉-25.12.27/群友/无锡/无锡歌单.jpg"
# MODEL_PATH = "zai-org/GLM-OCR"
#
# # 2. 加载模型和处理器 (务必加上 trust_remote_code)
# processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModelForImageTextToText.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# ).eval()
#
# # 3. 准备输入数据
# image = Image.open(img_path).convert("RGB")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "Text Recognition:"}
#         ],
#     }
# ]
#
# # 这一步将图片和文字转换为模型需要的 tensor
# inputs = processor.apply_chat_template(
#     messages,
#     images=[image],  # 图片对象在这里传入
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)
#
# # 4. 推理与计时
# if torch.cuda.is_available():
#     torch.cuda.synchronize()  # 确保之前的所有显存操作已就绪
#
# start_time = time.time()
#
# with torch.no_grad():
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=8192,
#         do_sample=False  # OCR 任务通常不需要随机性
#     )
#
# if torch.cuda.is_available():
#     torch.cuda.synchronize()  # 等待 GPU 计算完成
#
# end_time = time.time()
#
# # 5. 计算结果
# duration = end_time - start_time
# # 获取生成的 Token 数量（排除掉输入部分的长度）
# generated_tokens_count = len(generated_ids[0]) - len(inputs["input_ids"][0])
# tokens_per_sec = generated_tokens_count / duration
#
# # 6. 解码与输出
# output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#
# print("-" * 30)
# print(f"识别结果:\n{output_text}")
# print("-" * 30)
# print(f"⏱️  总耗时: {duration:.2f} 秒")
# print(f"🚀 生成速度: {tokens_per_sec:.2f} tokens/s")
# print(f"🔢 生成 Token 总数: {generated_tokens_count}")

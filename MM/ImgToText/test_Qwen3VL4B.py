# ModuleNotFoundError: No module named 'triton'

# import torch
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
#
# img_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
# model_id = "Qwen/Qwen3-VL-4B-Thinking-FP8"
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype="auto",  # 会自动识别并以 FP8/BF16 加载
#     device_map="auto",
#     trust_remote_code=True
# )
# processor = AutoProcessor.from_pretrained(model_id)
#
# # 准备输入
# messages = [
#     {"role": "user", "content": [
#         {"type": "image", "image": img_path},
#         {"type": "text", "text": "这张图里有什么？"}
#     ]}
# ]
#
# # 处理图像和文本
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# ).to(model.device)
#
# # 生成响应
# generated_ids = model.generate(**inputs, max_new_tokens=1024)
# output_text = processor.batch_decode(
#     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text[0])

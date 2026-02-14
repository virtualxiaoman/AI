from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

img_path = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
img_path = "描述这张图片"
prompt = "请用一个词描述图中动漫人物的表情。"
# prompt = "Please describe the character's expression with a single English word. Don't provide any other text content."
# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Thinking", dtype="auto", device_map="auto"
)
# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Thinking",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)
print(output_text)

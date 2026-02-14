import time
from transformers import AutoProcessor, AutoModelForImageTextToText

img_path = "G:/AAA重要文档、证书、照片、密钥/回忆/演唱会/流光协奏-武汉-25.12.27/群友/无锡/无锡歌单.jpg"

MODEL_PATH = "zai-org/GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": img_path
            },
            {
                "type": "text",
                "text": "Text Recognition:"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
start_time = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"推理耗时: {time.time() - start_time:.2f} 秒")
print(output_text)

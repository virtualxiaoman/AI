import time
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator

device = Accelerator().device
model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", use_fast=True)

image = "G:/AAA重要文档、证书、照片、密钥/回忆/演唱会/流光协奏-武汉-25.12.27/群友/无锡/无锡歌单.jpg"
inputs = processor(image, return_tensors="pt", device=device).to(device)

start_time = time.time()
generate_ids = model.generate(
    **inputs,
    do_sample=False,
    tokenizer=processor.tokenizer,
    stop_strings="<|im_end|>",
    max_new_tokens=4096,
)

processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"推理耗时: {time.time() - start_time:.2f} 秒")
print(output_text)

# # https://gemini.google.com/app/eb6b5115ae3c4579
# # https://chatgpt.com/c/6987f298-a260-8329-a3b4-07c499b98ce5
#
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import requests
#
# # 1. 加载模型和处理器
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#
# # 2. 准备图片
# img_url = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#
# # 3. 生成描述
# inputs = processor(raw_image, return_tensors="pt")
# out = model.generate(**inputs)
#
# # 4. 解码输出
# print(processor.decode(out[0], skip_special_tokens=True))
# # 输出类似: "a woman sitting on the beach with a dog"

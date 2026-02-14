# from transformers import AutoProcessor, AutoModelForImageTextToText
# import torch
#
# MODEL_PATH = "zai-org/GLM-OCR"
#
# # 载入处理器与模型
# processor = AutoProcessor.from_pretrained(MODEL_PATH)
# model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
#
# # 构造输入
# inputs = processor(
#     images=[
#         "G:/AAA重要文档、证书、照片、密钥/竞赛获奖证书 汇总备份处/中国大学生计算机设计大赛/中国大学生计算机设计大赛省级一等奖.png"],
#     text="Text Recognition:",
#     return_tensors="pt"
# )
#
# # 推理
# outputs = model.generate(**inputs)
# print(processor.decode(outputs[0]))

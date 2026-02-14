"""
该模型实现表情识别的效果不佳，故不使用，仅保留代码。
"""

# import os
# import requests
# from PIL import Image
# import torch
# from transformers import BlipProcessor, BlipForQuestionAnswering
#
#
# class ImageVQA:
#     def __init__(self, model_id="Salesforce/blip-vqa-base", device=None):
#         """
#         VQA 类：加载 BLIP VQA 模型并提供问答接口
#         :param model_id: HF 模型 ID（默认: "Salesforce/blip-vqa-base"）
#         :param device: 'cuda' or 'cpu' 或 None（自动检测）
#         """
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"正在加载 VQA 模型到 {self.device} ...")
#         self.processor = BlipProcessor.from_pretrained(model_id)
#         self.model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
#         print("VQA 模型加载完成。")
#
#     @staticmethod
#     def _load_image(img_path, timeout=15):
#         if img_path.startswith(('http://', 'https://')):
#             response = requests.get(img_path, stream=True, timeout=timeout)
#             response.raise_for_status()
#             return Image.open(response.raw).convert('RGB')
#         else:
#             if not os.path.exists(img_path):
#                 raise FileNotFoundError(f"本地路径不存在: {img_path}")
#             return Image.open(img_path).convert('RGB')
#
#     def answer(self, img_path, question, max_length=20):
#         """
#         对传入图片回答问题
#         :param img_path: 本地路径或 URL
#         :param question: 自然语言问题（字符串）
#         :param max_length: 生成答案的最大长度
#         :return: 字符串答案或异常信息
#         """
#         try:
#             raw_image = self._load_image(img_path)
#             inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
#             with torch.no_grad():
#                 out = self.model.generate(**inputs, max_length=max_length)
#             answer = self.processor.decode(out[0], skip_special_tokens=True)
#             return answer
#         except Exception as e:
#             return f"回答失败: {e}"
#
#
# if __name__ == "__main__":
#     vqa = ImageVQA()
#     path = "D:/Users/Administrator/Desktop/表情包/71FE2B9B7025D865169A3A38793591C9.jpg"
#     q1 = "What is the expression of the cartoon character in the picture?"
#     # print("Q:", q1)
#     print("A:", vqa.answer(path, q1))

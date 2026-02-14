import os
import requests
import time
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base", device=None):
        """
        初始化模型和处理器
        :param model_id: 模型在 HF 上的 ID
        :param device: 指定运行设备 ('cuda', 'cpu')，默认自动检测
        """
        # 自动选择设备
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载模型到 {self.device}...")

        # 加载一次，永久复用
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
        print("模型加载完成。")

    @staticmethod
    def _load_image(img_path):
        """内部方法：处理图片加载逻辑"""
        if img_path.startswith(('http://', 'https://')):
            response = requests.get(img_path, stream=True, timeout=15)
            response.raise_for_status()
            return Image.open(response.raw).convert('RGB')
        else:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"本地路径不存在: {img_path}")
            return Image.open(img_path).convert('RGB')

    def generate_caption(self, img_path):
        """
        对外公开的方法：传入路径/URL，返回描述
        """
        try:
            # 加载图片
            raw_image = self._load_image(img_path)

            # 推理
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)

            # 解码
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"生成失败: {str(e)}"


if __name__ == "__main__":
    captioner = ImageCaptioner()
    path_Shiroko = "F:/Picture/pixiv/BA/Shiroko/140776508_p0.png"
    path_Suzuran = "D:/Users/Administrator/Desktop/表情包/71FE2B9B7025D865169A3A38793591C9.jpg"
    print(f"描述: {captioner.generate_caption(path_Shiroko)}")
    print(f"描述: {captioner.generate_caption(path_Suzuran)}")
    # time_0 = time.time()
    # captioner = ImageCaptioner()
    # print(f"加载耗时: {time.time() - time_0:.2f} 秒")  # 9.26 秒
    # time_1 = time.time()
    # res1 = captioner.generate("F:/Picture/pixiv/BA/Shiroko/140776508_p0.png")
    # time_2 = time.time()
    # print(f"推理耗时: {time_2 - time_1}")  # 0.64 秒
    # print(f"本地图片描述: {res1}")  # a girl with long hair and blue eyes, wearing a white dress, standing in front of pink clouds
    # time_3 = time.time()
    # res2 = captioner.generate("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg")
    # time_4 = time.time()
    # print(f"推理耗时: {time_4 - time_3}")  # 1.80 秒
    # print(f"网络图片描述: {res2}")  # a woman sitting on the beach with her dog

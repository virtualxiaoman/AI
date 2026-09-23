from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms


class ImageLoader:
    """图片读取器"""

    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load_pil(self, image_path: str | Path) -> Image.Image:
        """读取PIL图片"""
        return Image.open(image_path).convert("RGB")

    def load_numpy(self, image_path: str | Path) -> np.ndarray:
        """读取numpy图片(H,W,C) uint8"""
        image = self.load_pil(image_path)
        return np.asarray(image)

    def preprocess(self, image: Image.Image):
        """PIL -> Tensor(C,H,W)"""
        return self.transform(image)

    def preprocess_numpy(self, image: np.ndarray):
        """numpy -> Tensor(C,H,W)"""
        image = Image.fromarray(image.astype(np.uint8))
        return self.transform(image)
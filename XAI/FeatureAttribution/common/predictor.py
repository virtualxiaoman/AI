from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .image_loader import ImageLoader


class Predictor:
    """
    统一预测接口。LIME SHAP GradCAM 全部调用这个类。
    """

    def __init__(self, model, image_size: int = 224, device: str | None = None, ):
        self.model = model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.loader = ImageLoader(image_size)

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """ tensor: (N,C,H,W) """
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        prob = torch.softmax(logits, dim=1)
        return prob.cpu()

    @torch.no_grad()
    def predict_pil(self, image: Image.Image):
        tensor = self.loader.preprocess(image)
        tensor = tensor.unsqueeze(0)
        return self.predict_tensor(tensor)

    @torch.no_grad()
    def predict_numpy(self, images: np.ndarray):
        """ LIME需要的接口 输入： (N,H,W,C) 返回： (N,num_classes) """
        tensors = []

        for image in images:
            tensor = self.loader.preprocess_numpy(image)
            tensors.append(tensor)

        tensors = torch.stack(tensors)
        prob = self.predict_tensor(tensors)
        return prob.numpy()

    def predict_single(self, image: np.ndarray):
        """
        单张图片预测 返回： class_id probability
        """
        prob = self.predict_numpy(image[None])[0]
        # top5 = np.argsort(prob)[::-1][:5]
        # for i in top5:
        #     print(i, prob[i])
        idx = int(np.argmax(prob))
        return idx, float(prob[idx])

from abc import ABC, abstractmethod

from Utils.models.cv.config import CVNetConfig


class CVModelLoader(ABC):
    """模型加载器基类"""

    def __init__(self, config: CVNetConfig):
        self.config = config

    @abstractmethod
    def load(self):
        """返回预训练模型"""
        raise NotImplementedError

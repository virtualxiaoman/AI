import timm
import torch.nn
from transformers import AutoModel, SiglipModel, CLIPModel

from Utils.models.base import CVModelLoader
from Utils.models.cv.config import CVNetConfig


class TIMMLoader(CVModelLoader):
    """timm 模型加载器"""

    def load(self):
        return timm.create_model(model_name=self.config.name, pretrained=self.config.pretrained)


class CLIPLoader(CVModelLoader):
    def load(self):
        return CLIPModel.from_pretrained(self.config.name)


class SigLIPLoader(CVModelLoader):
    def load(self):
        return SiglipModel.from_pretrained(self.config.name)


class TransformersLoader(CVModelLoader):
    """HuggingFace 模型加载器"""

    def load(self):
        if self.config.pretrained:
            return AutoModel.from_pretrained(self.config.name)

        return AutoModel.from_config(self.config.name)


class CVNetFactory:
    """视觉模型工厂"""

    registry = {
        # ===== CNN =====
        "resnet50": TIMMLoader,
        "efficientnet_b0": TIMMLoader,
        "efficientnet_b3": TIMMLoader,
        "convnext_base": TIMMLoader,

        # ===== Vision Transformer =====
        "vit_base_patch16_224": TIMMLoader,
        "vit_large_patch16_224": TIMMLoader,
        "swin_base_patch4_window7_224": TIMMLoader,

        # ===== Self-Supervised =====
        "vit_base_patch14_dinov2": TIMMLoader,

        # ===== Multimodal =====
        "openai/clip-vit-base-patch32": CLIPLoader,
        "google/siglip-base-patch16-224": SigLIPLoader,
    }

    @classmethod
    def register(cls, name: str, loader):
        """注册新的模型"""
        cls.registry[name] = loader

    @classmethod
    def create(cls, config: CVNetConfig | str):
        """
        创建模型

        Args:
            config: CVNetConfig 或 模型名称
        """
        if isinstance(config, str):
            config = CVNetConfig(name=config)

        if config.name not in cls.registry:
            raise ValueError(
                f"Unsupported model: {config.name}\n"
                f"Supported models:\n"
                f"{list(cls.registry.keys())}"
            )

        loader_cls = cls.registry[config.name]
        return loader_cls(config).load()


def test_all_models():
    """测试所有模型是否能够正常创建"""

    print("=" * 80)

    for model_name in CVNetFactory.registry:
        print(f"Testing: {model_name}")

        try:
            net = CVNetFactory.create(
                CVNetConfig(
                    name=model_name,
                    pretrained=False,  # 测试时不下载权重
                )
            )

            print(f"  ✓ Success")
            print(f"    Type: {type(net).__name__}")

        except Exception as e:
            print(f"  ✗ Failed")
            print(f"    {type(e).__name__}: {e}")

        print("-" * 80)


if __name__ == "__main__":
    test_all_models()
    torch.nn.Transformer

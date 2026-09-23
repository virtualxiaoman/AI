from dataclasses import dataclass


@dataclass(slots=True)
class CVNetConfig:
    """模型配置"""

    name: str
    pretrained: bool = True

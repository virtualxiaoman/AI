# 项目结构与 `Utils` 使用指南

> 本文基于当前仓库已有目录和代码编写，描述的是**现有实现范围**，而不是未来规划承诺。仓库仍处于持续整理阶段：部分领域主要包含方向说明、模型测试脚本或临时探索代码。

## 1. 项目定位与组织原则

本仓库用于沉淀 AI 学习笔记、模型试验、专题项目和论文复现。顶层按研究/应用领域划分；`Utils/` 则承担跨实验可复用的 PyTorch 基础能力。

推荐的代码流是：

```text
领域实验脚本（CV / MM / XAI / PR/...）
             │
             ├── Utils.data       数据集、DataLoader 与元数据
             ├── Utils.models     模型结构、预训练模型加载
             ├── Utils.losses     任务专用损失函数
             ├── Utils.trainer    训练、评估、日志、最佳模型保存
             └── Utils.inspect_net 参数量与张量形状分析
             │
             └── Datasets/、Models/、Temp/（本地数据、权重和临时产物）
```

原则如下：

1. **领域归属优先**：某个算法、任务或 demo 的业务逻辑放在对应领域目录，例如 CV 分类实验放在 `CV/Classification/`。
2. **稳定能力再抽取**：只有被多个实验使用、且接口足够稳定的代码才放入 `Utils/`。
3. **数据与权重分离**：数据集统一使用 `Datasets/`，训练权重使用 `Models/`，临时文件使用 `Temp/`；它们默认不纳入 Git。
4. **从根目录启动**：项目代码采用 `from Utils...` 等顶层导入。建议将当前工作目录保持为仓库根目录 `G:\Projects\py\AI`，并优先使用 `python -m <模块路径>` 启动，避免嵌套脚本运行时找不到顶层包。

---

## 2. 顶层目录说明

| 路径 | 当前职责与内容 |
| --- | --- |
| `CV/` | 计算机视觉实验区。目前主要是 `Classification/` 下的监督训练示例：ResNet18 + CIFAR-10、ResNet18 + JAFFE、孪生网络 + ATTfaces，以及 Faster R-CNN/VOC 测试入口；`test/` 中有训练器、AMP、MNIST/CIFAR-10 backbone 和网络信息的验证脚本。 |
| `EFF/` | Efficient AI（高效 AI）方向说明。目前以模型压缩、剪枝、蒸馏、量化、LoRA 等主题规划为主。 |
| `Learning/` | 学习路线、计算机视觉学习材料和阶段计划，含 Markdown 与部分导出的 PDF。 |
| `LLM/` | Large Language Models 方向入口，涵盖 Transformer Decoder、SFT/RLHF/LoRA、RAG、Prompt 与 Alignment；目前有 DeepSeek、Qwen 等模型测试脚本和临时目录。 |
| `MM/` | 多模态实验区，当前重点为 `ImgToText/`：图像描述、视觉问答、OCR 和文档解析相关模型的试验脚本；另有小型 demo 项目。 |
| `PR/` | Paper Reproduction（论文复现）工作区。`papers/` 用于一篇论文一个独立项目，`_template/` 提供模板，具体结构见 `PR/README.md`。 |
| `RL/` | Reinforcement Learning 方向说明，包括 DQN、PPO、SAC、模型式/免模型方法、World Model 与具身决策等主题。 |
| `SYS/` | AI 系统方向说明，包括 Agent、工具调用、大模型自动化与多智能体。 |
| `Utils/` | 通用工具层，是目前最完善的共享代码区域。详见第 3 至第 7 节。 |
| `XAI/` | Explainable AI 实验区。现有 LIME 图像特征归因相关实现、可视化图与 loss landscape 小项目。 |
| `Datasets/` | 本地数据存储根目录。`Utils.config.path.DATASETS_DIR` 指向此处；目录被 Git 忽略。 |
| `Models/` | 本地训练权重/模型文件的存储根目录。`Utils.config.path.MODELS_DIR` 指向此处；目录被 Git 忽略。 |
| `Temp/` | 临时代码、下载测试及中间产物，不应作为正式实现依赖。目录被 Git 忽略。 |
| `main.py` | 当前为 IDE/Python 环境验证性质的示例入口，并非项目统一 CLI。 |

> `EFF/`、`RL/`、`SYS/` 当前主要承担学习导航作用；与 `CV/`、`MM/`、`XAI/` 相比，尚未形成同等规模的可运行实现集合。

---

## 3. `Utils` 总览

`Utils/` 以“路径 → 数据 → 模型 → 损失 → 训练 → 检查”的层次组织：

```text
Utils/
├── config/
│   └── path.py                    # 仓库根目录、数据集和模型目录
├── data/
│   ├── base.py                    # 数据加载抽象基类与统一返回对象
│   └── cv/
│       ├── config.py              # CVDatasetConfig
│       ├── load_data.py           # CVDatasetFactory
│       ├── datasets/
│       │   ├── mnist.py           # CVMNISTLoader
│       │   └── cifar10.py         # CVCIFAR10Loader
│       └── utils/
│           ├── metadata.py        # 数据集类别/形状元数据提取
│           └── split.py           # 可复现训练/验证集切分
├── losses/
│   └── loss_fn.py                 # ContrastiveLoss、ArcFaceLoss
├── models/
│   ├── base.py                    # CVModelLoader 抽象接口
│   └── cv/
│       ├── config.py              # CVNetConfig
│       ├── load_net.py            # CVNetFactory 与 timm/HF 加载器
│       ├── classic_net.py         # LeNet-5、ResNet34、ResNet50
│       ├── xm_transformer.py      # 教学式 Encoder-Decoder Transformer
│       └── pretrained_net.py      # 当前为空的预留模块
├── trainer/
│   └── train_net.py               # FNN、孪生网络、ArcFace 训练器
├── inspect_net.py                 # 参数统计和 FX 形状流分析
├── evaluate/                      # 当前为空的预留目录
├── tools/                         # 当前为空的预留目录
└── visualization/                 # 当前为空的预留目录
```

### 3.1 模块边界

- `config`：只处理全局路径等基础配置，不应包含任务训练参数。
- `data`：负责构造 `Dataset`、`DataLoader`、数据集切分和元信息；不负责模型训练。
- `models`：负责网络结构与模型创建；不应掺杂特定实验的数据路径或训练循环。
- `losses`：存放可复用的损失层，尤其是需要自定义 `forward` 或推理 logits 的目标函数。
- `trainer`：负责标准训练循环、评估、AMP、调度器和最佳模型保存。
- `inspect_net`：独立的诊断工具，训练前用于检查结构、参数量与中间张量尺寸。
- `evaluate`、`tools`、`visualization`：已经预留但暂无实际 Python 实现。新增通用能力时可按职责补充，不要将与某一论文强耦合的代码直接搬入这些目录。

---

## 4. 路径管理：`Utils.config.path`

文件：`Utils/config/path.py`

```python
from Utils.config.path import DATASETS_DIR, MODELS_DIR, PROJECT_ROOT, TestPictures
```

该模块通过自身文件位置向上定位仓库根目录，提供：

| 名称 | 含义 |
| --- | --- |
| `PROJECT_ROOT` | 仓库根目录，即当前项目的 `AI/` 目录。 |
| `DATASETS_DIR` | `PROJECT_ROOT / "Datasets"`。 |
| `MODELS_DIR` | `PROJECT_ROOT / "Models"`。 |
| `TestPictures` | `DATASETS_DIR / "TestPictures"`，供图像解释等样例读取测试图片。 |

**使用建议**：新代码优先拼接这些 `Path` 对象，而非在嵌套脚本中写 `../../../Datasets/...` 形式的相对路径。

```python
from Utils.config.path import DATASETS_DIR, MODELS_DIR

jaffe_dir = DATASETS_DIR / "JAFFE" / "jaffe_split"
checkpoint = MODELS_DIR / "JAFFE" / "resnet18.pth"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
```

---

## 5. 数据模块：`Utils.data`

### 5.1 统一配置与返回对象

文件：

- `Utils/data/cv/config.py`
- `Utils/data/base.py`

`CVDatasetConfig` 是 CV 数据加载的配置对象：

```python
from Utils.data.cv.config import CVDatasetConfig

config = CVDatasetConfig(
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    train_transform=None,
    test_transform=None,
    val_ratio=0.1,
    random_seed=42,
    download=True,
)
```

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `batch_size` | `64` | 训练、验证和测试 DataLoader 的批大小。 |
| `num_workers` | `0` | DataLoader 工作进程数。Windows 环境建议从 `0` 开始验证。 |
| `pin_memory` | `True` | 是否使用 pinned memory。对 CUDA 训练通常有帮助。 |
| `train_transform` | `None` | 自定义训练集变换；为 `None` 时使用数据集内置默认变换。 |
| `test_transform` | `None` | 自定义测试/验证变换；为 `None` 时使用默认变换。 |
| `val_ratio` | `0` | 从原训练集划出的验证集比例；不大于 0 时不创建验证集。 |
| `random_seed` | `42` | 训练/验证切分使用的随机种子。 |
| `download` | `True` | 是否允许 torchvision 自动下载数据。 |

`CVDatasetBundle` 是加载器的统一返回值，含：

- `train_dataset`、`train_loader`（必有）；
- `val_dataset`、`val_loader`（仅当 `val_ratio > 0` 时可能有）；
- `test_dataset`、`test_loader`（当前 MNIST/CIFAR-10 加载器均提供）；
- `num_classes`、`class_names`、`channels`、`input_shape`（可获得时提供）；
- `mean`、`std`（该数据集默认归一化统计量）；
- `config`（本次实际使用的配置）。

`CVBaseDatasetLoader` 是数据加载器的抽象基类。子类只需设置 `dataset_name` 并实现 `load()`；其余部分可复用 `build_loader()`（训练集，`shuffle=True`）与 `build_test_loader()`（验证/测试集，`shuffle=False`）。

### 5.2 已实现数据集：MNIST 与 CIFAR-10

工厂位于 `Utils/data/cv/load_data.py`：

```python
from Utils.data.cv.load_data import CVDatasetFactory

bundle = CVDatasetFactory.create("MNIST")
# 或：bundle = CVDatasetFactory.create("CIFAR10", config)
```

当前工厂注册的名称严格为：

| 工厂名 | 加载器 | 本地目录 | 默认训练变换 |
| --- | --- | --- | --- |
| `"MNIST"` | `CVMNISTLoader` | `Datasets/MNIST/` | `ToTensor` + MNIST 均值/标准差归一化。 |
| `"CIFAR10"` | `CVCIFAR10Loader` | `Datasets/CIFAR10/` | 随机裁剪、随机水平翻转、`ToTensor` 与 CIFAR-10 归一化。 |

测试集默认不使用随机增强，只进行张量转换和归一化。

完整的 MNIST 使用示例：

```python
from Utils.data.cv.config import CVDatasetConfig
from Utils.data.cv.load_data import CVDatasetFactory

config = CVDatasetConfig(
    batch_size=128,
    val_ratio=0.1,
    random_seed=42,
    num_workers=0,
)
bundle = CVDatasetFactory.create("MNIST", config)

print(bundle.num_classes)   # 原始数据集可直接提取时为类别数
print(bundle.input_shape)   # 例如 (1, 28, 28)
for images, labels in bundle.train_loader:
    print(images.shape, labels.shape)
    break
```

### 5.3 切分和元数据的注意事项

- `split_train_val(dataset, ratio, seed)` 通过 `torch.utils.data.random_split` 返回训练子集和验证子集；设置种子后划分可复现。
- `extract_metadata(dataset)` 会尽可能从 `classes`、`targets`、`data` 属性中读取类别数、类别名和输入尺寸。
- **当前实现限制**：当 `val_ratio > 0` 时，训练集会成为 `torch.utils.data.Subset`；该对象通常不直接暴露原始数据集的 `classes` 或 `data` 属性，因此当前 `bundle` 中的元数据字段可能为 `None`。此时可从原始数据集、已知任务配置或 `train_dataset.dataset` 进一步读取信息。
- `CVDatasetFactory.create()` 对名称大小写敏感；传入未注册名称会直接抛出 `KeyError`。新增数据集前应先实现加载器，再注册到 `registry`。

### 5.4 自定义变换与新增数据集

自定义变换示例：

```python
from torchvision import transforms
from Utils.data.cv.config import CVDatasetConfig
from Utils.data.cv.load_data import CVDatasetFactory

train_tf = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

bundle = CVDatasetFactory.create(
    "MNIST",
    CVDatasetConfig(train_transform=train_tf, test_transform=test_tf),
)
```

新增一个通用 CV 数据集时，建议：

1. 在 `Utils/data/cv/datasets/` 新建 `<dataset>.py`，继承 `CVBaseDatasetLoader`。
2. 在 `load()` 中创建 train/test dataset，调用 `split_train_val()`，最后返回 `CVDatasetBundle`。
3. 在 `CVDatasetFactory.registry` 中登记稳定、明确的名称。
4. 在相应实验脚本或测试中验证加载器。若数据集仅为某篇论文服务，请优先放在该论文的 `PR/papers/<paper-id>/data/`，不要过早通用化。

---

## 6. 模型模块：`Utils.models`

### 6.1 配置与工厂

文件：

- `Utils/models/cv/config.py`
- `Utils/models/base.py`
- `Utils/models/cv/load_net.py`

`CVNetConfig` 目前只有两个字段：

```python
from Utils.models.cv.config import CVNetConfig

config = CVNetConfig(name="resnet50", pretrained=True)
```

`CVNetFactory` 根据模型名称选择加载器：

```python
from Utils.models.cv.config import CVNetConfig
from Utils.models.cv.load_net import CVNetFactory

net = CVNetFactory.create(CVNetConfig(name="resnet50", pretrained=True))
# 字符串也可用：net = CVNetFactory.create("resnet50")
```

当前注册表：

| 模型名 | 加载来源 |
| --- | --- |
| `resnet50`、`efficientnet_b0`、`efficientnet_b3`、`convnext_base` | `timm.create_model()` |
| `vit_base_patch16_224`、`vit_large_patch16_224`、`swin_base_patch4_window7_224` | `timm.create_model()` |
| `vit_base_patch14_dinov2` | `timm.create_model()` |
| `openai/clip-vit-base-patch32` | Hugging Face `CLIPModel.from_pretrained()` |
| `google/siglip-base-patch16-224` | Hugging Face `SiglipModel.from_pretrained()` |

注意：

- `TIMMLoader` 尊重 `pretrained` 配置。
- 当前 `CLIPLoader` 和 `SigLIPLoader` 始终调用 `from_pretrained()`，因此会加载对应预训练仓库；`pretrained=False` 并不会让它们构造随机初始化模型。
- `TransformersLoader` 支持通用 `AutoModel` 加载，但当前没有默认注册到 `CVNetFactory.registry`。
- `CVNetFactory.create()` 只支持已注册名称；不支持时会抛出 `ValueError` 并列出当前可用模型。
- 工厂创建的是原始上游模型。不同模型的输入预处理、输出字段和分类头形式不完全一致，训练前必须根据具体模型调整数据变换和 head。

注册扩展示例：

```python
from Utils.models.cv.load_net import CVNetFactory, TIMMLoader

CVNetFactory.register("my_timm_model", TIMMLoader)
net = CVNetFactory.create("my_timm_model")
```

只有在名称能够被 `timm` 正确识别时，上例才可运行。

### 6.2 本地经典网络

文件：`Utils/models/cv/classic_net.py`

当前提供：

- `LeNet5`：适合 MNIST 等单通道 28×28 输入的基础卷积网络；
- `resnet34(num_classes=...)`：本地 ResNet-34 构造函数；
- `resnet50(num_classes=...)`：本地 ResNet-50 构造函数；
- `BasicBlock`、`Bottleneck`、`ResNet`：上述 ResNet 构造所依赖的基础模块。

使用本地模型无需依赖模型工厂：

```python
from Utils.models.cv.classic_net import LeNet5

net = LeNet5()
```

### 6.3 教学式 Transformer

文件：`Utils/models/cv/xm_transformer.py`

该文件实现了一个完整的 Encoder–Decoder Transformer 教学版本，包含多头注意力、位置编码、前馈网络、编码器/解码器、掩码、标签平滑和 Noam 学习率调度器。核心入口：

```python
from Utils.models.cv.xm_transformer import (
    make_model, make_src_mask, make_tgt_mask, make_optimizer,
)

model = make_model(src_vocab=1000, tgt_vocab=1000, N=2, d_model=256)
optimizer = make_optimizer(model, d_model=256, warmup=4000)
```

它与 `models/cv/` 的路径同级，但从实现内容看更适合作为 Transformer 原理学习/实验模块，而不是通用视觉 backbone。使用时应自行准备 token 化、padding 约定、批处理和序列到序列训练循环。

`pretrained_net.py` 当前为空文件，尚无公开接口。

---

## 7. 损失、训练与模型诊断

### 7.1 自定义损失：`Utils.losses.loss_fn`

#### `ContrastiveLoss`

用于孪生网络嵌入学习：

```python
from Utils.losses.loss_fn import ContrastiveLoss

criterion = ContrastiveLoss(margin=1.0)
loss = criterion(embedding1, embedding2, pair_label)
```

当前标签约定是：

- `1`：同类/正样本对，最小化两 embedding 的欧氏距离；
- `0`：异类/负样本对，鼓励距离至少达到 `margin`。

该约定与部分论文或实现常用的标签方向不同，因此自定义数据集时必须保持一致。

#### `ArcFaceLoss`

用于 embedding 分类训练：

```python
from Utils.losses.loss_fn import ArcFaceLoss

loss_fn = ArcFaceLoss(num_classes=7, embedding_dim=128, s=30.0, m=0.5)
loss, logits = loss_fn(embeddings, labels)
```

模型应输出二维 embedding；`ArcFaceLoss.forward()` 返回 `(loss, logits)`。在评估阶段，`get_logits(embeddings)` 会提供未施加目标类别角度边际的缩放 cosine logits。

### 7.2 基础训练器：`NetTrainerFNN`

文件：`Utils/trainer/train_net.py`

`NetTrainerFNN` 适合输入批次结构为 `(X, y)`、前向形式为 `net(X)` 的标准前馈网络。它负责：

- 将网络和损失函数移动至自动选择的 CUDA/CPU 设备；
- 训练循环与每 epoch 的 loss 汇总；
- 按间隔记录训练指标，并可在训练中评估训练集/测试集；
- 对常见回归/分类损失自动推断 `eval_type`（`"loss"` 或 `"acc"`）；
- 可选 AMP（仅设备为 CUDA 时生效）；
- 每个 epoch 调用一次可选学习率调度器；
- 根据验证指标保存当前最佳完整模型对象；
- 记录 `train_loss_list`、`train_acc_list`、`test_loss_list`、`test_acc_list`、`time_list` 等历史数据；
- 通过 `view_parameters()` 打印模型结构、可训练参数数量及明细。

基本分类示例：

```python
import torch.nn as nn
import torch.optim as optim

from Utils.data.cv.load_data import CVDatasetFactory
from Utils.models.cv.classic_net import LeNet5
from Utils.trainer.train_net import NetTrainerFNN

bundle = CVDatasetFactory.create("MNIST")
net = LeNet5()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

trainer = NetTrainerFNN(
    train_loader=bundle.train_loader,
    test_loader=bundle.test_loader,
    net=net,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=10,
    eval_interval=1,
    use_amp=True,
)
trainer.train_net(net_save_path="Models/MNIST/lenet5.pth")
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `train_loader`、`test_loader` | 训练与评估 DataLoader；评估 loader 可为 `None`，但应相应关闭或调整训练期评估。 |
| `net` | `torch.nn.Module`，标准训练器要求 `net(X)` 返回预测张量。 |
| `loss_fn` | PyTorch 标准损失或兼容接口的损失模块。 |
| `optimizer` | PyTorch 优化器。 |
| `scheduler` | 可选学习率调度器；当前会在每个训练 epoch 后调用 `step()`。 |
| `epochs` | 训练轮数，默认 `100`。 |
| `eval_type` | `"acc"`、`"loss"` 或 `None`。`None` 时仅对已知的标准回归/分类损失自动推断。 |
| `eval_during_training` | 是否在训练过程中评估，默认 `True`。显存/耗时受限时可关闭。 |
| `eval_interval` | 评估和日志间隔，以 epoch 计。 |
| `device` | 可传入具体设备；未指定时优先 CUDA。 |
| `use_amp` | 自动混合精度开关；只有 CUDA 可用时才真正启用。 |
| `free_memory` | 设为 `True` 时会在训练迭代后删除部分局部张量引用。 |
| `net_name` | 用于训练完成日志的名称。 |

**使用限制与保存行为**：

- 自动评估仅覆盖实现中列出的常见 PyTorch 损失。使用 `ContrastiveLoss`、`ArcFaceLoss` 或其他自定义任务时，通常应显式传入合适的 `eval_type`，或使用对应专用训练器。
- `net_save_path` 仅在 `eval_during_training=True` 时用于最佳模型保存；当前保存方式为 `torch.save(self.net, path)`，即保存整个模型对象，而不是推荐性更强的 `state_dict`。
- 标准分类准确率逻辑假设多分类输出形状为 `[B, C]`；单输出分类时使用 sigmoid 后以 `0.5` 为阈值。

### 7.3 孪生网络训练器：`NetTrainerPair`

`NetTrainerPair` 继承 `NetTrainerFNN`，适用于 DataLoader 返回：

```text
(x1, x2, pair_label)
```

且网络前向为：

```text
embedding1, embedding2 = net(x1, x2)
```

示例：

```python
from Utils.losses.loss_fn import ContrastiveLoss
from Utils.trainer.train_net import NetTrainerPair

trainer = NetTrainerPair(
    train_loader=train_loader,
    test_loader=val_loader,
    net=siamese_net,
    loss_fn=ContrastiveLoss(margin=1.0),
    optimizer=optimizer,
    epochs=20,
    eval_type="acc",
)
trainer.train_net("Models/ATTfaces/siamese.pth")
```

该训练器用 `torch.nn.functional.pairwise_distance` 计算 pair distance，并以 **距离小于 `0.5` 判定同类** 来计算准确率。这个阈值是当前实现的固定值，应与 embedding 训练尺度和标签约定一起审视；若任务不同，建议修改/扩展评估策略而不是不加验证地复用。

### 7.4 ArcFace 训练器：`NetTrainerArcFace`

`NetTrainerArcFace` 用于：

```text
embeddings = net(X)
loss, logits = arcface_loss(embeddings, y)
```

它会使模型和 ArcFace 损失层均进入 train/eval 模式，并在评估中通过 `loss_fn.get_logits(outputs)` 计算分类准确率。

```python
from Utils.losses.loss_fn import ArcFaceLoss
from Utils.trainer.train_net import NetTrainerArcFace

loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_dim=128)
trainer = NetTrainerArcFace(
    train_loader=train_loader,
    test_loader=val_loader,
    net=embedding_net,
    loss_fn=loss_fn,
    optimizer=optimizer,
    eval_type="acc",
)
trainer.train_net("Models/ATTfaces/arcface.pth")
```

### 7.5 网络检查：`NetInspector` 与 `NetworkShapeAnalyzer`

文件：`Utils/inspect_net.py`

```python
import torch
from Utils.inspect_net import NetInspector, NetworkShapeAnalyzer
from Utils.models.cv.classic_net import LeNet5

net = LeNet5()

# 参数量、结构与参数明细
inspector = NetInspector(net)
inspector.view_parameters(
    view_net_struct=True,
    view_params_count=True,
    view_params_details=False,
)

# 用 torch.fx 追踪并查看各操作输出 shape
shape_analyzer = NetworkShapeAnalyzer(net)
shape_analyzer.analyze(torch.randn(2, 1, 28, 28))
```

`NetInspector` 复用训练器的参数查看能力，但只需要网络对象。`NetworkShapeAnalyzer` 使用 `torch.fx.symbolic_trace` 与 `ShapeProp` 推断 shape，因此网络的前向过程必须能够被 FX 正常追踪；包含复杂 Python 控制流、动态模块创建或特殊第三方算子的模型可能无法分析。

---

## 8. 新实验的推荐落位

### 8.1 常规专题实验

例如在 CV 中测试一个新分类方案：

```text
CV/Classification/<experiment-name>/
├── train.py
├── evaluate.py
├── config.py 或 configs/
├── README.md
└── assets/（可选）
```

- 实验特定网络、数据增强、指标和业务逻辑留在实验目录；
- 若数据加载器/损失/训练器经过多个项目验证后仍通用，再抽取到 `Utils/`；
- 权重写入 `Models/<experiment-name>/`，数据写入 `Datasets/<dataset-name>/`。

### 8.2 论文复现

新论文应从 `PR/_template/` 复制至 `PR/papers/<paper-id>/`，并按照 `PR/README.md` 的 `configs/`、`data/`、`src/`、`scripts/`、`outputs/`、`tests/` 等约定执行。论文专属实现不应直接混入 `Utils/`。

### 8.3 何时扩展 `Utils`

满足以下条件时才建议扩展：

1. 已至少在一个真实实验中验证；
2. 接口不依赖某一篇论文或某个数据集的私有细节；
3. 可以用简短文档说明输入、输出和使用限制；
4. 有最小示例或测试脚本可验证。

例如：新增的通用 Dataset loader、统一评估指标、可复用 checkpoint 工具、通用可视化函数，适合加入 `Utils/`；某个论文的 attention 变体、任务专属数据预处理，则应保留在该论文/项目目录。

---

## 9. 运行前检查清单

1. 在仓库根目录执行命令：
   ```powershell
   cd G:\Projects\py\AI
   ```
2. 确认所选脚本需要的第三方库已安装。当前代码涉及 PyTorch、torchvision、timm、transformers，以及部分 MM/XAI 脚本使用的额外模型生态依赖。
3. 需要本地数据时，将其放到 `Datasets/` 的预期位置；需要下载模型时确保网络、缓存与磁盘空间可用。
4. 训练前先调用 `NetInspector` 或打印首个 batch，确认输入通道、分辨率、标签范围与网络 head 一致。
5. 将 checkpoint 输出定位到 `Models/`，将一次性/中间结果定位到 `Temp/` 或项目专属输出目录。
6. 对新训练器、新模型或新数据加载器，先用小 batch、少量 epoch 完成最小闭环验证。

---

## 10. 当前已知边界

- 顶层尚未提供统一的打包配置、统一命令行入口或一份完整锁定依赖清单；`main.py` 也不是统一调度器。
- 若直接执行深层脚本文件而非从根目录用模块方式运行，Python 模块搜索路径可能导致 `Utils` 导入失败。
- `Utils` 当前最成熟的是 PyTorch/CV 训练链路；它还不是覆盖 LLM、RL、MM 全部任务类型的统一框架。
- 多模态和大模型测试脚本通常依赖各自的模型、权重、账号或运行环境，应逐个脚本检查其先决条件。

因此，本文档既可作为现有代码的使用说明，也可作为后续逐步规范化项目布局的基线。

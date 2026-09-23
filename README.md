# AI 项目

一个面向 **AI 学习、实验验证与项目沉淀** 的 Python 仓库。项目以计算机视觉（CV）为当前主要实现方向，同时为大语言模型（LLM）、多模态（MM）、高效 AI（EFF）、强化学习（RL）、AI 系统（SYS）、可解释 AI（XAI）和论文复现（PR）预留了清晰的工作区。

该仓库从早期 DataScience 项目演进而来，重点是按领域归档学习材料与实验代码，并通过 `Utils/` 复用数据加载、模型创建、损失函数、训练和网络分析能力。

## 项目结构

```text
AI/
├── CV/          # 计算机视觉：分类、检测及训练/模型测试样例
├── EFF/         # 高效 AI：压缩、蒸馏、量化、低秩适配等方向说明
├── Learning/    # 学习路线、知识整理与阶段规划
├── LLM/         # 大语言模型：Transformer、对齐、RAG 等探索入口
├── MM/          # 多模态：图文理解、OCR、图像描述及小型项目
├── PR/          # 论文复现：统一模板与独立论文复现实验空间
├── RL/          # 强化学习：算法与具身决策方向说明
├── SYS/         # AI 系统：Agent、工具调用与多智能体方向说明
├── Utils/       # 跨项目共享的 PyTorch 工具模块
├── XAI/         # 可解释 AI：LIME 等特征归因实验
├── Datasets/    # 本地数据集目录（默认不提交）
├── Models/      # 本地模型权重目录（默认不提交）
└── Temp/        # 临时代码与实验产物（默认不提交）
```

## 当前已具备的能力

- **CV 实验**：包含 ResNet、孪生网络、ArcFace、Faster R-CNN 等训练或验证脚本，以及 MNIST、CIFAR-10、JAFFE、ATTfaces、VOC 等相关实验目录。
- **共享训练链路**：`Utils/` 提供 MNIST/CIFAR-10 数据加载、常用视觉模型加载、经典网络、对比损失与 ArcFace 损失、通用/孪生/ArcFace 训练器，以及模型参数与形状分析工具。
- **多模态与 XAI 探索**：包含图像描述、视觉问答、OCR 模型测试，以及 LIME 图像解释示例。
- **论文复现规范**：`PR/` 提供每篇论文独立目录的模板、配置、输出和文档约定；当前不含具体论文复现实现。

## 使用方式

建议在仓库根目录运行代码，使 `Utils` 等顶层模块可被正常导入：

```powershell
cd G:\Projects\py\AI
python -m CV.test.test_backbone_MNIST
```

数据集放在 `Datasets/`，模型权重放在 `Models/`；两者及 `Temp/`、`.venv/`、`.idea/` 已由 `.gitignore` 忽略。不同实验对 Python、PyTorch、`torchvision`、`timm`、`transformers` 等依赖的要求可能不同，应按实际脚本所用模型补齐环境。

## 详细文档

- [项目结构与 Utils 使用指南](PROJECT_GUIDE.md)：各目录的职责、现有实现范围、`Utils/` 的组织方式、接口与示例。
- [论文复现工作区说明](PR/README.md)：新建论文复现实验时的目录规范。

## 约定

- 按领域组织实验；项目专属代码放入对应领域目录，稳定可复用的基础能力优先放入 `Utils/`。
- 大型数据集、权重、日志和临时产物不要提交到 Git。
- 新的论文复现项目应从 `PR/_template/` 复制，并创建于 `PR/papers/<paper-id>/`。

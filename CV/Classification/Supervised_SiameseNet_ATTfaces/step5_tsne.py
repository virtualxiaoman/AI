import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from CV.Classification.Supervised_SiameseNet_ATTfaces.step3_train_ArcFaceLoss import ResNet18Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../../../Models/ATTfaces/resnet18_arcface.pth"
data_root = "../../../Datasets/ATTfaces/raw"

try:
    model = torch.load(model_path, weights_only=False)
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型出错: {e}")
    print("提示：如果报错内容涉及 'AttributeError' 或结构不匹配，请确认保存时是否只保存了 state_dict。")
    exit()

model = model.to(device)
model.eval()

# 预处理 (必须与训练时的验证集 Transform 一致)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main(model, data_root, transform, device):
    print("正在提取全量数据特征以进行 t-SNE 可视化...")

    # 使用 ImageFolder 方便地加载所有数据
    dataset = datasets.ImageFolder(data_root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    embeddings_list = []
    labels_list = []

    # 提取特征
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            # 对特征进行归一化，ArcFace 的特征分布在超球面上，归一化后可视化效果更好
            # feats = F.normalize(feats, p=2, dim=1)  # 网络最后一层归一化了，所以不用
            embeddings_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"特征提取完毕，形状: {embeddings.shape}。开始运行 t-SNE...")

    # 运行 t-SNE
    # n_components=2: 降维到2D
    # init='pca': 初始化通常更稳定
    tsne = TSNE(n_components=2, init='pca', random_state=42, learning_rate='auto')
    X_tsne = tsne.fit_transform(embeddings)

    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='jet', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Class ID')
    plt.title(f"t-SNE Visualization of ResNet18-ArcFace Embeddings\n"
              f"({len(dataset)} samples, {len(dataset.classes)} classes)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main(model, data_root, val_transform, device)

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from CV.Classification.Supervised_SiameseNet_ATTfaces.step2_show import get_random_pair
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


def get_similarity(model, img_path1, img_path2, transform, device):
    """计算两张图片的余弦相似度"""
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    # 转换为 Tensor 并增加 Batch 维度
    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model(t1)
        emb2 = model(t2)

        # 归一化特征向量
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        # 计算余弦相似度 (Dot product of normalized vectors)
        similarity = torch.mm(emb1, emb2.t()).item()

    return img1, img2, similarity


def main():
    # --- 开始绘图 ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # 第一行：Same Person
    for i in range(3):
        p1, p2, label = get_random_pair(data_root, same=True)
        img1_pil, img2_pil, sim = get_similarity(model, p1, p2, val_transform, device)

        # 拼接显示
        combined_img = np.hstack((np.array(img1_pil.resize((112, 112))), np.array(img2_pil.resize((112, 112)))))

        axes[0, i].imshow(combined_img)
        axes[0, i].set_title(f"Same - Sim: {sim:.4f}", color='green', fontsize=10)
        axes[0, i].axis('off')

    # 第二行：Different People
    for i in range(3):
        p1, p2, label = get_random_pair(data_root, same=False)
        img1_pil, img2_pil, sim = get_similarity(model, p1, p2, val_transform, device)

        combined_img = np.hstack((np.array(img1_pil.resize((112, 112))), np.array(img2_pil.resize((112, 112)))))

        axes[1, i].imshow(combined_img)
        axes[1, i].set_title(f"Diff - Sim: {sim:.4f}", color='red', fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle("Face Verification", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()

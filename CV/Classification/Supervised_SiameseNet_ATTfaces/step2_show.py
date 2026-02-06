import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from CV.Classification.Supervised_SiameseNet_ATTfaces.step1_train_ContrastiveLoss import SiameseNet


def get_random_pair(root_dir, same=True):
    """
    从测试集中随机选择一对图片路径
    root_dir: ../../../Datasets/ATTfaces/split/test
    same: True 返回同一个人，False 返回不同人
    """
    # 获取所有人的文件夹列表 (s31, s32, ..., s40)
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()

    if same:
        # 选择同一个人
        class_name = random.choice(classes)
        class_path = os.path.join(root_dir, class_name)
        images = os.listdir(class_path)
        # 随机选两张
        img1_name, img2_name = random.sample(images, 2)
        return os.path.join(class_path, img1_name), os.path.join(class_path, img2_name), "Same Person"
    else:
        # 选择不同的人
        class1, class2 = random.sample(classes, 2)

        path1 = os.path.join(root_dir, class1)
        path2 = os.path.join(root_dir, class2)

        img1_name = random.choice(os.listdir(path1))
        img2_name = random.choice(os.listdir(path2))

        return os.path.join(path1, img1_name), os.path.join(path2, img2_name), "Different People"


# --- 图像预处理 ---
# transform_input: 用于输入模型（包含归一化）
# transform_display: 仅用于显示（Resize到同样大小，但不归一化，否则图片颜色会很奇怪）
transform_input = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_display = transforms.Compose([
    transforms.Resize((224, 224))
])


# --- 4. 主程序 ---
def main(model_path):
    test_root = "../../../Datasets/ATTfaces/split/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    # 准备 4 组数据：前两组 Same，后两组 Different
    pairs_config = [True, True, False, False]
    results = []

    for is_same in pairs_config:
        try:
            path1, path2, label_str = get_random_pair(test_root, same=is_same)

            # 打开图片
            img1_raw = Image.open(path1).convert('RGB')
            img2_raw = Image.open(path2).convert('RGB')

            # 预处理输入模型
            img1_tensor = transform_input(img1_raw).unsqueeze(0).to(device)  # Add batch dim
            img2_tensor = transform_input(img2_raw).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                feat1, feat2 = model(img1_tensor, img2_tensor)
                # 计算欧氏距离
                distance = F.pairwise_distance(feat1, feat2).item()

            # 保存用于绘图的数据
            # 将两张图拼在一起用于显示
            img1_show = transform_display(img1_raw)
            img2_show = transform_display(img2_raw)

            results.append({
                'img1': img1_show,
                'img2': img2_show,
                'label': label_str,
                'dist': distance
            })

        except Exception as e:
            print(f"Error processing pair: {e}")

    # --- 5. 绘图 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Siamese Network Verification Results', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        data = results[idx]

        # 拼接图片：将两张 PIL 图片转换为 numpy 数组并横向拼接
        im1 = np.array(data['img1'])
        im2 = np.array(data['img2'])
        combined_img = np.concatenate((im1, im2), axis=1)  # axis=1 是水平拼接

        ax.imshow(combined_img)

        # 根据距离判断颜色：距离越小越好（如果是同一个人）
        # 假设阈值大概在 1.0 左右 (取决于 margin 设置)
        text_color = 'green' if (data['dist'] < 0.5 and "Same" in data['label']) or \
                                (data['dist'] > 0.5 and "Different" in data['label']) else 'red'

        title = f"{data['label']}\nEuclidean Dist: {data['dist']:.4f}"
        ax.set_title(title, color=text_color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(model_path="../../../Models/ATTfaces/siamese_net.pth")

# # visualize_class_directions.py
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms, datasets
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # ---------- 配置 ----------
# data_root = "../../../Datasets/ATTfaces/raw"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# batch_size = 64
# embedding_dim = 512
#
# # 预处理（与你训练/验证一致）
# val_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])
# from CV.Classification.Supervised_SiameseNet_ATTfaces.step3_train_ArcFaceLoss import ResNet18Embedding
#
# model_path = "../../../Models/ATTfaces/resnet18_arcface.pth"
#
# try:
#     model = torch.load(model_path, weights_only=False)
#     print("模型加载成功！")
# except Exception as e:
#     print(f"加载模型出错: {e}")
#     print("提示：如果报错内容涉及 'AttributeError' 或结构不匹配，请确认保存时是否只保存了 state_dict。")
#     exit()
#
# model = model.to(device)
# model.eval()
#
#
# # ---------- 如果你还没计算 embeddings，这里可以提取 ----------
# def extract_embeddings(model, data_root, transform, device, max_samples=None):
#     """
#     返回 embeddings (N, D) 和 labels (N,)
#     """
#     dataset = datasets.ImageFolder(data_root, transform=transform)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     embs_list, labs_list = [], []
#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs = imgs.to(device)
#             e = model(imgs)  # 假定 model forward 已归一化
#             e = F.normalize(e, p=2, dim=1)  # 再确保归一化
#             embs_list.append(e.cpu().numpy())
#             labs_list.append(labels.numpy())
#             if max_samples is not None and sum(len(x) for x in embs_list) >= max_samples:
#                 break
#     embs = np.concatenate(embs_list, axis=0)
#     labs = np.concatenate(labs_list, axis=0)
#     return embs, labs, dataset
#
#
# # ---------- 画图函数：类方向 + 样本点 ----------
# def plot_class_directions(embs, labels, dataset, n_pca_components=2, show_samples=True,
#                           arrow_scale=1.0, tick_len=0.06, figsize=(8, 8)):
#     """
#     embs: (N, D) 已归一化
#     labels: (N,)
#     dataset: ImageFolder，用于类名
#     """
#     classes = dataset.classes
#     num_classes = len(classes)
#
#     # 1) 计算每类均值向量并归一化（方向向量）
#     class_means = np.zeros((num_classes, embs.shape[1]), dtype=np.float32)
#     counts = np.zeros((num_classes,), dtype=int)
#     for cls in range(num_classes):
#         sel = (labels == cls)
#         if sel.sum() == 0:
#             continue
#         mean_vec = embs[sel].mean(axis=0)
#         mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
#         class_means[cls] = mean_vec
#         counts[cls] = sel.sum()
#
#     # 2) 用 PCA (在 class_means 上 fit)，然后把 class_means 和样本 embs 都投影到 2D
#     pca = PCA(n_components=n_pca_components)
#     # 先对 class_means 做 PCA 拟合更能突出“方向分布”
#     valid_idx = (counts > 0)
#     pca.fit(class_means[valid_idx])
#     means_2d = pca.transform(class_means)  # (C,2)
#     embs_2d = pca.transform(embs)  # (N,2)
#
#     # 为可视化方便，让 class mean 的长度按 arrow_scale 放大（原点到点的向量）
#     # 但这些点是 PCA 的坐标，不再是单位长度；我们希望箭头从原点指向 means_2d
#     # 绘图：
#     plt.figure(figsize=figsize)
#     ax = plt.gca()
#
#     # optional: draw faint unit circle for reference (centered at origin)
#     # 计算半径为 max norm of means_2d 的值作为参考
#     maxr = np.max(np.linalg.norm(means_2d[valid_idx], axis=1)) * 1.15
#     theta = np.linspace(0, 2 * np.pi, 200)
#     ax.plot(maxr * np.cos(theta), maxr * np.sin(theta), color='lightgray', linewidth=0.8, zorder=0)
#
#     # draw sample scatter (背景)
#     if show_samples:
#         sc = ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=labels, cmap='tab20', s=8, alpha=0.6, zorder=1)
#
#     # draw arrows for each class mean
#     for cls in range(num_classes):
#         if not valid_idx[cls]:
#             continue
#         x, y = means_2d[cls]
#         # arrow
#         ax.arrow(0, 0, x * arrow_scale, y * arrow_scale, head_width=0.02 * maxr, head_length=0.04 * maxr,
#                  length_includes_head=True, linewidth=1.2, zorder=5, color=plt.cm.tab20(cls % 20))
#         # draw small perpendicular tick at tip (to mimic original figure's small bar)
#         tip = np.array([x * arrow_scale, y * arrow_scale])
#         v = tip
#         norm_v = np.linalg.norm(v) + 1e-12
#         u = v / norm_v
#         perp = np.array([-u[1], u[0]])
#         L = tick_len * maxr
#         p1 = tip + perp * (L / 2)
#         p2 = tip - perp * (L / 2)
#         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=plt.cm.tab20(cls % 20), linewidth=2.0, zorder=6)
#
#         # label (类名或编号)
#         ax.text(tip[0] * 1.08, tip[1] * 1.08, f"{cls}", fontsize=8, color=plt.cm.tab20(cls % 20), zorder=7)
#
#     ax.set_aspect('equal', 'box')
#     ax.set_xlabel("PCA dim 1")
#     ax.set_ylabel("PCA dim 2")
#     ax.set_title(f"Class directions (PCA of class means) — {len(dataset)} samples, {len(dataset.classes)} classes")
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.show()
#
#
# # ----------------------------
# # 使用示例（假定你已经载入 model）
# # ----------------------------
# if __name__ == "__main__":
#     # 1) 从模型/数据集中提取 embeddings（如果你已提取过，可跳过）
#     # model 应该已定义并加载好权重，且 model(imgs) 返回 L2 正规化后的 embedding
#     # e.g. model = torch.load(...); model.to(device); model.eval()
#     print("extracting embeddings, this may take a little time...")
#     embs, labs, dataset = extract_embeddings(model, data_root, val_transform, device, max_samples=None)
#     print("done. embs.shape=", embs.shape)
#
#     # 2) 绘制
#     plot_class_directions(embs, labs, dataset, show_samples=True, arrow_scale=1.0, tick_len=0.06, figsize=(8, 8))

# train_arcface.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from tqdm import tqdm

from Utils.loss_fn import ArcFaceLoss
from Utils.train_net import NetTrainerArcFace


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(512, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = self.backbone(x)  # [B, 512]
        x = self.embedding(x)  # [B, D]
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings (important)
        return x


#
# # ---------------- Utilities: retrieval recall@1 ----------------
# @torch.no_grad()
# def compute_recall_at_1(embeddings, labels):
#     """
#     embeddings: [N, D] normalized
#     labels: [N]
#     Return recall@1 (nearest neighbor excluding self).
#     """
#     device = embeddings.device
#     # cosine similarity matrix
#     sims = embeddings @ embeddings.t()  # [N, N]
#     N = sims.size(0)
#     # mask self
#     sims.fill_diagonal_(-1.0)
#     # top-1 index
#     top1 = sims.argmax(dim=1)  # [N]
#     preds = labels[top1]
#     correct = (preds == labels).sum().item()
#     return correct / N
#
# # ---------------- Training & Validation loops ----------------
# def train_one_epoch(model, loss_fn, train_loader, optimizer, device):
#     model.train()
#     loss_fn.train()
#     running_loss = 0.0
#     total = 0
#     correct = 0  # using logits top1 as proxy
#     for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
#         imgs = imgs.to(device)
#         labels = labels.to(device).long()
#         embeddings = model(imgs)               # [B, D] normalized
#         loss, logits = loss_fn(embeddings, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * imgs.size(0)
#         total += imgs.size(0)
#         preds = logits.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#     return running_loss / total, correct / total
#
# @torch.no_grad()
# def validate(model, arcface, val_loader, device):
#     model.eval()
#     arcface.eval()
#     total = 0
#     loss_sum = 0.0
#     correct = 0
#     # first compute logits-based top1
#     for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
#         imgs = imgs.to(device)
#         labels = labels.to(device).long()
#         embeddings = model(imgs)
#         loss, logits = arcface(embeddings, labels)
#         loss_sum += loss.item() * imgs.size(0)
#         total += imgs.size(0)
#         preds = logits.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#
#     # compute retrieval recall@1 using the whole validation set embeddings
#     all_embeddings = []
#     all_labels = []
#     for imgs, labels in val_loader:
#         imgs = imgs.to(device)
#         emb = model(imgs)  # normalized
#         all_embeddings.append(emb.cpu())
#         all_labels.append(labels)
#     all_embeddings = torch.cat(all_embeddings, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
#     recall1 = compute_recall_at_1(all_embeddings, all_labels)
#
#     return loss_sum / total, correct / total, recall1


def main():
    all_path = "../../../Datasets/ATTfaces/raw"
    train_path = "../../../Datasets/ATTfaces/split/train"
    test_path = "../../../Datasets/ATTfaces/split/test"
    embedding_dim = 256
    batch_size = 64
    epochs = 30
    lr = 0.005  # SGD recommended; if use Adam, reduce (1e-4)
    weight_decay = 5e-4
    s = 30.0
    m = 0.5
    use_pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_loader = datasets.ImageFolder(all_path, transform=transform)
    print("classes:", full_loader.classes)  # ['s1' ~ 's40']
    num_classes = len(full_loader.classes)
    train_names = [f"s{i}" for i in range(1, 31)]  # 's1'..'s30'
    test_names = [f"s{i}" for i in range(31, 41)]
    print("train_names: ", train_names)
    print("test_names: ", test_names)
    train_idx, test_idx = [], []
    for i, (path, class_idx) in enumerate(full_loader.samples):
        name = full_loader.classes[class_idx]
        if name in train_names:
            train_idx.append(i)
            # print("train: ", train_idx, i, path, name)
        elif name in test_names:
            test_idx.append(i)
            # print("test: ", test_idx, i, path, name)

    train_ds = Subset(full_loader, train_idx)
    test_ds = Subset(full_loader, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    # train_ds = datasets.ImageFolder(train_path, transform=transform)
    # test_ds = datasets.ImageFolder(test_path, transform=transform)
    # num_classes = len(train_ds.classes)
    # print(f"num_classes: {num_classes}")
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=use_pretrained).to(device)
    loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_dim=embedding_dim, s=s, m=m).to(device)
    optimizer = torch.optim.SGD(list(model.parameters()) + list(loss_fn.parameters()),
                                lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = NetTrainerArcFace(
        net=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        eval_interval=1,
        eval_type='acc'
    )
    trainer.train_net(net_save_path="../../../Models/ATTfaces/arcface_model.pth")

    # # ---------- training loop ----------
    # best_recall1 = 0.0
    # for epoch in range(1, epochs+1):
    #     train_loss, train_acc = train_one_epoch(model, loss_fn, train_loader, optimizer, device)
    #     val_loss, val_acc, val_recall1 = validate(model, loss_fn, val_loader, device)
    #     scheduler.step()
    #
    #     print(f"Epoch {epoch}/{epochs}  TrainLoss: {train_loss:.4f} TrainAcc(top1): {train_acc:.4f} "
    #           f"ValLoss: {val_loss:.4f} ValAcc(top1): {val_acc:.4f} ValRecall@1: {val_recall1:.4f}")
    #
    #     # save best
    #     if val_recall1 > best_recall1:
    #         best_recall1 = val_recall1
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state': model.state_dict(),
    #             'arcface_state': loss_fn.state_dict(),
    #             'optimizer': optimizer.state_dict()
    #         }, "best_arcface_checkpoint.pth")
    #         print("Saved best checkpoint.")


if __name__ == "__main__":
    main()

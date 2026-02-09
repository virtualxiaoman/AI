# https://chatgpt.com/c/69870e26-f090-832b-9d55-5681c0732334
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --------------------------
# Configuration - 修改这里
# --------------------------
VOC_ROOT = "../../../Datasets/PASCAL_VOC_2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
NUM_CLASSES = 21  # 20 classes + background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 2  # detection 通常 batch 小
LEARNING_RATE = 0.005
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# VOC 类别（顺序固定）
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat",  "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLS_TO_IDX = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 背景=0


# --------------------------
# Dataset
# --------------------------
class VOCDataset(Dataset):
    """
    解析 VOC Annotations -> 返回 (image_tensor, target_dict)
    target_dict 包含: boxes (Nx4), labels (N), image_id (1,), area (N), iscrowd (N)
    """

    def __init__(self, root: str, image_set: str = "train", transforms=None, keep_difficult: bool = False):
        """
        root: path to VOC2007 folder (contains Annotations, JPEGImages, ImageSets)
        image_set: "train", "val", "trainval", "test" 对应 ImageSets/Main/*.txt
        transforms: torchvision transforms applied to PIL image (should convert to tensor)
        """
        self.root = Path(root)
        self.transforms = transforms
        self.keep_difficult = keep_difficult

        ids_file = self.root / "ImageSets" / "Main" / f"{image_set}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"Image set file not found: {ids_file}")
        with open(ids_file, "r") as f:
            self.ids = [line.strip().split()[0] for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        ann_path = self.root / "Annotations" / f"{image_id}.xml"

        # PIL image -> apply transforms (ToTensor)
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        # parse xml
        boxes = []
        labels = []
        iscrowd = []
        areas = []
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            if (not self.keep_difficult) and difficult == 1:
                continue
            name = obj.find("name").text
            if name not in CLS_TO_IDX:
                # 跳过未知类（通常不发生）
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLS_TO_IDX[name])
            iscrowd.append(0)
            areas.append((xmax - xmin) * (ymax - ymin))

        if len(boxes) == 0:
            # 若某张图无标注（极少），返回空张量
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)] if image_id.isdigit() else [idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd
        }

        return img, target


# collate_fn for variable-size targets
def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------
# Model builder
# --------------------------
def get_model(num_classes: int):
    # 加载预训练模型并替换预测头
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --------------------------
# Trainer class (封装训练/验证/保存)
# --------------------------
class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader = None,
                 device=DEVICE, lr=LEARNING_RATE, epochs=NUM_EPOCHS, save_dir: str = SAVE_DIR):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.save_dir = save_dir

    def train_one_epoch(self, epoch_idx: int, print_freq: int = 10):
        self.model.train()
        metric_loss = 0.0
        t0 = time.time()
        for i, (images, targets) in enumerate(self.train_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            metric_loss += losses.item()

            if (i + 1) % print_freq == 0:
                avg_loss = metric_loss / (i + 1)
                print(f"Epoch [{epoch_idx}] Iter [{i + 1}/{len(self.train_loader)}]  avg_loss: {avg_loss:.4f}")

        epoch_time = time.time() - t0
        avg_epoch_loss = metric_loss / len(self.train_loader)
        print(f"Epoch {epoch_idx} finished. avg_loss={avg_epoch_loss:.4f} time={epoch_time:.1f}s")
        self.lr_scheduler.step()
        return avg_epoch_loss

    def evaluate(self, max_images: int = 50):
        """
        简单的验证：在 val_loader 上计算 average loss（如果提供 target），并返回平均损失。
        说明：这不是 mAP，若需要 mAP 请使用 COCO evaluator（需额外集成）。
        """
        if self.val_loader is None:
            print("No val_loader provided.")
            return None
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.val_loader):
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                # model 返回 loss dict（当提供 targets 时）
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                n += 1
                if n >= max_images:
                    break
        avg_loss = total_loss / n if n > 0 else 0.0
        print(f"Validation (approx) avg_loss={avg_loss:.4f} over {n} batches")
        return avg_loss

    def save_checkpoint(self, epoch_idx: int):
        path = os.path.join(self.save_dir, f"fasterrcnn_epoch{epoch_idx}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        best_val = math.inf
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)
            val_loss = self.evaluate(max_images=50) if self.val_loader else None
            self.save_checkpoint(epoch)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(self.save_dir, "fasterrcnn_best.pth")
                torch.save(self.model.state_dict(), best_path)
                print(f"Saved best model: {best_path}")


# --------------------------
# Utility: visualize predictions (一张图)
# --------------------------
def visualize_sample(model, dataset: Dataset, device=DEVICE, idx: int = 0, score_thresh: float = 0.5):
    model.eval()
    img, target = dataset[idx]
    with torch.no_grad():
        pred = model([img.to(device)])[0]
    img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    # draw GT boxes (green)
    for box in target["boxes"].cpu().numpy():
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    # draw predicted boxes (red)
    for box, label, score in zip(pred["boxes"].cpu().numpy(), pred["labels"].cpu().numpy(),
                                 pred["scores"].cpu().numpy()):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{VOC_CLASSES[label - 1]}:{score:.2f}", color='r', fontsize=12, backgroundcolor='white')
    plt.axis("off")
    plt.show()


# --------------------------
# Main: prepare dataloaders and start training
# --------------------------
def main():
    # transforms
    train_transforms = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.ToTensor()
    ])
    test_transforms = T.Compose([
        T.ToTensor()
    ])

    # datasets: image_set 可以是 "train", "val", "trainval", "test"
    train_dataset = VOCDataset(VOC_ROOT, image_set="train", transforms=train_transforms, keep_difficult=False)
    val_dataset = VOCDataset(VOC_ROOT, image_set="val", transforms=test_transforms, keep_difficult=False)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # model
    model = get_model(NUM_CLASSES)

    # trainer
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, lr=LEARNING_RATE, epochs=NUM_EPOCHS)
    trainer.train()

    # 可视化一张样例（训练后）
    visualize_sample(model, val_dataset, device=DEVICE, idx=5, score_thresh=0.5)


if __name__ == "__main__":
    main()

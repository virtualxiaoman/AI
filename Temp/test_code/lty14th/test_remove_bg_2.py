from pathlib import Path

import cv2
import numpy as np

def cv_imread(path: str | Path, flags=cv2.IMREAD_COLOR):
    """
    支持中文路径读取图片
    """
    path = str(path)

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)

    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")

    return img

def cv_imwrite(path: str | Path, image):
    """
    支持中文路径保存图片
    """
    path = Path(path)

    suffix = path.suffix.lower()
    if not suffix:
        raise ValueError("输出路径必须包含扩展名")

    success, encoded = cv2.imencode(suffix, image)

    if not success:
        raise RuntimeError(f"图片编码失败：{path}")

    encoded.tofile(str(path))

def remove_background(
    image_path: str | Path,
    bg_color=(224, 247, 255),
    tolerance=12,
    min_area=300,
):
    """
    去除纯色背景（适用于JPG）。

    Parameters
    ----------
    image_path
        图片路径

    bg_color
        背景RGB颜色

    tolerance
        Lab颜色距离阈值（建议8~20）

    min_area
        内部背景最小面积。
        小于该面积的不删除（例如人物身上的挂件）。

    Outputs
    -------
    xxx_mask.png
        白=保留
        黑=透明

    xxx_remove_bg.png
        去背景后的PNG
    """

    image_path = Path(image_path)
    print(f"绝对路径为：{image_path.resolve()}")
    # ----------------------------
    # 读取图片
    # ----------------------------
    bgr = cv_imread(image_path)

    h, w = bgr.shape[:2]

    # ----------------------------
    # Lab颜色空间
    # ----------------------------
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    bg_rgb = np.uint8([[list(bg_color)]])
    bg_lab = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB)[0, 0]

    dist = np.linalg.norm(
        lab.astype(np.float32) - bg_lab.astype(np.float32),
        axis=2,
    )

    candidate = (dist < tolerance).astype(np.uint8)

    # ----------------------------
    # 连通域
    # ----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    background = np.zeros_like(candidate)

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        ys, xs = np.where(labels == label)

        touch_border = (
            np.any(xs == 0)
            or np.any(xs == w - 1)
            or np.any(ys == 0)
            or np.any(ys == h - 1)
        )

        # 删除：
        # ① 与边界相连
        # ② 内部但面积较大（头发空洞等）
        if touch_border or area >= min_area:
            background[labels == label] = 255

    # ----------------------------
    # Alpha
    # ----------------------------
    alpha = 255 - background

    # 去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)

    # 略微羽化
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    # ----------------------------
    # 保存mask
    # ----------------------------
    mask_path = image_path.with_name(
        image_path.stem + "_mask.png"
    )

    cv_imwrite(mask_path, alpha)

    # ----------------------------
    # 保存RGBA
    # ----------------------------
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    out_path = image_path.with_name(
        image_path.stem + "_remove_bg.png"
    )

    cv_imwrite(out_path, rgba)

    return mask_path, out_path


if __name__ == "__main__":

    mask, out = remove_background(
        "7依-love-两个天依.jpg",
        tolerance=1,
        min_area=150,
    )

    print(mask)
    print(out)
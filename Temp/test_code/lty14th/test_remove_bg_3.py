from pathlib import Path

import cv2
import numpy as np


def black_to_transparent(
        image_path: str,
        threshold: int = 40,
) -> str:
    """
    将接近黑色的区域变成透明背景。

    Parameters
    ----------
    image_path : str
        图片路径
    threshold : int
        黑色阈值（0~255）
        越大，认为是黑色的范围越大。

    Returns
    -------
    str
        输出图片路径
    """

    # 支持中文路径读取
    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED
    )

    if img is None:
        raise FileNotFoundError(image_path)

    # 转RGBA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    elif img.shape[2] != 4:
        raise ValueError("不支持的图片格式")

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # 与黑色距离
    distance = np.sqrt(
        b.astype(np.float32) ** 2 +
        g.astype(np.float32) ** 2 +
        r.astype(np.float32) ** 2
    )

    # 接近黑色
    mask = distance <= threshold

    # Alpha置0
    img[mask, 3] = 0

    path = Path(image_path)
    output_path = str(path.with_name(path.stem + "_transparent.png"))

    # 支持中文路径保存
    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("图片保存失败")

    encoded.tofile(output_path)

    return output_path


if __name__ == "__main__":
    out = black_to_transparent(
        "14th/msedge_20260712_224546_547.png",
        threshold=40
    )
    print(out)

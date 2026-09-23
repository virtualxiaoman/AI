from rembg import remove
from PIL import Image
import numpy as np
import os

input_dir = "G:/AAA/小满的人设/AI生成/小满の表情包"
output_dir = "G:/AAA/小满的人设/AI生成/小满の表情包-透明底-批量转化"

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp")

# 白背景阈值（略低于FEFEFE，避免误伤）
BG_THRESHOLD = 250


def refine_alpha(original_img, alpha_img):
    """
    关键：只移除“背景白”，保留人物白色区域
    """
    rgb = np.array(original_img.convert("RGB"))
    out = np.array(alpha_img.convert("RGBA"))

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # rembg 输出的 alpha
    alpha = out[..., 3]

    # 1. 找“极白区域”
    near_white = (r > BG_THRESHOLD) & (g > BG_THRESHOLD) & (b > BG_THRESHOLD)

    # 2. 关键限制：只有“本来就接近透明的区域”才允许变透明
    #    也就是：避免误删袖子等主体区域
    safe_to_remove = near_white & (alpha < 200)

    # 3. 只删除背景白
    alpha[safe_to_remove] = 0

    out[..., 3] = alpha

    return Image.fromarray(out)


for root, dirs, files in os.walk(input_dir):

    relative_path = os.path.relpath(root, input_dir)
    current_output_dir = os.path.join(output_dir, relative_path)
    os.makedirs(current_output_dir, exist_ok=True)

    for file in files:

        if not file.lower().endswith(SUPPORTED_EXT):
            continue

        input_path = os.path.join(root, file)
        output_name = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(current_output_dir, output_name)

        try:
            img = Image.open(input_path).convert("RGBA")

            result = remove(
                img,
                alpha_matting=True,
                alpha_matting_foreground_threshold=245,
                alpha_matting_background_threshold=5,
                alpha_matting_erode_size=8
            )

            result = refine_alpha(img, result)

            result.save(output_path)

            print(f"[OK] {input_path}")

        except Exception as e:
            print(f"[ERROR] {input_path}")
            print(e)
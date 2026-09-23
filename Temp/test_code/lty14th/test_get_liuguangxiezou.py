import cv2
import numpy as np
import glob
import os


def extract_text_perfect(image_folder="images", output_name="extracted_text_transparent.png"):
    """
    从多张背景不同的截图中完美提取中心金色文字，保留细微笔锋并清除四周残留灯光
    【新加入功能】：基于灰度亮度分析，精准将任何偏黑色的过渡边缘转化为透明背景
    """
    # 1. 自动获取文件夹下所有的截图（支持多种常见图片格式）
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_paths:
        print(f"【错误】在 '{image_folder}' 文件夹中未找到任何截图图片！")
        print("请在代码同级目录下建立该文件夹，并把多张截图放进去。")
        return

    print(f"成功找到 {len(image_paths)} 张截图，开始加载并进行多图融合...")
    print(image_paths)

    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        print("【错误】未能成功读取任何图片，请检查图片文件是否损坏。")
        return

    # 2. 多图取中值融合（利用动态背景的变幻，自动过滤掉大部分闪烁的荧光棒和舞台散光）
    image_stack = np.array(images)
    median_image = np.median(image_stack, axis=0).astype(np.uint8)

    # 3. 转换到 HSV 色彩空间，方便精确锁定制定的金黄色
    hsv = cv2.cvtColor(median_image, cv2.COLOR_BGR2HSV)

    # ==============================================================================
    # 原始步骤 1：大幅放宽颜色下限，死守并保留最细微、最暗淡的笔锋细节
    # ==============================================================================
    lower_gold = np.array([10, 15, 25])
    upper_gold = np.array([45, 255, 255])

    # 生成初步的颜色遮罩
    color_mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # ==============================================================================
    # 原始步骤 2：建立中央保护几何遮罩，强行切除图像最左、最右及四周边缘的静态黄色灯光
    # ==============================================================================
    h, w = color_mask.shape
    geo_mask = np.zeros_like(color_mask)

    x_start = int(w * 0.06)
    x_end = int(w * 0.94)
    y_start = int(h * 0.02)
    y_end = int(h * 0.98)

    # 在全黑的几何遮罩核心区域绘制一个纯白（255）的实心矩形
    cv2.rectangle(geo_mask, (x_start, y_start), (x_end, y_end), 255, -1)

    # 将颜色遮罩与几何遮罩进行“位与（AND）”操作
    final_mask = cv2.bitwise_and(color_mask, geo_mask)

    # ==============================================================================
    # 🔥 新增核心改进：精准剔除任何偏黑色、偏暗色像素（彻底消除边缘黑线）
    # ==============================================================================
    # 1. 提取图像的灰度图（代表每个像素的绝对亮度）
    gray = cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY)

    # 2. 设置亮度控制区间（可调参数）：
    # low_thresh: 亮度低于此值的像素直接强制变 100% 透明
    # high_thresh: 亮度高于此值的像素保持原本的不透明度
    low_thresh = 91  # 如果发现白底上还有一丝黑边，可以把这个值微调调大（如 45 或 50）
    high_thresh = 100  # 保持在这个区间能让边缘产生自然的半透明羽化效果

    # 3. 计算亮度权重矩阵 (范围映射到 0.0 ~ 1.0)
    brightness_ramp = np.clip((gray.astype(np.float32) - low_thresh) / (high_thresh - low_thresh), 0, 1)

    # 4. 用亮度权重去调制我们原本的 final_mask 遮罩
    # 任何偏黑色的过渡像素，其 Alpha 通道值会被强制降低甚至直接归零
    final_mask = (final_mask.astype(np.float32) * brightness_ramp).astype(np.uint8)
    # ==============================================================================

    # 4. 后期平滑优化（让淡出后的边缘更加圆润）
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    # 5. 分离融合后图像的 B, G, R 通道
    b, g, r = cv2.split(median_image)

    # 6. 将 B, G, R 和我们净化过黑边后的 final_mask（作为 Alpha 透明通道）打包合并
    bgra_output = cv2.merge((b, g, r, final_mask))

    # 7. 保存最终成果（必须是 .png 格式才能支持和维持透明通道）
    cv2.imwrite(output_name, bgra_output)
    print("\n========================================================")
    print(f"【成功】精细化提取完成！已通过亮度分析剔除所有偏黑边线。")
    print(f"【结果】带透明通道的纯净文字图片已保存为: {output_name}")
    print("========================================================")


from pathlib import Path
from PIL import Image


def remove_black_background(image_path: str):
    """
    将 PNG 图片中的纯黑色背景 (0,0,0) 替换为透明背景。

    参数：
        image_path (str): 待处理图片路径

    返回：
        str: 保存后的图片路径
    """
    image_path = Path(image_path)

    # 转为 RGBA
    img = Image.open(image_path).convert("RGBA")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            # 纯黑色或者接近的变透明
            threshold = 50
            if r <= threshold and g <= threshold and b <= threshold:
                pixels[x, y] = (0, 0, 0, 0)

    # 保存路径：文件名_remove_bg.png
    output_path = image_path.with_name(
        f"{image_path.stem}_remove_bg{image_path.suffix}"
    )

    img.save(output_path)

    return str(output_path)


def text_to_black(image_path: str, threshold: int = 240):
    """
    将PNG中不是接近白色的像素全部变成黑色。

    Parameters
    ----------
    image_path : str
        图片路径
    threshold : int
        白色判定阈值(0~255)。
        当 R、G、B 三个通道都 >= threshold 时认为是白色。
        默认240，适合大多数情况。
    """

    img = Image.open(image_path).convert("RGBA")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            # 保留透明像素
            if a == 0:
                continue

            # 不是接近白色 -> 变黑
            if not (r >= threshold and g >= threshold and b >= threshold):
                pixels[x, y] = (0, 0, 0, a)

    save_path = os.path.splitext(image_path)[0] + "_black.png"
    img.save(save_path)

    return save_path


def remove_small_noise(image_path: str, min_area: int = 20):
    """
    去除PNG中的小黑色噪点（按连通域面积）。

    Parameters
    ----------
    image_path : str
        图片路径（白底黑字）
    min_area : int
        连通域最小面积，小于该面积的黑色区域会被删除。
        推荐：
            5~10：去除极小噪点
            15~30：一般文字
            50：较激进
    """

    # 灰度读取
    img_gbr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2GRAY)

    # 黑字白底 -> 二值图（黑色=255方便做连通域）
    binary = np.where(img < 128, 255, 0).astype(np.uint8)

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    result = binary.copy()

    # 0是背景
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            result[labels == i] = 0

    # 转回白底黑字
    output = np.where(result == 255, 0, 255).astype(np.uint8)

    save_path = os.path.splitext(image_path)[0] + "_denoise.png"
    cv2.imwrite(save_path, output)

    return save_path


if __name__ == "__main__":
    # extract_text_perfect(image_folder="liuguangxiezou", output_name="liuguangxiezou/extracted_text_transparent.png")
    # output = remove_black_background("../lty14th/liuguangxiezou/ChatGPT Image 2026年7月4日 22_24_18.png")
    # print("保存到：", output)
    output = text_to_black("liuguangxiezou/aaa.png")
    print(output)
    output = remove_small_noise(output, min_area=200)
    print(output)

from pathlib import Path
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from Utils.config.path import TestPictures
from Utils.models.cv.config import CVNetConfig
from Utils.models.cv.load_net import CVNetFactory
from XAI.FeatureAttribution.common.image_loader import ImageLoader
from XAI.FeatureAttribution.common.predictor import Predictor
from XAI.FeatureAttribution.common.visualization import Visualizer
from XAI.FeatureAttribution.lime.explainer import LimeExplainer


def main():
    # 创建模型
    net = CVNetFactory.create(CVNetConfig(name="resnet50", pretrained=True))
    # 创建工具
    predictor = Predictor(net)
    loader = ImageLoader()
    visualizer = Visualizer()
    explainer = LimeExplainer(predictor)
    # 图片路径
    image_path = TestPictures / "cat.jpg"
    # 加载图片
    image = loader.load_numpy(image_path)
    # 原始预测
    class_id, prob = predictor.predict_single(image)
    print("=" * 60)
    print(f"Prediction : {class_id}")
    print(f"Probability: {prob:.4f}")
    print("=" * 60)
    # LIME解释
    result = explainer.explain(image)
    # 输出
    print(f"LIME Label : {result.label}")
    # 可视化
    visualizer.show_three_images(image, result.mask, result.overlay, "Original", "Mask", "LIME")
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(image, result.explanation.segments))
    plt.axis("off")
    plt.title("Superpixels")
    plt.show()


if __name__ == "__main__":
    main()

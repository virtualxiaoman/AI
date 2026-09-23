import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """通用可视化"""

    @staticmethod
    def show_image(image: np.ndarray, title: str = ""):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_two_images(image1: np.ndarray, image2: np.ndarray, title1="Image", title2="Result",):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image1)
        axes[0].set_title(title1)
        axes[0].axis("off")

        axes[1].imshow(image2)
        axes[1].set_title(title2)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_three_images(image1, image2, image3, title1="Original", title2="Mask", title3="Overlay",):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image1)
        axes[0].set_title(title1)
        axes[0].axis("off")

        axes[1].imshow(image2)
        axes[1].set_title(title2)
        axes[1].axis("off")

        axes[2].imshow(image3)
        axes[2].set_title(title3)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
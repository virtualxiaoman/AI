from dataclasses import dataclass

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from XAI.FeatureAttribution.common.predictor import Predictor


@dataclass(slots=True)
class LimeResult:
    explanation: object
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    label: int


class LimeExplainer:
    def __init__(self, predictor: Predictor, num_samples: int = 2000, top_labels: int = 5, hide_color: int = 0, ):
        self.predictor = predictor
        self.num_samples = num_samples
        self.top_labels = top_labels
        self.hide_color = hide_color
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image: np.ndarray, ) -> LimeResult:
        explanation = self.explainer.explain_instance(image, classifier_fn=self.predictor.predict_numpy,
                                                      top_labels=self.top_labels, hide_color=self.hide_color,
                                                      num_samples=self.num_samples,
                                                      segmentation_fn=lambda img: slic(
                                                          img,
                                                          n_segments=150,
                                                          compactness=10,
                                                          sigma=1,
                                                      ), )
        label = explanation.top_labels[0]
        lime_image, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=100, hide_rest=False)
        overlay = mark_boundaries(lime_image, mask, )
        return LimeResult(explanation=explanation, image=lime_image, mask=mask, overlay=overlay, label=label, )

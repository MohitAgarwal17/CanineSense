"""
Stage 3: explain the emotion prediction with Grad-CAM++.

Targets the last block of the final ConvNeXt stage (learn.model[0].model.stages[-1].blocks[-1]),
then buckets the heatmap into a 3x3 spatial grid to produce a plain-English explanation
alongside the raw heatmap.
"""
from __future__ import annotations

import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .preprocessing import load_and_preprocess

_REGION_LABELS = [
    ["upper-left", "upper-center (forehead/ears)", "upper-right"],
    ["mid-left", "center (nose/mouth)", "mid-right"],
    ["lower-left", "lower-center (chin/mouth)", "lower-right"],
]

_EMOTION_CUES = {
    "angry": "bared teeth, tensed jaw, or a furrowed brow",
    "happy": "an open mouth, relaxed tongue, or bright eyes",
    "sad": "drooping eyes or lowered ears",
    "relaxed": "a neutral, evenly-lit expression with no sharp tension points",
}


def _get_target_layers(learn):
    return [learn.model[0].model.stages[-1].blocks[-1]]


def generate_gradcam(image_path, learn, class_idx: int):
    """Runs Grad-CAM++ for a given image and target class.

    Returns (grayscale_cam, visualization, img_np):
        grayscale_cam  : HxW float heatmap in [0, 1]
        visualization  : HxWx3 uint8 image with heatmap overlaid on the input
        img_np         : the padded/resized input image before normalization
    """
    model = learn.model.eval()
    img_np, input_tensor = load_and_preprocess(image_path)

    target_layers = _get_target_layers(learn)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return grayscale_cam, visualization, img_np


def explain_gradcam(grayscale_cam: np.ndarray, emotion: str, region_labels=None) -> tuple[str, np.ndarray]:
    """Turns a Grad-CAM heatmap into a plain-English explanation.

    Returns (explanation_text, grid_pct) where grid_pct is a 3x3 array of
    the percentage of total attention in each region.
    """
    h, w = grayscale_cam.shape
    grid = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cell = grayscale_cam[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            grid[i, j] = cell.sum()

    grid_pct = grid / grid.sum() * 100
    labels = region_labels or _REGION_LABELS

    flat = [(grid_pct[i, j], labels[i][j]) for i in range(3) for j in range(3)]
    flat.sort(reverse=True)

    top_region, top_pct = flat[0][1], flat[0][0]
    second_region, second_pct = flat[1][1], flat[1][0]

    explanation = (
        f"The model predicted '{emotion}' primarily based on the {top_region} "
        f"region ({top_pct:.1f}% of total attention), with secondary focus on "
        f"the {second_region} ({second_pct:.1f}%). This is consistent with "
        f"{_EMOTION_CUES.get(emotion, 'the visible facial expression')} in that area."
    )
    return explanation, grid_pct

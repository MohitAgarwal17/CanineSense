"""
Stage 3: explain the emotion prediction with Grad-CAM++.

Targets the last block of the final ConvNeXt stage (learn.model[0].model.stages[-1].blocks[-1]),
then buckets the heatmap into a 3x3 spatial grid to produce a plain-English explanation
alongside the raw heatmap.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .preprocessing import content_box, load_and_preprocess

# Purely positional labels. The 3x3 grid is a rough spatial summary of the photo;
# a cell is only "nose" or "ears" if the face happens to fill the frame, so we
# do not claim anatomy here.
_REGION_LABELS = [
    ["upper-left", "upper-center", "upper-right"],
    ["mid-left", "center", "mid-right"],
    ["lower-left", "lower-center", "lower-right"],
]

# Typical cues per emotion, used as general context only (see explain_gradcam).
_EMOTION_CUES = {
    "angry": "bared teeth, a tensed jaw, or a furrowed brow",
    "happy": "an open mouth, a relaxed tongue, or bright eyes",
    "sad": "drooping eyes or lowered ears",
    "relaxed": "a neutral expression with no sharp tension points",
}


def _get_target_layers(learn):
    return [learn.model[0].model.stages[-1].blocks[-1]]


def get_content_box(image_path) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the real photo inside the 400x400 model input.

    Reads only the image size (not the pixels), then asks preprocessing where
    the padding starts, so it always matches what the model actually saw.
    """
    with Image.open(image_path) as im:
        return content_box(im.size)


def padding_attention_fraction(grayscale_cam: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Fraction (0-1) of total heat that landed on mirrored padding. Useful for the bias analysis."""
    total = grayscale_cam.sum()
    if total == 0:
        return 0.0
    left, top, right, bottom = box
    inside = grayscale_cam[top:bottom, left:right].sum()
    return float(1.0 - inside / total)


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
    targets = [ClassifierOutputTarget(class_idx)]

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    finally:
        # Always remove the hooks from the layer, even if an error happens,
        # so repeated calls in a loop do not stack hooks.
        cam.activations_and_grads.release()

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return grayscale_cam, visualization, img_np


def explain_gradcam(
    grayscale_cam: np.ndarray,
    emotion: str,
    region_labels=None,
    content_box: tuple[int, int, int, int] | None = None,
) -> tuple[str, np.ndarray]:
    """Turns a Grad-CAM heatmap into a plain-English explanation.

    content_box: optional (left, top, right, bottom) from get_content_box().
                 If given, the heatmap is cropped to the real photo first, so the
                 3x3 grid describes the photo and ignores mirrored padding.

    Returns (explanation_text, grid_pct) where grid_pct is a 3x3 array of
    the percentage of attention in each region.
    """
    cam = grayscale_cam
    if content_box is not None:
        left, top, right, bottom = content_box
        cam = grayscale_cam[top:bottom, left:right]  # numpy indexes [rows=y, cols=x]

    h, w = cam.shape
    grid = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cell = cam[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            grid[i, j] = cell.sum()

    total = grid.sum()
    if total == 0:
        # An all-zero heatmap has no attention to describe; avoid dividing by zero.
        return "The model's attention map was empty, so no region can be reported.", grid

    grid_pct = grid / total * 100
    labels = region_labels or _REGION_LABELS

    flat = [(grid_pct[i, j], labels[i][j]) for i in range(3) for j in range(3)]
    flat.sort(reverse=True)

    top_region, top_pct = flat[0][1], flat[0][0]
    second_region, second_pct = flat[1][1], flat[1][0]
    cues = _EMOTION_CUES.get(emotion, "certain facial cues")

    explanation = (
        f"The model predicted '{emotion}' with attention concentrated in the "
        f"{top_region} region ({top_pct:.1f}% of attention), followed by the "
        f"{second_region} region ({second_pct:.1f}%). For reference, '{emotion}' is "
        f"typically associated with {cues}; this shows where the model looked, "
        f"not whether those cues are present."
    )
    return explanation, grid_pct
"""
Shared image preprocessing for training and inference.

This is the single source of truth for how an image becomes a model input.
Both training notebooks and every inference path (batch eval, single-image
predict, Grad-CAM) must use pad_to_square_resize() here -- do not
reimplement it elsewhere. An earlier version of this project had a
production-inference path that padded with black + resized to 448, while
training used reflect-padding + 400 -- that mismatch silently hurt accuracy.
Don't reintroduce it.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMG_SIZE = 400
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_to_square_resize(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Reflect-pad an image to a square, then resize.

    Replicates fastai's Resize(size, method='pad', pad_mode='reflection'),
    which both models were trained with.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    arr = np.array(img_resized)
    arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return Image.fromarray(arr)


def load_and_preprocess(image_path, size: int = IMG_SIZE):
    """Load an image from disk and return (raw_np_0to1, normalized_tensor).

    raw_np_0to1   : HxWx3 float array in [0, 1], the padded/resized image
                    BEFORE normalization -- used as the background for
                    Grad-CAM overlays.
    normalized_tensor : torch tensor [1, 3, size, size], model-ready input.
    """
    pil_img = Image.open(image_path).convert("RGB")
    processed = pad_to_square_resize(pil_img, size)

    img_np = np.array(processed).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)

    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    input_tensor = ((img_tensor - mean_t) / std_t).float()

    return img_np, input_tensor

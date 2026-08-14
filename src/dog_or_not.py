"""
Stage 1: is there a dog in this image at all?

Wraps models/dog_or_not_convnext_tiny.pkl (ConvNeXt Tiny, fastai, 99.86% test acc).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model_io import safe_load_learner
from .preprocessing import load_and_preprocess

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dog_or_not_convnext_tiny.pkl"

_learn = None


def _get_learner():
    global _learn
    if _learn is None:
        _learn = safe_load_learner(MODEL_PATH)
    return _learn


def predict_dog_or_not(image_path) -> dict:
    """Returns {'is_dog': bool, 'label': str, 'confidence': float 0-100}."""
    learn = _get_learner()
    model = learn.model.eval()
    _, input_tensor = load_and_preprocess(image_path)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    cl_idx = int(probs.argmax())
    label = learn.dls.vocab[cl_idx]
    confidence = float(probs[cl_idx]) * 100

    return {"is_dog": label == "dog", "label": label, "confidence": confidence}

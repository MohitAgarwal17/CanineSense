"""
Stage 2: which emotion is the dog showing?

Wraps models/emotion_convnext_tiny_94f1_tta.pkl
(ConvNeXt Tiny, fastai, two-phase fine-tune, 94% weighted F1 with TTA).
Classes: angry, happy, relaxed, sad.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .model_io import safe_load_learner
from .preprocessing import load_and_preprocess

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "emotion_convnext_tiny_94f1_tta.pkl"

_learn = None


def _get_learner():
    global _learn
    if _learn is None:
        _learn = safe_load_learner(MODEL_PATH)
    return _learn


def predict_emotion(image_path) -> dict:
    """Returns {'emotion': str, 'confidence': float, 'probs': {label: pct}, 'class_idx': int}."""
    learn = _get_learner()
    model = learn.model.eval()
    _, input_tensor = load_and_preprocess(image_path)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    vocab = learn.dls.vocab
    cl_idx = int(probs.argmax())
    emotion = vocab[cl_idx]
    confidence = float(probs[cl_idx]) * 100
    probs_dict = {vocab[i]: float(probs[i]) * 100 for i in range(len(vocab))}

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probs": probs_dict,
        "class_idx": cl_idx,
    }


def get_learner():
    """Exposed for gradcam_utils, which needs the raw model + target layers."""
    return _get_learner()

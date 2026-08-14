"""
dog-emotion-recognition source package.

2 trained models, wrapped by 6 small modules:

    preprocessing.py   shared image transform (must match training exactly)
    model_io.py        cross-platform-safe fastai learner loading
    dog_or_not.py       stage 1 -- is this even a dog? (dog_or_not_convnext_tiny.pkl)
    emotion.py          stage 2 -- which emotion? (emotion_convnext_tiny_94f1_tta.pkl)
    gradcam_utils.py     Grad-CAM++ heatmap + plain-English region explanation
    pipeline.py          ties stages 1+2+3 together into one predict() call

Usage:
    from src.pipeline import predict
    result = predict("some_dog.jpg")
"""

from .pipeline import predict

__all__ = ["predict"]

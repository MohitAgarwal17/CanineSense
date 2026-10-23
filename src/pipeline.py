"""
Combined inference pipeline: dog-or-not gate -> emotion classifier -> Grad-CAM++ explanation.

This is the single entry point the rest of the project should call.
"""
from __future__ import annotations

from .dog_or_not import predict_dog_or_not
from .emotion import get_learner, predict_emotion
from .gradcam_utils import explain_gradcam, generate_gradcam, get_content_box


def predict(image_path, explain: bool = True) -> dict:
    """Run the full pipeline on a single image.

    Args:
        image_path: path to an image file.
        explain: if True (default), also run Grad-CAM++ and attach a
            heatmap + plain-English explanation. Set False to skip Grad-CAM
            for faster batch throughput when you only need the label.

    Returns a dict. If the gate decides the image isn't a dog, only
    'is_dog', 'label', and 'confidence' are present. Otherwise the emotion
    fields (and, if explain=True, the Grad-CAM fields) are added.
    """
    gate_result = predict_dog_or_not(image_path)

    if not gate_result["is_dog"]:
        return gate_result

    emotion_result = predict_emotion(image_path)
    result = {**gate_result, **emotion_result}

    if explain:
        learn = get_learner()
        grayscale_cam, visualization, _ = generate_gradcam(
            image_path, learn, emotion_result["class_idx"]
        )
        box = get_content_box(image_path)  # where the real photo sits inside the padded input
        explanation, grid_pct = explain_gradcam(
            grayscale_cam, emotion_result["emotion"], content_box=box
        )
        result["gradcam_heatmap"] = grayscale_cam
        result["gradcam_visualization"] = visualization
        result["explanation"] = explanation
        result["attention_grid_pct"] = grid_pct

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.pipeline <image_path>")
        sys.exit(1)

    out = predict(sys.argv[1])
    if not out["is_dog"]:
        print(f"Not a dog (predicted '{out['label']}', {out['confidence']:.1f}% confidence)")
    else:
        print(f"Emotion: {out['emotion']} ({out['confidence']:.1f}% confidence)")
        if "explanation" in out:
            print(out["explanation"])
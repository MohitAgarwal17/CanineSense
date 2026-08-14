import sys
import warnings
from pathlib import Path
from PIL import Image

# Suppress verbose warnings from fastai loading
warnings.filterwarnings("ignore", category=UserWarning)

from src.pipeline import predict


def run_prediction(image_path: str):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found at {path}")
        return

    # Run inference silently
    result = predict(str(path))

    # Stage 1: Non-Dog Output
    if not result["is_dog"]:
        print(f"\nRESULT: NOT A DOG ({result['confidence']:.1f}% confidence)")
        return

    # Stage 2: Dog Output (Clean & Short)
    print("\n========================================")
    print(f"DOG DETECTED | Emotion: {result['emotion'].upper()} ({result['confidence']:.1f}%)")
    print("========================================")

    if "explanation" in result:
        print(f"\nFocus: {result['explanation']}\n")

    # Display Grad-CAM Image in a pop-up window without saving to disk
    if "gradcam_visualization" in result:
        cam_image = Image.fromarray(result["gradcam_visualization"])
        cam_image.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_setup.py \"path/to/image.jpg\"")
    else:
        run_prediction(sys.argv[1])
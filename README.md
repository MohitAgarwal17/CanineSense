# Dog Emotion Recognition

A two-stage computer vision pipeline that first checks whether an image contains a dog, then
classifies the dog's emotion (angry / happy / relaxed / sad), and explains the prediction with
Grad-CAM++ visual attention maps.

**Pipeline:** `image → dog-or-not gate → emotion classifier → Grad-CAM++ explanation`

## Results

| Model | Architecture | Metric | Score |
|---|---|---|---|
| Dog-or-Not gate | ConvNeXt Tiny (fastai) | Test accuracy | 99.86% |
| Emotion Classifier | ConvNeXt Tiny (fastai, two-phase fine-tune + TTA) | Weighted F1 (4-class) | 94% |

Both models are trained with `fastai`'s `vision_learner` on top of `timm`'s `convnext_tiny`,
using `fp16` training, label smoothing (emotion model), and a frozen→unfrozen two-phase
fine-tuning schedule for the emotion model.

## Datasets

- **Dog vs Not-Dog** — [`danielshanbalico/dog-vs-not-dog`](https://www.kaggle.com/datasets/danielshanbalico/dog-vs-not-dog) on Kaggle. Used to train the gating model that filters non-dog images before emotion inference.
- **Dog Emotion Dataset (cleaned)** — [`mohitagarwal17/dog-emotion-datasetcleaned-version`](https://www.kaggle.com/datasets/mohitagarwal17/dog-emotion-datasetcleaned-version) on Kaggle. ~4,000 images across 4 classes (angry, happy, relaxed, sad), published and maintained by me.

## Explainability (XAI)

The emotion model's predictions are explained using **Grad-CAM++**, targeting the last block of
the final ConvNeXt stage. The heatmap is additionally summarized into a human-readable
explanation by bucketing attention into a 3×3 spatial grid (e.g. "predicted 'happy' primarily
based on the center (nose/mouth) region, consistent with an open mouth or relaxed tongue").

## Repository structure

```
notebooks/    training + experimentation notebooks (source of truth for how the models were trained)
src/          reusable, cleaned-up Python modules used by the inference pipeline
examples/     a handful of small sample images for demoing the pipeline
models/       NOT committed — populated locally via src/download_models.py
```

## Setup

```bash
git clone <repo-url>
cd dog-emotion-recognition
pip install -r requirements.txt
python src/download_models.py   # fetches .pkl weights from GitHub Releases
```

## Usage

```python
from src.pipeline import predict

result = predict("examples/sample_images/some_dog.jpg")
print(result)
# {'is_dog': True, 'emotion': 'happy', 'confidence': 96.3, 'explanation': '...'}
```

## Model weights

Model weights are **not stored in git history** — they're distributed as
[GitHub Release](../../releases) assets and pulled on demand by `src/download_models.py`, to
keep the repository small and avoid large-file history bloat.

| File | Size | Source |
|---|---|---|
| `dog_or_not_convnext_tiny.pkl` | ~fastai learner export | GitHub Release / Kaggle Models |
| `emotion_convnext_tiny_94f1_tta.pkl` | ~fastai learner export | GitHub Release / Kaggle Models |

## Notes on preprocessing consistency

Both training and inference use identical preprocessing: reflect-pad to a square, resize to
400×400, then normalize with ImageNet mean/std. This is implemented once in
`src/preprocessing.py` and reused by every entry point (training dataloaders, batch evaluation,
and Grad-CAM) to avoid train/inference skew.

## License

Choose and add a license (MIT is a common default for portfolio ML projects) — see `LICENSE`.

## Acknowledgements

- [fastai](https://github.com/fastai/fastai) / [timm](https://github.com/huggingface/pytorch-image-models) for the ConvNeXt Tiny backbone and training loop.
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for the Grad-CAM++ implementation.
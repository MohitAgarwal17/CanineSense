"""
Utilities for loading trusted fastai exported Learners.

The models in this project were exported with:
    Learner.export(...)

Therefore they must be loaded with:
    fastai.load_learner(...)

These are trusted local model files, so we explicitly use
full-object pickle loading (weights_only=False).
"""

from __future__ import annotations

import pathlib
import platform
from pathlib import Path
import pickle

import torch
from fastai.learner import load_learner

# Fix PosixPath -> WindowsPath for models exported on Linux/Colab
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


# Robust unpickler fallback for missing 'Resolver' objects exported by timm
class _PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Resolver":
            class Resolver:
                def __init__(self, *args, **kwargs):
                    pass
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                def __getstate__(self):
                    return self.__dict__
            return Resolver
        return super().find_class(module, name)


def safe_load_learner(model_path: Path):
    """
    Load a trusted fastai exported Learner.

    Parameters
    ----------
    model_path : Path
        Path to a .pkl file produced by Learner.export().

    Returns
    -------
    fastai Learner
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist:\n{model_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model path is not a file:\n{model_path}"
        )

    print(f"Loading fastai learner: {model_path}")

    # PyTorch 2.6+ changed torch.load's default behavior.
    # We explicitly request full-object loading because an exported
    # fastai Learner contains much more than just model weights.
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs["weights_only"] = False
        kwargs["map_location"] = "cpu"
        kwargs["pickle_module"] = _PatchedUnpickler_Module
        return original_torch_load(*args, **kwargs)

    # Wrap our custom unpickler in a module proxy for torch.load
    class _PatchedUnpickler_Module:
        Unpickler = _PatchedUnpickler
        load = staticmethod(pickle.load)
        loads = staticmethod(pickle.loads)

    torch.load = torch_load_compat

    try:
        learn = load_learner(
            model_path,
            cpu=True,
            pickle_module=_PatchedUnpickler_Module,
        )
    except Exception as e:
        raise RuntimeError(
            f"\n"
            f"Could not load fastai learner.\n\n"
            f"Model:\n"
            f"  {model_path}\n\n"
            f"Original error:\n"
            f"  {type(e).__name__}: {e}\n"
        ) from e
    finally:
        # Never leave torch.load monkey-patched.
        torch.load = original_torch_load

    print("Learner loaded successfully.")

    return learn
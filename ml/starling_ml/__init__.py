"""Self-contained ML package for the starling pair transfer-classification model.

Pipeline:
    1. precompute_embeddings.py  -> cache frozen MolFormer + per-field MiniLM embeddings
    2. data.py                   -> build compact (a, b, label) memmaps + PairDataset
    3. model.py                  -> TransferPairModel (siamese MLPs + residual SwiGLU head)
    4. train.py / evaluate.py    -> HF Trainer training + evaluation

This package is intentionally decoupled from the PyArrow/RDKit data pipeline in `scripts/`.
"""

__all__ = ["config", "data", "model"]

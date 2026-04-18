
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def rho2_distance(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Thesis adaptation of paper-style rho2:
      rho2 = 1 - mean(min(a/b, b/a))

    For this rPVI experiment, vec_a and vec_b are sorted vectors of
    utterance-level rPVI values for the two trial sides.

    eps is added for numerical stability in case any rPVI value is 0.
    """
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Vector shape mismatch: {a.shape} vs {b.shape}")

    if a.size < 1:
        return np.nan

    a = a + eps
    b = b + eps

    ratio_min = np.minimum(a / b, b / a)
    return 1.0 - float(np.mean(ratio_min))


def eer_from_distances(distances: list[float], labels: list[int]) -> float:
    """
    Convert distances into similarity scores and compute EER (%).
    labels: 1 = same-speaker, 0 = different-speaker
    """
    if not distances:
        return np.nan

    scores = -np.asarray(distances, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if len(np.unique(labels)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = 100.0 * 0.5 * (fpr[idx] + fnr[idx])
    return float(eer)
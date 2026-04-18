from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


MODE_TO_FEATURE_COLS = {
    "all": ["rpvi_all", "npvi_all", "varco_all"],
    "vowel": ["rpvi_vowel", "npvi_vowel", "varco_vowel"],
    "consonant": ["rpvi_consonant", "npvi_consonant", "varco_consonant"],
}

MODE_TO_COUNT_COL = {
    "all": "n_all",
    "vowel": "n_vowels",
    "consonant": "n_consonants",
}


def filter_mode_df(metrics_df: pd.DataFrame, mode: str, min_adj_pairs: int) -> pd.DataFrame:
    if mode not in MODE_TO_FEATURE_COLS:
        raise ValueError(f"Unknown mode: {mode}")

    feature_cols = MODE_TO_FEATURE_COLS[mode]
    count_col = MODE_TO_COUNT_COL[mode]
    need = min_adj_pairs + 1

    df_ok = metrics_df[
        metrics_df[count_col] >= need
    ].dropna(subset=feature_cols).copy()

    return df_ok


def build_structures(metrics_df: pd.DataFrame, mode: str, min_adj_pairs: int):
    df_ok = filter_mode_df(metrics_df, mode, min_adj_pairs)
    feature_cols = MODE_TO_FEATURE_COLS[mode]

    speaker2utts = defaultdict(list)
    utt2vec = {}

    for row in df_ok.itertuples(index=False):
        utt = str(row.utter_id)
        spk = str(row.speaker_id)
        vec = np.array([getattr(row, col) for col in feature_cols], dtype=float)

        if np.any(np.isnan(vec)):
            continue

        utt2vec[utt] = vec
        speaker2utts[spk].append(utt)

    for spk, utts in speaker2utts.items():
        seen = set()
        uniq = []
        for utt in utts:
            if utt not in seen:
                uniq.append(utt)
                seen.add(utt)
        speaker2utts[spk] = uniq

    return speaker2utts, utt2vec


def build_trials(speaker2utts: dict[str, list[str]], k: int, n_trials: int, rng: np.random.Generator):
    eligible = {spk: utts for spk, utts in speaker2utts.items() if len(utts) >= 2 * k}
    speakers = list(eligible.keys())

    if len(speakers) < 2:
        return []

    trials = []
    half = n_trials // 2

    for _ in range(half):
        spk = rng.choice(speakers)
        utts = eligible[spk]
        picks = rng.choice(utts, size=2 * k, replace=False)
        side_a = picks[:k].tolist()
        side_b = picks[k:].tolist()
        trials.append((side_a, side_b, 1))

    for _ in range(n_trials - half):
        spk_a, spk_b = rng.choice(speakers, size=2, replace=False)
        utts_a = rng.choice(eligible[spk_a], size=k, replace=False).tolist()
        utts_b = rng.choice(eligible[spk_b], size=k, replace=False).tolist()
        trials.append((utts_a, utts_b, 0))

    return trials


def build_side_profile(utts: list[str], utt2vec: dict[str, np.ndarray]) -> np.ndarray | None:
    vecs = [utt2vec[u] for u in utts if u in utt2vec]
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)
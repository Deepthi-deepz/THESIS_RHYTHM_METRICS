from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


MODE_TO_VALUE_COL = {
    "all": "rpvi_all",
    "vowel": "rpvi_vowel",
    "consonant": "rpvi_consonant",
}

MODE_TO_COUNT_COL = {
    "all": "n_all",
    "vowel": "n_vowels",
    "consonant": "n_consonants",
}


def filter_mode_df(metrics_df: pd.DataFrame, mode: str, min_adj_pairs: int) -> pd.DataFrame:
    """
    Filter utterances eligible for a given table mode and adjacent-pair threshold.

    Threshold rule:
      t adjacent pairs => need at least t+1 durations in the relevant sequence
    """
    if mode not in MODE_TO_VALUE_COL:
        raise ValueError(f"Unknown mode: {mode}")

    value_col = MODE_TO_VALUE_COL[mode]
    count_col = MODE_TO_COUNT_COL[mode]
    need = min_adj_pairs + 1

    df_ok = metrics_df[
        (metrics_df[count_col] >= need)
    ].dropna(subset=[value_col]).copy()

    return df_ok


def build_structures(metrics_df: pd.DataFrame, mode: str, min_adj_pairs: int):
    """
    Return:
      speaker2utts: dict[speaker_id] -> list[utter_id]
      utt2value: dict[utter_id] -> rpvi scalar
    """
    df_ok = filter_mode_df(metrics_df, mode, min_adj_pairs)
    value_col = MODE_TO_VALUE_COL[mode]

    speaker2utts = defaultdict(list)
    utt2value = {}

    for row in df_ok.itertuples(index=False):
        utt = str(row.utter_id)
        spk = str(row.speaker_id)
        val = float(getattr(row, value_col))
        utt2value[utt] = val
        speaker2utts[spk].append(utt)

    # Deduplicate while preserving order
    for spk, utts in speaker2utts.items():
        seen = set()
        unique_utts = []
        for utt in utts:
            if utt not in seen:
                unique_utts.append(utt)
                seen.add(utt)
        speaker2utts[spk] = unique_utts

    return speaker2utts, utt2value


def build_trials(speaker2utts: dict[str, list[str]], k: int, n_trials: int, rng: np.random.Generator):
    """
    Build balanced same-speaker and different-speaker trials.
    Each side contains k utterances.
    """
    eligible = {spk: utts for spk, utts in speaker2utts.items() if len(utts) >= 2 * k}
    speakers = list(eligible.keys())

    if len(speakers) < 2:
        return []

    trials = []
    half = n_trials // 2

    # Same-speaker
    for _ in range(half):
        spk = rng.choice(speakers)
        utts = eligible[spk]
        picks = rng.choice(utts, size=2 * k, replace=False)
        side_a = picks[:k].tolist()
        side_b = picks[k:].tolist()
        trials.append((side_a, side_b, 1))

    # Different-speaker
    for _ in range(n_trials - half):
        spk_a, spk_b = rng.choice(speakers, size=2, replace=False)
        utts_a = rng.choice(eligible[spk_a], size=k, replace=False).tolist()
        utts_b = rng.choice(eligible[spk_b], size=k, replace=False).tolist()
        trials.append((utts_a, utts_b, 0))

    return trials


def build_side_vector(utts: list[str], utt2value: dict[str, float]) -> np.ndarray | None:
    """
    Side profile = sorted utterance-level rPVI values for that side.
    """
    vals = [utt2value[u] for u in utts if u in utt2value]
    if not vals:
        return None

    vec = np.asarray(sorted(vals), dtype=float)
    return vec
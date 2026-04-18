from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PHONES_39 = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]


def _load_one_alignment_csv(csv_path: str | Path, use_phones39: bool = True) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Alignment CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"utter_id", "speaker_id", "phone", "start_time", "end_time"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["phone"] = df["phone"].astype(str)
    df["base_phone"] = df["phone"].str.replace(r"\d$", "", regex=True)
    df["stress"] = df["phone"].str.extract(r"(\d)$", expand=False).fillna("0").astype(int)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    df["duration"] = df["end_time"] - df["start_time"]

    df = df[df["duration"] > 0].copy()

    if use_phones39:
        df = df[df["base_phone"].isin(PHONES_39)].copy()

    if df.empty:
        raise RuntimeError(f"No tokens remain after filtering for: {csv_path}")

    df["utter_id"] = df["utter_id"].astype(str)
    df["speaker_id"] = df["speaker_id"].astype(str)

    return df[[
        "utter_id", "speaker_id", "base_phone", "stress",
        "start_time", "end_time", "duration"
    ]].copy()


def load_alignments(csv_paths: str | Path | list[str] | list[Path], use_phones39: bool = True) -> pd.DataFrame:
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    dfs = [_load_one_alignment_csv(p, use_phones39=use_phones39) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        raise RuntimeError("No tokens remain after concatenating alignment CSVs.")

    spk_counts = df.groupby("utter_id")["speaker_id"].nunique()
    bad = spk_counts[spk_counts > 1]
    if not bad.empty:
        bad_utts = bad.index.tolist()[:10]
        raise ValueError(
            "Found utterance IDs assigned to multiple speakers across CSVs. "
            f"Examples: {bad_utts}"
        )

    return df


def varco(durations: np.ndarray) -> float:
    """
    Varco = 100 * std / mean

    Uses sample std (ddof=1), matching the earlier rhythm implementation style.
    Returns NaN if:
      - fewer than 2 durations
      - mean is 0
    """
    d = np.asarray(durations, dtype=float)
    if d.size < 2:
        return np.nan

    mean_d = float(np.mean(d))
    if mean_d == 0.0:
        return np.nan

    return float(100.0 * np.std(d, ddof=1) / mean_d)


def extract_utterance_varco_metrics(df: pd.DataFrame, vowels: Iterable[str]) -> pd.DataFrame:
    vowels = set(vowels)
    rows = []

    for utt_id, g in df.groupby("utter_id", sort=False):
        g = g.sort_values("start_time").reset_index(drop=True)
        speaker_id = str(g["speaker_id"].iloc[0])

        all_durs = g["duration"].to_numpy(dtype=float)

        v_mask = g["base_phone"].isin(vowels).to_numpy()
        vowel_durs = g.loc[v_mask, "duration"].to_numpy(dtype=float)
        consonant_durs = g.loc[~v_mask, "duration"].to_numpy(dtype=float)

        rows.append({
            "utter_id": str(utt_id),
            "speaker_id": speaker_id,
            "n_all": int(len(all_durs)),
            "n_vowels": int(len(vowel_durs)),
            "n_consonants": int(len(consonant_durs)),
            "varco_all": varco(all_durs),
            "varco_vowel": varco(vowel_durs),
            "varco_consonant": varco(consonant_durs),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No utterance-level Varco rows were produced.")

    return out
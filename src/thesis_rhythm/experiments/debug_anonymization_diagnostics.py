from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thesis_rhythm.features.rhythm_bundle import load_alignments, extract_utterance_rhythm_bundle
from thesis_rhythm.metrics.distances import rho2_distance, eer_from_distances
from thesis_rhythm.trials.rhythm_bundle_trials import build_structures, build_trials, build_side_profile


PROFILE_TO_COLS = {
    "all": ["rpvi_all", "npvi_all", "varco_all"],
    "vowel": ["rpvi_vowel", "npvi_vowel", "varco_vowel"],
    "consonant": ["rpvi_consonant", "npvi_consonant", "varco_consonant"],
}

PROFILE_TO_COUNT = {
    "all": "n_all",
    "vowel": "n_vowels",
    "consonant": "n_consonants",
}


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def dataset_basic_checks(df: pd.DataFrame) -> None:
    print_header("BASIC DATASET CHECKS")
    print(f"tokens: {len(df):,}")
    print(f"utts:   {df['utter_id'].nunique():,}")
    print(f"spks:   {df['speaker_id'].nunique():,}")

    print("\nDuration statistics:")
    print(df["duration"].describe())

    bad = df.groupby("utter_id")["speaker_id"].nunique()
    bad = bad[bad > 1]
    print(f"\nutter_id collision count (same utt_id mapped to >1 speaker): {len(bad)}")
    if len(bad) > 0:
        print("Examples:", bad.head(10).to_dict())


def feature_checks(metrics_df: pd.DataFrame) -> None:
    print_header("UTTERANCE-LEVEL FEATURE CHECKS")

    feature_sets = {
        "all": ["rpvi_all", "npvi_all", "varco_all"],
        "vowel": ["rpvi_vowel", "npvi_vowel", "varco_vowel"],
        "consonant": ["rpvi_consonant", "npvi_consonant", "varco_consonant"],
    }

    for name, cols in feature_sets.items():
        valid = metrics_df[cols].dropna().shape[0]
        print(f"valid {name:10s}: {valid:,}")

    print("\nCount-column summary:")
    print(metrics_df[["n_all", "n_vowels", "n_consonants"]].describe())

    print("\nFeature distribution summary:")
    cols = [
        "rpvi_all", "npvi_all", "varco_all",
        "rpvi_vowel", "npvi_vowel", "varco_vowel",
        "rpvi_consonant", "npvi_consonant", "varco_consonant",
    ]
    print(metrics_df[cols].describe().T[["mean", "std", "min", "max"]])


def eligibility_checks(
    metrics_df: pd.DataFrame,
    utterances_per_side: list[int],
    min_adj_pairs: list[int],
) -> pd.DataFrame:
    print_header("ELIGIBILITY CHECKS PER PROFILE / CELL")

    rows = []

    for mode in ["all", "vowel", "consonant"]:
        for k in utterances_per_side:
            for t in min_adj_pairs:
                speaker2utts, utt2vec = build_structures(metrics_df, mode=mode, min_adj_pairs=t)

                eligible_speakers = {s: u for s, u in speaker2utts.items() if len(u) >= 2 * k}
                eligible_utts = len(utt2vec)
                eligible_spks = len(speaker2utts)
                spks_with_2k = len(eligible_speakers)

                row = {
                    "profile": mode,
                    "k": k,
                    "t": t,
                    "eligible_utts": eligible_utts,
                    "eligible_spks": eligible_spks,
                    "spks_with_2k": spks_with_2k,
                }
                rows.append(row)

    out = pd.DataFrame(rows)
    print(out.head(30))
    return out


def evaluate_cell(
    metrics_df: pd.DataFrame,
    mode: str,
    k: int,
    t: int,
    n_trials: int,
    seed: int,
    n_debug_trials: int = 3,
) -> tuple[float, int, int]:
    speaker2utts, utt2vec = build_structures(metrics_df, mode=mode, min_adj_pairs=t)

    rng = np.random.default_rng(seed)
    trials = build_trials(speaker2utts, k=k, n_trials=n_trials, rng=rng)

    distances = []
    labels = []

    printed = 0
    for side_a, side_b, lab in trials:
        vec_a = build_side_profile(side_a, utt2vec)
        vec_b = build_side_profile(side_b, utt2vec)

        if vec_a is None or vec_b is None:
            continue

        dist = rho2_distance(vec_a, vec_b)
        if np.isnan(dist):
            continue

        distances.append(dist)
        labels.append(lab)

        if printed < n_debug_trials:
            print("\n--- Debug trial ---")
            print("label:", "same" if lab == 1 else "different")
            print("side_a_utts:", side_a)
            print("side_b_utts:", side_b)
            print("vec_a:", vec_a)
            print("vec_b:", vec_b)
            print("rho2:", dist)
            printed += 1

    eer = eer_from_distances(distances, labels)
    return eer, len(trials), len(distances)


def best_cell_summary(
    metrics_df: pd.DataFrame,
    utterances_per_side: list[int],
    min_adj_pairs: list[int],
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    print_header("BEST-CELL SUMMARY")

    rows = []

    for mode in ["all", "vowel", "consonant"]:
        best = None

        for k in utterances_per_side:
            for t in min_adj_pairs:
                eer, trials_built, trials_scored = evaluate_cell(
                    metrics_df=metrics_df,
                    mode=mode,
                    k=k,
                    t=t,
                    n_trials=n_trials,
                    seed=seed,
                    n_debug_trials=0,
                )

                if np.isnan(eer):
                    continue

                if best is None or eer < best["best_eer"]:
                    speaker2utts, utt2vec = build_structures(metrics_df, mode=mode, min_adj_pairs=t)
                    eligible_speakers = {s: u for s, u in speaker2utts.items() if len(u) >= 2 * k}

                    best = {
                        "profile": mode,
                        "best_eer": float(eer),
                        "best_row_k": k,
                        "best_col_t": t,
                        "eligible_utts_best": len(utt2vec),
                        "eligible_spks_best": len(speaker2utts),
                        "spks_with_2k_best": len(eligible_speakers),
                        "trials_built_best": trials_built,
                        "trials_scored_best": trials_scored,
                    }

        rows.append(best if best is not None else {
            "profile": mode,
            "best_eer": np.nan,
            "best_row_k": np.nan,
            "best_col_t": np.nan,
            "eligible_utts_best": np.nan,
            "eligible_spks_best": np.nan,
            "spks_with_2k_best": np.nan,
            "trials_built_best": np.nan,
            "trials_scored_best": np.nan,
        })

    out = pd.DataFrame(rows)
    print(out)
    return out


def run_debug(config_path: str) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = str(cfg["dataset_name"])
    alignment_csvs = cfg["alignment_csvs"]
    utterances_per_side = list(cfg["utterances_per_side"])
    min_adj_pairs = list(cfg["min_adj_pairs"])
    random_seed = int(cfg["random_seed"])
    n_trials = int(cfg["n_trials_per_condition"])
    use_phones39 = bool(cfg["phoneset"]["use_phones39"])
    vowels = list(cfg["phoneset"]["vowels"])

    print_header(f"CONFIG CHECK: {dataset_name}")
    print("config path:", config_path)
    print("alignment_csvs:")
    for p in alignment_csvs:
        print("  ", p)

    df = load_alignments(alignment_csvs, use_phones39=use_phones39)
    dataset_basic_checks(df)

    metrics_df = extract_utterance_rhythm_bundle(df, vowels=vowels)
    feature_checks(metrics_df)

    elig_df = eligibility_checks(metrics_df, utterances_per_side, min_adj_pairs)

    # Save eligibility diagnostics next to config for convenience
    diag_dir = Path("results") / "debug_diagnostics" / dataset_name
    diag_dir.mkdir(parents=True, exist_ok=True)
    elig_path = diag_dir / "eligibility_checks.csv"
    elig_df.to_csv(elig_path, index=False)
    print(f"\nSaved: {elig_path}")

    summary_df = best_cell_summary(
        metrics_df,
        utterances_per_side=utterances_per_side,
        min_adj_pairs=min_adj_pairs,
        n_trials=n_trials,
        seed=random_seed,
    )
    summary_path = diag_dir / "best_cell_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print_header("MANUAL DEBUG CELL")
    print("Using profile=all, k=5, t=1 for sample trial inspection")
    eer, trials_built, trials_scored = evaluate_cell(
        metrics_df=metrics_df,
        mode="all",
        k=5,
        t=1,
        n_trials=min(50, n_trials),
        seed=random_seed,
        n_debug_trials=3,
    )
    print("\nmanual debug cell summary:")
    print("profile=all k=5 t=1")
    print("eer:", eer)
    print("trials_built:", trials_built)
    print("trials_scored:", trials_scored)


def main():
    parser = argparse.ArgumentParser(description="Debug anonymization rhythm comparison.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_debug(args.config)


if __name__ == "__main__":
    main()
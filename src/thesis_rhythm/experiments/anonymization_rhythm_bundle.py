from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thesis_rhythm.features.rhythm_bundle import load_alignments, extract_utterance_rhythm_bundle
from thesis_rhythm.metrics.distances import rho2_distance, eer_from_distances
from thesis_rhythm.reporting.save_tables import save_table
from thesis_rhythm.trials.rhythm_bundle_trials import (
    build_side_profile,
    build_structures,
    build_trials,
)


def evaluate_one_table(
    metrics_df: pd.DataFrame,
    mode: str,
    utterances_per_side: list[int],
    min_adj_pairs: list[int],
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    all_rows = []

    for k in utterances_per_side:
        row_vals = []

        for t in min_adj_pairs:
            speaker2utts, utt2vec = build_structures(metrics_df, mode=mode, min_adj_pairs=t)

            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            trials = build_trials(speaker2utts, k=k, n_trials=n_trials, rng=rng)

            if not trials:
                row_vals.append(np.nan)
                continue

            distances = []
            labels = []

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

            row_vals.append(eer_from_distances(distances, labels))

        all_rows.append(row_vals)

    return pd.DataFrame(all_rows, index=utterances_per_side, columns=min_adj_pairs)


def save_summary(dataset_name: str, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []
    for profile_name, df in tables.items():
        if df.isna().all().all():
            rows.append({
                "dataset": dataset_name,
                "profile": profile_name,
                "best_eer": np.nan,
                "best_row": np.nan,
                "best_col": np.nan,
            })
            continue

        stacked = df.stack(dropna=True)
        best_idx = stacked.idxmin()

        rows.append({
            "dataset": dataset_name,
            "profile": profile_name,
            "best_eer": float(stacked.min()),
            "best_row": best_idx[0],
            "best_col": best_idx[1],
        })

    summary_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rhythm_bundle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print(summary_df)


def run_experiment(config_path: str) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = str(cfg["dataset_name"])
    alignment_csvs = list(cfg["alignment_csvs"])
    output_dir = Path(cfg["output_dir"])
    random_seed = int(cfg["random_seed"])
    n_trials = int(cfg["n_trials_per_condition"])
    utterances_per_side = list(cfg["utterances_per_side"])
    min_adj_pairs = list(cfg["min_adj_pairs"])
    use_phones39 = bool(cfg["phoneset"]["use_phones39"])
    vowels = list(cfg["phoneset"]["vowels"])

    print(f"Dataset: {dataset_name}")
    print("Loading alignments from:")
    for p in alignment_csvs:
        print(f"  - {p}")

    df = load_alignments(alignment_csvs, use_phones39=use_phones39)

    print(
        f"Loaded tokens={len(df):,}, "
        f"utts={df['utter_id'].nunique():,}, "
        f"spks={df['speaker_id'].nunique():,}"
    )

    metrics_df = extract_utterance_rhythm_bundle(df, vowels=vowels)
    print(f"Utterance-level rows: {len(metrics_df):,}")

    tables = {}

    mode_to_filename = {
        "all": "rhythm_bundle_all_rho2_table.csv",
        "vowel": "rhythm_bundle_vowel_rho2_table.csv",
        "consonant": "rhythm_bundle_consonant_rho2_table.csv",
    }

    for mode in ["all", "vowel", "consonant"]:
        print(f"\nRunning rhythm bundle | mode={mode}")

        table_df = evaluate_one_table(
            metrics_df=metrics_df,
            mode=mode,
            utterances_per_side=utterances_per_side,
            min_adj_pairs=min_adj_pairs,
            n_trials=n_trials,
            seed=random_seed,
        )

        out_path = output_dir / mode_to_filename[mode]
        save_table(table_df, out_path)
        tables[mode] = table_df

    save_summary(dataset_name=dataset_name, tables=tables, output_dir=output_dir)
    print(f"\nDone. Rhythm-bundle anonymization results written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run anonymization comparison with rhythm bundle profiles.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

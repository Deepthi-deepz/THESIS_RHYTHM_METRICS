from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


from thesis_rhythm.features.npvi import load_alignments, extract_utterance_npvi_metrics
from thesis_rhythm.metrics.distances import rho2_distance, eer_from_distances
from thesis_rhythm.reporting.save_tables import save_table
from thesis_rhythm.trials.npvi_trials import build_side_vector, build_structures, build_trials

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
            speaker2utts, utt2value = build_structures(metrics_df, mode=mode, min_adj_pairs=t)

            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            trials = build_trials(speaker2utts, k=k, n_trials=n_trials, rng=rng)

            if not trials:
                row_vals.append(np.nan)
                continue

            distances = []
            labels = []

            for side_a, side_b, lab in trials:
                vec_a = build_side_vector(side_a, utt2value)
                vec_b = build_side_vector(side_b, utt2value)

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


def run_experiment(config_path: str) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    alignment_csvs = list(cfg["alignment_csvs"])
    output_dir = Path(cfg["output_dir"])
    random_seed = int(cfg["random_seed"])
    n_trials = int(cfg["n_trials_per_condition"])
    utterances_per_side = list(cfg["utterances_per_side"])
    min_adj_pairs = list(cfg["min_adj_pairs"])
    use_phones39 = bool(cfg["phoneset"]["use_phones39"])
    vowels = list(cfg["phoneset"]["vowels"])
    tables = list(cfg["tables"])

    print("Loading alignments from:")
    for p in alignment_csvs:
        print(f"  - {p}")

    df = load_alignments(alignment_csvs, use_phones39=use_phones39)

    print(
        f"Loaded tokens={len(df):,}, "
        f"utts={df['utter_id'].nunique():,}, "
        f"spks={df['speaker_id'].nunique():,}"
    )

    metrics_df = extract_utterance_npvi_metrics(df, vowels=vowels)
    print(f"Utterance-level rows: {len(metrics_df):,}")

    for table_cfg in tables:
        table_name = str(table_cfg["name"])
        mode = str(table_cfg["mode"])

        print(f"\nRunning table: {table_name} (mode={mode})")

        table_df = evaluate_one_table(
            metrics_df=metrics_df,
            mode=mode,
            utterances_per_side=utterances_per_side,
            min_adj_pairs=min_adj_pairs,
            n_trials=n_trials,
            seed=random_seed,
        )

        out_path = output_dir / f"{table_name}_rho2_table.csv"
        save_table(table_df, out_path)

    print(f"\nDone. Final tables written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute nPVI rho2 tables on train-960.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
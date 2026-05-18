from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thesis_rhythm.features.rpvi import load_alignments as load_alignments_rpvi
from thesis_rhythm.features.rpvi import extract_utterance_rpvi_metrics
from thesis_rhythm.features.npvi import load_alignments as load_alignments_npvi
from thesis_rhythm.features.npvi import extract_utterance_npvi_metrics
from thesis_rhythm.features.varco import load_alignments as load_alignments_varco
from thesis_rhythm.features.varco import extract_utterance_varco_metrics

from thesis_rhythm.metrics.distances import rho2_distance, eer_from_distances
from thesis_rhythm.reporting.save_tables import save_table
from thesis_rhythm.trials.rpvi_trials import (
    build_side_vector as build_side_vector_rpvi,
    build_structures as build_structures_rpvi,
    build_trials as build_trials_rpvi,
)
from thesis_rhythm.trials.npvi_trials import (
    build_side_vector as build_side_vector_npvi,
    build_structures as build_structures_npvi,
    build_trials as build_trials_npvi,
)
from thesis_rhythm.trials.varco_trials import (
    build_side_vector as build_side_vector_varco,
    build_structures as build_structures_varco,
    build_trials as build_trials_varco,
)


def evaluate_one_table(
    metrics_df: pd.DataFrame,
    mode: str,
    utterances_per_side: list[int],
    min_adj_pairs: list[int],
    n_trials: int,
    seed: int,
    build_structures_fn,
    build_trials_fn,
    build_side_vector_fn,
) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    all_rows = []

    for k in utterances_per_side:
        row_vals = []

        for t in min_adj_pairs:
            speaker2utts, utt2value = build_structures_fn(metrics_df, mode=mode, min_adj_pairs=t)

            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            trials = build_trials_fn(speaker2utts, k=k, n_trials=n_trials, rng=rng)

            if not trials:
                row_vals.append(np.nan)
                continue

            distances = []
            labels = []

            for side_a, side_b, lab in trials:
                vec_a = build_side_vector_fn(side_a, utt2value)
                vec_b = build_side_vector_fn(side_b, utt2value)

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


def save_metric_summary(dataset_name: str, metric_name: str, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []
    for profile_name, df in tables.items():
        if df.isna().all().all():
            rows.append({
                "dataset": dataset_name,
                "metric": metric_name,
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
            "metric": metric_name,
            "profile": profile_name,
            "best_eer": float(stacked.min()),
            "best_row": best_idx[0],
            "best_col": best_idx[1],
        })

    summary_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{metric_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print(summary_df)


def run_metric_block(
    dataset_name: str,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    metric_name: str,
    utterances_per_side: list[int],
    min_adj_pairs: list[int],
    n_trials: int,
    seed: int,
    build_structures_fn,
    build_trials_fn,
    build_side_vector_fn,
) -> None:
    mode_to_filename = {
        "all": f"{metric_name}_all_rho2_table.csv",
        "vowel": f"{metric_name}_vowel_rho2_table.csv",
        "consonant": f"{metric_name}_consonant_rho2_table.csv",
    }

    tables = {}

    for mode in ["all", "vowel", "consonant"]:
        print(f"\nRunning {metric_name} | mode={mode}")

        table_df = evaluate_one_table(
            metrics_df=metrics_df,
            mode=mode,
            utterances_per_side=utterances_per_side,
            min_adj_pairs=min_adj_pairs,
            n_trials=n_trials,
            seed=seed,
            build_structures_fn=build_structures_fn,
            build_trials_fn=build_trials_fn,
            build_side_vector_fn=build_side_vector_fn,
        )

        out_path = output_dir / mode_to_filename[mode]
        save_table(table_df, out_path)
        tables[mode] = table_df

    save_metric_summary(dataset_name, metric_name, tables, output_dir)


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

    # rPVI
    df_rpvi = load_alignments_rpvi(alignment_csvs, use_phones39=use_phones39)
    rpvi_df = extract_utterance_rpvi_metrics(df_rpvi, vowels=vowels)
    run_metric_block(
        dataset_name=dataset_name,
        output_dir=output_dir,
        metrics_df=rpvi_df,
        metric_name="rpvi",
        utterances_per_side=utterances_per_side,
        min_adj_pairs=min_adj_pairs,
        n_trials=n_trials,
        seed=random_seed,
        build_structures_fn=build_structures_rpvi,
        build_trials_fn=build_trials_rpvi,
        build_side_vector_fn=build_side_vector_rpvi,
    )

    # nPVI
    df_npvi = load_alignments_npvi(alignment_csvs, use_phones39=use_phones39)
    npvi_df = extract_utterance_npvi_metrics(df_npvi, vowels=vowels)
    run_metric_block(
        dataset_name=dataset_name,
        output_dir=output_dir,
        metrics_df=npvi_df,
        metric_name="npvi",
        utterances_per_side=utterances_per_side,
        min_adj_pairs=min_adj_pairs,
        n_trials=n_trials,
        seed=random_seed,
        build_structures_fn=build_structures_npvi,
        build_trials_fn=build_trials_npvi,
        build_side_vector_fn=build_side_vector_npvi,
    )

    # Varco
    df_varco = load_alignments_varco(alignment_csvs, use_phones39=use_phones39)
    varco_df = extract_utterance_varco_metrics(df_varco, vowels=vowels)
    run_metric_block(
        dataset_name=dataset_name,
        output_dir=output_dir,
        metrics_df=varco_df,
        metric_name="varco",
        utterances_per_side=utterances_per_side,
        min_adj_pairs=min_adj_pairs,
        n_trials=n_trials,
        seed=random_seed,
        build_structures_fn=build_structures_varco,
        build_trials_fn=build_trials_varco,
        build_side_vector_fn=build_side_vector_varco,
    )

    print(f"\nDone. Separate-metric anonymization results written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run anonymization comparison with separate rhythm metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from thesis_rhythm.features.rhythm_bundle import (
    load_alignments,
    extract_utterance_rhythm_bundle,
)
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
    min_eligible_speakers: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng_master = np.random.default_rng(seed)
    all_rows = []
    diag_rows = []

    for k in utterances_per_side:
        row_vals = []

        for t in min_adj_pairs:
            speaker2utts, utt2vec = build_structures(
                metrics_df,
                mode=mode,
                min_adj_pairs=t,
            )

            eligible_speakers = {
                speaker_id: utts
                for speaker_id, utts in speaker2utts.items()
                if len(utts) >= 2 * k
            }

            diag = {
                "profile": mode,
                "k": k,
                "t": t,
                "eligible_utts": len(utt2vec),
                "eligible_spks": len(speaker2utts),
                "spks_with_2k": len(eligible_speakers),
                "min_eligible_speakers": min_eligible_speakers,
                "trials_built": 0,
                "trials_scored": 0,
                "eer": np.nan,
                "skipped_reason": "",
            }

            if len(eligible_speakers) < min_eligible_speakers:
                diag["skipped_reason"] = "too_few_eligible_speakers"
                diag_rows.append(diag)
                row_vals.append(np.nan)
                continue

            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            trials = build_trials(
                speaker2utts,
                k=k,
                n_trials=n_trials,
                rng=rng,
            )

            diag["trials_built"] = len(trials)

            if not trials:
                diag["skipped_reason"] = "no_trials"
                diag_rows.append(diag)
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

            diag["trials_scored"] = len(distances)

            if len(distances) == 0:
                diag["skipped_reason"] = "no_scored_trials"
                diag_rows.append(diag)
                row_vals.append(np.nan)
                continue

            if len(set(labels)) < 2:
                diag["skipped_reason"] = "missing_same_or_different_trials"
                diag_rows.append(diag)
                row_vals.append(np.nan)
                continue

            eer = eer_from_distances(distances, labels)

            if np.isnan(eer):
                diag["skipped_reason"] = "nan_eer"
                diag_rows.append(diag)
                row_vals.append(np.nan)
                continue

            diag["eer"] = float(eer)
            diag_rows.append(diag)
            row_vals.append(float(eer))

        all_rows.append(row_vals)

    table_df = pd.DataFrame(
        all_rows,
        index=utterances_per_side,
        columns=min_adj_pairs,
    )

    diag_df = pd.DataFrame(diag_rows)
    return table_df, diag_df


def save_summary(
    dataset_name: str,
    tables: dict[str, pd.DataFrame],
    diagnostics: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    rows = []

    for profile_name, df in tables.items():
        if df.isna().all().all():
            rows.append({
                "dataset": dataset_name,
                "profile": profile_name,
                "best_eer": np.nan,
                "best_row_k": np.nan,
                "best_col_t": np.nan,
                "eligible_utts_best": np.nan,
                "eligible_spks_best": np.nan,
                "spks_with_2k_best": np.nan,
                "trials_built_best": np.nan,
                "trials_scored_best": np.nan,
            })
            continue

        stacked = df.stack(dropna=True)
        best_k, best_t = stacked.idxmin()
        best_eer = float(stacked.min())

        diag_df = diagnostics[profile_name]
        best_diag = diag_df[
            (diag_df["k"] == best_k)
            & (diag_df["t"] == best_t)
        ]

        if len(best_diag) > 0:
            best_diag = best_diag.iloc[0]
            eligible_utts_best = best_diag["eligible_utts"]
            eligible_spks_best = best_diag["eligible_spks"]
            spks_with_2k_best = best_diag["spks_with_2k"]
            trials_built_best = best_diag["trials_built"]
            trials_scored_best = best_diag["trials_scored"]
        else:
            eligible_utts_best = np.nan
            eligible_spks_best = np.nan
            spks_with_2k_best = np.nan
            trials_built_best = np.nan
            trials_scored_best = np.nan

        rows.append({
            "dataset": dataset_name,
            "profile": profile_name,
            "best_eer": best_eer,
            "best_row_k": best_k,
            "best_col_t": best_t,
            "eligible_utts_best": eligible_utts_best,
            "eligible_spks_best": eligible_spks_best,
            "spks_with_2k_best": spks_with_2k_best,
            "trials_built_best": trials_built_best,
            "trials_scored_best": trials_scored_best,
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

    min_eligible_speakers = int(cfg.get("min_eligible_speakers", 100))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Minimum eligible speakers per cell: {min_eligible_speakers}")
    print("Loading alignments from:")

    for p in alignment_csvs:
        print(f"  - {p}")

    df = load_alignments(
        alignment_csvs,
        use_phones39=use_phones39,
    )

    print(
        f"Loaded tokens={len(df):,}, "
        f"utts={df['utter_id'].nunique():,}, "
        f"spks={df['speaker_id'].nunique():,}"
    )

    metrics_df = extract_utterance_rhythm_bundle(
        df,
        vowels=vowels,
    )

    print(f"Utterance-level rows: {len(metrics_df):,}")

    tables = {}
    diagnostics = {}

    mode_to_filename = {
        "all": "rhythm_bundle_all_rho2_table.csv",
        "vowel": "rhythm_bundle_vowel_rho2_table.csv",
        "consonant": "rhythm_bundle_consonant_rho2_table.csv",
    }

    for mode in ["all", "vowel", "consonant"]:
        print(f"\nRunning rhythm bundle | mode={mode}")

        table_df, diag_df = evaluate_one_table(
            metrics_df=metrics_df,
            mode=mode,
            utterances_per_side=utterances_per_side,
            min_adj_pairs=min_adj_pairs,
            n_trials=n_trials,
            seed=random_seed,
            min_eligible_speakers=min_eligible_speakers,
        )

        out_path = output_dir / mode_to_filename[mode]
        save_table(table_df, out_path)

        diag_path = output_dir / f"rhythm_bundle_{mode}_diagnostics.csv"
        diag_df.to_csv(diag_path, index=False)

        print(f"Saved: {diag_path}")

        tables[mode] = table_df
        diagnostics[mode] = diag_df

    save_summary(
        dataset_name=dataset_name,
        tables=tables,
        diagnostics=diagnostics,
        output_dir=output_dir,
    )

    print(f"\nDone. Rhythm-bundle anonymization results written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anonymization comparison with rhythm bundle profiles."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

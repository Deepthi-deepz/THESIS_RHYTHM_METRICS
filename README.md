# THESIS_RHYTHM_METRICS


This repository contains experiments for analyzing speech temporal dynamics in the context of speaker verification and voice anonymization.

## Overview

We study how rhythm-based features can encode speaker identity and how much of this information survives anonymization.

## Experiments

- rPVI, nPVI, Varco-based speaker verification
- rho2 distance metric evaluation
- Experiments on:
  - train-clean-100
  - train-960
- Rhythm bundle speaker profiles:
  - all phones
  - vowels only
  - consonants only

## Anonymization Comparison

We compare:

- Original train-clean-360
- SAS-1 (preserves phoneme durations)
- SAS-2 (modifies phoneme durations)

Goal:
Evaluate how temporal dynamics contribute to speaker identity leakage after anonymization.

## Structure

- `src/` → core implementation
- `configs/` → experiment configs
- `results/` → outputs (not tracked)

## Note

This work is inspired by temporal dynamics analysis for speaker verification and anonymization, but extends it using rhythm-based feature representations.
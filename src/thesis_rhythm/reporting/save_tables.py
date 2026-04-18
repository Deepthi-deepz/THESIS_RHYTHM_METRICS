from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Saved: {path}")
    print(df)
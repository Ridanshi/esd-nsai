"""Regenerate data/dermatology_raw.csv from UCI.

Run this once with a working network connection whenever the cached
dataset needs refreshing. Deployment (Streamlit Cloud, etc.) never needs
to run this itself — it just reads the committed CSV.
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dermatology_raw.csv"


def main():
    raw = fetch_ucirepo(id=33)
    X = raw.data.features
    y = raw.data.targets.iloc[:, 0].rename("class")
    df = pd.concat([X, y], axis=1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

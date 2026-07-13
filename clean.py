from __future__ import annotations

from pathlib import Path
from py_compile import main

import pandas as pd

IN = Path("data/matches.parquet")
OUT = Path("data/clean.parquet")

# All 42 temas wew visible to be uniqe, so mapping will remain empty.
NAME_MAP: dict[str, str] = {}

def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply NAME_MAP to HomeTeam/AwayTeam. A no-op wherever NAME_MAP is empty."""
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].replace(NAME_MAP)
    df["AwayTeam"] = df["AwayTeam"].replace(NAME_MAP)
    return df

def main() -> None:
    df = pd.read_parquet(IN)
    df = normalize_names(df)
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    counts = pd.concat([df.HomeTeam, df.AwayTeam]).value_counts()
    # the minimum amount of matches for a team to have played in a 38-game season is 38
    # all the season in current dataset are finished, so we can safely assume that any team with less than 38 matches is suspicious
    suspicious = counts[counts < 38]
    print(f"rows {len(df):,}  teams {counts.size}")
    if len(suspicious):
        print("SUSPICIOUS (appears <38 times -- check for an unmapped alias):")
        print(suspicious.to_string())
    else:
        print("no suspicious low-count team names")


if __name__ == "__main__":  
    main()
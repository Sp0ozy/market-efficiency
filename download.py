from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://www.football-data.co.uk/mmz4281"
DIV = "E0"
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2009, 2026)]  # '0910' .. '2526'

RAW = Path("data/raw")
OUT = Path("data/matches.parquet")

# Older seasons don't have the PSC* (Pinnacle closing) columns at all --
# we keep whichever of these actually exist in a given file.
KEEP = [
    "Div", "season", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "PSH", "PSD", "PSA",      # Pinnacle pre-match
    "PSCH", "PSCD", "PSCA",   # Pinnacle CLOSING <- the benchmark
]


def fetch(season: str) -> Path:
    """Download one season's CSV into the cache. No-op if already cached."""
    path = RAW / f"{DIV}_{season}.csv"
    if not path.exists():
        url = f"{BASE}/{season}/{DIV}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            path.write_bytes(r.read())
        time.sleep(0.5) 
    return path


def load(path: Path, season: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df = df.assign(season=season, Div=DIV)
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")  # UK date format
    return df


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in SEASONS:
        path = fetch(season)
        frames.append(load(path, season))

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    data.to_parquet(OUT, index=False)

    # ---- read this: it defines what you can even measure later ----------
    print(f"rows {len(data):,}   {data.Date.min().date()} -> {data.Date.max().date()}")

    print("\nPinnacle CLOSING coverage by season:")
    coverage = data.assign(has_close=data["PSCH"].notna()).groupby("season")["has_close"].mean()
    print(coverage.round(2).to_string())

    print("\nteam names -- scan for duplicates of the same club:")
    print(sorted(pd.concat([data.HomeTeam, data.AwayTeam]).unique()))
    


if __name__ == "__main__":
    main()
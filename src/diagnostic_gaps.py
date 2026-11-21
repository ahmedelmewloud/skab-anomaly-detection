from pathlib import Path
import pandas as pd

DATA = Path("data/raw/SKAB/data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

rows = []
for path in sorted(DATA.rglob("*.csv")):
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    steps = df["datetime"].diff().dt.total_seconds().dropna().round()
    gap_len = (steps[steps > 1] - 1).astype(int)       # missing seconds per gap
    dist = gap_len.index.to_series().diff().dropna()   # rows between two gaps
    rows.append({
        "file": str(path.relative_to(DATA)),
        "n_gaps": len(gap_len),
        "gaps_1s": int((gap_len == 1).sum()),
        "gaps_2_to_5s": int(((gap_len >= 2) & (gap_len <= 5)).sum()),
        "gaps_over_5s": int((gap_len > 5).sum()),
        "longest_gap_s": int(gap_len.max()) if len(gap_len) else 0,
        "median_rows_between_gaps": dist.median() if len(dist) else None,
        "std_rows_between_gaps": round(dist.std(), 1) if len(dist) > 1 else None,
    })

res = pd.DataFrame(rows)
res.to_csv(OUT / "journal_gaps.csv", index=False)
print(res.to_string())

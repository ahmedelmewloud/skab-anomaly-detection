from pathlib import Path
import pandas as pd

DATA = Path("data/raw/SKAB/data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
LABELS = ["anomaly", "changepoint"]  # labels: for evaluation only


def diagnose(path: Path) -> dict:
    df = pd.read_csv(path, sep=";")
    n_rows = len(df)

    # 1. Dates: convert and COUNT the failures
    t = pd.to_datetime(df["datetime"], errors="coerce")
    n_nat = int(t.isna().sum())
    df = df.assign(datetime=t).dropna(subset=["datetime"])

    # 2. Duplicates and order (before sorting!)
    n_dup = int(df["datetime"].duplicated().sum())
    monotone = bool(df["datetime"].is_monotonic_increasing)

    # 3. Sort and deduplicate
    df = (df.drop_duplicates(subset="datetime")
            .sort_values("datetime")
            .set_index("datetime"))

    # 4. Gaps: differences between consecutive timestamps
    steps = df.index.to_series().diff().dropna()
    median_step = steps.median()
    gap_mask = steps > 1.5 * median_step
    n_gaps = int(gap_mask.sum())
    n_missing_steps = int(((steps[gap_mask] / median_step) - 1).round().sum())

    # 5. Completeness = observed / expected
    duration = df.index[-1] - df.index[0]
    expected = int(round(duration / median_step)) + 1
    completeness = len(df) / expected

    # 6. Missing values in the sensors
    sensors = [c for c in df.columns if c not in LABELS]
    n_nan = int(df[sensors].isna().sum().sum())

    return {
        "file": str(path.relative_to(DATA)),
        "rows": n_rows,
        "bad_dates": n_nat,
        "duplicates": n_dup,
        "monotone": monotone,
        "median_step_s": median_step.total_seconds(),
        "gaps": n_gaps,
        "missing_steps": n_missing_steps,
        "completeness": round(completeness, 4),
        "nan_values": n_nan,
        "n_sensors": len(sensors),
    }


if __name__ == "__main__":
    files = sorted(DATA.rglob("*.csv"))
    journal = pd.DataFrame([diagnose(f) for f in files])
    journal.to_csv(OUT / "journal_diagnostic.csv", index=False)
    print(journal.to_string())

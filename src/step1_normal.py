import warnings
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

PATH = Path("data/raw/SKAB/data/anomaly-free/anomaly-free.csv")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

df = (pd.read_csv(PATH, sep=";", parse_dates=["datetime"])
        .set_index("datetime").sort_index())

# ---- 1. Profile of the normal (the reference) ----
med = df.median()
profile = pd.DataFrame({
    "median": med,
    "mad": (df - med).abs().median(),
    "mean": df.mean(),
    "std": df.std(),
    "min": df.min(),
    "max": df.max(),
})
profile.to_csv(OUT / "normal_profile.csv")
print("=== NORMAL PROFILE ===")
print(profile.round(3).to_string())

# ---- 2. Stationarity: ADF + KPSS read together ----
def verdict(adf_ok, kpss_ok):
    if adf_ok and kpss_ok:
        return "stationary"
    if adf_ok and not kpss_ok:
        return "stationary by difference -> difference"
    if not adf_ok and kpss_ok:
        return "around a trend -> remove trend"
    return "non-stationary -> difference"

n = len(df) // 3
rows = []
for c in df.columns:
    x = df[c]
    p_adf = adfuller(x, autolag="AIC")[1]
    p_kpss = kpss(x, regression="c", nlags="auto")[1]
    # simple drift check: mean of last third vs first third, in std units
    drift = (x.iloc[-n:].mean() - x.iloc[:n].mean()) / x.std()
    rows.append({
        "sensor": c,
        "p_ADF": round(p_adf, 4),
        "p_KPSS": round(p_kpss, 4),
        "drift_in_std": round(drift, 2),
        "verdict": verdict(p_adf < 0.05, p_kpss >= 0.05),
    })
stat = pd.DataFrame(rows)
stat.to_csv(OUT / "stationarity_normal.csv", index=False)
print("\n=== STATIONARITY ===")
print(stat.to_string(index=False))

# ---- 3. Visual check: signal + rolling mean (300 s) ----
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for ax, c in zip(axes.ravel(), df.columns):
    ax.plot(df.index, df[c], lw=0.4, alpha=0.6)
    ax.plot(df.index, df[c].rolling(300).mean(), lw=1.5, color="red")
    ax.set_title(c, fontsize=9)
fig.suptitle("Normal file: signal (blue) and rolling mean over 300 s (red)")
fig.tight_layout()
fig.savefig(OUT / "normal_signals.png", dpi=120)
print("\nFigure saved: results/normal_signals.png")

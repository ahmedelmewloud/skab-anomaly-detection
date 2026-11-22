from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/raw/SKAB/data")
NORMAL = ROOT / "anomaly-free" / "anomaly-free.csv"
DEV = sorted((ROOT / "valve1").glob("*.csv"))   # développement ; valve2 et other = test, NON utilisés
OUT = Path("results")
OUT.mkdir(exist_ok=True)
LABELS = ["anomaly", "changepoint"]
VAR_TARGET = 0.90
CALIB = [60, 120, 300]                 # lignes (≈ secondes) de calibration au début de chaque fichier
MODES = ["centre", "centre_echelle"]   # ce que l'on recalibre
QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
MAX_MULTS = [1.25, 1.5, 2.0]
PERSIST = [1, 3, 5, 10, 20, 30]
MIN_RECALL, FA_BUDGET = 0.9, 2.0       # même règle qu'en A


def load(path):
    return (pd.read_csv(path, sep=";", parse_dates=["datetime"])
              .set_index("datetime").sort_index())


# ---- 1. Axes de l'ACP : appris sur anomaly-free UNIQUEMENT (comme en A) ----
normal = load(NORMAL)
sensors = [c for c in normal.columns if c not in LABELS]
scaler = StandardScaler().fit(normal[sensors])
Xn = scaler.transform(normal[sensors])
cum = np.cumsum(PCA().fit(Xn).explained_variance_ratio_)
pca = PCA(n_components=int(np.argmax(cum >= VAR_TARGET)) + 1).fit(Xn)


def spe_calibrated(df, n_calib, mode):
    """SPE d'un fichier, recalibré avec ses n_calib premières lignes (passé seulement)."""
    X = df[sensors].values
    mu = X[:n_calib].mean(axis=0)
    if mode == "centre":
        sd = scaler.scale_                                       # échelle du normal
    else:
        sd = np.maximum(X[:n_calib].std(axis=0), scaler.scale_)  # jamais plus petite que celle du normal
    Xc = (X - mu) / sd
    t = Xc @ pca.components_.T
    return ((Xc - t @ pca.components_) ** 2).sum(axis=1)


# ---- 2. Vérification (dev-set seulement) : la calibration est-elle bien « normale » ? ----
firsts = []
for p in DEV:
    d = load(p)
    a = np.where(d["anomaly"].values == 1)[0]
    firsts.append(int(a[0]))
print(f"Dev-set : {len(DEV)} fichiers. 1re anomalie à la ligne : min = {min(firsts)}, médiane = {int(np.median(firsts))}")
assert max(CALIB) < min(firsts), "fenêtre de calibration contaminée par une anomalie"


def evaluate(items, tau, k, hours):
    n_ev = n_det = n_false = 0
    delays = []
    for idx, s, y in items:
        alert = pd.Series((s > tau).astype(int)).rolling(k).sum().eq(k).values   # causal
        d = np.diff(np.r_[0, y, 0])
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            n_ev += 1
            hit = np.where(alert[a:b])[0]
            if len(hit):
                n_det += 1
                delays.append((idx[a + hit[0]] - idx[a]).total_seconds())
        starts = alert & ~np.r_[False, alert[:-1]]
        n_false += int((starts & (y == 0)).sum())
    return {"recall": n_det / n_ev,
            "median_delay_s": float(np.median(delays)) if delays else np.nan,
            "false_per_hour": n_false / hours, "false_alerts": n_false}


rows = []
for mode in MODES:
    for n_calib in CALIB:
        items = []
        for p in DEV:
            d = load(p)
            s = spe_calibrated(d, n_calib, mode)
            # on évalue après la calibration seulement
            items.append((d.index[n_calib:], s[n_calib:], d["anomaly"].values[n_calib:].astype(int)))
        pooled = np.concatenate([s[y == 0] for _, s, y in items])
        hours = len(pooled) / 3600
        taus = {f"q{q}": float(np.quantile(pooled, q)) for q in QUANTILES}
        taus.update({f"{m}xmax": m * float(pooled.max()) for m in MAX_MULTS})
        for lab, tau in taus.items():
            for k in PERSIST:
                rows.append({"mode": mode, "calib_s": n_calib, "seuil": lab, "tau": tau,
                             "persist": k, "heures_normales": hours,
                             **evaluate(items, tau, k, hours)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "step4B_tradeoff_dev.csv", index=False)

# ---- 3. Pour chaque (mode, calibration) : meilleur réglage selon la règle ----
print(f"\n=== RÈGLE : rappel >= {MIN_RECALL} et fausses alertes/h <= {FA_BUDGET} ===")
best_rows = []
for (mode, n_calib), g in res.groupby(["mode", "calib_s"]):
    ok = g[(g.recall >= MIN_RECALL) & (g.false_per_hour <= FA_BUDGET)]
    if len(ok):
        b = ok.sort_values(["median_delay_s", "false_per_hour"]).iloc[0]; regle = "oui"
    else:
        b = g[g.false_per_hour <= FA_BUDGET].sort_values(
            ["recall", "median_delay_s"], ascending=[False, True]).iloc[0]; regle = "non (repli)"
    best_rows.append({**b.to_dict(), "regle_respectee": regle})
best = pd.DataFrame(best_rows)
show = ["mode", "calib_s", "seuil", "tau", "persist", "recall", "median_delay_s",
        "false_per_hour", "regle_respectee"]
print(best[show].round(2).to_string(index=False))

# Choix final : on privilégie les réglages qui respectent la règle, puis rappel, puis retard
best["_r"] = (best.regle_respectee == "oui").astype(int)
final = best.sort_values(["_r", "recall", "median_delay_s"], ascending=[False, False, True]).iloc[0]
print("\n=== RÉGLAGE RETENU POUR B (dev-set) ===")
print(final[show].to_frame().T.round(2).to_string(index=False))
pd.DataFrame([final[show]]).to_csv(OUT / "step4B_chosen.csv", index=False)

# ---- 4. Figure : compromis pour le réglage de calibration retenu ----
g = res[(res["mode"] == final["mode"]) & (res.calib_s == final.calib_s)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k in PERSIST:
    r = g[g.persist == k].sort_values("tau")
    axes[0].plot(r.false_per_hour, r.median_delay_s, marker="o", label=f"k = {k} s")
    axes[1].plot(r.false_per_hour, r.recall, marker="o", label=f"k = {k} s")
for ax, yl in zip(axes, ["retard médian d'alerte (s)", "rappel par événement"]):
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("fausses alertes par heure")
    ax.set_ylabel(yl)
axes[1].axhline(MIN_RECALL, ls="--", color="gray")
axes[1].axvline(FA_BUDGET, ls="--", color="gray")
axes[0].legend()
fig.suptitle(f"Expérience B ({final['mode']}, calibration {int(final.calib_s)} lignes) : compromis (SPE, valve1)")
fig.tight_layout()
fig.savefig(OUT / "step4B_tradeoff.png", dpi=120)
print("\nFigure : results/step4B_tradeoff.png")

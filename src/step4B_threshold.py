import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (DEV, OUT, MIN_RECALL, FA_BUDGET, load, fit_normal, spe_calibrated,
                    alerts, counts, choose)

# DEV = valve1 ; valve2 et other = test, NON utilisés
CALIB = [60, 120, 300]                 # lignes (≈ secondes) de calibration au début de chaque fichier
MODES = ["centre", "centre_echelle"]   # ce que l'on recalibre
QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
MAX_MULTS = [1.25, 1.5, 2.0]
PERSIST = [1, 3, 5, 10, 20, 30]

# ---- 1. Axes de l'ACP : appris sur anomaly-free UNIQUEMENT (comme en A) ----
_, sensors, scaler, pca, _ = fit_normal()

# ---- 2. Vérification (dev-set seulement) : la calibration est-elle bien « normale » ? ----
dev = [load(p) for p in DEV]
firsts = [int(np.where(d["anomaly"].values == 1)[0][0]) for d in dev]
print(f"Dev-set : {len(DEV)} fichiers. 1re anomalie à la ligne : min = {min(firsts)}, médiane = {int(np.median(firsts))}")
assert max(CALIB) < min(firsts), "fenêtre de calibration contaminée par une anomalie"


def evaluate(items, tau, k, hours):
    c = counts([(idx, y, alerts(s, tau, k)) for idx, s, y in items])
    n_false = c["n_alert"] - c["n_just"]
    return {"recall": c["n_det"] / c["n_ev"],
            "median_delay_s": float(np.median(c["delays"])) if c["delays"] else np.nan,
            "false_per_hour": n_false / hours, "false_alerts": n_false}


rows = []
for mode in MODES:
    for n_calib in CALIB:
        items = []
        for d in dev:
            s = spe_calibrated(d, sensors, scaler, pca, n_calib, mode)
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
    b, rule_ok = choose(g)
    best_rows.append({**b.to_dict(), "regle_respectee": "oui" if rule_ok else "non (repli)"})
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (DEV, OUT, MIN_RECALL, FA_BUDGET, load, fit_normal, spe_score,
                    alerts, events, counts, choose)

# DEV = valve1 ; valve2 et other = test, NON utilisés
QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
MAX_MULTS = [1.25, 1.5, 2.0]
PERSIST = [1, 3, 5, 10, 20, 30]

# ---- 1. Modèle A : le normal est appris sur anomaly-free UNIQUEMENT ----
_, sensors, scaler, pca, _ = fit_normal()

# ---- 2. Scores du dev-set ----
dev = []
for p in DEV:
    d = load(p)
    dev.append((d.index, spe_score(d, sensors, scaler, pca), d["anomaly"].values.astype(int)))
pooled_normal = np.concatenate([s[y == 0] for _, s, y in dev])
hours_normal = len(pooled_normal) / 3600
n_events = sum(len(list(events(y))) for _, _, y in dev)
print(f"Dev-set : {len(dev)} fichiers, {n_events} événements, {hours_normal:.1f} h de fonctionnement normal")

# ---- 3. Seuils candidats, exprimés sur l'échelle du dev-set ----
taus = {f"q{q}": float(np.quantile(pooled_normal, q)) for q in QUANTILES}
mx = float(pooled_normal.max())
taus.update({f"{m}xmax": m * mx for m in MAX_MULTS})
labels = list(taus)
print("Seuils testés :", {k: round(v) for k, v in taus.items()})


def evaluate(tau, k):
    c = counts([(idx, y, alerts(s, tau, k)) for idx, s, y in dev])
    n_false = c["n_alert"] - c["n_just"]
    return {"recall": c["n_det"] / c["n_ev"],
            "median_delay_s": float(np.median(c["delays"])) if c["delays"] else np.nan,
            "false_per_hour": n_false / hours_normal,
            "false_alerts": n_false}


rows = [{"seuil": lab, "tau": tau, "persist": k, **evaluate(tau, k)}
        for lab, tau in taus.items() for k in PERSIST]
res = pd.DataFrame(rows)
res.to_csv(OUT / "step4A_tradeoff_dev.csv", index=False)

for col, title in [("false_per_hour", "FAUSSES ALERTES PAR HEURE"),
                   ("median_delay_s", "RETARD MÉDIAN (secondes)"),
                   ("recall", "RAPPEL PAR ÉVÉNEMENT")]:
    print(f"\n=== {title} (lignes = persistance k, colonnes = seuil) ===")
    print(res.pivot(index="persist", columns="seuil", values=col)
             .reindex(columns=labels).round(2).to_string())

# ---- 4. Choix selon la règle écrite à l'avance ----
print(f"\n=== RÈGLE : rappel >= {MIN_RECALL} et fausses alertes/h <= {FA_BUDGET} ===")
best, rule_ok = choose(res)
if not rule_ok:
    print("Aucun réglage ne respecte la règle. Meilleurs rappels sous le budget de fausses alertes :")
    under = res[res.false_per_hour <= FA_BUDGET].sort_values(
        ["recall", "median_delay_s"], ascending=[False, True]).head(5)
    print(under.round(2).to_string(index=False))
print("Réglage retenu :" if rule_ok else "Réglage retenu (repli) :")
print(best.to_frame().T.round(2).to_string(index=False))
pd.DataFrame([{**best.to_dict(), "regle_respectee": "oui" if rule_ok else "non (repli)"}]).to_csv(
    OUT / "step4A_chosen.csv", index=False)

# ---- 5. Figure ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k in PERSIST:
    r = res[res.persist == k].sort_values("tau")
    axes[0].plot(r.false_per_hour, r.median_delay_s, marker="o", label=f"k = {k} s")
    axes[1].plot(r.false_per_hour, r.recall, marker="o", label=f"k = {k} s")
for ax, yl in zip(axes, ["retard médian d'alerte (s)", "rappel par événement"]):
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("fausses alertes par heure")
    ax.set_ylabel(yl)
axes[1].axhline(MIN_RECALL, ls="--", color="gray")
axes[1].axvline(FA_BUDGET, ls="--", color="gray")
axes[0].legend()
fig.suptitle("Expérience A : compromis fausses alertes / retard (SPE, valve1)")
fig.tight_layout()
fig.savefig(OUT / "step4A_tradeoff.png", dpi=120)
print("\nFigure : results/step4A_tradeoff.png")

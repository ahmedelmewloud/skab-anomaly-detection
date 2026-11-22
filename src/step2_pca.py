import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, OUT, load, fit_normal, pca_scores

DEMO = ROOT / "valve1" / "1.csv"   # exploration visuelle seulement

# ---- 1. Apprendre le normal (fichier sans anomalie uniquement) ----
normal, sensors, scaler, pca, cum = fit_normal()
k = pca.n_components_
Xn = scaler.transform(normal[sensors])

print(f"Composantes gardées : {k} sur {len(sensors)} "
      f"({cum[k-1]*100:.1f} % de la variance)")
print("Variance cumulée par nombre de composantes :", np.round(cum, 3))

loadings = pd.DataFrame(pca.components_.T, index=sensors,
                        columns=[f"PC{i+1}" for i in range(k)])
loadings.to_csv(OUT / "step2_pca_loadings.csv")
print("\n=== POIDS DES CAPTEURS SUR CHAQUE AXE ===")
print(loadings.round(2).to_string())

# ---- 2. Scores sur le normal (données d'apprentissage) ----
t2_n, spe_n, _, _ = pca_scores(Xn, pca)
summary = pd.DataFrame({
    "T2": [np.median(t2_n), np.quantile(t2_n, 0.99), t2_n.max()],
    "SPE": [np.median(spe_n), np.quantile(spe_n, 0.99), spe_n.max()],
}, index=["mediane", "q99", "max"])
summary.to_csv(OUT / "step2_normal_scores.csv")
print("\n=== SCORES SUR LE FICHIER NORMAL ===")
print(summary.round(2).to_string())

# ---- 3. Un fichier d'anomalie, pour regarder (pas pour régler quoi que ce soit) ----
demo = load(DEMO)
Xd = scaler.transform(demo[sensors])             # transform seulement, jamais fit
t2_d, spe_d, c_t2, c_spe = pca_scores(Xd, pca)
anom = demo["anomaly"].values == 1

print(f"\n=== FICHIER {DEMO.parent.name}/{DEMO.name} ===")
print(pd.DataFrame({
    "T2 médian": [np.median(t2_d[~anom]), np.median(t2_d[anom])],
    "SPE médian": [np.median(spe_d[~anom]), np.median(spe_d[anom])],
}, index=["avant/hors anomalie", "pendant anomalie"]).round(2).to_string())

share = pd.DataFrame({
    "part_SPE_hors_anomalie": c_spe[~anom].mean(axis=0) / c_spe[~anom].mean(axis=0).sum(),
    "part_SPE_pendant_anomalie": c_spe[anom].mean(axis=0) / c_spe[anom].mean(axis=0).sum(),
}, index=sensors)
share.to_csv(OUT / "step2_contributions_demo.csv")
print("\n=== CONTRIBUTION DE CHAQUE CAPTEUR AU SPE ===")
print(share.round(2).to_string())

# ---- 4. Figure ----
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, s, name in zip(axes, [t2_d, spe_d], ["T2", "SPE"]):
    ax.semilogy(demo.index, s, lw=0.6)
    ax.fill_between(demo.index, s.min(), s.max(), where=anom,
                    color="red", alpha=0.2, label="anomalie (étiquette)")
    ax.set_ylabel(name)
axes[0].legend()
fig.suptitle(f"T2 et SPE, fichier {DEMO.parent.name}/{DEMO.name} (échelle log)")
fig.tight_layout()
fig.savefig(OUT / "step2_scores_demo.png", dpi=120)
print("\nFigure : results/step2_scores_demo.png")

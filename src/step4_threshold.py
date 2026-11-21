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
MULTS = [1, 2, 3, 5, 10, 20, 50]     # seuil = m x q99 du SPE sur le normal
PERSIST = [1, 3, 5, 10, 20, 30]      # secondes d'affilée au-dessus du seuil
MIN_RECALL = 0.9                     # règle de choix (hypothèse à justifier)
FA_BUDGET = 2.0                      # fausses alertes par heure maximum


def load(path):
    return (pd.read_csv(path, sep=";", parse_dates=["datetime"])
              .set_index("datetime").sort_index())


# ---- 1. Modèle du normal (fichier normal UNIQUEMENT), comme à l'étape 2 ----
normal = load(NORMAL)
sensors = [c for c in normal.columns if c not in LABELS]
scaler = StandardScaler().fit(normal[sensors])
Xn = scaler.transform(normal[sensors])
cum = np.cumsum(PCA().fit(Xn).explained_variance_ratio_)
k_pca = int(np.argmax(cum >= VAR_TARGET)) + 1
pca = PCA(n_components=k_pca).fit(Xn)


def spe_score(df):
    Xc = scaler.transform(df[sensors]) - pca.mean_
    t = Xc @ pca.components_.T
    return ((Xc - t @ pca.components_) ** 2).sum(axis=1)


ref = np.quantile(spe_score(normal), 0.99)
print(f"ACP : {k_pca} composantes | référence q99 du SPE (normal) = {ref:.2f}")

# ---- 2. Scores des fichiers de développement ----
dev = []
for p in DEV:
    d = load(p)
    dev.append((d.index, spe_score(d), d["anomaly"].values.astype(int)))
hours_normal = sum((y == 0).sum() for _, _, y in dev) / 3600
n_events_total = sum(int(((np.diff(np.r_[0, y, 0])) == 1).sum()) for _, _, y in dev)
print(f"Jeu de développement : {len(dev)} fichiers, {n_events_total} événements, "
      f"{hours_normal:.1f} h de fonctionnement normal")


def evaluate(tau, k):
    n_ev = n_det = n_false = 0
    delays = []
    for idx, s, y in dev:
        above = (s > tau).astype(int)
        alert = pd.Series(above).rolling(k).sum().eq(k).values   # causal : passé seulement
        d = np.diff(np.r_[0, y, 0])
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(starts, ends):
            n_ev += 1
            hit = np.where(alert[a:b])[0]
            if len(hit):
                n_det += 1
                delays.append((idx[a + hit[0]] - idx[a]).total_seconds())
        edges = alert & ~np.r_[False, alert[:-1]]                 # début de chaque alerte
        n_false += int((edges & (y == 0)).sum())
    return {
        "recall": n_det / n_ev,
        "median_delay_s": float(np.median(delays)) if delays else np.nan,
        "false_per_hour": n_false / hours_normal,
        "false_alerts": n_false,
    }


rows = []
for m in MULTS:
    for k in PERSIST:
        rows.append({"mult": m, "persist": k, **evaluate(m * ref, k)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "step4_tradeoff_dev.csv", index=False)

for col, title in [("false_per_hour", "FAUSSES ALERTES PAR HEURE"),
                   ("median_delay_s", "RETARD MÉDIAN (secondes)"),
                   ("recall", "RAPPEL PAR ÉVÉNEMENT")]:
    print(f"\n=== {title} (lignes = persistance k, colonnes = multiple m) ===")
    print(res.pivot(index="persist", columns="mult", values=col).round(2).to_string())

# ---- 3. Choix selon la règle écrite à l'avance ----
ok = res[(res.recall >= MIN_RECALL) & (res.false_per_hour <= FA_BUDGET)]
print(f"\n=== RÈGLE : rappel >= {MIN_RECALL} et fausses alertes/h <= {FA_BUDGET} ===")
if ok.empty:
    print("Aucun réglage ne respecte la règle : il faut la discuter.")
    best = None
else:
    best = ok.sort_values(["median_delay_s", "false_per_hour"]).iloc[0]
    print(f"Réglage retenu : seuil = {int(best.mult)} x q99 (= {best.mult * ref:.1f}), "
          f"persistance = {int(best.persist)} s")
    print(best.round(2).to_string())
    pd.DataFrame([best]).to_csv(OUT / "step4_chosen.csv", index=False)

# ---- 4. Figure : le compromis ----
fig, ax = plt.subplots(figsize=(8, 5))
for k in PERSIST:
    r = res[res.persist == k].sort_values("mult")
    ax.plot(r.false_per_hour, r.median_delay_s, marker="o", label=f"k = {k} s")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_xlabel("fausses alertes par heure")
ax.set_ylabel("retard médian d'alerte (s)")
ax.set_title("Compromis fausses alertes / retard (SPE, valve1)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "step4_tradeoff.png", dpi=120)
print("\nFigure : results/step4_tradeoff.png")

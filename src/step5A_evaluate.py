from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/raw/SKAB/data")
NORMAL = ROOT / "anomaly-free" / "anomaly-free.csv"
TEST = sorted((ROOT / "valve2").glob("*.csv")) + sorted((ROOT / "other").glob("*.csv"))
OUT = Path("results")
LABELS = ["anomaly", "changepoint"]
VAR_TARGET = 0.90
MIN_RECALL, FA_BUDGET = 0.9, 2.0      # même règle qu'à l'étape 4
N_RANDOM, SEED = 500, 42


def load(path):
    return (pd.read_csv(path, sep=";", parse_dates=["datetime"])
              .set_index("datetime").sort_index())


# ---- 1. Modèle A : appris sur anomaly-free UNIQUEMENT ----
normal = load(NORMAL)
sensors = [c for c in normal.columns if c not in LABELS]
scaler = StandardScaler().fit(normal[sensors])
Xn = scaler.transform(normal[sensors])
cum = np.cumsum(PCA().fit(Xn).explained_variance_ratio_)
pca = PCA(n_components=int(np.argmax(cum >= VAR_TARGET)) + 1).fit(Xn)


def spe_score(df):
    Xc = scaler.transform(df[sensors]) - pca.mean_
    t = Xc @ pca.components_.T
    return ((Xc - t @ pca.components_) ** 2).sum(axis=1)


# ---- 2. Réglage fixé sur le dev-set (étape 4), relu dans le fichier ----
res = pd.read_csv(OUT / "step4A_tradeoff_dev.csv")
ok = res[(res.recall >= MIN_RECALL) & (res.false_per_hour <= FA_BUDGET)]
if len(ok):
    best = ok.sort_values(["median_delay_s", "false_per_hour"]).iloc[0]
else:
    best = res[res.false_per_hour <= FA_BUDGET].sort_values(
        ["recall", "median_delay_s"], ascending=[False, True]).iloc[0]
tau, k = float(best["tau"]), int(best["persist"])
print(f"Réglage fixé sur le dev-set : seuil = {tau:.1f} ({best['seuil']}), persistance = {k} s")
print(f"  (sur le dev-set : rappel {best['recall']:.2f}, {best['false_per_hour']:.2f} fausses alertes/h)")

# ---- 3. Test : alertes réelles ----
items = []
for p in TEST:
    d = load(p)
    s = spe_score(d)
    y = d["anomaly"].values.astype(int)
    alert = pd.Series((s > tau).astype(int)).rolling(k).sum().eq(k).values   # causal
    items.append((f"{p.parent.name}/{p.name}", d.index, y, alert, float((s > tau).mean())))


def counts(its):
    c = {"n_ev": 0, "n_det": 0, "n_alert": 0, "n_just": 0, "delays": [], "normal_s": 0}
    for idx, y, alert in its:
        d = np.diff(np.r_[0, y, 0])
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            c["n_ev"] += 1
            hit = np.where(alert[a:b])[0]
            if len(hit):
                c["n_det"] += 1
                c["delays"].append((idx[a + hit[0]] - idx[a]).total_seconds())
        starts = alert & ~np.r_[False, alert[:-1]]
        c["n_alert"] += int(starts.sum())
        c["n_just"] += int((starts & (y == 1)).sum())
        c["normal_s"] += int((y == 0).sum())
    return c


def summarize(c):
    return {"rappel": c["n_det"] / c["n_ev"] if c["n_ev"] else np.nan,
            "precision": c["n_just"] / c["n_alert"] if c["n_alert"] else np.nan,
            "retard_median_s": float(np.median(c["delays"])) if c["delays"] else np.nan,
            "fausses_alertes_par_h": (c["n_alert"] - c["n_just"]) / (c["normal_s"] / 3600),
            "n_alertes": c["n_alert"], "n_evenements": c["n_ev"]}


# Détail par fichier
rows = []
for name, idx, y, alert, frac in items:
    c = counts([(idx, y, alert)])
    rows.append({"fichier": name, "evenements": c["n_ev"], "detectes": c["n_det"],
                 "alertes": c["n_alert"], "justifiees": c["n_just"],
                 "retard_s": float(np.median(c["delays"])) if c["delays"] else np.nan,
                 "part_au_dessus_seuil": round(frac, 3)})
per_file = pd.DataFrame(rows)
per_file.to_csv(OUT / "step5A_test_events.csv", index=False)
print("\n=== DÉTAIL PAR FICHIER DE TEST ===")
print(per_file.to_string(index=False))

real = summarize(counts([(i, y, a) for _, i, y, a, _ in items]))

# ---- 4. Détecteur aléatoire : alertes réelles décalées circulairement ----
rng = np.random.default_rng(SEED)
rand = pd.DataFrame([
    summarize(counts([(i, y, np.roll(a, rng.integers(0, len(a)))) for _, i, y, a, _ in items]))
    for _ in range(N_RANDOM)])

cols = ["rappel", "precision", "retard_median_s", "fausses_alertes_par_h"]
summary = pd.DataFrame({
    "detecteur_A": {c: real[c] for c in cols},
    "aleatoire_moyenne": rand[cols].mean(),
    "aleatoire_q05": rand[cols].quantile(0.05),
    "aleatoire_q95": rand[cols].quantile(0.95),
})
summary.to_csv(OUT / "step5A_summary.csv")
print(f"\n=== TEST : {real['n_evenements']} événements, {real['n_alertes']} alertes ===")
print(summary.round(2).to_string())
print("\nPart des 500 tirages aléatoires au moins aussi bons que le détecteur :")
print("  rappel    :", round(float((rand.rappel >= real["rappel"]).mean()), 3))
print("  précision :", round(float((rand.precision.fillna(0) >= np.nan_to_num(real["precision"])).mean()), 3))

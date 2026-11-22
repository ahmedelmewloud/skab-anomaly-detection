"""Étape 6 : incertitude (bootstrap par fichier) et ablation sur le jeu de test.

Tous les réglages sont choisis sur le dev-set (fichiers de l'étape 4) ; le test ne sert qu'à mesurer.
"""
import numpy as np
import pandas as pd
from common import (TEST, OUT, load, fit_normal, spe_score, spe_calibrated, alerts, counts,
                    summarize, choose)

N_BOOT, SEED = 2000, 42
COLS = ["rappel", "precision", "retard_median_s", "fausses_alertes_par_h"]

_, sensors, scaler, pca, _ = fit_normal()
test = [(f"{p.parent.name}/{p.name}", load(p)) for p in TEST]


def run_A(tau, k):
    out = []
    for _, d in test:
        s = spe_score(d, sensors, scaler, pca)
        out.append((d.index, d["anomaly"].values.astype(int), alerts(s, tau, k)))
    return out


def run_B(mode, n_calib, tau, k):
    out = []
    for _, d in test:
        s = spe_calibrated(d, sensors, scaler, pca, n_calib, mode)[n_calib:]
        out.append((d.index[n_calib:], d["anomaly"].values[n_calib:].astype(int), alerts(s, tau, k)))
    return out


def bootstrap(items, rng):
    """IC à 95 % en tirant les fichiers avec remise (un fichier = une expérience indépendante)."""
    n = len(items)
    boot = pd.DataFrame([summarize(counts([items[i] for i in rng.integers(0, n, n)]))
                         for _ in range(N_BOOT)])
    return boot[COLS].quantile(0.025), boot[COLS].quantile(0.975)


# ---- 1. Les deux détecteurs retenus (réglages des étapes 4A et 4B) ----
a = pd.read_csv(OUT / "step4A_chosen.csv").iloc[0]
b = pd.read_csv(OUT / "step4B_chosen.csv").iloc[0]
detectors = {
    "A": run_A(float(a.tau), int(a.persist)),
    "B": run_B(b["mode"], int(b.calib_s), float(b.tau), int(b.persist)),
}

rng = np.random.default_rng(SEED)
rows = []
for name, items in detectors.items():
    real = summarize(counts(items))
    lo, hi = bootstrap(items, rng)
    for c in COLS:
        rows.append({"detecteur": name, "metrique": c, "valeur": real[c],
                     "ic95_bas": lo[c], "ic95_haut": hi[c]})
ci = pd.DataFrame(rows)
ci.to_csv(OUT / "step6_uncertainty.csv", index=False)
print(f"=== INCERTITUDE : bootstrap sur les {len(test)} fichiers de test ({N_BOOT} tirages) ===")
print(ci.round(2).to_string(index=False))

# ---- 2. Ablation : on retire un ingrédient de B à la fois, réglage re-choisi sur le dev-set ----
dev = pd.read_csv(OUT / "step4B_tradeoff_dev.csv")


def dev_choice(g):
    best, rule_ok = choose(g)
    return best


variants = {
    "B retenu (centre+échelle, 300 s, persistance)": dev[(dev["mode"] == b["mode"]) & (dev.calib_s == b.calib_s)],
    "B sans recalibrage de l'échelle (centre seul, 300 s)": dev[(dev["mode"] == "centre") & (dev.calib_s == b.calib_s)],
    "B avec calibration courte (120 s)": dev[(dev["mode"] == b["mode"]) & (dev.calib_s == 120)],
    "B avec calibration très courte (60 s)": dev[(dev["mode"] == b["mode"]) & (dev.calib_s == 60)],
    "B sans persistance (k = 1)": dev[(dev["mode"] == b["mode"]) & (dev.calib_s == b.calib_s) & (dev.persist == 1)],
}
rows = []
for label, g in variants.items():
    ch = dev_choice(g)
    items = run_B(ch["mode"], int(ch.calib_s), float(ch.tau), int(ch.persist))
    r = summarize(counts(items))
    rows.append({"variante": label, "mode": ch["mode"], "calib_s": int(ch.calib_s),
                 "seuil": ch.seuil, "persist": int(ch.persist),
                 "rappel_dev": ch.recall, "fa_h_dev": ch.false_per_hour,
                 **{c: r[c] for c in COLS}})
ch = a
items = detectors["A"]
r = summarize(counts(items))
rows.append({"variante": "A : aucun recalibrage (normal appris une fois)", "mode": "-", "calib_s": 0,
             "seuil": ch.seuil, "persist": int(ch.persist), "rappel_dev": ch.recall,
             "fa_h_dev": ch.false_per_hour, **{c: r[c] for c in COLS}})
abl = pd.DataFrame(rows)
abl.to_csv(OUT / "step6_ablation.csv", index=False)
print("\n=== ABLATION (réglages choisis sur dev, mesurés sur test) ===")
print(abl.round(2).to_string(index=False))

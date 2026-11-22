import numpy as np
import pandas as pd
from common import TEST, OUT, load, fit_normal, spe_score, alerts, counts, summarize, random_baseline

N_RANDOM, SEED = 500, 42

# ---- 1. Modèle A : appris sur anomaly-free UNIQUEMENT ----
_, sensors, scaler, pca, _ = fit_normal()

# ---- 2. Réglage fixé sur le dev-set (étape 4A) ----
best = pd.read_csv(OUT / "step4A_chosen.csv").iloc[0]
tau, k = float(best["tau"]), int(best["persist"])
print(f"Réglage fixé sur le dev-set : seuil = {tau:.1f} ({best['seuil']}), persistance = {k} s"
      f" | règle respectée : {best['regle_respectee']}")
print(f"  (sur le dev-set : rappel {best['recall']:.2f}, {best['false_per_hour']:.2f} fausses alertes/h)")

# ---- 3. Test : alertes réelles ----
items = []
for p in TEST:
    d = load(p)
    s = spe_score(d, sensors, scaler, pca)
    y = d["anomaly"].values.astype(int)
    items.append((f"{p.parent.name}/{p.name}", d.index, y, alerts(s, tau, k), float((s > tau).mean())))

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

triples = [(i, y, a) for _, i, y, a, _ in items]
real = summarize(counts(triples))

# ---- 4. Détecteur aléatoire : alertes réelles décalées circulairement ----
rand = random_baseline(triples, N_RANDOM, SEED)

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
print(f"\nPart des {N_RANDOM} tirages aléatoires au moins aussi bons que le détecteur :")
print("  rappel    :", round(float((rand.rappel >= real["rappel"]).mean()), 3))
print("  précision :", round(float((rand.precision.fillna(0) >= np.nan_to_num(real["precision"])).mean()), 3))

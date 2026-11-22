import numpy as np
import pandas as pd
from common import (TEST, OUT, load, fit_normal, spe_calibrated, alerts, counts, summarize,
                    random_baseline)

N_RANDOM, SEED = 500, 42

# ---- 1. Axes de l'ACP : appris sur anomaly-free UNIQUEMENT ----
_, sensors, scaler, pca, _ = fit_normal()

# ---- 2. Réglage fixé sur le dev-set (étape 4B) ----
ch = pd.read_csv(OUT / "step4B_chosen.csv").iloc[0]
mode, n_calib = ch["mode"], int(ch["calib_s"])
tau, k = float(ch["tau"]), int(ch["persist"])
print(f"Réglage fixé sur le dev-set : recalibrage '{mode}' sur {n_calib} lignes, "
      f"seuil = {tau:.2f}, persistance = {k} s")
print(f"  (sur le dev-set : rappel {float(ch['recall']):.2f}, {float(ch['false_per_hour']):.2f} fausses alertes/h)")

# ---- 3. Test : alertes réelles (évaluées après la calibration) ----
items = []
for p in TEST:
    d = load(p)
    s = spe_calibrated(d, sensors, scaler, pca, n_calib, mode)[n_calib:]
    y_all = d["anomaly"].values.astype(int)
    y = y_all[n_calib:]
    items.append((f"{p.parent.name}/{p.name}", d.index[n_calib:], y, alerts(s, tau, k),
                  float((s > tau).mean()), bool(y_all[:n_calib].any())))

rows = []
for name, idx, y, alert, frac, contam in items:
    c = counts([(idx, y, alert)])
    rows.append({"fichier": name, "evenements": c["n_ev"], "detectes": c["n_det"],
                 "alertes": c["n_alert"], "justifiees": c["n_just"],
                 "retard_s": float(np.median(c["delays"])) if c["delays"] else np.nan,
                 "part_au_dessus_seuil": round(frac, 3),
                 "anomalie_dans_calibration": contam})
per_file = pd.DataFrame(rows)
per_file.to_csv(OUT / "step5B_test_events.csv", index=False)
print("\n=== DÉTAIL PAR FICHIER DE TEST ===")
print(per_file.to_string(index=False))
print(f"\nFichiers dont la fenêtre de calibration contient une anomalie : "
      f"{int(per_file.anomalie_dans_calibration.sum())} / {len(per_file)}")

cols = ["rappel", "precision", "retard_median_s", "fausses_alertes_par_h"]


def report(its, label):
    """Détecteur B et détecteur aléatoire sur un ensemble de fichiers."""
    triples = [(i, y, a) for _, i, y, a, _, _ in its]
    real = summarize(counts(triples))
    rand = random_baseline(triples, N_RANDOM, SEED)
    summary = pd.DataFrame({
        "detecteur_B": {c: real[c] for c in cols},
        "aleatoire_moyenne": rand[cols].mean(),
        "aleatoire_q05": rand[cols].quantile(0.05),
        "aleatoire_q95": rand[cols].quantile(0.95),
    })
    print(f"\n=== TEST {label} : {real['n_evenements']} événements, {real['n_alertes']} alertes ===")
    print(summary.round(2).to_string())
    print(f"\nPart des {N_RANDOM} tirages aléatoires au moins aussi bons que le détecteur :")
    print("  rappel    :", round(float((rand.rappel >= real["rappel"]).mean()), 3))
    print("  précision :", round(float((rand.precision.fillna(0) >= np.nan_to_num(real["precision"])).mean()), 3))
    return summary


# ---- 4. Tous les fichiers de test, puis sans ceux dont la calibration est contaminée ----
summary = report(items, "(tous les fichiers)")
summary.to_csv(OUT / "step5B_summary.csv")

clean = [it for it in items if not it[5]]
if len(clean) < len(items):
    summary_clean = report(clean, "(calibration non contaminée)")
    summary_clean.to_csv(OUT / "step5B_summary_clean_calib.csv")

# ---- 5. Comparaison A / B sur le même jeu de test ----
fa = OUT / "step5A_summary.csv"
if fa.exists():
    a = pd.read_csv(fa, index_col=0)["detecteur_A"]
    print("\n=== COMPARAISON A (normal appris sur anomaly-free) / B (recalibrage) ===")
    print(pd.DataFrame({"A": a, "B": summary["detecteur_B"]}).round(2).to_string())

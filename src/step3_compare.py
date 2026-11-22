import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from common import DEV, OUT, load, fit_normal, pca_scores

# DEV = valve1 ; valve2 et other = test, NON utilisés ici
SEED = 42
LAGS = 3          # le résidu utilise les 3 secondes précédentes

# ---------- 1. Apprendre le normal (fichier sans anomalie UNIQUEMENT) ----------
normal, sensors, scaler, pca, _ = fit_normal()
k = pca.n_components_


def scale(df):
    return pd.DataFrame(scaler.transform(df[sensors]), index=df.index, columns=sensors)


Xn = scale(normal)

# Isolation Forest
iso = IsolationForest(n_estimators=200, random_state=SEED).fit(Xn)


# Résidu de prévision : Ridge sur les retards 1..LAGS (strictement causal)
def lagged(X):
    return pd.concat([X.shift(k).add_suffix(f"_lag{k}") for k in range(1, LAGS + 1)], axis=1)


F = lagged(Xn).dropna()
Y = Xn.loc[F.index]
reg = Ridge(alpha=1.0).fit(F, Y)
res_std = (Y - reg.predict(F)).std().values     # échelle du résidu sur le normal


def forecast_score(X):
    F = lagged(X)
    ok = F.notna().all(axis=1)
    s = pd.Series(np.nan, index=X.index)
    r = (X.loc[ok] - reg.predict(F[ok])) / res_std
    s[ok] = (r ** 2).sum(axis=1)
    return s.values


print(f"ACP : {k} composantes | Isolation Forest : 200 arbres | Résidu : Ridge, {LAGS} retards")

# ---------- 2. Comparer sur le jeu de développement ----------
rows = []
for p in DEV:
    d = load(p)
    X = scale(d)
    y = d["anomaly"].values
    t2, spe, _, _ = pca_scores(X.values, pca)
    sc = {
        "T2": t2,
        "SPE": spe,
        "IsolationForest": -iso.score_samples(X),
        "Residu": forecast_score(X),
    }
    ok = ~np.isnan(sc["Residu"])                 # on retire les premières lignes sans retards
    row = {"file": p.name, "anomaly_rate(hasard)": round(float(y[ok].mean()), 3)}
    for name, s in sc.items():
        row[name] = round(average_precision_score(y[ok], s[ok]), 3)
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv(OUT / "step3_comparison_dev.csv", index=False)
print("\n=== AVERAGE PRECISION PAR FICHIER (valve1, développement) ===")
print(res.to_string(index=False))
print("\n=== MÉDIANE SUR LES FICHIERS ===")
print(res.drop(columns="file").median().round(3).to_string())

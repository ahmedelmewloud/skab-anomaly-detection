from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/raw/SKAB/data")
NORMAL = ROOT / "anomaly-free" / "anomaly-free.csv"
DEV = sorted((ROOT / "valve1").glob("*.csv"))   # développement ; valve2 et other = test, NON utilisés ici
OUT = Path("results")
OUT.mkdir(exist_ok=True)
LABELS = ["anomaly", "changepoint"]
SEED = 42
LAGS = 3          # le résidu utilise les 3 secondes précédentes
VAR_TARGET = 0.90


def load(path):
    return (pd.read_csv(path, sep=";", parse_dates=["datetime"])
              .set_index("datetime").sort_index())


# ---------- 1. Apprendre le normal (fichier sans anomalie UNIQUEMENT) ----------
normal = load(NORMAL)
sensors = [c for c in normal.columns if c not in LABELS]
scaler = StandardScaler().fit(normal[sensors])


def scale(df):
    return pd.DataFrame(scaler.transform(df[sensors]), index=df.index, columns=sensors)


Xn = scale(normal)

# ACP (même modèle que l'étape 2)
cum = np.cumsum(PCA().fit(Xn).explained_variance_ratio_)
k = int(np.argmax(cum >= VAR_TARGET)) + 1
pca = PCA(n_components=k).fit(Xn)


def pca_scores(X):
    Xc = X.values - pca.mean_
    t = Xc @ pca.components_.T
    t2 = (t ** 2 / pca.explained_variance_).sum(axis=1)
    spe = ((Xc - t @ pca.components_) ** 2).sum(axis=1)
    return t2, spe


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
    t2, spe = pca_scores(X)
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

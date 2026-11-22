"""Code partagé par les étapes 2 à 5 : chargement, modèle du normal, scores, évaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/raw/SKAB/data")
NORMAL = ROOT / "anomaly-free" / "anomaly-free.csv"
DEV = sorted((ROOT / "valve1").glob("*.csv"))   # développement
TEST = sorted((ROOT / "valve2").glob("*.csv")) + sorted((ROOT / "other").glob("*.csv"))  # test
OUT = Path("results")
OUT.mkdir(exist_ok=True)
LABELS = ["anomaly", "changepoint"]   # étiquettes : pour l'évaluation uniquement
VAR_TARGET = 0.90                     # part de variance que l'ACP doit garder
MIN_RECALL, FA_BUDGET = 0.9, 2.0      # règle de choix : rappel minimum, fausses alertes/h maximum


def load(path):
    return (pd.read_csv(path, sep=";", parse_dates=["datetime"])
              .set_index("datetime").sort_index())


def fit_normal():
    """Normalisation + ACP apprises sur anomaly-free UNIQUEMENT."""
    normal = load(NORMAL)
    sensors = [c for c in normal.columns if c not in LABELS]
    scaler = StandardScaler().fit(normal[sensors])
    Xn = scaler.transform(normal[sensors])
    cum = np.cumsum(PCA().fit(Xn).explained_variance_ratio_)
    k = int(np.argmax(cum >= VAR_TARGET)) + 1
    pca = PCA(n_components=k).fit(Xn)
    return normal, sensors, scaler, pca, cum


def pca_scores(Xs, pca, center=True):
    """T2, SPE et contributions, pour des données déjà normalisées."""
    Xc = np.asarray(Xs) - pca.mean_ if center else np.asarray(Xs)
    t = Xc @ pca.components_.T                    # coordonnées sur les axes
    t2 = (t ** 2 / pca.explained_variance_).sum(axis=1)
    resid = Xc - t @ pca.components_             # ce que l'ACP ne reconstruit pas
    spe = (resid ** 2).sum(axis=1)
    w = t / pca.explained_variance_
    contrib_t2 = Xc * (w @ pca.components_)      # somme des colonnes = T2
    contrib_spe = resid ** 2                     # somme des colonnes = SPE
    return t2, spe, contrib_t2, contrib_spe


def spe_score(df, sensors, scaler, pca):
    """SPE avec la normalisation du fichier normal (expérience A)."""
    return pca_scores(scaler.transform(df[sensors]), pca)[1]


def spe_calibrated(df, sensors, scaler, pca, n_calib, mode):
    """SPE recalibré avec les n_calib premières lignes du fichier (expérience B, passé seulement)."""
    X = df[sensors].values
    mu = X[:n_calib].mean(axis=0)
    if mode == "centre":
        sd = scaler.scale_                                       # échelle du normal
    else:
        sd = np.maximum(X[:n_calib].std(axis=0), scaler.scale_)  # jamais plus petite que celle du normal
    return pca_scores((X - mu) / sd, pca, center=False)[1]   # déjà centré sur le fichier


def alerts(s, tau, k):
    """Alerte quand le score dépasse tau pendant k lignes consécutives (causal)."""
    return pd.Series((s > tau).astype(int)).rolling(k).sum().eq(k).values


def events(y):
    """Début et fin (exclue) de chaque événement étiqueté."""
    d = np.diff(np.r_[0, y, 0])
    return zip(np.where(d == 1)[0], np.where(d == -1)[0])


def counts(items):
    """items : liste de (index temporel, étiquettes, alertes). Comptage par événement."""
    c = {"n_ev": 0, "n_det": 0, "n_alert": 0, "n_just": 0, "delays": [], "normal_s": 0}
    for idx, y, alert in items:
        for a, b in events(y):
            c["n_ev"] += 1
            hit = np.where(alert[a:b])[0]
            if len(hit):
                c["n_det"] += 1
                c["delays"].append((idx[a + hit[0]] - idx[a]).total_seconds())
        starts = alert & ~np.r_[False, alert[:-1]]   # une alerte = un début de période d'alerte
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


def choose(res, recall="recall", fa="false_per_hour", delay="median_delay_s"):
    """Règle écrite à l'avance. Renvoie (réglage, règle respectée ?)."""
    ok = res[(res[recall] >= MIN_RECALL) & (res[fa] <= FA_BUDGET)]
    if len(ok):
        return ok.sort_values([delay, fa]).iloc[0], True
    return res[res[fa] <= FA_BUDGET].sort_values([recall, delay], ascending=[False, True]).iloc[0], False


def random_baseline(items, n, seed):
    """Détecteur aléatoire : les vraies alertes décalées circulairement dans chaque fichier."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        summarize(counts([(i, y, np.roll(a, rng.integers(0, len(a)))) for i, y, a in items]))
        for _ in range(n)])

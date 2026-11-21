# Détection d'anomalies d'une pompe hydraulique (SKAB)

Mini-projet AML313 · Séries temporelles de capteurs IoT · Master SupNum · Sujet 4

## Objectif
Détecter le plus tôt possible les anomalies d'une pompe à eau, avec peu de fausses alertes.

## Données
SKAB (Skoltech Anomaly Benchmark) : https://github.com/waico/SKAB
- 8 capteurs (vibrations, courant, pression, températures, tension, débit), 1 mesure par seconde.
- `anomaly-free` : fichier sans anomalie, utilisé pour apprendre le « normal ».
- `valve1`, `valve2`, `other` : fichiers avec anomalies étiquetées (`anomaly`, `changepoint`), utilisés uniquement pour évaluer.

## Structure du dépôt
- `data/raw/` : données brutes téléchargées (non versionnées)
- `src/` : code du projet
- `scripts/` : outils annexes (téléchargement des données)
- `results/` : tableaux et figures produits
- `notebooks/` : exploration
- `article/` : brouillons de l'article

## Reproduire les résultats

1. Cloner le dépôt
```bash
git clone https://github.com/ahmedelmewloud/skab-anomaly-detection.git
cd skab-anomaly-detection
```

2. Créer l'environnement et installer les dépendances
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Télécharger les données
```bash
./scripts/download_data.sh
```

4. Diagnostic des fichiers bruts (dates, doublons, ordre, trous)
```bash
python src/diagnostic.py
python src/diagnostic_gaps.py
```
Sorties : `results/journal_diagnostic.csv`, `results/journal_gaps.csv`

5. Étape 1 : profil du « normal » et stationnarité
```bash
python src/step1_normal.py
```
Sorties : `results/normal_profile.csv`, `results/stationarity_normal.csv`, `results/normal_signals.png`

## État d'avancement
- [x] Diagnostic des données brutes
- [x] Étape 1 : apprendre le normal et vérifier la stationnarité
- [ ] Étape 2 : ACP (T² et SPE)
- [ ] Étape 3 : Isolation Forest et résidu de prévision
- [ ] Étape 4 : seuil et persistance
- [ ] Étape 5 : évaluation par événement

## Résultats de l'étape 1
- Courant, tension et pression : stationnaires.
- Accéléromètres, débit, température, thermocouple : dérive lente (mise en régime probable de la pompe).
- Fichier normal : 556 trous isolés d'une seconde (≈ 5,6 %), conservés sans imputation.

## Limites
- Le « normal » appris n'est pas strictement constant.
- La pression a un MAD nul : une échelle minimale sera nécessaire pour le z-score robuste.

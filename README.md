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

Tout relancer en une commande (depuis la racine, environnement activé) :
```bash
./scripts/run_all.sh
```
Détail des étapes :

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




5. Profil du normal et stationnarité (fichier `anomaly-free` uniquement)
```bash
python src/step1_normal.py
```
Sorties : `results/normal_profile.csv`, `results/stationarity_normal.csv`, `results/normal_signals.png`

6. ACP, T², SPE et contributions des capteurs (exemple visuel sur `valve1/1.csv`)
```bash
python src/step2_pca.py
```
Sorties : `results/step2_pca_loadings.csv`, `results/step2_normal_scores.csv`, `results/step2_contributions_demo.csv`, `results/step2_scores_demo.png`

7. Comparaison des scores (T², SPE, Isolation Forest, résidu de prévision) sur le dev-set `valve1`
```bash
python src/step3_compare.py
```
Sortie : `results/step3_comparison_dev.csv`

8. Seuil et persistance, réglés sur le dev-set `valve1`
```bash
python src/step4A_threshold.py   # A : normal appris une fois sur anomaly-free
python src/step4B_threshold.py   # B : recalibrage sur le début de chaque fichier
```
Sorties : `results/step4A_tradeoff_dev.csv`, `results/step4A_chosen.csv`, `results/step4A_tradeoff.png`, `results/step4B_tradeoff_dev.csv`, `results/step4B_chosen.csv`, `results/step4B_tradeoff.png`

9. Évaluation par événement sur le jeu de test (`valve2` + `other`), comparée à un détecteur aléatoire
```bash
python src/step5A_evaluate.py    # lit results/step4A_chosen.csv
python src/step5B_evaluate.py    # lit results/step4B_chosen.csv (et step5A_summary.csv pour la comparaison A/B)
```
Sorties : `results/step5A_test_events.csv`, `results/step5A_summary.csv`, `results/step5B_test_events.csv`, `results/step5B_summary.csv`, `results/step5B_summary_clean_calib.csv` (sans les fichiers dont la fenêtre de calibration contient une anomalie)

10. Incertitude (bootstrap par fichier) et ablation sur le jeu de test
```bash
python src/step6_uncertainty_ablation.py   # lit step4A_chosen.csv, step4B_chosen.csv, step4B_tradeoff_dev.csv
```
Sorties : `results/step6_uncertainty.csv`, `results/step6_ablation.csv`

L'article se trouve dans `article/rapport.md`.

L'ordre compte : l'étape 5A dépend de 4A, l'étape 5B dépend de 4B (et de 5A pour la comparaison). Toutes les commandes se lancent depuis la racine du dépôt.

## Code partagé
`src/common.py` regroupe ce qui est commun aux étapes 2 à 5 : chemins et découpage dev/test, chargement des fichiers, apprentissage du normal (normalisation + ACP à 90 % de variance), scores T²/SPE, recalibrage par fichier, alertes avec persistance, comptage par événement, règle de choix (rappel ≥ 0,9 et ≤ 2 fausses alertes/h) et détecteur aléatoire.

## Points d'attention
- La persistance `k` est comptée en lignes, pas en secondes : avec ≈ 5 % de secondes manquantes, `k = 10` correspond en pratique à un peu plus de 10 s.
- Aucun réglage ne respecte la règle sur le dev-set, ni en A ni en B : les réglages retenus sont des replis (colonne `regle_respectee`).

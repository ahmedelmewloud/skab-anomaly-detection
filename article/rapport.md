# Détection précoce d'anomalies sur une pompe hydraulique (SKAB) : un « normal » appris une fois ne suffit pas

Mini-projet AML313 · Séries temporelles de capteurs IoT · Master SupNum · Sujet 4

Code et résultats : https://github.com/ahmedelmewloud/skab-anomaly-detection

## Résumé

Nous cherchons à détecter le plus tôt possible les anomalies d'une pompe à eau instrumentée (jeu SKAB, 8 capteurs à 1 Hz), avec peu de fausses alertes. Le comportement normal est appris sur un unique enregistrement sans anomalie. Parmi quatre scores non supervisés comparés sur un jeu de développement, l'erreur de reconstruction d'une ACP (SPE) est nettement la meilleure (précision moyenne médiane 0,86 contre 0,35 pour le hasard). Un seuil et une persistance sont ensuite réglés sur le développement, selon une règle fixée à l'avance, puis évalués une seule fois sur un jeu de test séparé, événement par événement. Appliqué tel quel, le modèle échoue : il ne détecte que 22 % des événements de test (IC 95 % : 6–44 %), sans faire mieux qu'un détecteur aléatoire. La raison est un changement de régime entre enregistrements : le « normal » appris ne correspond plus au fonctionnement des autres journées. En recentrant et en remettant à l'échelle les données sur les 300 premières secondes de chaque enregistrement, on détecte 78 % des événements (IC 95 % : 56–94 %), avec un retard médian de 48 s et 68 % d'alertes justifiées. Cela reste au-dessus du budget de fausses alertes (6,4/h au lieu de 2/h), et le rappel n'est pas meilleur que celui d'un détecteur aléatoire émettant autant d'alertes : le gain réel est la précision.

## 1. Question et données

**Objectif.** Sur un banc d'essai de pompe, déclencher une alerte aussi tôt que possible après le début d'une anomalie (fermeture de vanne, cavitation, déséquilibre du rotor…) en limitant les fausses alertes. Le cadre est celui d'une surveillance en ligne : à chaque seconde, la décision ne peut utiliser que le passé.

**Données.** SKAB (Skoltech Anomaly Benchmark, https://github.com/waico/SKAB, DOI 10.34740/KAGGLE/DSV/1693952). Chaque fichier est une expérience indépendante d'environ 15 à 20 minutes, échantillonnée à 1 Hz, avec 8 capteurs : deux accéléromètres (RMS), courant, pression, température du moteur, thermocouple (température de l'eau), tension et débit. Deux colonnes d'étiquettes, `anomaly` et `changepoint`, ne servent qu'à l'évaluation ; `changepoint` n'est pas utilisée.

**Découpage et dates de coupure.** Les fichiers étant des expériences séparées, le découpage se fait par fichier, jamais par tirage aléatoire de lignes. Il est fixé dans le code dès l'étape de comparaison des modèles.

| Rôle | Fichiers | Dates d'enregistrement | Lignes | Événements |
|---|---|---|---|---|
| Apprentissage du normal | `anomaly-free` (1) | 8 fév. 2020, 13:30–16:16 | 9 405 | 0 |
| Développement (réglages) | `valve1` (16) | 9 mars 2020, 10:14–15:34 | 18 160 | 16 |
| Test (mesure finale) | `valve2` (4) | 9 mars 2020, 15:56–17:14 | 4 312 | 4 |
| Test (mesure finale) | `other` (14) | 8 fév. 2020, 16:06–19:32 et 1ᵉʳ mars 2020, 15:44–17:20 | 14 929 | 14 |

Pour les vannes, le test (`valve2`) est strictement postérieur au développement (`valve1`). Ce n'est pas le cas des fichiers `other`, enregistrés avant `valve1`. Aucune donnée de test n'entre dans un réglage, mais le seuil appliqué à `other` provient d'enregistrements plus récents : c'est une évaluation « hors distribution » plus qu'une simulation de déploiement chronologique (voir Limites). Chaque fichier de test contient exactement un événement, soit 18 événements en tout.

## 2. Préparation

**Diagnostic de la grille** (`src/diagnostic.py`, `src/diagnostic_gaps.py`, sorties `results/journal_*.csv`). Pour les 35 fichiers : aucune date illisible, aucun doublon d'horodatage, ordre toujours croissant, aucune valeur manquante dans les capteurs. Le pas médian est d'une seconde. En revanche, la grille a des trous : environ 5 % des secondes manquent (complétude de 0,94 pour le fichier normal, entre 0,73 et 0,96 ailleurs). Ce sont presque toujours des trous isolés d'une seconde, réguliers (environ un tous les 15 à 25 points). Quelques fichiers ont des trous longs : 246 s (`other/2`), 75 s (`valve1/2`), 64 s (`valve1/7`), 63 s (`valve2/1`), 53 s (`valve1/4`), 32 s (`other/13`).

**Choix de traitement.** Nous gardons la fréquence native (1 Hz) sans agrégation, puisque l'objectif est la précocité de l'alerte. Les trous sont conservés sans imputation : tous les scores sont calculés point par point (aucun ne demande une grille régulière), et une imputation par interpolation utiliserait le point suivant, ce qui serait une fuite. Les retards sont mesurés sur les horodatages réels, pas en nombre de lignes.

**Profil du normal et stationnarité** (`src/step1_normal.py`). Le normal est appris uniquement sur `anomaly-free` (9 405 lignes, ≈ 2 h 45). Pour chaque capteur, nous calculons médiane, MAD, moyenne, écart-type, minimum et maximum. Les tests ADF et KPSS, lus ensemble et confirmés par la figure `results/normal_signals.png`, montrent que le courant, la tension et la pression sont stationnaires. Les deux accéléromètres, le débit, la température et le thermocouple dérivent lentement, de 1,5 à 2,2 écarts-types entre le premier et le dernier tiers. Les accéléromètres et le débit évoluent surtout pendant la première heure puis se stabilisent ; les températures évoluent plus lentement. Nous interprétons cela, à titre d'hypothèse, comme une mise en régime de la pompe. Nous ne différencions pas les signaux, afin de conserver l'information de niveau, qui porte l'essentiel des anomalies. La pression a un MAD nul (elle reste la plupart du temps sur une même valeur), ce qui interdit un z-score robuste sans échelle minimale.

**Normalisation.** Un `StandardScaler` est ajusté sur le seul fichier normal, puis seulement appliqué (`transform`) aux autres fichiers.

## 3. Baseline

Deux références sont définies avant tout modèle.

1. **Score au hasard, pour le classement des points.** Sa précision moyenne (average precision) est égale à la proportion de points anormaux, soit 0,35 sur `valve1`. Tout score doit faire mieux.
2. **Détecteur aléatoire, pour l'évaluation par événement.** Dans chaque fichier, la série d'alertes réelle du détecteur est décalée circulairement d'une quantité tirée au hasard (500 tirages, graine 42). Ce détecteur émet le même nombre d'alertes, de la même durée, mais sans lien avec les anomalies. C'est une référence exigeante : comme une anomalie couvre environ 35 % de chaque fichier, des alertes nombreuses et longues tombent souvent dedans par hasard. Un détecteur n'est utile que s'il dépasse le 95ᵉ centile de cette référence.

## 4. Modèle

**Scores comparés** (`src/step3_compare.py`), tous appris sur le fichier normal seulement :

- **T²** de Hotelling sur une ACP gardant 90 % de la variance (5 axes sur 8, 93,4 %). PC1 regroupe accéléromètres, températures et débit (le régime thermo-mécanique), PC2 courant et tension, PC3 la pression, PC4 les accéléromètres et PC5 l'opposition courant/tension.
- **SPE** (erreur de reconstruction de cette même ACP) : la part du point que les 5 axes du normal n'expliquent pas.
- **Isolation Forest** (200 arbres, graine 42).
- **Résidu de prévision** : une régression Ridge prédit chaque capteur à partir des 3 secondes précédentes (strictement causale) ; le score est la somme des carrés des résidus normalisés.

| Score (`valve1`, 16 fichiers) | Précision moyenne médiane |
|---|---|
| Hasard (taux d'anomalie) | 0,351 |
| T² | 0,366 |
| Isolation Forest | 0,350 |
| Résidu de prévision | 0,496 |
| **SPE** | **0,858** |

Le SPE domine sur presque tous les fichiers ; T² et Isolation Forest ne font pas mieux que le hasard. Nous retenons donc le SPE. L'ACP fournit aussi la contribution de chaque capteur au score, utile pour l'interprétation (`src/step2_pca.py`).

**Règle d'alerte.** Une alerte se déclenche quand le SPE dépasse un seuil τ pendant k points consécutifs (persistance, fenêtre glissante alignée à droite donc causale). Une alerte est comptée une fois, à son début.

**Deux variantes.**

- **A — normal appris une fois.** Le SPE est calculé avec la normalisation du fichier normal.
- **B — recalibrage par enregistrement.** Les axes de l'ACP restent ceux du normal, mais chaque fichier est recentré sur la moyenne de ses n premières secondes et, en option, remis à l'échelle par leur écart-type (jamais plus petit que celui du normal). Ces n secondes sont supposées normales ; c'est vérifié sur le développement, où la première anomalie arrive au plus tôt à la ligne 566. Le recalibrage n'utilise que le passé ; l'évaluation commence après la fenêtre.

## 5. Protocole de validation

**Réglage sur le développement uniquement** (`src/step4A_threshold.py`, `src/step4B_threshold.py`). Seuils candidats : quantiles 0,5 à 1 du SPE des points normaux de `valve1`, et 1,25 à 2 fois son maximum ; persistances k ∈ {1, 3, 5, 10, 20, 30}. Pour B, on teste aussi la longueur de calibration (60, 120, 300 s) et le mode (centre seul, centre et échelle).

**Règle de choix écrite avant de voir les résultats** : rappel par événement ≥ 0,9 et au plus 2 fausses alertes par heure de fonctionnement normal ; parmi les réglages admissibles, le plus petit retard médian. Si aucun réglage n'est admissible, on garde, sous le budget de fausses alertes, le meilleur rappel puis le plus petit retard.

**Aucun réglage ne respecte la règle**, ni en A ni en B : les deux réglages retenus sont des replis, notés comme tels dans `results/step4*_chosen.csv`.

![Figure 2 — Compromis sur le développement pour B (centre + échelle, 300 s). À gauche, le retard médian ; à droite, le rappel par événement, en fonction des fausses alertes par heure, pour chaque persistance k. Les pointillés marquent la règle (rappel ≥ 0,9, au plus 2 fausses alertes/h) : aucun point n'est dans la zone admissible.](../results/step4B_tradeoff.png)

| | Réglage retenu | Rappel (dev) | Retard médian (dev) | Fausses alertes/h (dev) |
|---|---|---|---|---|
| A | τ = quantile 0,95, k = 20 s | 0,44 | 59 s | 1,52 |
| B | centre + échelle, 300 s, τ = quantile 0,9, k = 10 s | 0,62 | 59 s | 1,53 |

**Mesure finale sur le test, une seule fois** (`src/step5A_evaluate.py`, `src/step5B_evaluate.py`). Métriques par événement :

- **rappel** : part des événements contenant au moins une alerte ;
- **retard** : délai entre le début de l'événement et la première alerte qui y tombe ;
- **précision** : part des alertes qui commencent pendant une anomalie ;
- **fausses alertes par heure** : alertes qui commencent hors anomalie, rapportées à la durée normale.

**Incertitude** (`src/step6_uncertainty_ablation.py`). Les fichiers étant des expériences indépendantes, nous rééchantillonnons les 18 fichiers de test avec remise (bootstrap, 2 000 tirages, graine 42) pour obtenir un intervalle à 95 %.

**Contrôle des fuites temporelles.**

- Pas de découpage aléatoire : le découpage se fait par fichier et il est fixé dans le code.
- Pas de fenêtre centrée : la persistance, les retards du résidu et la fenêtre de calibration ne regardent que le passé.
- Pas de prétraitement ajusté sur toutes les données : la normalisation et l'ACP sont ajustées sur le fichier normal seul, les seuils sur le développement seul.
- Les étiquettes de test servent seulement au calcul des métriques. En particulier, le fichier `other/2`, dont l'anomalie commence à la ligne 104, donc dans la fenêtre de calibration, n'est pas retiré du résultat principal ; il est seulement signalé, et un résultat sans lui est donné à part.

## 6. Résultats

**Pourquoi A échoue.** Sur `valve1/1`, le SPE médian vaut environ 3 700 hors anomalie, contre 0,36 sur le fichier normal (99ᵉ centile : 3,3). Il ne bouge presque pas pendant l'anomalie (3 781). Le débit (71 %) et la température (19 %) portent ce score, à l'identique avant et pendant l'anomalie (figure 1). L'enregistrement entier est donc « anormal » par rapport au normal de février : le score mesure le changement de jour de fonctionnement, pas l'anomalie. Le seuil réglé sur `valve1` (≈ 4 300) devient alors arbitraire pour les autres journées : sur 11 des 14 fichiers `other`, le SPE ne le dépasse jamais, et sur les 3 autres jamais assez longtemps pour déclencher une alerte.

![Figure 1 — T² et SPE du modèle A sur `valve1/1` (échelle log). La zone rouge est l'anomalie étiquetée : les deux scores restent des milliers de fois au-dessus du normal (SPE médian 0,36) pendant tout l'enregistrement.](../results/step2_scores_demo.png)

**Test (18 fichiers, 18 événements).** Intervalles à 95 % par bootstrap sur les fichiers ; pour le détecteur aléatoire, moyenne et intervalle 5–95 % sur 500 tirages.

| | Rappel | Précision | Retard médian | Fausses alertes/h |
|---|---|---|---|---|
| A | 0,22 [0,06 ; 0,44] | 0,62 [0,33 ; 1,00] | 75 s [56 ; 299] | 1,4 [0,0 ; 3,2] |
| Aléatoire (alertes de A) | 0,15 [0,06 ; 0,22] | 0,35 [0,08 ; 0,62] | 125 s | 2,5 [1,4 ; 3,5] |
| **B** | **0,78 [0,56 ; 0,94]** | **0,68 [0,50 ; 0,85]** | **48 s [11 ; 78]** | **6,4 [2,4 ; 10,9]** |
| Aléatoire (alertes de B) | 0,74 [0,61 ; 0,83] | 0,42 [0,26 ; 0,60] | 55 s | 13,0 [8,4 ; 16,8] |

Lecture :

- **A n'est pas meilleur que le hasard.** 21 % des tirages aléatoires ont un rappel au moins égal au sien, et 8 % une précision au moins égale. Ses 4 détections sont toutes sur `valve2`.
- **B ne gagne pas sur le rappel.** Un détecteur aléatoire qui émet autant d'alertes atteint le même rappel dans 50 % des tirages.
- **B gagne nettement sur la qualité des alertes.** Sa précision (0,68) dépasse le 95ᵉ centile aléatoire (0,60) ; seuls 0,4 % des tirages font aussi bien. Ses fausses alertes (6,4/h) sont sous le 5ᵉ centile aléatoire (8,4/h). Autrement dit, ses alertes tombent bien plus souvent dans les anomalies que par hasard.
- **B reste hors budget** : 6,4 fausses alertes/h, au lieu de 1,5 sur le développement. Le réglage se généralise mal d'une campagne d'essais à l'autre.
- Sans le fichier `other/2` (calibration contaminée) : rappel 0,82 et 6,7 fausses alertes/h ; les conclusions ne changent pas.

Les intervalles sont larges : avec 18 événements, un événement de plus ou de moins déplace le rappel de 0,06. Les différences A/B sur le rappel et le retard sont nettes ; les différences entre variantes proches de B ne le sont pas.

## 7. Ablation

On retire un ingrédient de B à la fois. Pour chaque variante, le seuil et la persistance sont re-choisis sur le développement par la même règle, puis mesurés sur le test.

| Variante | Réglage (dev) | Rappel | Précision | Retard médian | Fausses alertes/h |
|---|---|---|---|---|---|
| B retenu (centre + échelle, 300 s, k = 10) | q0,9, k = 10 | 0,78 | 0,68 | 48 s | 6,4 |
| sans remise à l'échelle (centre seul) | q0,9, k = 10 | 0,67 | 0,70 | 31 s | 3,5 |
| calibration de 120 s | q0,9, k = 10 | 0,61 | 0,80 | 25 s | 1,4 |
| calibration de 60 s | q0,9, k = 20 | 0,61 | 0,82 | 37 s | 0,9 |
| sans persistance (k = 1) | q1,0, k = 1 | 0,44 | 0,87 | 39 s | 3,5 |
| sans recalibrage (A) | q0,95, k = 20 | 0,22 | 0,62 | 75 s | 1,4 |

- **Le recalibrage est l'ingrédient décisif** : sans lui, le rappel passe de 0,78 à 0,22.
- **La persistance est nécessaire.** Sans elle, il faut un seuil au maximum du normal pour tenir le budget sur le développement, et le rappel tombe à 0,44.
- **La remise à l'échelle et la calibration longue ajoutent du rappel, mais doublent les fausses alertes.** La variante à 120 s est la seule qui respecte le budget de fausses alertes sur le test (1,4/h), avec un rappel de 0,61 et le retard le plus court (25 s). Le développement ne permettait pas de la préférer (rappel 0,44 contre 0,62) : la règle de choix, appliquée honnêtement, a retenu un réglage plus « agressif ».

## 8. Limites

- **Le normal appris n'est pas stable.** Le fichier sans anomalie montre une mise en régime d'environ une heure, et les autres journées sont dans un autre régime. Un seul enregistrement normal ne représente pas la variabilité réelle du banc. C'est la cause principale de l'échec de A.
- **Le recalibrage suppose un début d'enregistrement sain.** C'est vrai sur 17 des 18 fichiers de test, faux sur `other/2`. En exploitation, rien ne garantit qu'une anomalie ne soit pas déjà en cours au redémarrage.
- **Peu d'événements.** Avec 16 événements en développement et 18 en test, les intervalles de confiance sont larges, et la règle de choix (rappel ≥ 0,9) n'a jamais été atteinte. Le budget de fausses alertes mesuré sur le développement ne tient pas sur le test.
- **Chronologie imparfaite.** Les fichiers `other` sont antérieurs au développement. Certains horodatages se chevauchent (`other/5` commence 10 minutes avant la fin de `anomaly-free` ; `other/2` et `other/3`, `other/12` et `other/13` se chevauchent aussi), ce qui est impossible sur un banc unique : les horodatages de SKAB ne sont pas entièrement fiables.
- **Persistance comptée en lignes.** Avec environ 5 % de secondes manquantes, k = 10 correspond en pratique à un peu plus de 10 s ; près d'un trou long, l'écart peut être important.
- **Baseline limitée.** Il n'y a pas de seuil simple par capteur (z-score robuste) comme baseline métier ; et l'étiquette `changepoint`, comme la nature des anomalies `other`, n'est pas exploitée.
- **Une seule famille de modèles réglée.** Seul le SPE a été réglé en seuil et persistance ; le résidu de prévision, deuxième sur le développement, n'a pas été testé avec recalibrage.

## 9. Reproductibilité

Le code, les données de sortie et cet article sont disponibles sur https://github.com/ahmedelmewloud/skab-anomaly-detection. Le dépôt se relance entièrement avec une commande, depuis la racine et l'environnement activé : `./scripts/run_all.sh`. Elle télécharge les données puis exécute les étapes dans l'ordre. L'environnement est figé (`requirements.txt`, Python 3.12). Les graines sont fixées (Isolation Forest, détecteur aléatoire, bootstrap : 42). Le code commun est dans `src/common.py`. Tous les tableaux de cet article proviennent des fichiers de `results/`.

## Référence

Katser, I. D., Kozitsin, V. O. (2020). *Skoltech Anomaly Benchmark (SKAB)*. Kaggle. DOI : 10.34740/KAGGLE/DSV/1693952.

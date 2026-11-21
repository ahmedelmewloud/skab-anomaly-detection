## Apprentissage du « normal » et stationnarité

Le comportement normal est appris uniquement sur le fichier sans anomalie (9 405 lignes, 8 capteurs, ≈ 2 h 45 à 1 Hz). Pour chaque capteur, nous calculons médiane, MAD, moyenne, écart-type, minimum et maximum. Les tests ADF et KPSS, lus ensemble et confirmés par la figure, montrent que le courant, la tension et la pression sont stationnaires. Les deux accéléromètres, le débit, la température et le thermocouple dérivent lentement : les accéléromètres et le débit évoluent surtout pendant la première heure, puis se stabilisent ; les deux températures évoluent plus lentement. Nous interprétons cela, à titre d'hypothèse, comme une mise en régime de la pompe. Nous ne différencions pas les signaux, afin de conserver l'information de niveau. Le fichier présente 556 trous isolés d'une seconde (≈ 5,6 %), conservés sans imputation. La pression a un MAD nul, ce qui impose une échelle minimale pour le z-score robuste.

## Limites (à reprendre dans la section « Limites »)

Le « normal » appris n'est pas strictement constant ; cette dérive peut provoquer de fausses alertes si les données de test sont enregistrées dans un autre régime.

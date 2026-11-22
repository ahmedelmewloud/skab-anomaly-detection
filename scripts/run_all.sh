#!/usr/bin/env bash
# Relance tout le projet, des données brutes jusqu'aux tableaux et figures de results/.
# Usage (depuis la racine du dépôt, environnement activé) : ./scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

./scripts/download_data.sh
for step in diagnostic diagnostic_gaps step1_normal step2_pca step3_compare \
            step4A_threshold step4B_threshold step5A_evaluate step5B_evaluate \
            step6_uncertainty_ablation; do
  echo "===== $step ====="
  "$PY" "src/$step.py"
done
echo "Terminé : résultats dans results/"

#!/usr/bin/env bash
# Runs the app's original signal-processing algorithms (BaselineEstimator) and quality gate on a dataset.
# Usage: ./run_baseline.sh DATASET_DIR baseline.csv      (requires a JDK 8+)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SRC="$ROOT/app/src/main/java/com/example/yo7a/healthwatcher"
OUT="$(mktemp -d)"
javac -nowarn -d "$OUT" -cp "$ROOT/lib/commons-math3-3.6.1.jar" \
  "$SRC"/Math/*.java "$SRC"/ml/PpgSignal.java "$SRC"/ml/PpgDsp.java "$SRC"/ml/PPGFeatureExtractor.java \
  "$SRC"/ml/QualityReport.java "$SRC"/ml/SignalQualityAnalyzer.java "$SRC"/ml/UserProfile.java \
  "$SRC"/ml/BaselineEstimator.java "$HERE"/java/BaselineCli.java
java -cp "$OUT:$ROOT/lib/commons-math3-3.6.1.jar" BaselineCli "$1" "$2"
echo "wrote $2"

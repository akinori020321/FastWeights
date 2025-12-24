#!/usr/bin/env bash
set -euo pipefail

# この .sh が置かれているディレクトリに移動（どこから実行してもOKにする）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

scripts=(
  "plot_learning_curve_acc_loss_fw.py"
  "FW_sweep.py"
  "LN_sweep.py"
  "SC_sweep.py"
  "T_bind_sweep.py"
  "Wait_sweep.py"
  "alpha.py"
  "plot_learning_curve_acc_loss_fw_vs_rnn.py"
)

# まず通常スクリプト群を実行
for s in "${scripts[@]}"; do
  if [[ ! -f "$s" ]]; then
    echo "[WARN] missing: $s (skip)"
    continue
  fi
  echo "============================================================"
  echo "[RUN] $s"
  "$PYTHON" "$s"
  echo "[OK ] $s"
done

# ------------------------------------------------------------
# learning curve (T_bind) は汎用スクリプトで複数 tbind を回す
# ------------------------------------------------------------
LC_SCRIPT="plot_learning_curve_Tbind.py"
TBINDS=(2 5)

if [[ ! -f "$LC_SCRIPT" ]]; then
  echo "[WARN] missing: $LC_SCRIPT (skip)"
else
  for tb in "${TBINDS[@]}"; do
    echo "============================================================"
    echo "[RUN] $LC_SCRIPT --tbind $tb"
    "$PYTHON" "$LC_SCRIPT" --tbind "$tb"
    echo "[OK ] $LC_SCRIPT (T_bind=$tb)"
  done
fi

echo "============================================================"
echo "[DONE] all scripts finished."

#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# A-matrix Eigenvalue Growth Plot Batch
# ==========================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_SCRIPT="${BASE_DIR}/plot_eigen_growth.py"

# 出力ディレクトリ
RESULT_DIR="${BASE_DIR}/plot_eig_growth_results"
mkdir -p "$RESULT_DIR"

echo "----------------------------------------"
echo "[RUN] A-matrix (eigen growth) Plot Batch"
echo "----------------------------------------"

SRC_DIR="${BASE_DIR}/../results_A_kv"

for csv in "${SRC_DIR}"/Amat_kv_*_S*.csv; do
    [[ -f "$csv" ]] || continue

    filename=$(basename "$csv")
    echo "[A-matrix] Processing $filename"

    # 実行
    python3 "$PLOT_SCRIPT" --csv "$csv" --topk 10

    # 出力された PNG を move
    core=$(echo "$filename" | sed -E 's/Amat_kv_(.*)\.csv/\1/')
    png="eig_growth_${core}.png"

    if [[ -f "$png" ]]; then
        mv "$png" "$RESULT_DIR/"
        echo "  → Saved: $RESULT_DIR/$png"
    else
        echo "[WARN] PNG not found: $png"
    fi
done

echo "----------------------------------------"
echo "[DONE] All A-matrix CSV processed!"
echo "----------------------------------------"

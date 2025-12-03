#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_SCRIPT="${BASE_DIR}/plot_A_kv.py"
RESULT_DIR="${BASE_DIR}/plot_results"
mkdir -p "$RESULT_DIR"

SRC_DIR="${BASE_DIR}/../results_A_kv"

echo "----------------------------------------"
echo "[RUN] KV A-dynamics + S-loop Plot Batch"
echo "----------------------------------------"


############################################
# 1) A-dynamics
############################################
for csv in "${SRC_DIR}"/A_kv_*; do
    [[ -f "$csv" ]] || continue

    filename=$(basename "$csv")
    echo "[A-dynamics] Processing $filename"

    core=$(echo "$filename" | sed -E 's/^A_kv_([A-Za-z]+)_S[0-9]+.*$/\1/')
    S=$(echo   "$filename" | sed -E 's/^A_kv_[A-Za-z]+_S([0-9]+).*$/\1/')

    out_png="${RESULT_DIR}/A_kv_${core}_S${S}.png"

    python3 "$PLOT_SCRIPT" \
        --csv "$csv" \
        --out_png "$out_png"

    echo "  → Saved A-dynamics: $out_png"
done



############################################
# 2) S-loop  → A_kv の suffix を自動取得
############################################
for sloop_csv in "${SRC_DIR}"/Sloop_kv_*; do
    [[ -f "$sloop_csv" ]] || continue

    filename=$(basename "$sloop_csv")
    echo "[S-loop] Processing $filename"

    # core / S / suffix を抽出
    core=$(echo "$filename" | sed -E 's/^Sloop_kv_([A-Za-z]+)_S[0-9]+.*$/\1/')
    S=$(echo "$filename"   | sed -E 's/^Sloop_kv_[A-Za-z]+_S([0-9]+).*$/\1/')
    suffix=$(echo "$filename" | sed -E "s/^Sloop_kv_${core}_S${S}//")

    # A_kv 側も同じ suffix を持ったファイルを探す
    a_csv="${SRC_DIR}/A_kv_${core}_S${S}${suffix}"

    if [[ ! -f "$a_csv" ]]; then
        echo "[WARN] Missing A-dynamics CSV → $a_csv"
        continue
    fi

    out_png="${RESULT_DIR}/Sloop_kv_${core}_S${S}_grid.png"

    python3 "$PLOT_SCRIPT" \
        --csv "$a_csv" \
        --sloop_csv "$sloop_csv" \
        --out_png "$out_png" \
        --s_max 5

    echo "  → Saved S-loop grid: $out_png"
done


echo "----------------------------------------"
echo "[DONE] All CSV files processed!"
echo "----------------------------------------"

#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# KV A-dynamics + S-loop Plot Batch Script
#  → plots/plot_results に出力をまとめる
# ==========================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_SCRIPT="${BASE_DIR}/plot_A_kv.py"

# ★ 出力ディレクトリ
RESULT_DIR="${BASE_DIR}/plot_results"

mkdir -p "$RESULT_DIR"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "[ERROR] plot_A_kv.py が見つかりません: $PLOT_SCRIPT"
    exit 1
fi

echo "----------------------------------------"
echo "[RUN] KV A-dynamics + S-loop Plot Batch (auto mode)"
echo "----------------------------------------"

# ==========================================
# results_A_kv フォルダ内を処理
# ==========================================
SRC_DIR="${BASE_DIR}/../results_A_kv"

# ==========================================
# 1) A-dynamics CSV（A_kv_*）をすべて処理
# ==========================================
for csv in "${SRC_DIR}"/A_kv_*_S*.csv; do
    [[ -f "$csv" ]] || continue

    filename=$(basename "$csv")
    echo "[A-dynamics] Processing $filename"

    # fw / tanh を抽出
    core=$(echo "$filename" | sed -E 's/A_kv_([A-Za-z]+)_S[0-9]+.*/\1/')

    # S を抽出
    S=$(echo "$filename" | sed -E 's/.*_S([0-9]+).*/\1/')

    out_png="${RESULT_DIR}/A_kv_${core}_S${S}.png"

    python3 "$PLOT_SCRIPT" \
        --csv "$csv" \
        --out_png "$out_png"

    echo "  → Saved A-dynamics: $out_png"
done


# ==========================================
# 2) S-loop CSV（Sloop_kv_*）をすべて処理
# ==========================================
for csv in "${SRC_DIR}"/Sloop_kv_*_S*.csv; do
    [[ -f "$csv" ]] || continue

    filename=$(basename "$csv")
    echo "[S-loop] Processing $filename"

    # ===============================
    # lam085 を含む完全一致で対応 A_kv CSV を生成
    # ===============================
    a_csv="${SRC_DIR}/${filename/Sloop_kv_/A_kv_}"

    # 存在チェック（必須）
    if [[ ! -f "$a_csv" ]]; then
        echo "[WARN] 対応する A_kv CSV が存在しません → $a_csv"
        continue
    fi

    # 出力名（そのまま lam085 まで維持）
    out_png="${RESULT_DIR}/${filename/Sloop_kv_/Sloop_kv_}_grid.png"

    python3 "$PLOT_SCRIPT" \
        --csv "$a_csv" \
        --sloop_csv "$csv" \
        --out_png "$out_png"

    echo "  → Saved S-loop grid: $out_png"
done

echo "----------------------------------------"
echo "[DONE] All CSV files processed!"
echo "----------------------------------------"

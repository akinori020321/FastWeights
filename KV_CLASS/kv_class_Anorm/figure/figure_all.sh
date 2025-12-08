#!/usr/bin/env bash
set -euo pipefail

#############################################
# 0. パスの基準をこの .sh があるディレクトリに固定
#############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

#############################################
# 1. LayerNorm の生ログを全部出す (LN.py)
#############################################
OUTDIR="plots/results_LN_stats"   # LNパラメータのログ保存先
CKPT_DIR="../checkpoints"         # checkpoint が入っているディレクトリ

CORE_LIST="rnn fw tanh"

mkdir -p "${OUTDIR}"

echo "==============================================="
echo "[STEP 1] Dump LayerNorm stats (LN.py)"
echo "  OUTDIR   = ${OUTDIR}"
echo "  CKPT_DIR = ${CKPT_DIR}"
echo "==============================================="

for CORE in ${CORE_LIST}; do
    echo "-----------------------------------------------"
    echo "[CORE] ${CORE}"
    echo "-----------------------------------------------"

    # 例: kv_fw_S1_fw1_eta0300_lam0850_seed0.pt
    mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -type f -name "kv_${CORE}_S*_fw*_eta*_lam*_seed*.pt" | sort)

    if [ ${#CKPT_LIST[@]} -eq 0 ]; then
        echo "[WARN] No checkpoints found for CORE=${CORE}"
        continue
    fi

    for CKPT in "${CKPT_LIST[@]}"; do
        echo "[RUN] checkpoint = ${CKPT}"

        BASE=$(basename "${CKPT}" .pt)
        LOG="${OUTDIR}/LN_${BASE}.txt"

        echo "[INFO] Save LN stats → ${LOG}"
        python3 LN.py --ckpt "${CKPT}" | tee "${LOG}"
        echo ""
    done
done

echo "==============================================="
echo "[STEP 1 DONE] All LayerNorm dumps completed."
echo "==============================================="

#############################################
# 2. LN 全体の統計を集計＆プロット (plot_ln_stats.py)
#############################################
echo "==============================================="
echo "[STEP 2] Aggregate LN stats & make plots (plot_ln_stats.py)"
echo "==============================================="

python3 plot_ln_stats.py

echo "==============================================="
echo "[STEP 2 DONE]"
echo "==============================================="

#############################################
# 3. A-dynamics (specA / ||h||) のプロット (plot_Akv.py)
#############################################
echo "==============================================="
echo "[STEP 3] Plot A-dynamics curves (plot_Akv.py)"
echo "==============================================="

python3 plot_Akv.py

echo "==============================================="
echo "[STEP 3 DONE]"
echo "==============================================="

#############################################
# 4. h_t のヒートマップ & S-loop 可視化
#    - h_heatmap.py : 各 t の cos 行列ヒートマップ
#    - sloop_plot.py: S-loop (||h_s||, ||Ah_s||, ratio, cos など)
#############################################
echo "==============================================="
echo "[STEP 4] Plot h heatmaps (h_heatmap.py)"
echo "==============================================="

python3 h_heatmap.py

echo "-----------------------------------------------"
echo "[STEP 4] Plot S-loop diagnostics (sloop_plot.py)"
echo "-----------------------------------------------"

python3 sloop_plot.py

echo "==============================================="
echo "ALL ANALYSES COMPLETED."
echo "  - LN raw logs     : ${OUTDIR}"
echo "  - LN summary plots: ../plots/results_LN_stats"
echo "  - A-dynamics      : plots/figure_norm"
echo "  - h heatmaps      : plots/h_heatmap"
echo "  - S-loop plots    : plots/sloop"
echo "==============================================="

#############################################
# 5. h_t の PCA 時系列（全体カラーバー付き）
#############################################
echo "==============================================="
echo "[STEP 5] Plot PCA time-series (h_PCA.py)"
echo "==============================================="

python3 h_PCA.py

echo "==============================================="
echo "[STEP 5 DONE]"
echo "==============================================="

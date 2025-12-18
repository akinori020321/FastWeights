#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
OUTDIR="results_A_kv"   # 結果ディレクトリ

DG=30           # d_g
DH=60           # d_h
TBIND=12        # T_bind
DUPLICATE=3     # duplicate（run_A_kv 側のデフォルトと合わせておく）
BETA=1.0        # beta
BATCH=1         # batch_size

GPU="--gpu"
USE_FW="--use_fw"
USE_LN="--use_ln"

# ★★ 追加：クラス数 / Wait ステップ数 / Bind ノイズ（環境変数で上書き可） ★★
NUM_CLASSES=${NUM_CLASSES:-8}
NUM_WAIT=${NUM_WAIT:-2}          # run_A_kv.py の --num_wait に渡す
BIND_NOISE_STD=${BIND_NOISE_STD:-0.4}  # run_A_kv.py の --bind_noise_std に渡す

# Sweep する core タイプ
CORE_LIST="rnn fw tanh"

# checkpoint ディレクトリ
CKPT_DIR="checkpoints"

echo "==============================================="
echo "Settings:"
echo "  d_g           = ${DG}"
echo "  d_h           = ${DH}"
echo "  T_bind        = ${TBIND}"
echo "  duplicate     = ${DUPLICATE}"
echo "  beta          = ${BETA}"
echo "  batch_size    = ${BATCH}"
echo "  num_classes   = ${NUM_CLASSES}"
echo "  num_wait      = ${NUM_WAIT}"
echo "  bind_noise_std= ${BIND_NOISE_STD}"
echo "==============================================="

# ==========================================
# 実行
# ==========================================
mkdir -p "${OUTDIR}"

for CORE in ${CORE_LIST}; do
    echo "==============================================="
    echo "[CORE] ${CORE}"
    echo "==============================================="

    # 例: kv_fw_S1_fw1_eta03_lam0950_seed0.pt
    mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -type f -name "kv_${CORE}_S*_noise*_eta*_lam*_seed*.pt" | sort)

    if [ ${#CKPT_LIST[@]} -eq 0 ]; then
        echo "[WARN] No checkpoints found for CORE=${CORE} in ${CKPT_DIR}"
        continue
    fi

    for CKPT in "${CKPT_LIST[@]}"; do
        echo "-----------------------------------------------"
        echo "[RUN] checkpoint = ${CKPT}"
        echo "-----------------------------------------------"

        BASE=$(basename "${CKPT}" .pt)

        # S をファイル名から取得
        #   kv_fw_S3_fw1_eta03_lam0950_seed0 → S=3
        S=$(echo "${BASE}" | sed -E 's/.*_S([0-9]+)_.*/\1/')

        # 先頭の "kv_CORE_S<S>_" を削る
        #   kv_fw_S3_fw1_eta03_lam0950_seed0 → fw1_eta03_lam0950_seed0
        REST_HEADLESS=${BASE#kv_${CORE}_S${S}_}

        # 先頭の "fwX_" を削る (fw0_ / fw1_ どちらにも対応)
        #   fw1_eta03_lam0950_seed0 → eta03_lam0950_seed0
        REST=${REST_HEADLESS#fw?_}

        # 最終的な CSV 名:
        #   A_kv_fw_S3_eta03_lam0950_seed0.csv
        OUTCSV="${OUTDIR}/A_kv_${CORE}_S${S}_${REST}.csv"

        echo "[INFO] S=${S}, REST=${REST}"
        echo "[INFO] out_csv = ${OUTCSV}"

        python3 run_A_kv.py \
            --ckpt "${CKPT}" \
            --out_csv "${OUTCSV}" \
            --core_type "${CORE}" \
            --d_g ${DG} \
            --d_h ${DH} \
            --T_bind ${TBIND} \
            --duplicate ${DUPLICATE} \
            --beta ${BETA} \
            --batch_size ${BATCH} \
            --S ${S} \
            --num_classes "${NUM_CLASSES}" \
            --num_wait "${NUM_WAIT}" \
            --bind_noise_std "${BIND_NOISE_STD}" \
            ${USE_FW} \
            ${USE_LN} \
            ${GPU}

        echo "[DONE] Saved → ${OUTCSV}"
        echo ""
    done
done

echo "==============================================="
echo "All A-dynamics runs completed."
echo "==============================================="

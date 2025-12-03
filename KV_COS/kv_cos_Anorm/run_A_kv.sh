#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
OUTDIR="results_A_kv"   # 結果ディレクトリ

DG=30
DH=80
TBIND=12        
DUPLICATE=3     
BETA=0.1        
BATCH=1         

GPU="--gpu"
ETA=0.3
USE_FW="--use_fw"
USE_LN="--use_ln"
SEED=0

# ==========================================
# ★★ 追加：クラス数設定 ★★
# ==========================================
NUM_CLASSES=${NUM_CLASSES:-30}

# Sweep する λ
LAMBDA_LIST="0.80 0.95"

# Sweep する S
S_LIST="1 3 5"

# Sweep する core タイプ
CORE_LIST="fw tanh"

# ==========================================
# 実行
# ==========================================
mkdir -p "${OUTDIR}"

for CORE in ${CORE_LIST}; do
    echo "==============================================="
    echo "[CORE] ${CORE}"
    echo "==============================================="

    for LAMBDA in ${LAMBDA_LIST}; do

        LAMSTR=$(printf "%.2f" ${LAMBDA} | sed 's/\.//')

        for S in ${S_LIST}; do

            CKPT="checkpoints/kv_${CORE}_S${S}_lam${LAMSTR}0_seed${SEED}.pt"

            if [ ! -f "${CKPT}" ]; then
                echo "[WARN] Checkpoint not found: ${CKPT}"
                echo "       → skip (CORE=${CORE}, S=${S}, lambda=${LAMBDA})"
                continue
            fi

            echo "-----------------------------------------------"
            echo "[RUN] A-dynamics | core=${CORE} | S=${S} | lambda=${LAMBDA}"
            echo "Using checkpoint: ${CKPT}"
            echo "-----------------------------------------------"

            OUTCSV="${OUTDIR}/A_kv_${CORE}_S${S}_lam${LAMSTR}.csv"

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
                --lambda_decay ${LAMBDA} \
                --eta ${ETA} \
                --num_classes "${NUM_CLASSES}" \
                ${USE_FW} \
                ${USE_LN} \
                ${GPU} \
                --seed ${SEED}

            echo "[DONE] Saved → ${OUTCSV}"
            echo ""
        done
    done
done

echo "==============================================="
echo "All A-dynamics runs completed."
echo "==============================================="

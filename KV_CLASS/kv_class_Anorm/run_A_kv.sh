#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
OUTDIR="results_A_kv"   # 結果ディレクトリ

DG=30
DH=80
NUM_CLASSES=10
TBIND=4
BATCH=1

GPU="--gpu"
LAMBDA_LIST="0.85 0.95"      # ★ 複数 λ に対応
ETA=0.3
USE_FW="--use_fw"
USE_LN="--use_ln"
SEED=0

# Sweep する S の値（学習時 S）
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

        # λ をファイル名用に整形（例: 0.85 → 085, 0.95 → 095）
        LAMSTR=$(printf "%.2f" ${LAMBDA} | sed 's/\.//')

        for S in ${S_LIST}; do

            # ===== λ入りcheckpoint名 =====
            CKPT="checkpoints/kv_${CORE}_S${S}_lam${LAMSTR}_seed${SEED}.pt"

            if [ ! -f "${CKPT}" ]; then
                echo "[WARN] Checkpoint not found: ${CKPT}"
                echo "       → スキップ (${CORE}, S=${S}, lambda=${LAMBDA})"
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
                --num_classes ${NUM_CLASSES} \
                --T_bind ${TBIND} \
                --batch_size ${BATCH} \
                --S ${S} \
                --lambda_decay ${LAMBDA} \
                --eta ${ETA} \
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

#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
EPOCHS=${EPOCHS:-150}
TRAIN_STEPS=${TRAIN_STEPS:-600}
VAL_STEPS=${VAL_STEPS:-200}
BATCH=${BATCH:-32}

# ==========================================
# ★★ 追加：class 数設定 ★★
# ==========================================
NUM_CLASSES=${NUM_CLASSES:-8}

# ==========================================
# モデル設定
# ==========================================
CORE_TYPE=${CORE_TYPE:-"fw"}

# ==========================================
# FW on/off
# ==========================================
USE_FW_LIST=${USE_FW_LIST:-"1"}

# ==========================================
# 次元設定
# ==========================================
DG=${DG:-30}
DH=${DH:-60}

# ==========================================
# Fast Weights パラメータ
# ==========================================
LAMBDA=${LAMBDA:-0.85}
ETA=${ETA:-0.30}

# ==========================================
# 最適化設定
# ==========================================
LR=${LR:-3e-3}
WD=${WD:-5e-4}
CLIP=${CLIP:-0.7}

# ==========================================
# Bind/Wait
# ==========================================
T_BIND=${T_BIND:-9}
DELTA_WAIT=${DELTA_WAIT:-0}
DUPLICATE=${DUPLICATE:-3}

# ==========================================
# Sweep
# ==========================================
S_LIST=${S_LIST:-"1"}
NOISE_LIST=${NOISE_LIST:-"1.0 0.8 0.5 0.3"}
BETA=${BETA:-0.0}
SEEDS=${SEEDS:-"0"}

# ==========================================
# 実行環境
# ==========================================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力先
# ==========================================
OUT_DIR=${OUT_DIR:-"results_recon_fw"}
if [[ -d "${OUT_DIR}" ]]; then
  BACKUP="${OUT_DIR}_backup_$(date +"%Y%m%d_%H%M%S")"
  echo "[WARN] Moving old results → ${BACKUP}"
  mv "${OUT_DIR}" "${BACKUP}"
fi
mkdir -p "${OUT_DIR}"

# ==========================================
# GPU フラグ
# ==========================================
GPU_FLAG=""
if [[ "${GPU}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpu"
fi

# ==========================================
# Sweep loop
# ==========================================
for USE_FW in ${USE_FW_LIST}; do
  for S_FIXED in ${S_LIST}; do
    for SIGMA in ${NOISE_LIST}; do
      for SEED in ${SEEDS}; do

        FW_FLAG="--use_fw"
        [[ "${USE_FW}" == "0" ]] && FW_FLAG="--no-use_fw"

        echo "=========================================="
        echo "[RUN] recon | core=${CORE_TYPE} | FW=${USE_FW} | S=${S_FIXED} | dup=${DUPLICATE} | σ=${SIGMA} | seed=${SEED}"
        echo "=========================================="

        OUT_CSV="${OUT_DIR}/recon_${CORE_TYPE}_fw${USE_FW}_beta${BETA}_dup${DUPLICATE}_sigma${SIGMA}_S${S_FIXED}_seed${SEED}.csv"

        ${PYTHON_BIN} run.py \
          --epochs "${EPOCHS}" \
          --train_steps "${TRAIN_STEPS}" \
          --val_steps "${VAL_STEPS}" \
          --batch_size "${BATCH}" \
          --d_g "${DG}" \
          --d_h "${DH}" \
          --T_bind "${T_BIND}" \
          --duplicate "${DUPLICATE}" \
          --lambda_decay "${LAMBDA}" \
          --eta "${ETA}" \
          --S "${S_FIXED}" \
          --seed "${SEED}" \
          --lr "${LR}" \
          --weight_decay "${WD}" \
          --grad_clip "${CLIP}" \
          --beta "${BETA}" \
          --delta_wait "${DELTA_WAIT}" \
          --core_type "${CORE_TYPE}" \
          --bind_noise_std "${SIGMA}" \
          --query_noise_std "${SIGMA}" \
          --num_classes "${NUM_CLASSES}" \
          --use_ln \
          ${FW_FLAG} \
          ${GPU_FLAG} \
          --out_csv "${OUT_CSV}"

      done
    done
  done
done

echo "=========================================="
echo "✅ All reconstruction experiments completed."
echo "Results saved in: ${OUT_DIR}/"
echo "=========================================="

#!/usr/bin/env bash
set -euo pipefail

############################################
# ========== 基本設定 ==========
############################################
EPOCHS=${EPOCHS:-150}
TRAIN_STEPS=${TRAIN_STEPS:-10000}
VAL_STEPS=${VAL_STEPS:-2000}
BATCH=${BATCH:-32}

############################################
# ========== クラス / 次元設定 ==========
############################################
NUM_CLASSES=${NUM_CLASSES:-8}
DG=${DG:-30}
DH=${DH:-60}

############################################
# ========== モデル設定 ==========
############################################
# core_type は tanh-FW 固定
CORE_TYPE="tanh"

############################################
# ========== 最適化 ==========
############################################
LR=${LR:-3e-3}
WD=${WD:-5e-4}
CLIP=${CLIP:-0.7}

############################################
# ========== Bind / Wait 設定 ==========
############################################
T_BIND=${T_BIND:-12}
DELTA_WAIT=${DELTA_WAIT:-0}
DUPLICATE=${DUPLICATE:-4}

############################################
# ========== FWパラメータ ==========
############################################
LAMBDA=${LAMBDA:-0.95}
ETA=${ETA:-0.30}

############################################
# ========== Sweep 設定 ==========
############################################
S_LIST=${S_LIST:-"1"}              # S-loop sweep
NOISE_LIST=${NOISE_LIST:-"0.4"}  # noise sweep
BETA=${BETA:-0.0}
SEEDS=${SEEDS:-"0"}

############################################
# ========== 実行環境 ==========
############################################
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

############################################
# ========== 出力ディレクトリ ==========
############################################
OUT_DIR=${OUT_DIR:-"results_recon_tanh"}

if [[ -d "${OUT_DIR}" ]]; then
  BACKUP="${OUT_DIR}_backup_$(date +"%Y%m%d_%H%M%S")"
  echo "[WARN] Moving old results → ${BACKUP}"
  mv "${OUT_DIR}" "${BACKUP}"
fi
mkdir -p "${OUT_DIR}"

############################################
# ========== GPUフラグ ==========
############################################
GPU_FLAG=""
if [[ "${GPU}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpu"
fi

############################################
# ========== 設定ログ ==========
############################################
LOG="${OUT_DIR}/run_config.txt"
{
  echo "[INFO] TANH Reconstruction Sweep"
  echo "NUM_CLASSES=${NUM_CLASSES}"
  echo "DG=${DG}, DH=${DH}"
  echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}"
  echo "T_BIND=${T_BIND}, DUPLICATE=${DUPLICATE}, DELTA_WAIT=${DELTA_WAIT}"
  echo "LAMBDA=${LAMBDA}, ETA=${ETA}, BETA=${BETA}"
  echo "S_LIST=${S_LIST}"
  echo "NOISE_LIST=${NOISE_LIST}"
  echo "SEEDS=${SEEDS}"
  echo "CORE_TYPE=${CORE_TYPE}"
} > "${LOG}"

############################################
# ========== Sweep 実行 ==========
############################################
for S_FIXED in ${S_LIST}; do
  for SIGMA in ${NOISE_LIST}; do
    for SEED in ${SEEDS}; do

      echo "=========================================="
      echo "[RUN] recon-${CORE_TYPE} | dup=${DUPLICATE} | S=${S_FIXED} | σ=${SIGMA} | seed=${SEED}"
      echo "=========================================="

      OUT_CSV="${OUT_DIR}/recon_${CORE_TYPE}_beta${BETA}_dup${DUPLICATE}_sigma${SIGMA}_S${S_FIXED}_seed${SEED}.csv"

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
        ${GPU_FLAG} \
        --out_csv "${OUT_CSV}"

    done
  done
done

echo "=========================================="
echo "  ✅ All TANH reconstruction experiments completed."
echo "  Results saved in: ${OUT_DIR}/"
echo "=========================================="

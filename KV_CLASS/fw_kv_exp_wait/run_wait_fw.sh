#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
EPOCHS=${EPOCHS:-150}
TRAIN_STEPS=${TRAIN_STEPS:-10000}
VAL_STEPS=${VAL_STEPS:-2000}
BATCH=${BATCH:-32}

# ==========================================
# モデル設定（FW固定）
# ==========================================
CORE_TYPE="fw"

# ==========================================
# 入力・次元設定
# ==========================================
DG=${DG:-100}
DH=${DH:-50}
NUM_CLASSES=${NUM_CLASSES:-10}

# ==========================================
# 最適化
# ==========================================
LR=${LR:-3e-3}
WD=${WD:-5e-4}
CLIP=${CLIP:-0.7}

# ==========================================
# 容量・待機設定
# ==========================================
T_BIND=${T_BIND:-5}

# ==========================================
# Fast Weights パラメータ（Sweep対応）
# ==========================================
LAMBDA=${LAMBDA:-0.95}
ETA=${ETA:-0.30}
S_LIST=${S_LIST:-"1"} 

# ==========================================
# Sweep 対象
# ==========================================
DELTA_WAIT_LIST=${DELTA_WAIT_LIST:-"0 2 4 6 8 10 12 15 18"}
SIGMA=${SIGMA:-0.00}
BETA=${BETA:-1.0}
SEEDS=${SEEDS:-"0"}

# ==========================================
# 実行環境
# ==========================================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力ディレクトリ（安全設計）
# ==========================================
OUT_DIR=${OUT_DIR:-"results_wait_fw"}

if [[ -d "${OUT_DIR}" ]]; then
  BACKUP="${OUT_DIR}_backup_$(date +"%Y%m%d_%H%M%S")"
  echo "[WARN] Moving old results → ${BACKUP}"
  mv "${OUT_DIR}" "${BACKUP}"
fi
mkdir -p "${OUT_DIR}"

# ==========================================
# GPUフラグ
# ==========================================
GPU_FLAG=""
if [[ "${GPU}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpu"
fi

# ==========================================
# 設定ログ保存
# ==========================================
CONFIG_FILE="${OUT_DIR}/run_config.txt"

{
  echo "[INFO] FW Wait Sweep Configuration"
  echo "DG=${DG}, DH=${DH}, NUM_CLASSES=${NUM_CLASSES}"
  echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}"
  echo "T_BIND=${T_BIND}"
  echo "LAMBDA=${LAMBDA}, ETA=${ETA}"
  echo "S_LIST=${S_LIST}"
  echo "DELTA_WAIT_LIST=${DELTA_WAIT_LIST}"
  echo "SIGMA=${SIGMA}, BETA=${BETA}"
  echo "SEEDS=${SEEDS}"
} > "${CONFIG_FILE}"

echo "[INFO] Saved config → ${CONFIG_FILE}"

# ==========================================
# 実験ループ（Wait × S × seed）
# ==========================================
for S_FIXED in ${S_LIST}; do
  for DELTA_WAIT in ${DELTA_WAIT_LIST}; do
    for SEED in ${SEEDS}; do

      OUT_CSV="${OUT_DIR}/fw_sigma${SIGMA}_wait${DELTA_WAIT}_S${S_FIXED}_seed${SEED}.csv"

      echo "[RUN] FW | σ=${SIGMA} | Δ_wait=${DELTA_WAIT} | S=${S_FIXED} | seed=${SEED}"

      ${PYTHON_BIN} run.py \
        --epochs "${EPOCHS}" \
        --train_steps "${TRAIN_STEPS}" \
        --val_steps "${VAL_STEPS}" \
        --batch_size "${BATCH}" \
        --d_g "${DG}" \
        --d_h "${DH}" \
        --num_classes "${NUM_CLASSES}" \
        --T_bind "${T_BIND}" \
        --noise_std "${SIGMA}" \
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
        ${GPU_FLAG} \
        --out_csv "${OUT_CSV}"

    done
  done
done

echo "=========================================="
echo "  All Wait × S × seed FW experiments done."
echo "  Results saved in → ${OUT_DIR}/"
echo "=========================================="
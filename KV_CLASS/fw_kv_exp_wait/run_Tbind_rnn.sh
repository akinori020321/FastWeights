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
# モデル設定
# ==========================================
CORE_TYPE=${CORE_TYPE:-"fw"}   # fw / rnn / tanh など

# ==========================================
# 入力次元
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
# Fast Weights パラメータ（使わない）
# ==========================================
LAMBDA=${LAMBDA:-0.05}
ETA=${ETA:-0.}
S_LIST=${S_LIST:-"1"} 

# ==========================================
# Sweep 対象
# ==========================================
T_BIND_LIST=${T_BIND_LIST:-"2 4 6 8 10 12 15 18"}
FW_LIST=${FW_LIST:-"0"}
DELTA_WAIT=0
SIGMA=0.00
BETA=${BETA:-1.0}
SEEDS=${SEEDS:-"0"}

# ==========================================
# 実行環境
# ==========================================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力ディレクトリ
# ==========================================
OUT_DIR=${OUT_DIR:-"results_Tbind_rnn"}

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
# 設定ログ
# ==========================================
CONFIG_FILE="${OUT_DIR}/run_config.txt"
{
  echo "[INFO] T_bind Capacity Sweep Configuration"
  echo "CORE_TYPE=${CORE_TYPE}"
  echo "DG=${DG}, DH=${DH}, NUM_CLASSES=${NUM_CLASSES}"
  echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}"
  echo "LAMBDA=${LAMBDA}, ETA=${ETA}"
  echo "S_LIST=${S_LIST}"
  echo "T_BIND_LIST=${T_BIND_LIST}"
  echo "DELTA_WAIT=${DELTA_WAIT}, SIGMA=${SIGMA}, BETA=${BETA}"
  echo "SEEDS=${SEEDS}"
} > "${CONFIG_FILE}"

echo "[INFO] Saved config → ${CONFIG_FILE}"

# ==========================================
# Sweep Loop（T_bind × S × FW × seed）
# ==========================================
for T_BIND in ${T_BIND_LIST}; do
  for S in ${S_LIST}; do
    for FW in ${FW_LIST}; do
      for SEED in ${SEEDS}; do

        FW_FLAG="--use_fw"
        [[ "${FW}" == "0" ]] && FW_FLAG="--no_fw"

        OUT_CSV="${OUT_DIR}/${CORE_TYPE}_fw${FW}_T${T_BIND}_S${S}_seed${SEED}.csv"

        echo "[RUN] CORE=${CORE_TYPE} | FW=${FW} | T_bind=${T_BIND} | S=${S} | seed=${SEED}"

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
          --S "${S}" \
          --seed "${SEED}" \
          --lr "${LR}" \
          --weight_decay "${WD}" \
          --grad_clip "${CLIP}" \
          --beta "${BETA}" \
          --delta_wait "${DELTA_WAIT}" \
          --core_type "${CORE_TYPE}" \
          --use_ln \
          ${FW_FLAG} \
          ${GPU_FLAG} \
          --out_csv "${OUT_CSV}"

      done
    done
  done
done

echo "=========================================="
echo "  All T_bind × S × FW × seed experiments done."
echo "  Results saved in → ${OUT_DIR}/"
echo "=========================================="

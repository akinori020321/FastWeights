#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
EPOCHS=${EPOCHS:-500}
TRAIN_STEPS=${TRAIN_STEPS:-10000}
VAL_STEPS=${VAL_STEPS:-2000}
BATCH=${BATCH:-32}

# ==========================================
# モデル設定
# （FW / RNN / tanh などを切り替えたい場合は
#  CORE_TYPE を変える）
# ==========================================
CORE_TYPE=${CORE_TYPE:-"tanh"}   # "fw", "rnn", "tanh" など

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
# Fast Weights パラメータ
# ==========================================
LAMBDA=${LAMBDA:-}
ETA=${ETA:-}
S_LIST=${S_LIST:-""} 

# ==========================================
# Sweep 対象
# ==========================================
T_BIND_LIST=${T_BIND_LIST:-"2 4 6 8 10 12 14 16 18 20"}
DELTA_WAIT=0                           
SIGMA=0.00                                
BETA=${BETA:-1.0}
SEEDS=${SEEDS:-"0 1 2"}

# ==========================================
# 実行環境
# ==========================================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力ディレクトリ（安全設計）
# ==========================================
OUT_DIR=${OUT_DIR:-"results_Tbind_${CORE_TYPE}"}

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
  echo "[INFO] T_bind Capacity Sweep Configuration"
  echo "CORE_TYPE=${CORE_TYPE}"
  echo "DG=${DG}, DH=${DH}, NUM_CLASSES=${NUM_CLASSES}"
  echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}"
  echo "LAMBDA=${LAMBDA}, ETA=${ETA}"
  echo "T_BIND_LIST=${T_BIND_LIST}"
  echo "DELTA_WAIT=${DELTA_WAIT}, SIGMA=${SIGMA}, BETA=${BETA}"
  echo "SEEDS=${SEEDS}"
} > "${CONFIG_FILE}"

echo "[INFO] Saved config → ${CONFIG_FILE}"

# ==========================================
# Sweep ループ（T_bind × seed）
# ==========================================
for S_FIXED in ${S_LIST}; do
  for T_BIND in ${T_BIND_LIST}; do
    for SEED in ${SEEDS}; do

      OUT_CSV="${OUT_DIR}/${CORE_TYPE}_T${T_BIND}_S${S_FIXED}_seed${SEED}.csv"

      echo "[RUN] ${CORE_TYPE} | T_bind=${T_BIND} | S=${S_FIXED} | seed=${SEED}"

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
        --use_ln --use_fw \
        ${GPU_FLAG} \
        --out_csv "${OUT_CSV}"

    done
  done
done

echo "=========================================="
echo "  All T_bind × seed experiments done."
echo "  Results saved in → ${OUT_DIR}/"
echo "=========================================="

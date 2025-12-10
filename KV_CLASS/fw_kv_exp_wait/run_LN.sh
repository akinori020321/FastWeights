#!/usr/bin/env bash
set -euo pipefail

# ==============================
# 基本設定
# ==============================
EPOCHS=${EPOCHS:-150}
TRAIN_STEPS=${TRAIN_STEPS:-10000}
VAL_STEPS=${VAL_STEPS:-2000}
BATCH=${BATCH:-32}

# ==============================
# モデル設定（FW基準）
# ==============================
CORE_TYPE="fw"

# ==============================
# 次元設定
# ==============================
DG=${DG:-100}
DH=${DH:-50}
NUM_CLASSES=${NUM_CLASSES:-10}

# ==============================
# 最適化
# ==============================
LR=${LR:-3e-3}
WD=${WD:-5e-4}
CLIP=${CLIP:-0.7}

# ==============================
# タスク設定
# ==============================
T_BIND=${T_BIND:-5}
DELTA_WAIT=${DELTA_WAIT:-2}
BETA=${BETA:-1.0}

# ==============================
# Fast Weights パラメータ
# ==============================
LAMBDA=${LAMBDA:-0.0}
ETA=${ETA:-0.0}
S_LIST=${S_LIST:-"1"}

# ==============================
# Sweep 対象
# ==============================
SEEDS=${SEEDS:-"0"}
FW_LIST=${FW_LIST:-"0"}
LN_LIST=${LN_LIST:-"0 1"}

# ==============================
# 実行環境
# ==============================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力設定（安全設計）
# ==========================================
OUT_DIR=${OUT_DIR:-"results_fw_ln_sweep"}

if [[ -d "${OUT_DIR}" ]]; then
  BACKUP="${OUT_DIR}_backup_$(date +"%Y%m%d_%H%M%S")"
  echo "[WARN] Moving old results directory → ${BACKUP}"
  mv "${OUT_DIR}" "${BACKUP}"
fi
mkdir -p "${OUT_DIR}"

# GPUフラグ
GPU_FLAG=""
if [[ "${GPU}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpu"
fi

# ==========================================
# 実験パラメータログ保存
# ==========================================
CONFIG_FILE="${OUT_DIR}/run_config.txt"

echo "[INFO] FW × LN × S × seed Sweep" > "${CONFIG_FILE}"
echo "DG=${DG}, DH=${DH}, NUM_CLASSES=${NUM_CLASSES}" >> "${CONFIG_FILE}"
echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}" >> "${CONFIG_FILE}"
echo "T_BIND=${T_BIND}, DELTA_WAIT=${DELTA_WAIT}" >> "${CONFIG_FILE}"
echo "LAMBDA=${LAMBDA}, ETA=${ETA}, BETA=${BETA}" >> "${CONFIG_FILE}"
echo "S_LIST=${S_LIST}" >> "${CONFIG_FILE}"
echo "FW_LIST=${FW_LIST}, LN_LIST=${LN_LIST}" >> "${CONFIG_FILE}"
echo "SEEDS=${SEEDS}" >> "${CONFIG_FILE}"

echo "[INFO] Configuration saved to ${CONFIG_FILE}"

# ==========================================
# 関数：1条件実行
# ==========================================
run_one () {
  local FW="$1"
  local LN="$2"
  local S="$3"
  local SEED="$4"

  local FW_FLAG="--use_fw"; [[ "$FW" == "0" ]] && FW_FLAG="--no_fw"
  local LN_FLAG="--use_ln"; [[ "$LN" == "0" ]] && LN_FLAG="--no_ln"

  local OUT_CSV="${OUT_DIR}/fw${FW}_ln${LN}_S${S}_T${T_BIND}_wait${DELTA_WAIT}_seed${SEED}.csv"

  echo "=== RUN FW=${FW} LN=${LN} S=${S} seed=${SEED} ==="

  ${PYTHON_BIN} run.py \
    --epochs "${EPOCHS}" \
    --train_steps "${TRAIN_STEPS}" \
    --val_steps "${VAL_STEPS}" \
    --batch_size "${BATCH}" \
    --d_g "${DG}" \
    --d_h "${DH}" \
    --num_classes "${NUM_CLASSES}" \
    --T_bind "${T_BIND}" \
    --S "${S}" \
    --beta "${BETA}" \
    --delta_wait "${DELTA_WAIT}" \
    --lambda_decay "${LAMBDA}" \
    --eta "${ETA}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --grad_clip "${CLIP}" \
    --core_type "${CORE_TYPE}" \
    $FW_FLAG $LN_FLAG $GPU_FLAG \
    --out_csv "${OUT_CSV}"
}

# ==========================================
# Sweep Loop
# ==========================================
echo "==== Running Sweep FW × LN × S × seed (Δ_wait=${DELTA_WAIT}) ===="

for S in ${S_LIST}; do
  for FW in ${FW_LIST}; do
    for LN in ${LN_LIST}; do
      for SEED in ${SEEDS}; do
        run_one "$FW" "$LN" "$S" "$SEED"
      done
    done
  done
done

echo "============================================"
echo "  All experiments completed!"
echo "  Results saved in: ${OUT_DIR}/"
echo "============================================"
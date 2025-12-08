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
# モデル設定
# ==============================
CORE_TYPE="tanh"

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
DELTA_WAIT=${DELTA_WAIT:2}
BETA=${BETA:-1.0}

# ==============================
# Sweep 対象（★★変更点★★）
# ==============================
# 小さい → 大きい で 4段階ずつ
LAMBDA_LIST=${LAMBDA_LIST:-"0.65 0.85 0.95"}
ETA_LIST=${ETA_LIST:-"0.10 0.30 0.50"}

SEEDS=${SEEDS:-"0"}
S_LIST=${S_LIST:-"3"}      # 固定でOK。変更したい場合はここ
FW_LIST=${FW_LIST:-"1"}    # FWは常にONで良いので 1 のまま
LN_LIST=${LN_LIST:-"1"}    # LNも固定で良い

# ==============================
# 実行環境
# ==============================
GPU=${GPU:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}

# ==========================================
# 出力設定
# ==========================================
OUT_DIR=${OUT_DIR:-"results_SC3_lambda_eta_sweep"}

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
# 実験パラメータログ
# ==========================================
CONFIG_FILE="${OUT_DIR}/run_config.txt"

echo "[INFO] FW × LN（固定） × S（固定） × lambda × eta × seed Sweep" > "${CONFIG_FILE}"
echo "DG=${DG}, DH=${DH}, NUM_CLASSES=${NUM_CLASSES}" >> "${CONFIG_FILE}"
echo "LR=${LR}, WD=${WD}, CLIP=${CLIP}" >> "${CONFIG_FILE}"
echo "T_BIND=${T_BIND}, DELTA_WAIT=${DELTA_WAIT}, BETA=${BETA}" >> "${CONFIG_FILE}"
echo "LAMBDA_LIST=${LAMBDA_LIST}" >> "${CONFIG_FILE}"
echo "ETA_LIST=${ETA_LIST}" >> "${CONFIG_FILE}"
echo "S_LIST=${S_LIST}" >> "${CONFIG_FILE}"
echo "FW_LIST=${FW_LIST}, LN_LIST=${LN_LIST}" >> "${CONFIG_FILE}"
echo "SEEDS=${SEEDS}" >> "${CONFIG_FILE}"

echo "[INFO] Configuration saved to ${CONFIG_FILE}"

# ==========================================
# 関数：1条件実行
# ==========================================
run_one () {
  local LAM="$1"
  local ETA="$2"
  local S="$3"
  local SEED="$4"

  local OUT_CSV="${OUT_DIR}/lam${LAM}_eta${ETA}_S${S}_seed${SEED}.csv"

  echo "=== RUN λ=${LAM} η=${ETA} S=${S} seed=${SEED} ==="

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
    --lambda_decay "${LAM}" \
    --eta "${ETA}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --grad_clip "${CLIP}" \
    --core_type "${CORE_TYPE}" \
    --use_fw --use_ln $GPU_FLAG \
    --out_csv "${OUT_CSV}"
}

# ==========================================
# Sweep Loop（★★ λ × η ★★）
# ==========================================
echo "==== Running Sweep λ × η × S × seed (Δ_wait=${DELTA_WAIT}) ===="

for S in ${S_LIST}; do
  for LAM in ${LAMBDA_LIST}; do
    for ETA in ${ETA_LIST}; do
      for SEED in ${SEEDS}; do
        run_one "$LAM" "$ETA" "$S" "$SEED"
      done
    done
  done
done

echo "============================================"
echo "  All experiments completed!"
echo "  Results saved in: ${OUT_DIR}/"
echo "============================================"

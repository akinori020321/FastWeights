#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# 基本設定
# ==========================================
OUTDIR="results_A_kv"   # 結果ディレクトリ

DG=100
DH=50
NUM_CLASSES=10
TBIND=5
BATCH=1

# ★ 追加：待機ステップ数
NUM_WAIT=2   # ← 好きな値に変更（例：3回 wait を挿入）

GPU="--gpu"
USE_FW="--use_fw"
USE_LN="--use_ln"

# Sweep する S の値
S_LIST="1 3"

# Sweep する core タイプ
CORE_LIST="rnn fw tanh"

# ==========================================
# 実行
# ==========================================
mkdir -p "${OUTDIR}"

# チェックポイントは checkpoints/* のタイムスタンプディレクトリから検索
CKPT_DIR="checkpoints"

for CORE in ${CORE_LIST}; do
    echo "==============================================="
    echo "[CORE] ${CORE}"
    echo "==============================================="

    # checkpoints/*/kv_CORE_S*_fw*_eta*_lam*_seed*.pt を検索
    mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -type f -name "kv_${CORE}_S*_fw*_eta*_lam*_seed*.pt" | sort)

    if [ ${#CKPT_LIST[@]} -eq 0 ]; then
        echo "[WARN] No checkpoints found for CORE=${CORE}"
        continue
    fi

    # 見つかった checkpoint をすべて処理
    for CKPT in "${CKPT_LIST[@]}"; do
        echo "-----------------------------------------------"
        echo "[RUN] checkpoint = ${CKPT}"
        echo "-----------------------------------------------"

        # ===== S をファイル名からパース =====
        S=$(basename "${CKPT}" | sed -E 's/.*_S([0-9]+)_.*/\1/')

        # ===== 出力ファイル名作成 =====
        BASE=$(basename "${CKPT}" .pt)

        REST=${BASE#kv_${CORE}_S${S}_fw1_}

        OUTCSV="${OUTDIR}/A_kv_${CORE}_S${S}_${REST}.csv"

        # ===== run_A_kv 実行 =====
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
            --num_wait ${NUM_WAIT} \
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

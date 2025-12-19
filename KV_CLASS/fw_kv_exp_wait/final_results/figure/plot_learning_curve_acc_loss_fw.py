#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_learning_curve_acc_loss_fw.py
----------------------------------
- ../T_bind/results_Tbind_fw4/ から
  fw_fw1_T4_S1_seed*_beta1.00_wait2.csv を全取得
- seedごとの val_acc / val_loss を読み込み
- epochごとに mean±std を計算して
  上: Acc, 下: Loss の2段で描画
- eps/pdf 保存
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# パス設定（スクリプト基準で相対）
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(THIS_DIR, "..", "T_bind", "results_Tbind_fw4")
OUT_DIR  = os.path.join(THIS_DIR, "fig")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# CSV列名の自動推定
# ======================================================
def pick_col(df: pd.DataFrame, candidates, kind: str):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(f"[Error] {kind} column not found. candidates={candidates}, available={list(df.columns)}")


def load_one_csv(path: str):
    df = pd.read_csv(path)

    epoch_col = pick_col(df, ["epoch", "ep", "step"], "epoch")
    acc_col   = pick_col(df, ["val_acc", "valid_acc", "acc_val", "acc"], "val_acc")
    loss_col  = pick_col(df, ["val_loss", "valid_loss", "loss_val", "loss", "ce"], "val_loss")

    ep   = df[epoch_col].astype(float).to_numpy()
    acc  = df[acc_col].astype(float).to_numpy()
    loss = df[loss_col].astype(float).to_numpy()
    return ep, acc, loss


def extract_seed(path: str) -> int:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


# ======================================================
# 複数seedの集約（epochで整列し、mean/std）
# ======================================================
def aggregate(files):
    if len(files) == 0:
        raise FileNotFoundError("No CSV files found.")

    frames_acc = []
    frames_loss = []

    for f in files:
        ep, acc, loss = load_one_csv(f)
        seed = extract_seed(f)

        s_acc = pd.Series(acc, index=ep, name=f"seed{seed}")
        s_loss = pd.Series(loss, index=ep, name=f"seed{seed}")

        frames_acc.append(s_acc)
        frames_loss.append(s_loss)

    acc_df = pd.concat(frames_acc, axis=1).sort_index()
    loss_df = pd.concat(frames_loss, axis=1).sort_index()

    # epochごとに mean/std（欠損があっても計算可能）
    acc_mean = acc_df.mean(axis=1, skipna=True).to_numpy()
    acc_std  = acc_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    loss_mean = loss_df.mean(axis=1, skipna=True).to_numpy()
    loss_std  = loss_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    epochs = acc_df.index.to_numpy(dtype=float)
    return epochs, acc_mean, acc_std, loss_mean, loss_std


def moving_average(y, w: int):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="fw_fw1_T4_S1_seed*_beta1.00_wait2.csv")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (default 1)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "learning_curve_acc_loss_fw_T4_wait2_beta1.00.eps"))
    ap.add_argument("--title", default="")  # 必要なら
    args = ap.parse_args()

    glob_pat = os.path.join(CSV_DIR, args.pattern)
    files = sorted(glob.glob(glob_pat), key=extract_seed)

    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {glob_pat}")

    epochs, acc_m, acc_s, loss_m, loss_s = aggregate(files)

    # smoothing（meanだけに適用）
    acc_m_s  = moving_average(acc_m, args.smooth)
    loss_m_s = moving_average(loss_m, args.smooth)

    # ===== plot =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.2), sharex=True)

    # Acc
    ax1.plot(epochs, acc_m_s, linewidth=1.8, label="FW (mean)")
    ax1.fill_between(epochs, acc_m - acc_s, acc_m + acc_s, alpha=0.18, label="±1 std")
    ax1.set_ylabel("Validation Acc")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", frameon=True)

    # Loss
    ax2.plot(epochs, loss_m_s, linewidth=1.8, label="FW (mean)")
    ax2.fill_between(epochs, loss_m - loss_s, loss_m + loss_s, alpha=0.18, label="±1 std")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss (CE)")
    ax2.grid(True, alpha=0.25)

    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")

    # epsならpdfも併産
    if args.out.lower().endswith(".eps"):
        fig.savefig(args.out[:-4] + ".pdf", bbox_inches="tight")

    print(f"[OK] matched {len(files)} files")
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()

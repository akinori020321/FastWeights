#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_tbind_curve_only.py
========================================================
Bind=4 学習曲線（Loss のみ：FW seed曲線 + FW mean±std + RNN mean）
    - ../T_bind/results_Tbind_fw4/ から
        fw_fw1_T4_S1_seed*_beta1.00_wait2.csv
      -> seed曲線 + mean±std + mean（黒破線）
    - ../T_bind/results_Tbind_rnn/ から bind=4（T4）のCSVを探索
      -> mean（赤実線）のみ

追加:
  - FW の「最終epochのLossが最小」の seed を1本だけ描く図（RNNなし）
    -> 図の見方・収束イメージ説明用

出力:
  out_dir/ に
    - learning_curve_acc_fw_vs_rnn_T4_wait2_beta1.00.(png|eps)
    - learning_curve_acc_fw_best_T4_wait2_beta1.00.(png|eps)

変更点（今回）:
  - 図タイトルは表示しない（上部タイトル削除）
  - 縦軸ラベルは "Negative log likelihood" に統一
  - 縦軸・横軸ラベルを大きくする
========================================================
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# スクリプト自身のディレクトリ
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# 入力パス
# ======================================================
# Bind=4 learning curve 用
FW_CSV_DIR_DEFAULT  = os.path.join(THIS_DIR, "..", "T_bind", "results_Tbind_fw4")
RNN_CSV_DIR_DEFAULT = os.path.join(THIS_DIR, "..", "T_bind", "results_Tbind_rnn")

# ======================================================
# ★出力先
# ======================================================
OUT_DIR_DEFAULT = os.path.join(THIS_DIR, "Tbind_and_curve_fig")
os.makedirs(OUT_DIR_DEFAULT, exist_ok=True)


# ======================================================
# Bind=4 learning curve: 共通ユーティリティ
# ======================================================
def pick_col(df: pd.DataFrame, candidates, kind: str):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(
        f"[Error] {kind} column not found. candidates={candidates}, available={list(df.columns)}"
    )


def load_one_csv(path: str):
    df = pd.read_csv(path)

    epoch_col = pick_col(df, ["epoch", "ep", "step"], "epoch")
    acc_col   = pick_col(df, ["val_acc", "valid_acc", "acc_val", "acc"], "val_acc")
    loss_col  = pick_col(df, ["val_loss", "valid_loss", "loss_val", "loss", "ce", "val_ce"], "val_loss")

    ep   = df[epoch_col].astype(float).to_numpy()
    acc  = df[acc_col].astype(float).to_numpy()
    loss = df[loss_col].astype(float).to_numpy()
    return ep, acc, loss


def extract_seed(path: str) -> int:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


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

    acc_mean = acc_df.mean(axis=1, skipna=True).to_numpy()
    acc_std  = acc_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    loss_mean = loss_df.mean(axis=1, skipna=True).to_numpy()
    loss_std  = loss_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    epochs = acc_df.index.to_numpy(dtype=float)

    return epochs, acc_mean, acc_std, loss_mean, loss_std, acc_df, loss_df


def moving_average(y, w: int):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()


def find_files_with_fallback(csv_dir: str, patterns):
    for pat in patterns:
        glob_pat = os.path.join(csv_dir, pat)
        files = sorted(glob.glob(glob_pat), key=extract_seed)
        if len(files) > 0:
            return files, pat
    return [], None


def _read_final_loss_from_csv(path: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # 大文字/小文字揺れ対策
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ["valid_loss", "val_loss", "loss_val", "loss", "ce", "val_ce"]:
        if key in cols_lower:
            col = cols_lower[key]
            try:
                return float(df[col].iloc[-1])
            except Exception:
                return None
    return None


# ======================================================
# (A) FW seed + mean±std（Loss版 / RNNなし）
# ======================================================
def plot_learning_curve_acc_fw_vs_rnn(
    fw_dir: str,
    rnn_dir: str,        # ← 互換のため残す（使わない）
    fw_pattern: str,
    rnn_pattern: str,    # ← 互換のため残す（使わない）
    smooth: int,
    out_dir: str,
    no_rnn: bool,        # ← 互換のため残す（使わない）
):
    out_prefix = os.path.join(out_dir, "learning_curve_acc_fw_vs_rnn_T4_wait2_beta1.00")

    # ======================================================
    # FW files
    # ======================================================
    fw_glob = os.path.join(fw_dir, fw_pattern)
    fw_files = sorted(glob.glob(fw_glob), key=extract_seed)
    if len(fw_files) == 0:
        raise FileNotFoundError(f"[FW] No files matched: {fw_glob}")

    fw_epochs, fw_acc_m, fw_acc_s, fw_loss_m, fw_loss_s, fw_acc_df, fw_loss_df = aggregate(fw_files)
    fw_loss_m_s = moving_average(fw_loss_m, smooth)

    # ======================================================
    # plot (LOSS only)  ★横幅1.2倍 + 文字小さく + RNNなし
    # ======================================================
    fig, ax1 = plt.subplots(1, 1, figsize=(6.72, 3.7))  # ★(5.6,3.7) の横だけ1.2倍

    # ★文字小さめに
    AXIS_LABEL_FONTSIZE = 13
    TICK_FONTSIZE = 8
    LEGEND_FONTSIZE = 11

    # ★ゴチャつき軽減（seed線を薄く＆細く）
    SEED_LW = 1.0
    SEED_ALPHA = 0.40
    MEAN_LW = 1.2
    BAND_ALPHA_FW = 0.10

    # mean ± std band（loss）
    ax1.fill_between(
        fw_epochs, fw_loss_m - fw_loss_s, fw_loss_m + fw_loss_s,
        alpha=BAND_ALPHA_FW, zorder=1, label="FW ±1 std"
    )

    # seed lines（loss）
    for col in fw_loss_df.columns:
        ax1.plot(
            fw_loss_df.index.values, fw_loss_df[col].values,
            linewidth=SEED_LW, alpha=SEED_ALPHA, zorder=2
        )

    # FW mean（loss）
    ax1.plot(
        fw_epochs, fw_loss_m_s,
        color="black", linestyle="--", linewidth=MEAN_LW,
        zorder=3, label="FW mean"
    )

    # ★ラベル
    ax1.set_ylabel("Negative log likelihood", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # ★loss 用：0〜(最大値+余白)
    y_max = float(np.nanmax(fw_loss_m + fw_loss_s))
    ax1.set_ylim(0.0, max(0.1, y_max * 1.05))

    # ★x軸はそのまま（物理幅だけ伸ばす）
    ax1.set_xlim(-3, 153)
    xt = list(np.arange(0, 141, 20)) + [150]
    ax1.set_xticks(xt)

    ax1.grid(True, alpha=0.25)

    # ★凡例は右上（小さめ）
    ax1.legend(loc="upper right", frameon=True, fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()

    out_png = out_prefix + ".png"
    out_eps = out_prefix + ".eps"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_eps, bbox_inches="tight")

    print(f"[OK] FW matched {len(fw_files)} files: dir={fw_dir} pattern={fw_pattern}")
    print("[OK] RNN not added (disabled by design).")
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_eps}")


# ======================================================
# (B) 追加：FWの「最終Lossが最小」1本だけ（RNNなし）
# ======================================================
def plot_learning_curve_acc_fw_best_only(
    fw_dir: str,
    fw_pattern: str,
    smooth: int,
    out_dir: str,
):
    out_prefix = os.path.join(out_dir, "learning_curve_acc_fw_best_T4_wait2_beta1.00")

    fw_glob = os.path.join(fw_dir, fw_pattern)
    fw_files = sorted(glob.glob(fw_glob), key=extract_seed)
    if len(fw_files) == 0:
        raise FileNotFoundError(f"[FW-best] No files matched: {fw_glob}")

    # 最終epochのLossが最小の1本を選ぶ
    best_path = None
    best_final = float("inf")

    for p in fw_files:
        fl = _read_final_loss_from_csv(p)
        if fl is None:
            continue
        if fl < best_final:
            best_final = fl
            best_path = p

    if best_path is None:
        raise RuntimeError("[FW-best] Could not determine best seed (final loss not found).")

    ep, _, loss = load_one_csv(best_path)
    loss_line = moving_average(loss, smooth)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8))

    # ★軸ラベルを大きく
    AXIS_LABEL_FONTSIZE = 19
    TICK_FONTSIZE = 14

    ax.plot(
        ep, loss_line,
        linewidth=2.0,
        alpha=0.95,
        zorder=2,
    )

    # ★ラベル（指定どおり）
    ax.set_ylabel("Negative log likelihood", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # ★ loss 用：0〜(最大値+余白)
    y_max = float(np.nanmax(loss_line))
    ax.set_ylim(0.0, max(0.1, y_max * 1.05))

    # ★ 20刻みは維持しつつ「150だけ」追加で表示（両端に少し余白）
    ax.set_xlim(-3, 153)
    xt = list(np.arange(0, 141, 20)) + [150]  # 0,20,...,140,150
    ax.set_xticks(xt)

    ax.grid(True, alpha=0.25)

    # ★タイトルは付けない（ユーザ要望）
    # ax.set_title(...)

    fig.tight_layout()

    out_png = out_prefix + ".png"
    out_eps = out_prefix + ".eps"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_eps, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] FW-best picked: {os.path.basename(best_path)}  (final_loss={best_final:.6f})")
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_eps}")


# ======================================================
# main
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    # ★出力先
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="output directory for figure")

    # --- curve ---
    ap.add_argument("--fw_dir", default=FW_CSV_DIR_DEFAULT)
    ap.add_argument("--fw_pattern", default="fw_fw1_T4_S1_seed*_beta1.00_wait2.csv")

    ap.add_argument("--rnn_dir", default=RNN_CSV_DIR_DEFAULT)
    ap.add_argument("--rnn_pattern", default="rnn_T4_seed*_beta1.00_wait2.csv")
    ap.add_argument("--no_rnn", action="store_true", help="disable adding RNN mean")

    ap.add_argument("--smooth", type=int, default=1)

    # ★ best-run（追加図）
    ap.add_argument("--skip_best", action="store_true", help="skip FW best-only curve plot")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # (A) 図（FW seed + mean±std + RNN mean）
    plot_learning_curve_acc_fw_vs_rnn(
        fw_dir=args.fw_dir,
        rnn_dir=args.rnn_dir,
        fw_pattern=args.fw_pattern,
        rnn_pattern=args.rnn_pattern,
        smooth=args.smooth,
        out_dir=args.out_dir,
        no_rnn=args.no_rnn,
    )

    # (B) 追加図（FW best 1本）
    if not args.skip_best:
        plot_learning_curve_acc_fw_best_only(
            fw_dir=args.fw_dir,
            fw_pattern=args.fw_pattern,
            smooth=args.smooth,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

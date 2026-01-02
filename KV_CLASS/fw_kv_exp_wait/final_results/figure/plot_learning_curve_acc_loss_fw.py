#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_learning_curve_acc_loss_fw.py
----------------------------------
- ../T_bind/results_Tbind_fw4/ から
  fw_fw1_T4_S1_seed*_beta1.00_wait2.csv を全取得
- seedごとの val_acc / val_loss を読み込み
- epochごとに mean±std を計算して
- ★Acc と Loss を「別々のpng」にして保存（上下2段を廃止）
- ★各seedの曲線も重ね描き（seed感度の可視化）
- ★mean/±stdは背景（目立たせない）
- FW_bind4_fig/ に png (+eps) を保存
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
OUT_DIR  = os.path.join(THIS_DIR, "FW_bind4_fig")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# CSV列名の自動推定
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

        s_acc  = pd.Series(acc,  index=ep, name=f"seed{seed}")
        s_loss = pd.Series(loss, index=ep, name=f"seed{seed}")

        frames_acc.append(s_acc)
        frames_loss.append(s_loss)

    acc_df  = pd.concat(frames_acc,  axis=1).sort_index()
    loss_df = pd.concat(frames_loss, axis=1).sort_index()

    acc_mean  = acc_df.mean(axis=1, skipna=True).to_numpy()
    acc_std   = acc_df.std(axis=1, ddof=0, skipna=True).to_numpy()
    loss_mean = loss_df.mean(axis=1, skipna=True).to_numpy()
    loss_std  = loss_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    epochs = acc_df.index.to_numpy(dtype=float)

    return epochs, acc_mean, acc_std, loss_mean, loss_std, acc_df, loss_df


def moving_average(y, w: int):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()


# ======================================================
# Plot helper（Acc / Loss を別 figure で保存）
# ======================================================
def plot_one_metric(
    epochs,
    mean,
    std,
    df: pd.DataFrame,
    ylabel: str,
    ylim=None,
    title: str = "",
    out_png: str = "",
    out_eps: str = "",
    smooth_mean=None,
    legend_loc: str = "best",
):
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 3.9))

    # 見た目パラメータ（seedを主役、mean/stdは背景）
    SEED_LW = 1.4
    SEED_ALPHA = 0.55
    MEAN_LW = 1.6
    BAND_ALPHA = 0.08

    # band（背景）
    ax.fill_between(
        epochs, mean - std, mean + std,
        alpha=BAND_ALPHA, zorder=1, label="±1 std"
    )

    # seed 曲線（主役）
    for col in df.columns:
        ax.plot(
            df.index.values, df[col].values,
            linewidth=SEED_LW, alpha=SEED_ALPHA, zorder=2
        )

    # mean（ガイド：黒破線）
    mean_line = smooth_mean if smooth_mean is not None else mean
    ax.plot(
        epochs, mean_line,
        color="black", linestyle="--", linewidth=MEAN_LW,
        zorder=3, label="mean"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc=legend_loc, frameon=True)

    if title:
        ax.set_title(title)

    fig.tight_layout()

    # save
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    if out_eps:
        fig.savefig(out_eps, bbox_inches="tight")

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="fw_fw1_T4_S1_seed*_beta1.00_wait2.csv")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (default 1)")
    # ★ 拡張子なしの出力プレフィックス（ここから _acc/_loss を作る）
    ap.add_argument(
        "--out_prefix",
        default=os.path.join(OUT_DIR, "learning_curve_fw_T4_wait2_beta1.00"),
        help="output prefix (no extension). saves *_acc.png and *_loss.png (+eps)"
    )
    ap.add_argument("--title", default="", help="optional title prefix")
    ap.add_argument("--no_eps", action="store_true", help="if set, do not save eps")
    args = ap.parse_args()

    glob_pat = os.path.join(CSV_DIR, args.pattern)
    files = sorted(glob.glob(glob_pat), key=extract_seed)

    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {glob_pat}")

    epochs, acc_m, acc_s, loss_m, loss_s, acc_df, loss_df = aggregate(files)

    # smoothing（meanだけに適用）
    acc_m_s  = moving_average(acc_m, args.smooth)
    loss_m_s = moving_average(loss_m, args.smooth)

    # 出力先ディレクトリ
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ========== Acc figure ==========
    out_png_acc = args.out_prefix + "_acc.png"
    out_eps_acc = "" if args.no_eps else (args.out_prefix + "_acc.eps")
    title_acc = (args.title + "  " if args.title else "") + "Validation Acc"

    plot_one_metric(
        epochs=epochs,
        mean=acc_m,
        std=acc_s,
        df=acc_df,
        ylabel="Validation Acc",
        ylim=(0.0, 1.0),
        title=title_acc,
        out_png=out_png_acc,
        out_eps=out_eps_acc,
        smooth_mean=acc_m_s,
        legend_loc="lower right",
    )

    # ========== Loss figure ==========
    out_png_loss = args.out_prefix + "_loss.png"
    out_eps_loss = "" if args.no_eps else (args.out_prefix + "_loss.eps")
    title_loss = (args.title + "  " if args.title else "") + "Validation Loss (CE)"

    plot_one_metric(
        epochs=epochs,
        mean=loss_m,
        std=loss_s,
        df=loss_df,
        ylabel="Validation Loss (CE)",
        ylim=None,
        title=title_loss,
        out_png=out_png_loss,
        out_eps=out_eps_loss,
        smooth_mean=loss_m_s,
        legend_loc="best",
    )

    print(f"[OK] matched {len(files)} files")
    print(f"[Saved] {out_png_acc}")
    if out_eps_acc:
        print(f"[Saved] {out_eps_acc}")
    print(f"[Saved] {out_png_loss}")
    if out_eps_loss:
        print(f"[Saved] {out_eps_loss}")


if __name__ == "__main__":
    main()

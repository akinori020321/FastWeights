#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_PCA_timeseries_color.py
----------------------------------------
各 H_kv_*.csv について
 - h_t を PCA 2 次元へ射影（実験ごとに fit）
 - t=0→T を色グラデーション（青→赤）でプロット
 - 右側に全体共通のカラーバーを配置
 - subplot に並べて表示
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re


# --------------------------------------------------
# CSV 読み込み
# --------------------------------------------------
def load_h_csv(path):
    df = pd.read_csv(path)
    H = df.select_dtypes(include=[np.number]).values
    return H


# --------------------------------------------------
# ファイル名 → タイトル
# --------------------------------------------------
def parse_title(fname):
    base = os.path.basename(fname)
    pattern = r"H_kv_(fw|tanh)_S([0-9]+)_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    m = re.match(pattern, base)

    if m is None:
        return base

    return f"{m.group(1)}_S{m.group(2)}_eta{m.group(3)}_lam{m.group(4)}_seed{m.group(5)}"


# --------------------------------------------------
# PCA プロット（実験ごとに subplot）
# --------------------------------------------------
def plot_pca_subplots(csv_list, out_png="h_pca_timeseries_color.png"):
    N = len(csv_list)
    ncols = 2
    nrows = int(np.ceil(N / ncols))

    # -----------------------------------------
    # 図全体を包む Figure + GridSpec
    # -----------------------------------------
    fig = plt.figure(figsize=(8*ncols, 6*nrows))
    gs = fig.add_gridspec(nrows, ncols)

    axes = []
    for r in range(nrows):
        row_axes = []
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        axes.append(row_axes)

    cmap = plt.cm.plasma

    # -----------------------------------------
    # 各 CSV を subplot に描画
    # -----------------------------------------
    for idx, csv_path in enumerate(csv_list):
        H = load_h_csv(csv_path)
        T = H.shape[0]

        # PCA fit
        pca = PCA(n_components=2)
        H_pca = pca.fit_transform(H)

        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # 色（t=0→T）
        colors = cmap(np.linspace(0, 1, T))

        ax.scatter(H_pca[:, 0], H_pca[:, 1], c=colors, s=40)

        ax.set_title(parse_title(csv_path), fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid()

    # 余った subplot を非表示
    for i in range(N, nrows*ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis("off")

    # --------------------------------------------------------
    # flatten Axes for colorbar
    # --------------------------------------------------------
    flat_axes = [ax for row in axes for ax in row]

    # --------------------------------------------------------
    # カラーバー（右側縦）
    # --------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T-1))
    sm.set_array([])

    # ★ 右側の余白を確保（tight_layout を使わない）
    fig.subplots_adjust(right=0.88)

    cbar = fig.colorbar(
        sm,
        ax=flat_axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.01
    )
    cbar.set_label("time step (t=0 → t=T)", fontsize=12)

    # 保存
    os.makedirs("plots/h_pca", exist_ok=True)
    out_path = os.path.join("plots/h_pca", out_png)
    plt.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    ap.add_argument("--out", type=str, default="h_pca_timeseries_color.png")
    args = ap.parse_args()

    csv_list = sorted(glob.glob(os.path.join(args.dir, "H_kv_*.csv")))
    print("[INFO] Found", len(csv_list), "files")

    plot_pca_subplots(csv_list, out_png=args.out)


if __name__ == "__main__":
    main()

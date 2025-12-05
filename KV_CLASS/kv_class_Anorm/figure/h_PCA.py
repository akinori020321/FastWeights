#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_PCA_timeseries_color.py
----------------------------------------
各 H_kv_*.csv について
 - h_t を PCA 2 次元へ射影（実験ごとに fit）
 - t=0→T を色グラデーション（青→赤）でプロット
 - t=5 の地点を赤い大丸で強調
 - それぞれを subplot に配置して見やすく可視化
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    axes = axes.reshape(nrows, ncols)

    for idx, csv_path in enumerate(csv_list):
        H = load_h_csv(csv_path)
        T = H.shape[0]

        # ---- PCA fit（この実験だけ） ----
        pca = PCA(n_components=2)
        H_pca = pca.fit_transform(H)

        # subplot
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # ---- 時系列カラー（青→赤） ----
        colors = plt.cm.plasma(np.linspace(0, 1, T))

        # 散布図（矢印なし）
        ax.scatter(H_pca[:, 0], H_pca[:, 1], c=colors, s=50)

        # t=5 を赤丸で強調
        if T > 5:
            ax.scatter(
                H_pca[5, 0], H_pca[5, 1],
                s=250,
                facecolor="none",
                edgecolor="red",
                linewidth=3
            )

        # ---- タイトル ----
        ax.set_title(parse_title(csv_path), fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid()

    # 余分 subplot を消す
    for i in range(N, nrows*ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis("off")

    # 保存
    os.makedirs("plots/h_pca", exist_ok=True)
    out_path = os.path.join("plots/h_pca", out_png)
    plt.tight_layout()
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

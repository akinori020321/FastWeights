#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_heatmap.py
----------------------------------------
各時刻 h_t の cosine similarity 行列 (T×T) を可視化し、
同一ディレクトリ内の H_kv_*.csv をすべて1枚の図に subplot として並べる。

特徴：
 - 対角成分は青で塗る
 - 同じクラスの value 同士は square ではなく、(i,j) 1セルのみ塗る
 - query と bind_key の対応セルも 1セルのみ塗る
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------
# H_t CSV を読み込む
# --------------------------------------------------
def load_h_csv(path):
    df = pd.read_csv(path)
    # 数値列のみ抽出
    numeric_df = df.select_dtypes(include=[np.number])
    H = numeric_df.values  # (T, d_h)
    return H, df["kind"].tolist(), df["class_id"].tolist()


# --------------------------------------------------
# cosine similarity matrix
# --------------------------------------------------
def cosine_matrix(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    C = np.matmul(Hn, Hn.T)
    return C


# --------------------------------------------------
# ファイル名パーサ
# --------------------------------------------------
def parse_filename(fname):
    base = os.path.basename(fname)

    pattern = r"H_kv_(fw|tanh)_S([0-9]+)_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)

    return core, S, eta, lam, seed


# --------------------------------------------------
# heatmap プロット
# --------------------------------------------------
def plot_heatmaps_all(csv_list, out_png="h_heatmap_all.png"):
    N = len(csv_list)

    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False
    )

    sns.set(font_scale=1.0)

    # ★ カラーマップ定義 (-1〜0 黒, 0〜1 赤→緑)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.0,  "black"),  # -1
            (0.5,  "black"),  #  0 未満は全部黒
            (0.5,  "red"),    #  0
            (1.0,  "green"),  #  1
        ]
    )

    # 共通 colorbar 軸
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    # 描画ループ
    for idx, csv_path in enumerate(csv_list):
        H, kinds, class_ids = load_h_csv(csv_path)
        C = cosine_matrix(H)

        core, S, eta, lam, seed = parse_filename(csv_path)

        T = len(kinds)

        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        # ---- Heatmap 本体 ----
        hm = sns.heatmap(
            C,
            vmin=-1.0,
            vmax=1.0,
            cmap=cmap,
            ax=ax,
            cbar=(idx == 0),
            cbar_ax=cbar_ax if idx == 0 else None
        )

        # ------------------------------------------------------------
        # ① 対角成分を青で塗りつぶす
        # ------------------------------------------------------------
        for t in range(T):
            ax.add_patch(
                patches.Rectangle(
                    (t, t), 1, 1,
                    fill=True, facecolor="blue", edgecolor="none", alpha=0.6
                )
            )

        # ------------------------------------------------------------
        # ② 同じクラスの value 同士のセル（枠線で強調）
        # ------------------------------------------------------------
        from collections import defaultdict
        class_to_values = defaultdict(list)

        for t in range(T):
            if kinds[t] == "value":
                class_to_values[class_ids[t]].append(t)

        import matplotlib.cm as cm
        color_map = cm.get_cmap("tab20", len(class_to_values))

        for idx_c, (cid, steps) in enumerate(class_to_values.items()):
            color = color_map(idx_c)

            # (i,j) の 1セルを「枠線のみ」で囲む（対角は除く）
            for i in steps:
                for j in steps:
                    if i == j:
                        continue
                    ax.add_patch(
                        patches.Rectangle(
                            (j, i), 1, 1,
                            fill=False,
                            edgecolor=color,
                            linewidth=2.0
                        )
                    )

        # ------------------------------------------------------------
        # ③ query と bind_key の1セル → 枠線で magenta 強調
        # ------------------------------------------------------------
        if "query" in kinds:
            q_step = kinds.index("query")
            bind_idx = class_ids[q_step]
            key_step = bind_idx * 2  # key の位置（bind 仕様）

            ax.add_patch(
                patches.Rectangle(
                    (key_step, q_step), 1, 1,
                    fill=False,
                    edgecolor="magenta",
                    linewidth=2.5
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (q_step, key_step), 1, 1,
                    fill=False,
                    edgecolor="magenta",
                    linewidth=2.5
                )
            )

        # ---- 各 subplot のタイトル ----
        title = f"{core}_S{S}_eta{eta}_lam{lam}_seed{seed}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t'")
        ax.set_ylabel("t")

    # 余り subplot を非表示
    for i in range(N, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_png, dpi=200)
    print(f"[SAVE] {out_png}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    ap.add_argument("--out", type=str, default="h_heatmap_all.png")
    args = ap.parse_args()

    # 保存先ディレクトリ
    save_dir = "plots/h_heatmap"
    os.makedirs(save_dir, exist_ok=True)

    # 保存ファイルパス
    out_path = os.path.join(save_dir, args.out)

    csv_list = sorted(glob.glob(os.path.join(args.dir, "H_kv_*.csv")))
    if len(csv_list) == 0:
        print("[ERROR] No H_kv_*.csv found.")
        return

    print(f"[INFO] Found {len(csv_list)} files")
    for c in csv_list:
        print(" -", c)

    plot_heatmaps_all(csv_list, out_png=out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_heatmap.py + dot-product heatmap
----------------------------------------
Cosine similarity と Dot-product の2種類のヒートマップを生成する。
クラス番号に基づいて枠線を色分けし、
対角青・クラス枠線・query 対応枠線を描画する。
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

    # ★ h[...] だけを特徴ベクトルとして使う
    h_cols = [c for c in df.columns if c.startswith("h[")]
    H = df[h_cols].values.astype(float)

    return H, df["kind"].tolist(), df["class_id"].tolist()


# --------------------------------------------------
# cosine similarity matrix
# --------------------------------------------------
def cosine_matrix(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    C = np.matmul(Hn, Hn.T)
    return C


# --------------------------------------------------
# ★ dot product matrix
# --------------------------------------------------
def dot_matrix(H):
    return np.matmul(H, H.T)


# --------------------------------------------------
# ファイル名パーサ
# --------------------------------------------------
def parse_filename(fname):
    base = os.path.basename(fname)
    pattern = r"H_kv_(fw|tanh|rnn)_S([0-9]+)_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
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
# core 名
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN-LN",
}


# --------------------------------------------------
# クラス枠用の色リスト（赤・緑から離れた色だけ）
# --------------------------------------------------
CLASS_OUTLINE_COLORS = [
    "#1f77b4",  # 濃い青
    "#9467bd",  # 紫
    "#17becf",  # シアン
    "#7f7f7f",  # グレー
    "#8c564b",  # ブラウン
    "#aec7e8",  # 明るい青
    "#c5b0d5",  # 明るい紫
    "#9edae5",  # 明るいシアン
]


# --------------------------------------------------
# 共通のヒートマップ描画（cos/dot 共通）
#  - 対角青
#  - 同じ class_id（bind / value / query）のペアを同じ色で枠囲み
#  - Query と同じクラスの Bind/Value との対応セルをマゼンタで強調
# --------------------------------------------------
def draw_single_heatmap(ax, M, kinds, class_ids):
    T = len(kinds)

    # ---- ① 対角を青 ----
    for t in range(T):
        ax.add_patch(
            patches.Rectangle(
                (t, t), 1, 1,
                fill=True, facecolor="blue", edgecolor="none", alpha=0.6
            )
        )

    # ---- ② 同じ class_id を持つステップの枠線強調 ----
    #  - class_id >= 0 だけ対象
    #  - kind が "bind" / "value" / "query" のものをまとめる
    from collections import defaultdict
    class_to_steps = defaultdict(list)
    for t in range(T):
        cid = class_ids[t]
        if cid < 0:
            continue
        k = kinds[t]
        if k not in ("bind", "value", "query"):
            continue
        class_to_steps[cid].append(t)

    colors = CLASS_OUTLINE_COLORS

    for idx_c, (cid, steps) in enumerate(sorted(class_to_steps.items())):
        color = colors[idx_c % len(colors)]
        for i in steps:
            for j in steps:
                if i == j:
                    continue  # 対角は青塗りに任せる
                ax.add_patch(
                    patches.Rectangle(
                        (j, i), 1, 1,
                        fill=False,
                        edgecolor=color,
                        linewidth=2.0
                    )
                )

    # ---- ③ query と同じクラスの bind/value をマゼンタで強調 ----
    if "query" in kinds:
        q_step = kinds.index("query")
        q_cid = class_ids[q_step]

        if q_cid >= 0:
            key_steps = [
                t for t in range(T)
                if kinds[t] in ("bind", "value") and class_ids[t] == q_cid
            ]

            for key_step in key_steps:
                for (x, y) in [(key_step, q_step), (q_step, key_step)]:
                    ax.add_patch(
                        patches.Rectangle(
                            (x, y), 1, 1,
                            fill=False,
                            edgecolor="magenta",
                            linewidth=2.5
                        )
                    )


# --------------------------------------------------
# cosine Heatmap（まとめて）
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

    # カラーマップ（既存そのまま：黒→赤→緑）
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.0, "black"),
            (0.5, "black"),
            (0.5, "red"),
            (1.0, "green"),
        ]
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    for idx, csv_path in enumerate(csv_list):
        H, kinds, class_ids = load_h_csv(csv_path)
        C = cosine_matrix(H)
        core, S, eta, lam, seed = parse_filename(csv_path)
        core_name = CORE_NAME.get(core, core)

        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        sns.heatmap(
            C,
            vmin=-1.0,
            vmax=1.0,
            cmap=cmap,
            ax=ax,
            cbar=(idx == 0),
            cbar_ax=cbar_ax if idx == 0 else None
        )

        draw_single_heatmap(ax, C, kinds, class_ids)

        title = f"{core_name}_S{S}_eta{eta}_lam{lam}_seed{seed}"
        ax.set_title(title, fontsize=10)

    # 余白 subplot を非表示
    for i in range(N, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r][c].axis("off")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_png, dpi=200)
    print(f"[SAVE] {out_png}")


# --------------------------------------------------
# dot-product Heatmap（各subplotに個別colorbar）
# --------------------------------------------------
def plot_heatmaps_all_dot(csv_list, out_png="h_heatmap_dot.png"):
    N = len(csv_list)
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False
    )

    # dot 用 cmap（赤→緑）
    cmap = LinearSegmentedColormap.from_list(
        "dot_cmap",
        [
            (0.0, "red"),
            (1.0, "green")
        ]
    )

    for idx, csv_path in enumerate(csv_list):
        H, kinds, class_ids = load_h_csv(csv_path)
        D = dot_matrix(H)

        core, S, eta, lam, seed = parse_filename(csv_path)
        core_name = CORE_NAME.get(core, core)

        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        vmin = float(D.min())
        vmax = float(D.max())

        hm = sns.heatmap(
            D,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            cbar=True
        )

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)

        draw_single_heatmap(ax, D, kinds, class_ids)

        title = f"{core_name}_S{S}_eta{eta}_lam{lam}_seed{seed} (dot)"
        ax.set_title(title, fontsize=10)

    # 余分 subplot を非表示
    for i in range(N, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r][c].axis("off")

    plt.tight_layout()
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

    save_dir = "plots/h_heatmap"
    os.makedirs(save_dir, exist_ok=True)

    csv_list = sorted(glob.glob(os.path.join(args.dir, "H_kv_*.csv")))
    if len(csv_list) == 0:
        print("[ERROR] No H_kv_*.csv found.")
        return

    print(f"[INFO] Found {len(csv_list)} files")
    for c in csv_list:
        print(" -", c)

    # cosine 版
    out_path = os.path.join(save_dir, args.out)
    plot_heatmaps_all(csv_list, out_png=out_path)

    # dot-product 版
    out_path_dot = os.path.join(save_dir, "h_heatmap_dot.png")
    plot_heatmaps_all_dot(csv_list, out_png=out_path_dot)


if __name__ == "__main__":
    main()

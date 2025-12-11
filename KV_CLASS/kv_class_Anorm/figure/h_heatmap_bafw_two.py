#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ba-FW 各条件の cosine ヒートマップ図（1ファイル1枚）
------------------------------------------------------
../results_A_kv/ 以下の H_kv_fw_*.csv を読み込み，
Ba-FW (各条件) の cosine 類似度ヒートマップを
それぞれ別々の PNG として出力する。

出力:
  plots/sloop_all/kv_dyn_cos_bafw_S{S}_eta{eta}_lam{lam}_seed{seed}.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
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
# ファイル名パーサ
#   H_kv_(fw|tanh|rnn)_S... を想定
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
# core 名を変換
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN-LN",
}


# --------------------------------------------------
# 共通のヒートマップ装飾（対角青・value 枠・query 対応枠）
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

    # ---- ② 同クラス value の枠線強調 ----
    from collections import defaultdict
    class_to_values = defaultdict(list)
    for t in range(T):
        if kinds[t] == "value":
            class_to_values[class_ids[t]].append(t)

    # 推奨APIで cmap を取得（DeprecationWarning 回避）
    color_map = mpl.colormaps["tab20"]

    for idx_c, (cid, steps) in enumerate(class_to_values.items()):
        color = color_map(idx_c)
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

    # ---- ③ query と bind_key 強調 ----
    if "query" in kinds:
        q_step = kinds.index("query")
        bind_idx = class_ids[q_step]
        key_step = bind_idx * 2  # Bind 時刻（key）が 2 * class_id という前提

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
# Ba-FW 各条件の cosine heatmap を 1枚ずつ描画
# --------------------------------------------------
def plot_bafw_each(csv_list, out_dir, out_prefix="kv_dyn_cos_bafw"):
    # cos 用カラーマップ（元と同じ）
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            (0.0, "black"),
            (0.5, "black"),
            (0.5, "red"),
            (1.0, "green"),
        ]
    )

    for csv_path in sorted(csv_list):
        H, kinds, class_ids = load_h_csv(csv_path)
        C = cosine_matrix(H)
        core, S, eta, lam, seed = parse_filename(csv_path)
        core_name = CORE_NAME.get(core, core)

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        sns.heatmap(
            C,
            vmin=-1.0,
            vmax=1.0,
            cmap=cmap,
            ax=ax,
            cbar=True
        )

        draw_single_heatmap(ax, C, kinds, class_ids)

        title = f"{core_name}_S{S}_eta{eta}_lam{lam}_seed{seed}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel("t")

        plt.tight_layout()

        fname = f"{out_prefix}_S{S}_eta{eta}_lam{lam}_seed{seed}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[SAVE] {out_path}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv",
                    help="H_kv_*.csv が置いてあるディレクトリ")
    ap.add_argument("--out_prefix", type=str,
                    default="kv_dyn_cos_bafw",
                    help="出力ファイル名のプレフィックス")
    args = ap.parse_args()

    save_dir = "plots/sloop_all"
    os.makedirs(save_dir, exist_ok=True)

    # Ba-FW (core=fw) のみ取得
    csv_list = sorted(glob.glob(os.path.join(args.dir, "H_kv_fw_*.csv")))
    if len(csv_list) == 0:
        print("[ERROR] No H_kv_fw_*.csv found.")
        return

    print(f"[INFO] Found {len(csv_list)} Ba-FW files")
    for c in csv_list:
        print(" -", c)

    plot_bafw_each(csv_list, out_dir=save_dir, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()

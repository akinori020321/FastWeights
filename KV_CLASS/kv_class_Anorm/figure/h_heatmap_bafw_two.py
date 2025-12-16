#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ba-FW / SC-FW 各条件の cosine ヒートマップ図（1ファイル1枚）
-------------------------------------------------------------
../results_A_kv/ 以下の H_kv_*.csv を読み込み，
Ba-FW (core=fw) および SC-FW (core=tanh) の
cosine 類似度ヒートマップをそれぞれ別々の PNG として出力する。

出力（例）:
  plots/sloop_all/kv_dyn_cos_bafw_S1_eta0300_lam0950_seed0.png
  plots/sloop_all/kv_dyn_cos_scfw_S1_eta0500_lam0950_seed0.png
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

    # h[...] だけを特徴ベクトルとして使う
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
# core 名を変換（タイトル表示用・ファイル名用）
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN-LN",
}

CORE_PREFIX = {
    "fw": "bafw",
    "tanh": "scfw",
    "rnn": "rnnln",
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

    # ★ 修正ポイント：tab20 の色リストから色を取る
    cmap = mpl.colormaps["tab20"]
    cmap_colors = cmap.colors  # RGBA のリスト

    for idx_c, (cid, steps) in enumerate(class_to_values.items()):
        color = cmap_colors[idx_c % len(cmap_colors)]
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
        # Bind 時刻（key）が 2 * class_id という前提
        key_step = bind_idx * 2

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
# 指定された CSV 群の cosine heatmap を 1枚ずつ描画
# --------------------------------------------------
def plot_each_core(csv_list, out_dir, out_prefix="kv_dyn_cos"):
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
        core_prefix = CORE_PREFIX.get(core, core)

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

        title = f"{core_name}, S={S}, η={int(eta)/1000.0:g}, λ={int(lam)/1000.0:g}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t (step)")
        ax.set_ylabel("t (step)")

        plt.tight_layout()

        fname = f"{out_prefix}_{core_prefix}_S{S}_eta{eta}_lam{lam}_seed{seed}.png"
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
                    default="kv_dyn_cos",
                    help="出力ファイル名のプレフィックス")
    args = ap.parse_args()

    save_dir = "plots/sloop_all"
    os.makedirs(save_dir, exist_ok=True)

    # Ba-FW (core=fw)
    csv_fw = sorted(glob.glob(os.path.join(args.dir, "H_kv_fw_*.csv")))
    if len(csv_fw) == 0:
        print("[WARN] No H_kv_fw_*.csv found.")
    else:
        print(f"[INFO] Found {len(csv_fw)} Ba-FW files")
        for c in csv_fw:
            print(" -", c)
        plot_each_core(csv_fw, out_dir=save_dir, out_prefix=args.out_prefix)

    # SC-FW (core=tanh)
    csv_scfw = sorted(glob.glob(os.path.join(args.dir, "H_kv_tanh_*.csv")))
    if len(csv_scfw) == 0:
        print("[WARN] No H_kv_tanh_*.csv found.")
    else:
        print(f"[INFO] Found {len(csv_scfw)} SC-FW files")
        for c in csv_scfw:
            print(" -", c)
        plot_each_core(csv_scfw, out_dir=save_dir, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()

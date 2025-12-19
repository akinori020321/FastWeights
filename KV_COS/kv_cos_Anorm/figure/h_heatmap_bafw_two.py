#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ba-FW / SC-FW 各条件の cosine ヒートマップ図（1ファイル1枚）
-------------------------------------------------------------
../results_A_kv/ 以下の H_kv_*.csv を読み込み，
Ba-FW (core=fw) および SC-FW (core=tanh) の
cosine 類似度ヒートマップをそれぞれ別々の PNG として出力する。

★ class色割当ルール（PCA / h_heatmap(all) と統一）:
  kind in ("bind","value","query") かつ class_id>=0 を t順に走査し、
  初出 class_id に順番に色を割り当てる。
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
from collections import defaultdict


# --------------------------------------------------
# H_t CSV を読み込む
# --------------------------------------------------
def load_h_csv(path):
    df = pd.read_csv(path)
    h_cols = [c for c in df.columns if c.startswith("h[")]
    H = df[h_cols].values.astype(float)
    return H, df["kind"].tolist(), df["class_id"].tolist()


# --------------------------------------------------
# cosine similarity matrix
# --------------------------------------------------
def cosine_matrix(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return np.matmul(Hn, Hn.T)


# --------------------------------------------------
# ファイル名パーサ（noise あり / なし両対応）
# --------------------------------------------------
def parse_filename(fname):
    base = os.path.basename(fname)
    pattern = (
        r"H_kv_(fw|tanh|rnn)_S([0-9]+)"
        r"(?:_noise([0-9]+))?"
        r"_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    )
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(4)
    lam = m.group(5)
    seed = m.group(6)
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
# ★ class 枠線用：最大限明るい固定色
# --------------------------------------------------
CLASS_OUTLINE_COLORS = [
    "#ff4fa3",  # vivid pink
    "#ff9f1a",  # vivid orange
    "#4dd9ff",  # bright cyan-blue
    "#6dff6d",  # bright green
]


# --------------------------------------------------
# ★ predicate 統一の cid->color を作る
#   kind in ("bind","value","query") かつ class_id>=0 を t順に走査
#   戻り値: (cid2color, ordered_cids)
# --------------------------------------------------
def build_cid2color_bvq(kinds, class_ids):
    cid2color = {}
    ordered_cids = []
    color_idx = 0

    for k, cid in zip(kinds, class_ids):
        cid = int(cid)
        if cid < 0:
            continue
        if k not in ("bind", "value", "query"):
            continue
        if cid in cid2color:
            continue

        cid2color[cid] = CLASS_OUTLINE_COLORS[color_idx % len(CLASS_OUTLINE_COLORS)]
        ordered_cids.append(cid)
        color_idx += 1

    return cid2color, ordered_cids


# --------------------------------------------------
# 共通のヒートマップ装飾
# --------------------------------------------------
def draw_single_heatmap(ax, M, kinds, class_ids):
    T = len(kinds)

    # ---- ① 対角を青塗り ----
    for t in range(T):
        ax.add_patch(
            patches.Rectangle(
                (t, t), 1, 1,
                fill=True, facecolor="blue", edgecolor="none", alpha=0.6
            )
        )

    # ---- ② bvq & cid>=0 の class_to_steps ----
    class_to_steps = defaultdict(list)
    for t in range(T):
        cid = int(class_ids[t])
        k = kinds[t]
        if cid < 0:
            continue
        if k not in ("bind", "value", "query"):
            continue
        class_to_steps[cid].append(t)

    # ---- ③ bvq & cid>=0 の初出順で cid2color ----
    cid2color, ordered = build_cid2color_bvq(kinds, class_ids)

    # ---- ④ 枠線描画（順序は ordered = 出現順） ----
    for cid in ordered:
        color = cid2color[cid]
        steps = class_to_steps.get(cid, [])
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

    # ---- ⑤ query と同じクラスの bind/value をマゼンタで強調 ----
    if "query" in kinds:
        q_step = kinds.index("query")
        q_cid = int(class_ids[q_step])

        if q_cid >= 0:
            key_steps = [
                t for t in range(T)
                if (kinds[t] in ("bind", "value")) and (int(class_ids[t]) == q_cid)
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
# cosine heatmap を 1枚ずつ描画
# --------------------------------------------------
def plot_each_core(csv_list, out_dir, out_prefix="kv_dyn_cos"):
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

        if core == "rnn":
            ax.set_title(f"{core_name}", fontsize=11)
        else:
            ax.set_title(f"{core_name}, S={S}, η={int(eta)/1000.0:g}, λ={int(lam)/1000.0:g}", fontsize=11)

        ax.set_xlabel("t (step)")
        ax.set_ylabel("t (step)")

        plt.tight_layout()

        fname = f"{out_prefix}_{core}_S{S}_eta{eta}_lam{lam}_seed{seed}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        out_path_eps = os.path.splitext(out_path)[0] + ".eps"
        plt.savefig(out_path_eps)
        plt.close()
        print(f"[SAVE] {out_path}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    ap.add_argument("--out_prefix", type=str, default="kv_dyn_cos")
    args = ap.parse_args()

    save_dir = "plots/sloop_all"
    os.makedirs(save_dir, exist_ok=True)

    csv_fw = sorted(glob.glob(os.path.join(args.dir, "H_kv_fw_*.csv")))
    csv_scfw = sorted(glob.glob(os.path.join(args.dir, "H_kv_tanh_*.csv")))
    csv_rnn = sorted(glob.glob(os.path.join(args.dir, "H_kv_rnn_*.csv")))

    if csv_fw:
        plot_each_core(csv_fw, save_dir, args.out_prefix)
    if csv_scfw:
        plot_each_core(csv_scfw, save_dir, args.out_prefix)
    if csv_rnn:
        plot_each_core(csv_rnn, save_dir, args.out_prefix)


if __name__ == "__main__":
    main()

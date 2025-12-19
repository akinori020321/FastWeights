#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_PCA_timeseries_color.py
----------------------------------------
各 H_kv_*.csv について
 - h_t を PCA 2 次元へ射影（実験ごとに fit）
 - t=0→T を色グラデーションでプロット
 - ★ class_id ごとに色付きの輪っか（※出現順で色割当）
      ただし predicate を統一：
        kind in ("bind","value","query") かつ class_id>=0 を t順に走査して初出順に色
 - ★ query のみ「星マーカー」
 - ★ 各図の右上に class 色バー（凡例）
 - ★ wait（class_id < 0）には輪っかを付けない
 - 右側に全体共通のカラーバー
 - ★ 1ファイルにつき 1枚の PNG として出力
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
import matplotlib.patches as mpatches


# --------------------------------------------------
# ★ 輪っか／クエリ星 用の明るい固定色（出現順で割当）
# --------------------------------------------------
OUTLINE_COLORS = [
    "#ff4fa3",  # vivid pink
    "#ff9f1a",  # vivid orange
    "#4dd9ff",  # bright cyan-blue
    "#6dff6d",  # bright green
]


# --------------------------------------------------
# CSV 読み込み
# --------------------------------------------------
def load_h_csv(path):
    df = pd.read_csv(path)

    h_cols = [c for c in df.columns if c.startswith("h[")]
    H = df[h_cols].values.astype(np.float32)

    class_ids = (
        df["class_id"].values.astype(int)
        if "class_id" in df.columns
        else np.full(len(H), -1, dtype=int)
    )
    kinds = (
        df["kind"].values
        if "kind" in df.columns
        else np.full(len(H), "other")
    )

    return H, class_ids, kinds


# --------------------------------------------------
# ファイル名 → タイトル / 出力名
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN+LN",
}

def parse_title(fname):
    base = os.path.basename(fname)
    pattern = (
        r"H_kv_(fw|tanh|rnn)_S(\d+)"
        r"(?:_noise(\d+))?"
        r"_eta(\d+)_lam(\d+)_seed(\d+)"
    )
    m = re.match(pattern, base)
    if m is None:
        return base

    core, S, noise, eta, lam, seed = m.groups()
    core_name = CORE_NAME.get(core, core)

    eta_f = int(eta) / 1000.0
    lam_f = int(lam) / 1000.0

    noise_str = f", noise={int(noise)/1000.0:g}" if noise else ""
    return f"{core_name}, S={int(S)}, η={eta_f:g}, λ={lam_f:g}{noise_str}"


def parse_outname(fname):
    base = os.path.basename(fname)
    pattern = (
        r"H_kv_(fw|tanh|rnn)_S(\d+)"
        r"(?:_noise(\d+))?"
        r"_eta(\d+)_lam(\d+)_seed(\d+)"
    )
    m = re.match(pattern, base)
    if m is None:
        return os.path.splitext(base)[0] + ".png"

    core, S, noise, eta, lam, seed = m.groups()
    core_name = CORE_NAME.get(core, core).lower().replace("+", "").replace("-", "")
    noise_str = f"_noise{noise}" if noise else ""
    return f"h_pca_{core_name}_S{S}{noise_str}_eta{eta}_lam{lam}.png"


# --------------------------------------------------
# ★ class_id の「出現順」で色を割り当てる（predicate 統一版）
#   predicate:
#     kind in ("bind","value","query") かつ class_id>=0
#   戻り値:
#     (cid2color: dict[int,str], ordered_cids: list[int])
# --------------------------------------------------
def build_class_color_map(kinds, class_ids):
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

        cid2color[cid] = OUTLINE_COLORS[color_idx % len(OUTLINE_COLORS)]
        ordered_cids.append(cid)
        color_idx += 1

    return cid2color, ordered_cids


# --------------------------------------------------
# PCA プロット（1ファイル=1図）
# --------------------------------------------------
def plot_each_csv(csv_list, out_dir):
    cmap = plt.cm.plasma

    os.makedirs(out_dir, exist_ok=True)

    for csv_path in sorted(csv_list):
        H, class_ids, kinds = load_h_csv(csv_path)
        T = H.shape[0]

        # ★ predicate 統一で class_id→色 を決める（このCSV内ローカル）
        cid2color, ordered_cids = build_class_color_map(kinds, class_ids)

        pca = PCA(n_components=2)
        H_pca = pca.fit_transform(H)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        colors_time = cmap(np.linspace(0, 1, T))

        # ---- マスク類 ----
        kinds_arr = np.asarray(kinds)
        is_query = (kinds_arr == "query")
        not_query = ~is_query
        has_class = (class_ids >= 0)
        is_bvq = np.isin(kinds_arr, ["bind", "value", "query"])  # ★ predicate統一

        # ========================
        # 非 query：点（全て）
        # ========================
        ax.scatter(
            H_pca[not_query, 0], H_pca[not_query, 1],
            c=colors_time[not_query],
            s=40, linewidths=0, zorder=2
        )

        # 非 query：輪っか（predicate 統一: bvq & class_id>=0 のみ）
        idx_ring = (not_query & has_class & is_bvq)
        ax.scatter(
            H_pca[idx_ring, 0],
            H_pca[idx_ring, 1],
            s=130,
            facecolors="none",
            edgecolors=[cid2color[int(cid)] for cid in class_ids[idx_ring]],
            linewidths=2.2,
            zorder=3
        )

        # ========================
        # query：★ 星（全て）
        # ========================
        ax.scatter(
            H_pca[is_query, 0], H_pca[is_query, 1],
            c=colors_time[is_query],
            marker="*",
            s=220,
            linewidths=0,
            zorder=4
        )

        # query：★ 星の輪っか（predicate 統一: bvq & class_id>=0 のみ）
        idx_qring = (is_query & has_class & is_bvq)
        ax.scatter(
            H_pca[idx_qring, 0],
            H_pca[idx_qring, 1],
            facecolors="none",
            edgecolors=[cid2color[int(cid)] for cid in class_ids[idx_qring]],
            marker="*",
            s=420,
            linewidths=2.5,
            zorder=5
        )

        ax.set_title(parse_title(csv_path), fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)

        # ------------------------
        # 凡例（右上）※出現順のまま並べる
        # ------------------------
        handles = [
            mpatches.Patch(
                facecolor="none",
                edgecolor=cid2color[cid],
                linewidth=2.5,
                label=f"class {cid}"
            )
            for cid in ordered_cids
        ]
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="*",
                color="k",
                linestyle="None",
                markersize=12,
                label="query"
            )
        )

        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=9,
            framealpha=0.85
        )

        # ------------------------
        # カラーバー（時間）
        # ------------------------
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(0, T - 1)
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label("time step (t=0 → t=T)")

        plt.tight_layout()

        out_png = parse_outname(csv_path)
        out_path = os.path.join(out_dir, out_png)
        plt.savefig(out_path, dpi=200)
        out_eps = os.path.splitext(out_path)[0] + ".eps"
        plt.savefig(out_eps)
        plt.close()
        print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    ap.add_argument("--out_dir", type=str, default="plots/h_pca")
    args = ap.parse_args()

    csv_list = sorted(glob.glob(os.path.join(args.dir, "H_kv_*.csv")))
    print("[INFO] Found", len(csv_list), "files")

    plot_each_csv(csv_list, args.out_dir)


if __name__ == "__main__":
    main()

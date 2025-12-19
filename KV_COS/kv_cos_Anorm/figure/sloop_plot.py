#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-loop Top5 指標可視化スクリプト（fw / tanh 両対応）
-----------------------------------------------------
入力：
  - SloopVec_kv_<core>_S*.csv   （h と base）
  - Amat_kv_<core>_S*.csv       （A 行列）

出力：
  plots/sloop/<model_id>/
    cos_summary.png
    ratio_summary.png
    norm_summary.png
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# cosine
# --------------------------------------------------
def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------
# ファイル名パーサ（noise あり/なし + rnn 対応）
# --------------------------------------------------
def parse_filename(fname):
    base = os.path.basename(fname)
    pattern = (
        r"SloopVec_kv_(fw|tanh|rnn)_S(\d+)"
        r"(?:_noise([0-9]+))?"
        r"_eta([0-9]+)_lam([0-9]+)_seed(\d+)"
        r"\.csv$"
    )
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    noise = m.group(3)   # None or "0600"
    eta = m.group(4)
    lam = m.group(5)
    seed = m.group(6)

    return core, S, noise, eta, lam, seed


# --------------------------------------------------
# A 行列を読み込む
# --------------------------------------------------
def load_A(csv_path):
    df = pd.read_csv(csv_path)

    A_list = []
    for _, row in df.iterrows():
        vals = row.values.astype(float)

        # （最も一般的ケース）1列余っていたら先頭を落とす
        if int(np.sqrt(vals.shape[0] - 1)) ** 2 == vals.shape[0] - 1:
            vals = vals[1:]  # drop first col

        # 正方チェック
        L = vals.shape[0]
        d = int(np.sqrt(L))
        if d * d != L:
            raise ValueError(f"A row size {L} cannot form square: {L}")

        A_list.append(vals.reshape(d, d))

    return A_list


# --------------------------------------------------
# 1 モデル分の可視化をする
# --------------------------------------------------
def process_model(sloop_csv, amat_csv, out_dir):

    print(f"[PROCESS] sloop={sloop_csv}")
    print(f"[PROCESS] amat ={amat_csv}")

    df = pd.read_csv(sloop_csv)
    A_list = load_A(amat_csv)

    groups = df.groupby("t")
    T = len(groups)

    # まとめプロット用
    cos_plots = []
    ratio_plots = []
    norm_plots = []

    # 各 t について subplot 用データを作る
    for t, g in groups:

        # base（s=-1）
        base_vec = g[g["s"] == -1].iloc[0]
        base_h = base_vec.filter(regex=r"h\[").values.astype(float)
        norm_base = np.linalg.norm(base_h)

        # A_t
        A = A_list[int(t)]

        # S-loop rows
        loop_rows = g[g["s"] >= 0]

        s_list = []
        cos_base = []
        cos_Ah = []
        ratio_base = []
        norm_h = []
        norm_Ah = []

        for _, row in loop_rows.iterrows():
            s = int(row["s"])
            h_s = row.filter(regex=r"h\[").values.astype(float)

            Ah = A @ h_s

            s_list.append(s)

            cos_base.append(cosine(base_h, h_s))   # 指標1
            cos_Ah.append(cosine(h_s, Ah))         # 指標2
            ratio_base.append(np.linalg.norm(Ah) / (norm_base + 1e-8))  # 指標3
            norm_h.append(np.linalg.norm(h_s))     # 指標4
            norm_Ah.append(np.linalg.norm(Ah))     # 指標5

        cos_plots.append((t, s_list, cos_base, cos_Ah))
        ratio_plots.append((t, s_list, ratio_base))
        norm_plots.append((t, s_list, norm_h, norm_Ah))

    # ======================================================
    # 図1 : cosine（cos(h0,hs), cos(hs,Ah)）
    # ======================================================
    cols = 3
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))
    axes = axes.flatten()

    for ax_idx, (t, s_list, cos_base, cos_Ah) in enumerate(cos_plots):
        ax = axes[ax_idx]
        ax.plot(s_list, cos_base, marker="o", label="cos(base, h_s)")
        ax.plot(s_list, cos_Ah, marker="o", label="cos(h_s, A h_s)")
        ax.set_title(f"t={t}")
        ax.grid(True)
        ax.legend()

        ax.set_ylim([-1.05, 1.05])
        ax.set_xticks(range(max(s_list) + 1))

    for j in range(len(cos_plots), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cos_summary.png"))
    plt.savefig(os.path.join(out_dir, "cos_summary.eps"), format="eps")
    plt.close()

    # ======================================================
    # 図2 : ratio（||Ah|| / ||base||）
    # ======================================================
    all_ratio_vals = []
    for _, _, ratio_base in ratio_plots:
        all_ratio_vals += ratio_base
    ymax_ratio = max(all_ratio_vals) if len(all_ratio_vals) > 0 else 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))
    axes = axes.flatten()

    for ax_idx, (t, s_list, ratio_base) in enumerate(ratio_plots):
        ax = axes[ax_idx]
        ax.plot(s_list, ratio_base, marker="o", color="red",
                label="||Ah|| / ||base||")
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_title(f"t={t}")
        ax.grid(True)
        ax.legend()
        ax.set_ylim([0, ymax_ratio * 1.05])
        ax.set_xticks(range(max(s_list) + 1))

    for j in range(len(ratio_plots), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_summary.png"))
    plt.savefig(os.path.join(out_dir, "ratio_summary.eps"), format="eps")
    plt.close()

    # ======================================================
    # 図3 : norm（||h_s|| , ||Ah_s||）
    # ======================================================
    all_vals = []
    for _, _, norm_h, norm_Ah in norm_plots:
        all_vals += norm_h + norm_Ah
    ymax = max(all_vals) if len(all_vals) > 0 else 1.0

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))
    axes = axes.flatten()

    for ax_idx, (t, s_list, norm_h, norm_Ah) in enumerate(norm_plots):
        ax = axes[ax_idx]
        ax.plot(s_list, norm_h, marker="o", label="||h_s||")
        ax.plot(s_list, norm_Ah, marker="o", label="||A h_s||")
        ax.set_title(f"t={t}")
        ax.grid(True)
        ax.legend()
        ax.set_ylim([0, ymax * 1.05])
        ax.set_xticks(range(max(s_list) + 1))

    for j in range(len(norm_plots), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "norm_summary.png"))
    plt.savefig(os.path.join(out_dir, "norm_summary.eps"), format="eps")
    plt.close()

    print(f"[DONE] Saved plots → {out_dir}")


# --------------------------------------------------
# main（../results_A_kv/ を前提）
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    args = ap.parse_args()

    save_root = "plots/sloop"
    os.makedirs(save_root, exist_ok=True)

    sloop_list = sorted(glob.glob(os.path.join(args.dir, "SloopVec_kv_*.csv")))
    if len(sloop_list) == 0:
        print("[ERROR] No SloopVec_kv_*.csv found.")
        return

    amat_list = sorted(glob.glob(os.path.join(args.dir, "Amat_kv_*.csv")))
    if len(amat_list) == 0:
        print("[ERROR] No Amat_kv_*.csv found.")
        return

    print(f"[INFO] Found {len(sloop_list)} S-loop files")
    print(f"[INFO] Found {len(amat_list)} A-matrix files")

    for s_csv in sloop_list:
        core, S, noise, eta, lam, seed = parse_filename(s_csv)

        # ★ rnn は A が無いのでスキップ
        if core == "rnn":
            print(f"[SKIP] rnn has no A-matrix: {os.path.basename(s_csv)}")
            continue

        # ★ S=0 もスキップ（S-loop が無い）
        if S == 0:
            print(f"[SKIP] S=0 (no S-loop): {os.path.basename(s_csv)}")
            continue

        noise_str = f"_noise{noise}" if noise is not None else ""
        target_name = f"Amat_kv_{core}_S{S}{noise_str}_eta{eta}_lam{lam}_seed{seed}.csv"

        amat_csv = None
        for a in amat_list:
            if os.path.basename(a) == target_name:
                amat_csv = a
                break

        if amat_csv is None:
            print(f"[WARN] Amat not found for: {s_csv}")
            continue

        model_id = f"{core}_S{S}{noise_str}_eta{eta}_lam{lam}_seed{seed}"
        out_dir = os.path.join(save_root, model_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[PROCESS] {model_id}")
        process_model(s_csv, amat_csv, out_dir)


if __name__ == "__main__":
    main()

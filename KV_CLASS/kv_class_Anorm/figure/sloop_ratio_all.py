#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-loop ratio over time (Ba-FW vs SC-FW)
---------------------------------------
- ../results_A_kv/ 以下の SloopVec_kv_*.csv と Amat_kv_*.csv を読む
- 各モデルについて、各時刻 t で
      ratio(t) = ||A_t h_t^{(last input)}||_2 / ||h_t^{base}||_2
  を計算
- Ba-FW(core=fw) と SC-FW(core=tanh) を 2 行サブプロットで別々に描画

★色分け：
  - (lambda, eta) = (0.95, 0.3) の線 → tab:orange（前と同じ橙）
  - それ以外 → tab:blue（前と同じ青）

出力:
  plots/sloop_all/ratio_over_time.png
  plots/sloop_all/ratio_over_time.eps
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# SloopVec ファイル名パーサ（rnn 対応）
# --------------------------------------------------
def parse_sloop_filename(fname):
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
    noise = m.group(3)
    eta = m.group(4)
    lam = m.group(5)
    seed = m.group(6)

    noise_str = f"_noise{noise}" if noise is not None else ""
    model_id = f"{core}_S{S}{noise_str}_eta{eta}_lam{lam}_seed{seed}"
    return core, S, eta, lam, seed, model_id


# --------------------------------------------------
# Amat ファイル名パーサ（fw / tanh のみ）
# --------------------------------------------------
def parse_amat_filename(fname):
    base = os.path.basename(fname)
    pattern = (
        r"Amat_kv_(fw|tanh)_S(\d+)"
        r"(?:_noise([0-9]+))?"
        r"_eta([0-9]+)_lam([0-9]+)_seed(\d+)"
        r"\.csv$"
    )
    m = re.match(pattern, base)
    if m is None:
        return None

    core = m.group(1)
    S = int(m.group(2))
    noise = m.group(3)
    eta = m.group(4)
    lam = m.group(5)
    seed = m.group(6)

    noise_str = f"_noise{noise}" if noise is not None else ""
    model_id = f"{core}_S{S}{noise_str}_eta{eta}_lam{lam}_seed{seed}"
    return core, S, eta, lam, seed, model_id


# --------------------------------------------------
# A 行列を読み込む
# --------------------------------------------------
def load_A(csv_path):
    df = pd.read_csv(csv_path)

    A_list = []
    for _, row in df.iterrows():
        vals = row.values.astype(float)

        # 先頭列が index 等の場合は落とす
        if int(np.sqrt(vals.shape[0] - 1)) ** 2 == vals.shape[0] - 1:
            vals = vals[1:]

        L = vals.shape[0]
        d = int(np.sqrt(L))
        if d * d != L:
            raise ValueError(f"A row size {L} cannot form square")

        A_list.append(vals.reshape(d, d))

    return A_list


# --------------------------------------------------
# ratio(t) 計算
# --------------------------------------------------
def compute_ratio_curve(sloop_csv, amat_csv):
    df = pd.read_csv(sloop_csv)
    A_list = load_A(amat_csv)

    groups = df.groupby("t")
    t_list, ratio_list = [], []

    for t, g in groups:
        t = int(t)

        # base: s = -1
        base_rows = g[g["s"] == -1]
        if len(base_rows) == 0:
            continue

        base_h = base_rows.iloc[0].filter(regex=r"h\[").values.astype(float)
        norm_base = np.linalg.norm(base_h) + 1e-8

        if t >= len(A_list):
            continue
        A = A_list[t]

        # loop rows: s >= 0
        loop_rows = g[g["s"] >= 0].sort_values("s")
        if len(loop_rows) == 0:
            continue

        # 実装都合：最後の 1 つ手前を「last input」相当として採用
        if len(loop_rows) == 1:
            target_row = loop_rows.iloc[0]
        else:
            target_row = loop_rows.iloc[-2]

        h_target = target_row.filter(regex=r"h\[").values.astype(float)
        Ah = A @ h_target
        ratio = np.linalg.norm(Ah) / norm_base

        t_list.append(t)
        ratio_list.append(ratio)

    t_arr = np.array(t_list)
    ratio_arr = np.array(ratio_list)
    order = np.argsort(t_arr)
    return t_arr[order], ratio_arr[order]


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv")
    args = ap.parse_args()

    sloop_list = sorted(glob.glob(os.path.join(args.dir, "SloopVec_kv_*.csv")))
    amat_list = sorted(glob.glob(os.path.join(args.dir, "Amat_kv_*.csv")))

    if len(sloop_list) == 0 or len(amat_list) == 0:
        print("[ERROR] Required CSV files not found.")
        return

    # Amat を model_id で引けるように map 化
    amat_map = {}
    for a_csv in amat_list:
        parsed = parse_amat_filename(a_csv)
        if parsed is None:
            continue
        _, _, _, _, _, model_id = parsed
        amat_map[model_id] = a_csv

    out_dir = "plots/sloop_all"
    os.makedirs(out_dir, exist_ok=True)

    # ★ core の割り当て：fw -> Ba-FW, tanh -> SC-FW
    curves_ba, curves_sc = [], []

    for s_csv in sloop_list:
        core, S, eta, lam, seed, model_id = parse_sloop_filename(s_csv)

        # rnn は A が無いのでスキップ
        if core == "rnn":
            print(f"[SKIP] rnn has no A-matrix: {os.path.basename(s_csv)}")
            continue

        if model_id not in amat_map:
            print(f"[WARN] Amat not found for {model_id}, skip.")
            continue

        print(f"[PROCESS] {model_id}")
        t_arr, ratio_arr = compute_ratio_curve(s_csv, amat_map[model_id])

        # 表示用に eta/lam を小数へ（eta0300 -> 0.3, lam0950 -> 0.95）
        eta_f = int(eta) / 1000.0
        lam_f = int(lam) / 1000.0

        # ★色： (lam, eta) = (0.95, 0.3) だけ橙、それ以外は青（トーンは tab:* で固定）
        color = "tab:orange" if (int(lam) == 950 and int(eta) == 300) else "tab:blue"

        if core == "fw":
            curves_ba.append((t_arr, ratio_arr, f"Ba-FW (eta={eta_f:.1f}, lam={lam_f:.2f}, seed={seed})", color))
        elif core == "tanh":
            curves_sc.append((t_arr, ratio_arr, f"SC-FW (eta={eta_f:.1f}, lam={lam_f:.2f}, seed={seed})", color))

    if len(curves_ba) == 0 and len(curves_sc) == 0:
        print("[ERROR] No valid curves to plot.")
        return

    # 両方あるなら 2 段，片方だけなら 1 枚
    if len(curves_ba) > 0 and len(curves_sc) > 0:
        fig, (ax_ba, ax_sc) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        for t, r, lab, col in curves_ba:
            ax_ba.plot(t, r, marker="o", label=lab, color=col)
        ax_ba.axhline(1.0, color="gray", linestyle="--")
        ax_ba.set_title("Ba-FW")
        ax_ba.grid(True)
        ax_ba.legend()

        for t, r, lab, col in curves_sc:
            ax_sc.plot(t, r, marker="o", label=lab, color=col)
        ax_sc.axhline(1.0, color="gray", linestyle="--")
        ax_sc.set_title("SC-FW")
        ax_sc.grid(True)
        ax_sc.legend()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        curves = curves_ba if len(curves_ba) > 0 else curves_sc
        title = "Ba-FW" if len(curves_ba) > 0 else "SC-FW"

        for t, r, lab, col in curves:
            ax.plot(t, r, marker="o", label=lab, color=col)
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    save_path = os.path.join(out_dir, "ratio_over_time.png")
    plt.savefig(save_path)

    save_path_eps = os.path.join(out_dir, "ratio_over_time.eps")
    plt.savefig(save_path_eps)

    plt.close()
    print(f"[DONE] Saved plot → {save_path}")


if __name__ == "__main__":
    main()

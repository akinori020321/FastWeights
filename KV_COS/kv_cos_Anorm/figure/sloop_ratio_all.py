#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S-loop ratio over time (SC-FW vs Ba-FW)
---------------------------------------
- ../results_A_kv/ 以下の SloopVec_kv_*.csv と Amat_kv_*.csv を読む
- 各モデルについて、各時刻 t で
      ratio(t) = ||A_t h_t^{(last input)}||_2 / ||h_t^{base}||_2
  を計算
  （S=1 なら、更新に実際に使われた h^{(0)} に対する A_t h^{(0)} を用いる）
- SC-FW(core=fw) と Ba-FW(core=tanh) を 2 行サブプロットで別々に描画

出力:
  plots/sloop_all/ratio_over_time.png
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# ファイル名パーサ
#   SloopVec_kv_(fw|tanh)_S<S>_eta<eta>_lam<lam>_seed<seed>.csv
# --------------------------------------------------
def parse_sloop_filename(fname):
    base = os.path.basename(fname)
    pattern = r"SloopVec_kv_(fw|tanh)_S(\d+)_eta([0-9]+)_lam([0-9]+)_seed(\d+)"
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)
    model_id = f"{core}_S{S}_eta{eta}_lam{lam}_seed{seed}"
    return core, S, eta, lam, seed, model_id


def parse_amat_filename(fname):
    base = os.path.basename(fname)
    pattern = r"Amat_kv_(fw|tanh)_S(\d+)_eta([0-9]+)_lam([0-9]+)_seed(\d+)"
    m = re.match(pattern, base)

    if m is None:
        return None

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)
    model_id = f"{core}_S{S}_eta{eta}_lam{lam}_seed{seed}"
    return core, S, eta, lam, seed, model_id


# --------------------------------------------------
# A 行列を読み込む
# --------------------------------------------------
def load_A(csv_path):
    df = pd.read_csv(csv_path)

    A_list = []
    for _, row in df.iterrows():
        vals = row.values.astype(float)

        # 1列余っていたら先頭を落とす
        if int(np.sqrt(vals.shape[0] - 1)) ** 2 == vals.shape[0] - 1:
            vals = vals[1:]

        L = vals.shape[0]
        d = int(np.sqrt(L))
        if d * d != L:
            raise ValueError(f"A row size {L} cannot form square: {L}")

        A_list.append(vals.reshape(d, d))

    return A_list


# --------------------------------------------------
# 1 モデル分の ratio(t) = ||A_t h_t^{last input}|| / ||h_t^base|| を計算
#   - base: s = -1 の行
#   - S-loop: s >= 0 の行から「最後の更新に入った h」を選ぶ
# --------------------------------------------------
def compute_ratio_curve(sloop_csv, amat_csv):
    df = pd.read_csv(sloop_csv)
    A_list = load_A(amat_csv)

    groups = df.groupby("t")

    t_list = []
    ratio_list = []

    for t, g in groups:
        t = int(t)

        # base（s = -1）
        base_rows = g[g["s"] == -1]
        if len(base_rows) == 0:
            # 想定外フォーマットならスキップ
            continue
        base_vec = base_rows.iloc[0]
        base_h = base_vec.filter(regex=r"h\[").values.astype(float)
        norm_base = np.linalg.norm(base_h) + 1e-8  # ゼロ除算防止

        # A_t
        if t >= len(A_list):
            # 安全のため範囲外ならスキップ
            continue
        A = A_list[t]

        # S-loop 部分（s >= 0）だけ取り出して s でソート
        loop_rows = g[g["s"] >= 0].sort_values("s")

        if len(loop_rows) == 0:
            # S=0 など、S-loop なしの場合はスキップ
            continue
        elif len(loop_rows) == 1:
            # 1 個しかなければそれを「更新に使った h」とみなす
            target_row = loop_rows.iloc[0]
        else:
            # 最後の 1 個前を「最後の更新で A に入った h」とみなす
            target_row = loop_rows.iloc[-2]

        h_target = target_row.filter(regex=r"h\[").values.astype(float)

        Ah = A @ h_target
        ratio = np.linalg.norm(Ah) / norm_base

        t_list.append(t)
        ratio_list.append(ratio)

    # t の昇順に並べ替え
    t_arr = np.array(t_list)
    ratio_arr = np.array(ratio_list)
    order = np.argsort(t_arr)
    return t_arr[order], ratio_arr[order]


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default="../results_A_kv",
        help="SloopVec_kv_*.csv / Amat_kv_*.csv のあるディレクトリ",
    )
    args = ap.parse_args()

    # ファイル探索
    sloop_list = sorted(glob.glob(os.path.join(args.dir, "SloopVec_kv_*.csv")))
    amat_list = sorted(glob.glob(os.path.join(args.dir, "Amat_kv_*.csv")))

    if len(sloop_list) == 0:
        print("[ERROR] No SloopVec_kv_*.csv found.")
        return
    if len(amat_list) == 0:
        print("[ERROR] No Amat_kv_*.csv found.")
        return

    # Amat 側を model_id -> path でマップ
    amat_map = {}
    for a_csv in amat_list:
        parsed = parse_amat_filename(a_csv)
        if parsed is None:
            continue
        _, _, _, _, _, model_id = parsed
        amat_map[model_id] = a_csv

    # 出力ディレクトリ
    out_dir = "plots/sloop_all"
    os.makedirs(out_dir, exist_ok=True)

    # 各モデルの曲線を SC-FW / Ba-FW に分けて保持
    curves_sc = []  # (t_arr, ratio_arr, label)
    curves_ba = []

    for s_csv in sloop_list:
        core, S, eta, lam, seed, model_id = parse_sloop_filename(s_csv)

        if model_id not in amat_map:
            print(f"[WARN] Amat not found for {model_id}, skip.")
            continue

        a_csv = amat_map[model_id]
        print(f"[PROCESS] {model_id}")
        t_arr, ratio_arr = compute_ratio_curve(s_csv, a_csv)

        # 凡例用の core 名を SC-FW / Ba-FW に変換
        if core == "fw":
            core_label = "SC-FW"
            curves_sc.append(
                (t_arr, ratio_arr,
                 f"{core_label}_S{S}_eta{eta}_lam{lam}_seed{seed}")
            )
        elif core == "tanh":
            core_label = "Ba-FW"
            curves_ba.append(
                (t_arr, ratio_arr,
                 f"{core_label}_S{S}_eta{eta}_lam{lam}_seed{seed}")
            )
        else:
            core_label = core.upper()
            curves_sc.append(
                (t_arr, ratio_arr,
                 f"{core_label}_S{S}_eta{eta}_lam{lam}_seed{seed}")
            )

    if len(curves_sc) == 0 and len(curves_ba) == 0:
        print("[ERROR] No valid curves to plot.")
        return

    # --------------------------------------------------
    # プロット：SC-FW / Ba-FW を上下に分ける
    # --------------------------------------------------
    if len(curves_sc) > 0 and len(curves_ba) > 0:
        fig, (ax_sc, ax_ba) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # 上段: SC-FW
        for t_arr, ratio_arr, label in curves_sc:
            ax_sc.plot(t_arr, ratio_arr, marker="o", label=label)
        ax_sc.axhline(1.0, color="gray", linestyle="--")
        ax_sc.set_ylabel(r"$\|A_t h_t\|_2 / \|h_t^{\mathrm{base}}\|_2$")
        ax_sc.set_title("SC-FW (self-consistent S-loop)")
        ax_sc.grid(True)
        ax_sc.legend()

        # 下段: Ba-FW
        for t_arr, ratio_arr, label in curves_ba:
            ax_ba.plot(t_arr, ratio_arr, marker="o", label=label)
        ax_ba.axhline(1.0, color="gray", linestyle="--")
        ax_ba.set_xlabel("t (step)")
        ax_ba.set_ylabel(r"$\|A_t h_t\|_2 / \|h_t^{\mathrm{base}}\|_2$")
        ax_ba.set_title("Ba-FW")
        ax_ba.grid(True)
        ax_ba.legend()

        fig.suptitle(
            r"ratio over time: "
            r"$\|A_t h_t\|_2 / \|h_t^{\mathrm{base}}\|_2$",
            y=0.99,
        )

    else:
        # どちらか片方しか無い場合は 1 枚だけ描画
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        if len(curves_sc) > 0:
            for t_arr, ratio_arr, label in curves_sc:
                ax.plot(t_arr, ratio_arr, marker="o", label=label)
            title_suffix = "SC-FW only"
        else:
            for t_arr, ratio_arr, label in curves_ba:
                ax.plot(t_arr, ratio_arr, marker="o", label=label)
            title_suffix = "Ba-FW only"

        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_xlabel("t (step)")
        ax.set_ylabel(r"$\|A_t h_t\|_2 / \|h_t^{\mathrm{base}}\|_2$")
        ax.set_title(
            r"ratio over time: "
            r"$\|A_t h_t\|_2 / \|h_t^{\mathrm{base}}\|_2$"
            f" ({title_suffix})"
        )
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, "ratio_over_time.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[DONE] Saved plot → {save_path}")


if __name__ == "__main__":
    main()

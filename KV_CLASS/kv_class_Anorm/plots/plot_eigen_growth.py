#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_eigen_growth.py
------------------------------------
A-matrix 固有値の “棘の成長” を可視化する図を作る
 - 上位固有値をカラーでプロット
 - 全ステップを 1 つの図にまとめる
 - Amat_kv_*_S*.csv に対応
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_A_csv(path):
    df = pd.read_csv(path)
    A_cols = [c for c in df.columns if c.startswith("A[")]
    steps = df["step"].values
    A = df[A_cols].values
    return steps, A, A_cols


def flatten_to_matrix(row_flat, d_h):
    """1 行の A ベクトルを (d_h, d_h) 行列へ"""
    return row_flat.reshape(d_h, d_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    # -------------------------------
    # Load A matrix CSV
    # -------------------------------
    steps, flatA, A_cols = load_A_csv(args.csv)
    d_h = int(np.sqrt(len(A_cols)))

    print(f"[INFO] d_h = {d_h}, steps = {len(steps)}")

    # -------------------------------
    # Compute eigenvalues at each step
    # -------------------------------
    eigvals_topk = []

    for i in range(len(steps)):
        A_t = flatten_to_matrix(flatA[i], d_h)
        w = np.linalg.eigvals(A_t)

        # real part の絶対値順にソートして上位 top-k を取得
        w_sorted = sorted(w, key=lambda x: abs(np.real(x)), reverse=True)
        eigvals_topk.append(w_sorted[:args.topk])

    eigvals_topk = np.array(eigvals_topk)  # shape (T, topk)

    # -------------------------------
    # Plot — Growth of top eigenvalues
    # -------------------------------
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    T = len(steps)
    x = steps

    for k in range(args.topk):
        vals = np.real(eigvals_topk[:, k])
        plt.plot(
            x,
            vals,
            color=cmap(k % 10),
            linewidth=2,
            label=f"eig {k+1}"
        )

    plt.title("Eigenvalue Growth (Top-k)", fontsize=20)
    plt.xlabel("Step t", fontsize=16)
    plt.ylabel("Eigenvalue (real part)", fontsize=16)
    plt.grid(True)
    plt.legend()

    # 出力名
    import os
    base = os.path.basename(args.csv)
    core = base.replace("Amat_kv_", "").replace(".csv", "")
    out_png = f"eig_growth_{core}.png"

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print(f"[DONE] Saved → {out_png}")


if __name__ == "__main__":
    main()

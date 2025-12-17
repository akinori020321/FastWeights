#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure/check_prototypes.py
------------------------------------------------------------
目的：
- 保存された checkpoint の mu / key_proto を読み出し、
- wait_vec（seed=999で再現）も加え、
- 「変な相関構造がない」ことを図2枚だけで確認する。

入出力：
- checkpoint: ../checkpoints/**/*.pt の中で最新の .pt を自動選択
- output: figure/plots/proto_check_YYYYmmdd_HHMMSS/ に png 2枚だけ保存
    1) cos_heatmap.png     : プロトタイプ間 cosine 類似度ヒートマップ（mu+key+wait）
    2) hist_offdiag.png    : off-diagonal cosine 分布（全ペア）

実行：
  python3 figure/check_prototypes.py
"""

from __future__ import annotations
import os
import glob
from datetime import datetime

import numpy as np
import torch


# =========================
# パス（figure/ を基準）
#   figure/ と checkpoints/ は同じ階層（兄弟ディレクトリ）
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../figure
PROJ_DIR = os.path.dirname(THIS_DIR)                    # .../（figure の親）
CKPT_ROOT = os.path.join(PROJ_DIR, "checkpoints")       # ../checkpoints
PLOTS_ROOT = os.path.join(THIS_DIR, "plots")            # figure/plots


# =========================
# 設定（引数なし）
# =========================
SEED_WAIT = 999
HEATMAP_MAX_N = 600   # 行列が大きすぎると見づらいので上限（必要なら増やす）


# =========================
# Utils
# =========================
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """各行を L2 正規化する（cos 計算の前処理）"""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def regen_wait_vec(d_g: int, seed_wait: int = 999) -> np.ndarray:
    """wait ベクトルを seed 固定で再現（学習スクリプトと同一）"""
    rng = np.random.RandomState(seed_wait)
    w = rng.randn(d_g).astype(np.float32)
    w /= (np.linalg.norm(w) + 1e-8)
    return w

def find_latest_ckpt(ckpt_root: str) -> str:
    """ckpt_root 以下の .pt を再帰探索し、更新時刻が最新のものを返す。"""
    paths = glob.glob(os.path.join(ckpt_root, "**", "*.pt"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No .pt found under: {ckpt_root}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def cosine_similarity_matrix(Xn: np.ndarray) -> np.ndarray:
    """
    Xn: (N, d) 各行が L2 正規化済みのベクトル
    返り値 S: (N, N),  S_ij = cos(x_i, x_j) = x_i^T x_j
    """
    return Xn @ Xn.T

def offdiag_values(S: np.ndarray) -> np.ndarray:
    """S の非対角成分（i≠j）だけを 1 次元に取り出す"""
    m = ~np.eye(S.shape[0], dtype=bool)
    return S[m].reshape(-1)


# =========================
# Plotters (2 figs only)
# =========================
def save_cosine_heatmap(
    S: np.ndarray,
    path: str,
    boundaries: list[int] | None = None,
    title: str = "",
):
    """
    S: cosine similarity matrix
    boundaries: mu / key / wait の境界線を入れるための index リスト
    """
    import matplotlib.pyplot as plt

    plt.figure()
    im = plt.imshow(S, aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("cosine similarity")

    if title:
        plt.title(title)

    plt.xlabel("prototype index")
    plt.ylabel("prototype index")

    # 境界線（mu / key / wait の区切り）
    if boundaries:
        ax = plt.gca()
        for b in boundaries:
            ax.axhline(b - 0.5)
            ax.axvline(b - 0.5)

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def save_hist(x: np.ndarray, path: str, title: str = ""):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(x, bins=80)
    if title:
        plt.title(title)
    plt.xlabel("cosine similarity (i≠j)")
    plt.ylabel("count")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print(f"[Debug] THIS_DIR   = {THIS_DIR}")
    print(f"[Debug] CKPT_ROOT  = {CKPT_ROOT}")
    print(f"[Debug] PLOTS_ROOT = {PLOTS_ROOT}")

    # ---- latest ckpt ----
    ckpt_path = find_latest_ckpt(CKPT_ROOT)
    print(f"[Info] Latest checkpoint: {ckpt_path}")

    # ---- output dir ----
    os.makedirs(PLOTS_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PLOTS_ROOT, f"proto_check_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] Output dir: {out_dir}")

    # ---- load ----
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "mu" not in state or "key_proto" not in state:
        raise KeyError("checkpoint must contain keys: 'mu' and 'key_proto'")

    mu = np.asarray(state["mu"], dtype=np.float32)          # (C, d_g)
    key = np.asarray(state["key_proto"], dtype=np.float32)  # (K, d_g)
    d_g = mu.shape[1]

    wait = regen_wait_vec(d_g, seed_wait=SEED_WAIT).astype(np.float32).reshape(1, -1)

    # ---- stack all prototypes ----
    # X = [mu; key; wait]
    X = np.concatenate([mu, key, wait], axis=0)  # (C+K+1, d)
    Xn = l2_normalize_rows(X)

    # boundaries for visualization
    C = mu.shape[0]
    K = key.shape[0]
    boundaries = [C, C + K]  # after mu, after key (wait is last)

    # if too big, truncate for heatmap only (hist uses full)
    N = Xn.shape[0]
    Xn_show = Xn
    boundaries_show = boundaries
    if N > HEATMAP_MAX_N:
        Xn_show = Xn[:HEATMAP_MAX_N]
        boundaries_show = [b for b in boundaries if b < HEATMAP_MAX_N]
        print(f"[Warn] N={N} is large; heatmap uses first {HEATMAP_MAX_N} rows only.")

    # ---- Fig 1: cosine similarity heatmap ----
    S_show = cosine_similarity_matrix(Xn_show)
    out_heat = os.path.join(out_dir, "cos_heatmap.png")
    save_cosine_heatmap(
        S_show,
        out_heat,
        boundaries=boundaries_show,
        title="Prototype cosine similarity heatmap (values μ, keys k, wait w)"
    )
    print(f"[SAVE] {out_heat}")

    # ---- Fig 2: off-diagonal histogram (FULL set) ----
    S_full = cosine_similarity_matrix(Xn)
    off = offdiag_values(S_full)
    out_hist = os.path.join(out_dir, "hist_offdiag.png")
    save_hist(
        off,
        out_hist,
        title="Off-diagonal cosine similarity distribution (all pairs, i≠j)"
    )
    print(f"[SAVE] {out_hist}")

    # console summary only（ファイルは増やさない）
    print("============================================================")
    print(f"[Summary] d_g={d_g}, C(mu)={C}, K(key)={K}, total={N}")
    print(f"[Summary] offdiag: mean={off.mean():.6f}, std={off.std():.6f}, max|.|={np.max(np.abs(off)):.6f}")
    print("============================================================")
    print(f"[DONE] Saved 2 figures in: {out_dir}")


if __name__ == "__main__":
    main()

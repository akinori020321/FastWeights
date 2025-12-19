#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ln_stats.py
----------------------------------------
figure/ 以下に置いて使う想定。
1つ上の階層にある checkpoints/*.pt から LayerNorm (ln_h) の統計量を集計し，
<root>/results_LN_stats に CSV とグラフを出力する。

使い方:
  cd figure
  python plot_ln_stats.py
    または
  python plot_ln_stats.py \
    --ckpt_pattern "../checkpoints/kv_*_S*_fw*_eta*_lam*_seed*.pt" \
    --out_dir "../results_LN_stats"
"""

from __future__ import annotations
import argparse
import glob
import os
import re

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_ln_params(ckpt_path: str):
    """checkpoint から ln_h.weight / ln_h.bias を取り出す"""

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "model_state" in state:
        raw_sd = state["model_state"]
    else:
        raw_sd = state

    sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

    w = sd.get("ln_h.weight", None)
    b = sd.get("ln_h.bias", None)

    if w is None or b is None:
        raise RuntimeError(f"ln_h.weight / ln_h.bias が見つかりません: {ckpt_path}")

    return w.detach().cpu().float().view(-1), b.detach().cpu().float().view(-1)


def stats_1d(x: torch.Tensor):
    """1 次元テンソルの統計量を dict で返す"""
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    min_v = x.min().item()
    max_v = x.max().item()
    n_pos = int((x > 0).sum().item())
    n_neg = int((x < 0).sum().item())
    n_zero = int((x == 0).sum().item())
    return dict(
        mean=mean,
        std=std,
        min=min_v,
        max=max_v,
        n_pos=n_pos,
        n_neg=n_neg,
        n_zero=n_zero,
    )


def parse_name(basename: str):
    """
    ファイル名から core, S, eta, lam, seed などを取り出す。
    例:
      kv_fw_S1_noise0400_eta0300_lam0950_seed0
      kv_fw_S1_noise0400_fw1_eta0300_lam0950_seed0
    """
    pat = re.compile(
        r"^kv_(?P<core>\w+)_S(?P<S>\d+)"
        r"(?:_noise(?P<noise>\d{4}))?"
        r"(?:_fw(?P<fw>\d))?"                 # ★ fw を任意にする
        r"_eta(?P<eta>\d{4})_lam(?P<lam>\d{4})_seed(?P<seed>\d+)$"
    )
    m = pat.match(basename)
    if m is None:
        return dict(core="?", S=None, fw=None, eta=None, lam=None, seed=None, noise=None)

    d = m.groupdict()
    eta = int(d["eta"]) / 1000.0
    lam = int(d["lam"]) / 1000.0
    noise = int(d["noise"]) / 1000.0 if d.get("noise") is not None else None
    fw = int(d["fw"]) if d.get("fw") is not None else None

    return dict(
        core=d["core"],
        S=int(d["S"]),
        fw=fw,
        eta=eta,
        lam=lam,
        seed=int(d["seed"]),
        noise=noise,
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  # 一つ上の階層

    # ここを変更
    plots_dir = os.path.join(root_dir, "figure/plots")
    default_ckpt_pattern = os.path.join(
        root_dir, "checkpoints", "kv_*_S*_noise*_eta*_lam*_seed*.pt"
    )
    default_out_dir = os.path.join(plots_dir, "results_LN_stats")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_pattern",
        type=str,
        default=default_ckpt_pattern,
        help="読み込む checkpoint の glob パターン",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help="CSV / 図 の出力ディレクトリ",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_list = sorted(glob.glob(args.ckpt_pattern))
    if not ckpt_list:
        print(f"[WARN] No checkpoints found for pattern: {args.ckpt_pattern}")
        return

    print(f"[Info] Found {len(ckpt_list)} checkpoints.")

    rows = []
    for ckpt in ckpt_list:
        base = os.path.splitext(os.path.basename(ckpt))[0]
        meta = parse_name(base)

        print(f"[Info] Processing {base}")

        w, b = load_ln_params(ckpt)
        w_stat = stats_1d(w)
        b_stat = stats_1d(b)

        row = {
            "ckpt": base,
            **meta,
            "w_mean": w_stat["mean"],
            "w_std": w_stat["std"],
            "w_min": w_stat["min"],
            "w_max": w_stat["max"],
            "w_n_pos": w_stat["n_pos"],
            "w_n_neg": w_stat["n_neg"],
            "w_n_zero": w_stat["n_zero"],
            "b_mean": b_stat["mean"],
            "b_std": b_stat["std"],
            "b_min": b_stat["min"],
            "b_max": b_stat["max"],
            "b_n_pos": b_stat["n_pos"],
            "b_n_neg": b_stat["n_neg"],
            "b_n_zero": b_stat["n_zero"],
        }
        rows.append(row)

    # ==============================
    # CSV 出力
    # ==============================
    import csv

    csv_path = os.path.join(args.out_dir, "ln_summary.csv")
    print(f"[Info] Writing CSV → {csv_path}")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ==============================
    # 図: w_mean / b_mean のバー
    # ==============================
    labels = [f"{r['core']}\nlam={r['lam']:.2f}" for r in rows]
    x = np.arange(len(rows))
    w_means = np.array([r["w_mean"] for r in rows])
    b_means = np.array([r["b_mean"] for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes[0]
    ax1.bar(x, w_means)
    ax1.set_ylabel("LN weight mean (ln_h.weight)")
    ax1.set_title("LayerNorm scale (gamma) mean per checkpoint")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax2 = axes[1]
    ax2.bar(x, b_means)
    ax2.set_ylabel("LN bias mean (ln_h.bias)")
    ax2.set_title("LayerNorm bias (beta) mean per checkpoint")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "ln_stats.png")
    print(f"[Info] Saving figure → {out_png}")
    fig.savefig(out_png, dpi=200)
    out_eps = os.path.join(args.out_dir, "ln_stats.eps")
    fig.savefig(out_eps, format="eps")
    plt.close(fig)

    print("[DONE]")


if __name__ == "__main__":
    main()

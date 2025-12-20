#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LN.py
----------------------------------------
Checkpoint (.pt) から LayerNorm の gamma(weight) / beta(bias) を取り出して
簡単な統計量を表示するスクリプト。

使い方:
  python LN.py --ckpt checkpoints/kv_fw_S1_fw1_eta0300_lam0850_seed0.pt
"""

from __future__ import annotations
import argparse
import torch
import io
from contextlib import redirect_stdout
import os


def is_ln_param(name: str) -> bool:
    """
    LayerNorm っぽいパラメータ名かどうかを判定する。
    - 名前に 'ln' または 'layernorm' を含み
    - かつ末尾が '.weight' or '.bias'
    """
    name_lower = name.lower()
    if ("ln" not in name_lower) and ("layernorm" not in name_lower):
        return False
    return name_lower.endswith(".weight") or name_lower.endswith(".bias")


def summarize_param(name: str, tensor: torch.Tensor) -> None:
    """1個のパラメータの統計を表示"""
    t = tensor.detach().cpu().float().view(-1)
    mean = t.mean().item()
    std = t.std(unbiased=False).item()
    min_v = t.min().item()
    max_v = t.max().item()
    n_pos = int((t > 0).sum().item())
    n_neg = int((t < 0).sum().item())
    n_zero = int((t == 0).sum().item())

    print(f"{name}")
    print(f"  shape : {tuple(tensor.shape)}")
    print(f"  mean  : {mean:.6f}, std: {std:.6f}")
    print(f"  min   : {min_v:.6f}, max: {max_v:.6f}")
    print(f"  #pos  : {n_pos}, #neg: {n_neg}, #zero: {n_zero}")
    print("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="学習済み checkpoint (.pt) のパス",
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))

    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"[Info] Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)

        # run.py / run_A_kv.py と同様の形式を想定:
        #   state["model_state"] に sd が入っている
        if isinstance(state, dict) and "model_state" in state:
            raw_sd = state["model_state"]
        else:
            # 念のため、直接 state_dict が入っているケースにも対応
            raw_sd = state

        # torch.compile で付く "_orig_mod." 接頭辞を削除
        sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

        # LayerNorm パラメータだけ抽出
        ln_items = [(k, v) for k, v in sd.items() if is_ln_param(k)]
        ln_items.sort(key=lambda kv: kv[0])

        if not ln_items:
            print("[WARN] 'ln' / 'layernorm' を含む weight / bias が見つかりませんでした。")
            text = buf.getvalue()
        else:
            print(f"[Info] Found {len(ln_items)} LayerNorm parameters.\n")

            for name, tensor in ln_items:
                summarize_param(name, tensor)

            text = buf.getvalue()

    print(text, end="")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, max(2.0, 0.24 * len(text.splitlines()) + 1.0)))
    plt.axis("off")
    plt.text(
        0.01, 0.99, text,
        va="top", ha="left",
        family="monospace",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.close()


if __name__ == "__main__":
    main()

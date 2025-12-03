#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_single.py
----------------------------------------
Single-clean A-dynamics 実験ランナー。

- 学習済み FW / Self-consistent FW コアをロード
- 同じ clean ベクトルを T_long ステップ連続して入力
- 各ステップごとに
    - ||A_t||_F
    - ρ(A_t)
    - ||h_t||
    - ||A_t h_t||
  を記録して CSV に保存する。
"""

from __future__ import annotations
import argparse
import os

import torch
import numpy as np

from fw_kv.utils import set_seed
from fw_kv.models.core_config import CoreCfg
from fw_kv.models.core_fw import CoreRNNFW as CoreFW
from fw_kv.models.core_tanh import CoreRNNFW as CoreTanh

import seq_gen
import utils_dyn


def build_core_model(args) -> torch.nn.Module:
    """
    CoreCfg からコアモデル（FW or Self-consistent）を構築。
    """
    cfg = CoreCfg(
        glimpse_dim=args.d_g,
        hidden_dim=args.d_h,
        lambda_decay=args.lambda_decay,
        eta=args.eta,
        epsilon=args.eps,
        inner_steps=args.S,
        use_layernorm=args.use_ln,
        use_A=True,      # この実験では FW 前提
    )

    if args.core_type == "fw":
        core_cls = CoreFW
    elif args.core_type == "tanh":
        core_cls = CoreTanh
    else:
        raise ValueError(f"Unsupported core_type: {args.core_type}")

    model = core_cls(cfg)
    return model


def main():
    ap = argparse.ArgumentParser(
        description="Single-clean A-dynamics runner"
    )

    # --------------------------------------------------
    # 基本設定
    # --------------------------------------------------
    ap.add_argument("--ckpt", type=str, required=True,
                    help="ロードする core モデルの checkpoint (.pt) パス")
    ap.add_argument("--out_csv", type=str, default="results_A_single/A_dynamics_single.csv",
                    help="結果を書き出す CSV パス")
    ap.add_argument("--core_type", type=str, default="fw",
                    choices=["fw", "tanh"],
                    help="使用するコアタイプ（fw: 古典FW, tanh: 自己整合FW）")

    # --------------------------------------------------
    # モデル次元・ハイパラ（学習時と揃える）
    # --------------------------------------------------
    ap.add_argument("--d_g", type=int, default=40, help="入力ベクトル次元")
    ap.add_argument("--d_h", type=int, default=100, help="隠れ次元")
    ap.add_argument("--lambda_decay", type=float, default=0.95,
                    help="Hebbian メモリの減衰係数 λ")
    ap.add_argument("--eta", type=float, default=0.50,
                    help="Hebbian 学習率 η")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="数値安定用 ε")
    ap.add_argument("--S", type=int, default=3,
                    help="Fast Weights の inner loop 回数 S")
    ap.add_argument("--use_ln", action="store_true",
                    help="隠れ状態に LayerNorm を入れる（学習時と揃えること）")

    # --------------------------------------------------
    # 実験設定
    # --------------------------------------------------
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--T_long", type=int, default=80,
                    help="clean を繰り返して入力するステップ数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="GPU が利用可能なら CUDA を使用")

    args = ap.parse_args()

    # ==================================================
    # Seed & Device
    # ==================================================
    set_seed(args.seed)
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"[Info] device = {device}")

    # ==================================================
    # Core モデル構築 & Checkpoint ロード
    # ==================================================
    model = build_core_model(args)
    print(f"[Info] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    print(f"[Info] core_type = {args.core_type}, "
          f"d_g = {args.d_g}, d_h = {args.d_h}, S = {args.S}")

    # ==================================================
    # Single-clean シーケンス生成
    # ==================================================
    z_seq, clean_vec = seq_gen.make_single_clean_sequence(
        d_g=args.d_g,
        batch_size=args.batch_size,
        T_long=args.T_long,
        device=device,
        seed=args.seed + 100,  # モデル seed と分けたい場合
    )
    T_total = z_seq.size(0)
    B = z_seq.size(1)
    d_h = model.d_h

    print(f"[Info] Generated single-clean sequence: "
          f"T={T_total}, B={B}, d_g={args.d_g}")

    # ==================================================
    # 初期状態 (h, A)
    # ==================================================
    h = torch.zeros(B, d_h, device=device)
    A = torch.zeros(B, d_h, d_h, device=device)

    # ==================================================
    # 時間発展の記録
    # ==================================================
    rows = []

    with torch.no_grad():
        for t in range(T_total):
            z_t = z_seq[t]  # (B, d_g)

            # 1ステップ更新
            h, A = model.forward(z_t, h, A)

            froA, specA, h_norm, Ah_norm = utils_dyn.compute_A_stats(A, h)

            rows.append([
                int(t),
                float(froA),
                float(specA),
                float(h_norm),
                float(Ah_norm),
            ])

    # ==================================================
    # CSV 書き出し
    # ==================================================
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import csv
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "froA", "specA", "h_norm", "Ah_norm"])
        for row in rows:
            writer.writerow(row)

    print(f"[DONE] Saved single-clean A dynamics → {args.out_csv}")


if __name__ == "__main__":
    main()

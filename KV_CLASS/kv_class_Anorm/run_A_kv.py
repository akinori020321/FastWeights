#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_A_kv.py
----------------------------------------
Key-Value Bind / Query（A-dynamics + S-loop + Query判定）実験ランナー
"""

from __future__ import annotations
import argparse
import os
import csv

import numpy as np
import torch

from fw_kv.utils import set_seed
from fw_kv.models.core_config import CoreCfg
from fw_kv.models.core_fw import batch_spectral_radius
from fw_kv.models.head import OutputHead
from seq_gen import make_keyvalue_sequence


# ======================================================
# Core モデル構築
# ======================================================
def build_core_model(args):
    cfg = CoreCfg(
        glimpse_dim=args.d_g,
        hidden_dim=args.d_h,
        inner_steps=args.S,
        lambda_decay=args.lambda_decay,
        eta=args.eta,
        use_layernorm=args.use_ln,
        use_A=args.use_fw,
    )

    if args.core_type == "fw":
        from fw_kv.models.core_fw import CoreRNNFW
        print("[Model] Using core_fw.py")
        return CoreRNNFW(cfg)

    elif args.core_type == "tanh":
        from fw_kv.models.core_tanh import CoreRNNFW
        print("[Model] Using core_tanh.py")
        return CoreRNNFW(cfg)

    else:
        raise ValueError("Unsupported core_type for A-dynamics")


# ======================================================
# main
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    # checkpoint
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="results_A_kv/A_kv_fw_S3.csv")

    # model設定
    ap.add_argument("--core_type", type=str, default="fw", choices=["fw", "tanh"])
    ap.add_argument("--d_g", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=128)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--lambda_decay", type=float, default=0.97)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--use_ln", action="store_true")
    ap.add_argument("--no_ln", dest="use_ln", action="store_false")
    ap.add_argument("--use_fw", action="store_true")
    ap.add_argument("--no_fw", dest="use_fw", action="store_false")
    ap.add_argument("--eps", type=float, default=1e-6)

    # KV config
    ap.add_argument("--T_bind", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_key_proto", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.set_defaults(use_ln=True, use_fw=True)

    args = ap.parse_args()

    # --------------------------------------------------
    # device
    # --------------------------------------------------
    set_seed(args.seed)
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"[Info] device = {device}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # --------------------------------------------------
    # model load
    # --------------------------------------------------
    model = build_core_model(args).to(device)
    print(f"[Info] Loading checkpoint: {args.ckpt}")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_sd = state["model_state"]
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    # --------------------------------------------------
    # μ 再生成
    # --------------------------------------------------
    rng_mu = np.random.RandomState(123)
    mu_value = rng_mu.randn(args.num_classes, args.d_g).astype(np.float32)
    mu_value /= (np.linalg.norm(mu_value, axis=1, keepdims=True) + 1e-8)

    # --------------------------------------------------
    # key proto
    # --------------------------------------------------
    num_key_proto = int(2.6 * args.num_classes) if args.num_key_proto is None else args.num_key_proto
    rng_k = np.random.RandomState(777)
    key_proto = rng_k.randn(num_key_proto, args.d_g).astype(np.float32)
    key_proto /= (np.linalg.norm(key_proto, axis=1, keepdims=True) + 1e-8)

    # --------------------------------------------------
    # classIDs for value side
    # --------------------------------------------------
    rng_cls = np.random.RandomState(args.seed + 10)
    class_ids = rng_cls.randint(0, args.num_classes, size=args.T_bind).tolist()
    print(f"[Info] class_ids = {class_ids}")

    # --------------------------------------------------
    # KV + Query sequence
    # --------------------------------------------------
    z_seq, event_list = make_keyvalue_sequence(
        mu_value=mu_value,
        key_proto=key_proto,
        batch_size=args.batch_size,
        class_ids=class_ids,
        device=device,
        seed=args.seed + 10,
    )

    T_total = z_seq.size(0)
    B = z_seq.size(1)
    print(f"[Info] Generated sequence: T={T_total}, B={B}")

    # --------------------------------------------------
    # head 構築
    # --------------------------------------------------
    head = OutputHead(args.d_h, args.num_classes).to(device)
    head.load_state_dict(state["head_state"], strict=True)
    head.eval()

    # --------------------------------------------------
    # forward_episode（内部で query 判定）
    # --------------------------------------------------
    _ = model.forward_episode(
        z_seq=z_seq,
        y=None,
        head=head,
        event_list=event_list,
        S=args.S,
        class_ids=class_ids,
        mu_value=mu_value,
        train=False,
        compute_stats=False,
    )

    # --------------------------------------------------
    # ☆ Query 結果表示
    # --------------------------------------------------
    print("===========================================")
    print(" QUERY RESULT (from Core)")
    print("===========================================")
    for k, v in model.log_query.items():
        print(f" {k:12s} : {v}")
    print("===========================================")

    # --------------------------------------------------
    # ☆ Query 結果 CSV 保存
    # --------------------------------------------------
    query_csv = args.out_csv.replace("A_kv_", "Query_kv_")
    with open(query_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "pred", "correct", "margin", "cosine"])
        q = model.log_query
        w.writerow([q["true"], q["pred"], q["correct"], q["margin"], q["cosine"]])

    print(f"[DONE] Saved Query CSV → {query_csv}")

    # --------------------------------------------------
    # A-dynamics CSV 保存
    # --------------------------------------------------
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "kind", "class_id", "froA", "specA", "h_norm", "Ah_norm"])
        for t in range(len(model.log_froA)):
            w.writerow([
                t,
                model.log_kind[t],
                int(model.log_class[t]),
                model.log_froA[t],
                model.log_specA[t],
                model.log_hnorm[t],
                model.log_Ahnorm[t],
            ])

    print(f"[DONE] Saved KV A-dynamics → {args.out_csv}")

    # --------------------------------------------------
    # Save A-matrix CSV (optional)
    # --------------------------------------------------
    A_csv = args.out_csv.replace("A_kv_", "Amat_kv_")

    with open(A_csv, "w", newline="") as f:
        w = csv.writer(f)

        # 見出し
        d_h = args.d_h
        header = ["step"] + [f"A[{i},{j}]" for i in range(d_h) for j in range(d_h)]
        w.writerow(header)

        # 書き込み
        for t, A_t in enumerate(model.log_A):
            A_flat = A_t.view(-1).tolist()  # flatten
            w.writerow([t] + A_flat)

    print(f"[DONE] A-matrix CSV → {A_csv}")

    # --------------------------------------------------
    # S-loop CSV 保存（通常版）
    # --------------------------------------------------
    sloop_csv = args.out_csv.replace("A_kv_", "Sloop_kv_")

    with open(sloop_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s", "h_norm", "Ah_norm", "cos_h0"])
        for t, s_list in enumerate(model.log_sloop):
            for d in s_list:
                w.writerow([
                    t,
                    d["s"],
                    d["h_norm"],
                    d["Ah_norm"],
                    d["cos_h0"],
                ])

    print(f"[DONE] Saved S-loop CSV → {sloop_csv}")

    # --------------------------------------------------
    # CosValue CSV 保存（Query の S-loop のみ）
    # --------------------------------------------------
    cos_csv = args.out_csv.replace("A_kv_", "CosValue_kv_")

    with open(cos_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s", "cos_value"])   # ヘッダー

        for t, s_list in enumerate(model.log_sloop):
            for d in s_list:
                if "cos_value" in d:  # ★ Query のみ
                    w.writerow([t, d["s"], d["cos_value"]])

    print(f"[DONE] Saved CosValue CSV → {cos_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_A_kv.py (Direction Reconstruction version)
----------------------------------------------
KV Bind → Query（方向復元）＋ A-dynamics / S-loop / CosValue ログ
"""

from __future__ import annotations
import argparse
import os
import csv

import torch
import numpy as np

from fw_kv.utils import set_seed
from fw_kv.models.core_config import CoreCfg
from fw_kv.models.head import OutputHead
from seq_gen import make_keyvalue_sequence_direction


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
        print("[Model] Using core_fw (direction unified)")
        return CoreRNNFW(cfg)

    elif args.core_type == "tanh":
        from fw_kv.models.core_tanh import CoreRNNFW
        print("[Model] Using core_tanh (direction unified)")
        return CoreRNNFW(cfg)

    else:
        raise ValueError("Unsupported core_type")


# ======================================================
# main
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    # checkpoint
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="results_A_kv/A_kv_fw_S1.csv")

    # model設定
    ap.add_argument("--core_type", type=str, default="fw", choices=["fw", "tanh"])
    ap.add_argument("--d_g", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=128)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--lambda_decay", type=float, default=0.97)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--use_ln", action="store_true")
    ap.add_argument("--no_ln", dest="use_ln", action="store_false")
    ap.add_argument("--use_fw", action="store_true")
    ap.add_argument("--no_fw", dest="use_fw", action="store_false")
    ap.add_argument("--eps", type=float, default=1e-6)

    # KV task
    ap.add_argument("--T_bind", type=int, default=60)
    ap.add_argument("--duplicate", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")

    ap.add_argument("--num_classes", type=int, default=10,
                help="使用するクラス数。key数は class×3 に自動設定する。")

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
    # load model
    # --------------------------------------------------
    model = build_core_model(args).to(device)

    print(f"[Info] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # ---- 互換性吸収：model_state ----
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    # --------------------------------------------------
    # head（fc.weight / fc.bias に自動変換）
    # --------------------------------------------------
    head = OutputHead(args.d_h, args.d_g).to(device)

    raw_head = state["head_state"]

    # ---- 互換性補正 ----
    if ("weight" in raw_head) and ("bias" in raw_head):
        # 旧フォーマット → 新フォーマットへ rename
        head_state = {
            "fc.weight": raw_head["weight"],
            "fc.bias": raw_head["bias"]
        }
        print("[Fix] Converted head_state: weight/bias → fc.weight/fc.bias")
    else:
        # すでに正しい形式
        head_state = raw_head

    head.load_state_dict(head_state, strict=True)
    head.eval()

    # --------------------------------------------------
    # num_classes と num_keys を固定
    # --------------------------------------------------
    num_classes = args.num_classes
    num_keys = int(2.6 * args.num_classes)

    print(f"[Info] num_classes={num_classes}, num_keys={num_keys}")

    # --------------------------------------------------
    # μ（value-class prototypes） → class には依存
    # --------------------------------------------------
    rng_mu = np.random.RandomState(123)
    mu_value = rng_mu.randn(num_classes, args.d_g).astype(np.float32)
    mu_value /= (np.linalg.norm(mu_value, axis=1, keepdims=True) + 1e-8)
    print(f"[Info] mu_value shape = {mu_value.shape}")

    # --------------------------------------------------
    # key prototypes → class とは独立
    # --------------------------------------------------
    rng_k = np.random.RandomState(777)
    key_proto = rng_k.randn(num_keys, args.d_g).astype(np.float32)
    key_proto /= (np.linalg.norm(key_proto, axis=1, keepdims=True) + 1e-8)
    print(f"[Info] key_proto shape = {key_proto.shape}")

    # --------------------------------------------------
    # KV sequence（Direction Task）
    #   → class_ids / key_ids はここでは作らない！
    #   → make_keyvalue_sequence_direction 側で扱う
    # --------------------------------------------------
    z_seq, event_list, clean_target = make_keyvalue_sequence_direction(
        d_g=args.d_g,
        batch_size=args.batch_size,
        T_bind=args.T_bind,
        duplicate=args.duplicate,
        beta=args.beta,
        device=device,
        seed=args.seed + 200,

        # ★ これだけ渡す（固定ベクトル）
        mu_value=mu_value,
        key_proto=key_proto,
    )

    T_total = z_seq.size(0)
    print(f"[Info] Sequence generated: T={T_total}, B={args.batch_size}")

    # --------------------------------------------------
    # forward (episode)
    # --------------------------------------------------
    loss, cosine, stats = model.forward_episode(
        z_seq=z_seq,
        clean_vec=clean_target,
        head=head,
        event_list=event_list,
        S=args.S,
        train=False,
        compute_stats=True,
    )

    # --------------------------------------------------
    # Print Query result
    # --------------------------------------------------
    print("====================================")
    print(" QUERY RESULT (Direction)")
    print("====================================")
    print(f" cosine : {cosine:.6f}")
    print("====================================")

    # --------------------------------------------------
    # Save Query CSV
    # --------------------------------------------------
    query_csv = args.out_csv.replace("A_kv_", "Query_kv_")

    with open(query_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cosine"])
        w.writerow([cosine])

    print(f"[DONE] Query CSV → {query_csv}")

    # --------------------------------------------------
    # Save A-dynamics CSV
    # --------------------------------------------------
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "kind", "cid", "froA", "specA", "h_norm", "Ah_norm"])

        for t in range(len(model.log_froA)):
            w.writerow([
                t,
                model.log_kind[t],
                model.log_class[t],      # ← ★ これを忘れていた！
                model.log_froA[t],
                model.log_specA[t],
                model.log_hnorm[t],
                model.log_Ahnorm[t],
            ])

    print(f"[DONE] A-dynamics CSV → {args.out_csv}")


    # --------------------------------------------------
    # Save A-matrix CSV (optional)
    # --------------------------------------------------
    A_csv = args.out_csv.replace("A_kv_", "Amat_kv_")

    with open(A_csv, "w", newline="") as f:
        w = csv.writer(f)

        d_h = args.d_h
        header = ["step"] + [f"A[{i},{j}]" for i in range(d_h) for j in range(d_h)]
        w.writerow(header)

        for t, A_t in enumerate(model.log_A):
            A_flat = A_t.view(-1).tolist()
            w.writerow([t] + A_flat)

    print(f"[DONE] A-matrix CSV → {A_csv}")

    # --------------------------------------------------
    # Save S-loop CSV
    # --------------------------------------------------
    sloop_csv = args.out_csv.replace("A_kv_", "Sloop_kv_")

    with open(sloop_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s", "h_norm", "Ah_norm", "cos_h0", "cos_value"])

        for t, s_list in enumerate(model.log_sloop):
            for d in s_list:
                cos_value = d["cos_value"] if "cos_value" in d else ""
                w.writerow([
                    t,
                    d["s"],
                    d["h_norm"],
                    d["Ah_norm"],
                    d["cos_h0"],
                    cos_value,
                ])

    print(f"[DONE] S-loop CSV → {sloop_csv}")

    # --------------------------------------------------
    # Save CosValue CSV（Query only）
    # --------------------------------------------------
    cos_csv = args.out_csv.replace("A_kv_", "CosValue_kv_")

    with open(cos_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s", "cos_value"])

        for t, s_list in enumerate(model.log_sloop):
            for d in s_list:
                if "cos_value" in d:
                    w.writerow([t, d["s"], d["cos_value"]])

    print(f"[DONE] CosValue CSV → {cos_csv}")

if __name__ == "__main__":
    main()

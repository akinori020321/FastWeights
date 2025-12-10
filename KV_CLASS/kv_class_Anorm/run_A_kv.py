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
    
    elif args.core_type == "rnn":
        from fw_kv.models.core_rnn_ln import CoreRNN_LN
        print("[Model] Using core_rnn_ln.py (Pure RNN + LayerNorm)")
        return CoreRNN_LN(cfg)

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
    ap.add_argument("--core_type", type=str, default="fw", choices=["fw", "tanh", "rnn"])
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

    ap.add_argument("--T_bind", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--num_wait", type=int, default=0,
                    help="Bind の後に挟む Wait ステップの数")

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
    # ★ 保存されていた μ / key_proto を復元
    # --------------------------------------------------
    if "mu" not in state or "key_proto" not in state:
        raise ValueError("Checkpoint does not contain 'mu' or 'key_proto'.")

    mu_value = state["mu"]
    key_proto = state["key_proto"]

    print(f"[Info] Loaded mu from checkpoint: {mu_value.shape}")
    print(f"[Info] Loaded key_proto from checkpoint: {key_proto.shape}")

    # --------------------------------------------------
    # ★ class_ids を学習時と同じように生成（Bind のクラス列）
    # --------------------------------------------------
    rng_cls = np.random.RandomState(args.seed + 10)
    class_ids = rng_cls.randint(0, args.num_classes, size=args.T_bind).tolist()
    print(f"[Info] class_ids = {class_ids}")

    # --------------------------------------------------
    # ★ 待機ベクトル（固定）の生成（学習時と完全に同じ方式）
    # --------------------------------------------------
    rng_wait = np.random.RandomState(999)  # 固定seed
    wait_vec = rng_wait.randn(args.d_g).astype(np.float32)
    wait_vec /= np.linalg.norm(wait_vec) + 1e-8  # unit norm

    # torch テンソルに変換して device に載せる
    wait_vec = torch.from_numpy(wait_vec).float().to(device)

    # --------------------------------------------------
    # KV + Query sequence（KVDataset と同じ方式）
    # --------------------------------------------------
    z_seq, event_list = make_keyvalue_sequence(
        mu_value=mu_value,
        key_proto=key_proto,
        batch_size=args.batch_size,
        class_ids=class_ids,
        device=device,
        seed=args.seed + 3000,
        num_wait=args.num_wait,           # ★ 追加
        wait_vec=wait_vec,                # ★ 追加
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
    # Save A-matrix CSV (FW / tanh のみ)
    # --------------------------------------------------
    A_csv = args.out_csv.replace("A_kv_", "Amat_kv_")

    # RNN の場合は A が存在しないためスキップ
    if args.core_type in ["rnn"]:  
        print("[INFO] core_type=rnn → A-matrix CSV を保存しません")
    else:
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
    # S-loop CSV 保存
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
    # CosValue CSV 保存
    # --------------------------------------------------
    cos_csv = args.out_csv.replace("A_kv_", "CosValue_kv_")

    with open(cos_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "s", "cos_value"])
        for t, s_list in enumerate(model.log_sloop):
            for d in s_list:
                if "cos_value" in d:
                    w.writerow([t, d["s"], d["cos_value"]])

    print(f"[DONE] Saved CosValue CSV → {cos_csv}")

    # --------------------------------------------------
    # h-full CSV 保存（★修正版）
    # --------------------------------------------------
    h_csv = args.out_csv.replace("A_kv_", "H_kv_")

    with open(h_csv, "w", newline="") as f:
        w = csv.writer(f)

        # header を変更：step, kind, class_id, h[0], ...
        header = ["step", "kind", "class_id"] + [f"h[{i}]" for i in range(args.d_h)]
        w.writerow(header)

        for t, h_t in enumerate(model.log_h_full):
            # h_t : (B, d_h)
            h_mean = h_t.mean(dim=0).tolist()

            kind = model.log_kind[t]          # "key" / "value" / "query"
            cid  = int(model.log_class[t])    # クラスID (query の場合は bind_idx)

            w.writerow([t, kind, cid] + h_mean)

    print(f"[DONE] Saved h-full CSV → {h_csv}")

    # --------------------------------------------------
    # S-loop（base + h_s ベクトル）生データ保存
    # --------------------------------------------------
    sloop_vec_csv = args.out_csv.replace("A_kv_", "SloopVec_kv_")

    # log_h_sloop / log_base の両方が存在するか確認
    if hasattr(model, "log_base") and hasattr(model, "log_h_sloop"):

        with open(sloop_vec_csv, "w", newline="") as f:
            w = csv.writer(f)

            # ---- header ----
            header = ["t", "s", "vec_type", "kind", "class_id"]
            header += [f"h[{i}]" for i in range(args.d_h)]
            w.writerow(header)

            T = len(model.log_base)

            for t in range(T):

                # kind, class_id 判定
                kind = model.log_kind[t] if t < len(model.log_kind) else "-"
                cid  = int(model.log_class[t]) if t < len(model.log_class) else -1

                # ---------------------------
                # (1) base（s = -1）
                # ---------------------------
                h_base = model.log_base[t]     # shape: (B, d_h)
                h_base_mean = h_base.mean(dim=0).tolist()

                w.writerow([t, -1, "base", kind, cid] + h_base_mean)

                # ---------------------------
                # (2) S-loop の h_s
                # ---------------------------
                h_s_list = model.log_h_sloop[t] if t < len(model.log_h_sloop) else []

                for s, h_s in enumerate(h_s_list):
                    h_s_mean = h_s.mean(dim=0).tolist()
                    w.writerow([t, s, "hs", kind, cid] + h_s_mean)

        print(f"[DONE] Saved S-loop vector CSV → {sloop_vec_csv}")

    else:
        print("[INFO] S-loop vector CSV は log_base / log_h_sloop が無いためスキップしました。")

if __name__ == "__main__":
    main()

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
        use_A=args.use_fw,   # ← rnn のときは main側で False に強制される
    )

    if args.core_type == "fw":
        from fw_kv.models.core_fw import CoreRNNFW
        print("[Model] Using core_fw (direction unified)")
        return CoreRNNFW(cfg)

    elif args.core_type == "tanh":
        from fw_kv.models.core_tanh import CoreRNNFW
        print("[Model] Using core_tanh (direction unified)")
        return CoreRNNFW(cfg)

    elif args.core_type == "rnn":
        # ★ RNN専用実装があるならそれを使う
        #    無い場合は安全策として core_fw を「A無効」で代替（重み互換を壊さない）
        try:
            from fw_kv.models.core_rnn import CoreRNN
            print("[Model] Using core_rnn (no Fast Weights)")
            return CoreRNN(cfg)
        except Exception as e:
            from fw_kv.models.core_fw import CoreRNNFW
            print(f"[WARN] core_rnn import failed ({e}). Fallback: core_fw with A disabled (RNN-LN mode)")
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
    ap.add_argument("--core_type", type=str, default="fw", choices=["rnn", "fw", "tanh"])
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

    # ★ 追加：bind_noise_std（Clean 部分のノイズ率）
    ap.add_argument("--bind_noise_std", type=float, default=0.0,
                    help="Bind（Clean）のノイズ混合率 r。KVDataset の bind_noise_std と対応。")

    # ★ Wait ステップ数（Δ_wait 用）
    ap.add_argument(
        "--num_wait",
        type=int,
        default=0,
        help="Bind の後に挟む Wait ステップの数",
    )

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")

    # これはチェックポイントの mu から上書きされる想定
    ap.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="使用するクラス数（チェックポイント上の mu.shape[0] と一致している必要あり）",
    )

    ap.set_defaults(use_ln=True, use_fw=True)
    args = ap.parse_args()

    # --------------------------------------------------
    # ★ core_type=rnn のときは FW(A) を必ず無効化
    #    （core名は rnn のまま運用 → ファイル名にも反映）
    # --------------------------------------------------
    if args.core_type == "rnn":
        args.use_fw = False
        print("[INFO] core_type=rnn → force use_fw=False (A disabled)")

    # --------------------------------------------------
    # ★ out_csv 自動命名（ユーザーが指定してない場合のみ）
    #    rnn / fw / tanh の core名がそのままファイル名に入る
    # --------------------------------------------------
    if args.out_csv == "results_A_kv/A_kv_fw_S1.csv":
        args.out_csv = (
            f"results_A_kv/A_kv_{args.core_type}"
            f"_S{args.S}"
            f"_eta{int(args.eta*1000):04d}"
            f"_lam{int(args.lambda_decay*1000):04d}"
            f"_seed{args.seed}.csv"
        )
        print(f"[INFO] Auto out_csv → {args.out_csv}")

    # --------------------------------------------------
    # シード & device
    # --------------------------------------------------
    set_seed(args.seed)
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"[Info] device = {device}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # --------------------------------------------------
    # Wait ベクトル（Δ_wait 用）
    #   → 学習時と同じように rng=999 で固定生成
    # --------------------------------------------------
    rng_wait = np.random.RandomState(999)
    wait_vec = rng_wait.randn(args.d_g).astype(np.float32)
    wait_vec /= np.linalg.norm(wait_vec) + 1e-8
    print(f"[Info] wait_vec generated (seed=999), norm={np.linalg.norm(wait_vec):.6f}")

    # --------------------------------------------------
    # load model & checkpoint
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
    # μ / key_proto を checkpoint から復元
    #   （学習時とまったく同じものを使用）
    # --------------------------------------------------
    if ("mu" not in state) or ("key_proto" not in state):
        raise ValueError("Checkpoint does not contain 'mu' or 'key_proto'.")

    mu_value = state["mu"]
    key_proto = state["key_proto"]

    # shape から num_classes / num_keys を決める
    num_classes = mu_value.shape[0]
    num_keys = key_proto.shape[0]

    # args.num_classes と違う場合は警告だけ出す
    if args.num_classes != num_classes:
        print(
            f"[WARN] args.num_classes={args.num_classes} "
            f"but checkpoint mu.shape[0]={num_classes}. "
            f"Using {num_classes}."
        )
        args.num_classes = num_classes

    print(f"[Info] Loaded mu_value from checkpoint: {mu_value.shape}")
    print(f"[Info] Loaded key_proto from checkpoint: {key_proto.shape}")
    print(f"[Info] num_classes={num_classes}, num_keys={num_keys}")

    # --------------------------------------------------
    # KV sequence（Direction Task）
    # --------------------------------------------------
    z_seq, event_list, clean_target = make_keyvalue_sequence_direction(
        d_g=args.d_g,
        batch_size=args.batch_size,
        T_bind=args.T_bind,
        duplicate=args.duplicate,
        beta=args.beta,
        device=device,
        seed=args.seed + 1400,
        mu_value=mu_value,
        key_proto=key_proto,
        bind_noise_std=args.bind_noise_std,
        query_noise_std=0.0,
        num_wait=args.num_wait,
        wait_vec=wait_vec,
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
    # Save A-matrix CSV（FWが有効なときだけ）
    # --------------------------------------------------
    A_csv = args.out_csv.replace("A_kv_", "Amat_kv_")

    if not args.use_fw:
        print("[INFO] use_fw=False → A-matrix CSV を保存しません")
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
    # h-full CSV 保存（★RNNでも落ちないように index ガード）
    # --------------------------------------------------
    h_csv = args.out_csv.replace("A_kv_", "H_kv_")

    with open(h_csv, "w", newline="") as f:
        w = csv.writer(f)

        header = ["step", "kind", "class_id"] + [f"h[{i}]" for i in range(args.d_h)]
        w.writerow(header)

        T = min(len(model.log_h_full), len(model.log_kind), len(model.log_class))

        for t in range(T):
            h_t = model.log_h_full[t]   # (B, d_h)
            h_mean = h_t.mean(dim=0).tolist()

            kind = model.log_kind[t]
            cid  = int(model.log_class[t])

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

            header = ["t", "s", "vec_type", "kind", "class_id"]
            header += [f"h[{i}]" for i in range(args.d_h)]
            w.writerow(header)

            T = len(model.log_base)

            for t in range(T):

                kind = model.log_kind[t] if t < len(model.log_kind) else "-"
                cid  = int(model.log_class[t]) if t < len(model.log_class) else -1

                # (1) base（s = -1）
                h_base = model.log_base[t]
                h_base_mean = h_base.mean(dim=0).tolist()

                w.writerow([t, -1, "base", kind, cid] + h_base_mean)

                # (2) S-loop の h_s
                h_s_list = model.log_h_sloop[t] if t < len(model.log_h_sloop) else []

                for s, h_s in enumerate(h_s_list):
                    h_s_mean = h_s.mean(dim=0).tolist()
                    w.writerow([t, s, "hs", kind, cid] + h_s_mean)

        print(f"[DONE] Saved S-loop vector CSV → {sloop_vec_csv}")

    else:
        print("[INFO] S-loop vector CSV は log_base / log_h_sloop が無いためスキップしました。")


if __name__ == "__main__":
    main()

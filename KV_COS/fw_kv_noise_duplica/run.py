# -*- coding: utf-8 -*-
"""
run_recon.py
--------------------------------
方向復元タスク（cosine reconstruction）専用 Runner
 - core_type: fw / rnn / tanh
 - Bind / Query にノイズ導入
 - Δ_wait 対応
 - 出力は clean 方向復元精度（cos-based）
"""

import argparse
import torch
import torch.optim as optim
import numpy as np

from fw_kv.utils import set_seed, write_csv_row
from fw_kv.data.kv import KVConfig, KVDataset        # ★ 修正版（clean_vec返す版）
from fw_kv.train import TrainCfg, run_epoch_seq  # ★ reconstruction版（後で作る）

# ★ CoreCfg（FW/RNNの設定）
from fw_kv.models.core_config import CoreCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--val_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--d_g", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=128)

    ap.add_argument("--T_bind", type=int, default=60)
    ap.add_argument("--duplicate", type=int, default=3)

    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--lambda_decay", type=float, default=0.97)
    ap.add_argument("--eta", type=float, default=0.3)
    # use_ln
    ap.add_argument("--use_ln", dest="use_ln", action="store_true")
    ap.add_argument("--no-use_ln", dest="use_ln", action="store_false")
    ap.set_defaults(use_ln=True)

    # use_fw
    ap.add_argument("--use_fw", dest="use_fw", action="store_true")
    ap.add_argument("--no-use_fw", dest="use_fw", action="store_false")
    ap.set_defaults(use_fw=True)


    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--out_csv", type=str, default="results/kv_recon.csv")

    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--delta_wait", type=int, default=0)
    ap.add_argument("--bind_noise_std", type=float, default=0.0)
    ap.add_argument("--query_noise_std", type=float, default=0.0)

    ap.add_argument("--num_classes", type=int, default=10,
                help="使用するクラス数。key数は class×3 に自動設定する。")

    ap.add_argument("--core_type", type=str, default="rnn",
                    choices=["fw", "rnn", "tanh"])

    args = ap.parse_args()

    # ======================================================
    # デバイス
    # ======================================================
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    # ======================================================
    # シード固定
    # ======================================================
    set_seed(args.seed)

    # ======================================================
    # Wait ベクトル（Δ_wait 用）
    # ======================================================
    rng_wait = np.random.RandomState(999)
    wait_vec = rng_wait.randn(args.d_g).astype(np.float32)
    wait_vec /= np.linalg.norm(wait_vec) + 1e-8

    # ======================================================
    # クラス / キーの固定設定
    # ======================================================
    num_classes = args.num_classes
    num_keys = args.num_classes
    # num_keys = int(2.6 * args.num_classes)

    print(f"[Info] num_classes={num_classes}, num_keys={num_keys}")

    # --------------------------------------------------
    # μ（value-class prototypes） → class_id に依存
    # --------------------------------------------------
    rng_mu = np.random.RandomState(123)
    mu_value = rng_mu.randn(num_classes, args.d_g).astype(np.float32)
    mu_value /= (np.linalg.norm(mu_value, axis=1, keepdims=True) + 1e-8)
    print(f"[Info] mu_value shape = {mu_value.shape}")

    # --------------------------------------------------
    # key prototypes → class と独立
    # --------------------------------------------------
    rng_k = np.random.RandomState(777)
    key_proto = rng_k.randn(num_keys, args.d_g).astype(np.float32)
    key_proto /= (np.linalg.norm(key_proto, axis=1, keepdims=True) + 1e-8)
    print(f"[Info] key_proto shape = {key_proto.shape}")

    # ======================================================
    # Dataset 構築（方向復元タスク専用版）
    # ======================================================
    kv_cfg_tr = KVConfig(
        d_g=args.d_g,
        T_bind=args.T_bind,
        duplicate=args.duplicate,
        device=device,
        seed=123,          # エピソード生成の RNG
        beta=args.beta,
        bind_noise_std=args.bind_noise_std
    )

    kv_cfg_va = KVConfig(
        d_g=args.d_g,
        T_bind=args.T_bind,
        duplicate=args.duplicate,
        device=device,
        seed=456,
        beta=args.beta,
        bind_noise_std=args.bind_noise_std
    )

    # ★ mu_value / key_proto を外部から渡す版（方向復元タスク）
    train_data = KVDataset(
        cfg=kv_cfg_tr,
        mu_value=mu_value,
        key_proto=key_proto,
        wait_vec=wait_vec,
        delta_wait=args.delta_wait
    )

    valid_data = KVDataset(
        cfg=kv_cfg_va,
        mu_value=mu_value,
        key_proto=key_proto,
        wait_vec=wait_vec,
        delta_wait=args.delta_wait
    )

    # ======================================================
    # Core モデル（FW/RNN/Tanh）
    # ======================================================
    core_cfg = CoreCfg(
        glimpse_dim=args.d_g, hidden_dim=args.d_h,
        inner_steps=args.S,
        lambda_decay=args.lambda_decay,
        eta=args.eta,
        use_layernorm=args.use_ln,
        use_A=args.use_fw
    )

    if args.core_type == "fw":
        from fw_kv.models.core_fw import CoreRNNFW as CoreFW
        model = CoreFW(core_cfg)
    elif args.core_type == "rnn":
        from fw_kv.models.core_rnn import CoreRNNFW as CoreRNN
        model = CoreRNN(core_cfg)
    else:
        from fw_kv.models.core_tanh import CoreRNNFW as CoreTanh
        model = CoreTanh(core_cfg)

    model = model.to(device)

    # ======================================================
    # head: d_h → d_g（方向復元）
    # ======================================================
    head = torch.nn.Linear(args.d_h, args.d_g).to(device)

    # ======================================================
    # Optimizer
    # ======================================================
    opt = optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # ======================================================
    # Train Config
    # ======================================================
    train_cfg = TrainCfg(
        epochs=args.epochs, train_steps=args.train_steps,
        val_steps=args.val_steps, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, device=device,
        stats_stride_train=1, stats_stride_eval=1
    )

    # ======================================================
    # CSV名
    # ======================================================
    tag = f"_{args.core_type}_recon_beta{args.beta:.2f}_wait{args.delta_wait}_dup{args.duplicate}"
    args.out_csv = args.out_csv.replace(".csv", f"{tag}.csv")

    # ======================================================
    # 学習ループ
    # ======================================================
    for ep in range(1, args.epochs + 1):
        tr = run_epoch_seq(model, head, train_data, opt, train_cfg, S=args.S)
        va = run_epoch_seq(model, head, valid_data, None, train_cfg, S=args.S)

        row = {
            "epoch": ep, "seed": args.seed,
            "core_type": args.core_type,
            "use_fw": int(args.use_fw), "use_ln": int(args.use_ln),
            "S": args.S, "lambda": args.lambda_decay,
            "eta": args.eta, "T_bind": args.T_bind,
            "duplicate": args.duplicate, "delta_wait": args.delta_wait,
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "valid_loss": va["loss"], "valid_acc": va["acc"],
            "specA": tr.get("specA", 0.0),
            "specA_val": va.get("specA", 0.0),
            "alpha_train": tr.get("alpha_dyn", 0.0),
            "alpha_val": va.get("alpha_dyn", 0.0),
        }

        write_csv_row(args.out_csv, row, header=(ep == 1))

        # ======== alpha の表示用 ========
        alpha_tr = tr.get("alpha_dyn", 0.0)
        alpha_va = va.get("alpha_dyn", 0.0)

        print(
            f"[Epoch {ep:03d}] "
            f"loss={tr['loss']:.3f} acc={tr['acc']:.3f} "
            f"| valid: loss={va['loss']:.3f} acc={va['acc']:.3f} "
            f"| specA={tr['specA']:.3f}/{va['specA']:.3f} "
            f"| alpha={alpha_tr:.3f}/{alpha_va:.3f} "
            f"(S={args.S}, Δ_wait={args.delta_wait}, dup={args.duplicate}, core={args.core_type})"
        )

    # === 保存ディレクトリ作成（タイムスタンプ版） ===
    import os
    from datetime import datetime

    # 親ディレクトリ
    BASE_DIR = "checkpoints"
    os.makedirs(BASE_DIR, exist_ok=True)

    # タイムスタンプ生成（例：20251123_121530）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # タイムスタンプ付きの新しい保存先ディレクトリ
    SAVE_DIR = os.path.join(BASE_DIR, timestamp)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # lambda を小数3桁にしてファイル名に含める
    lam_str = f"{args.lambda_decay:.3f}".replace(".", "")

    # use_fw（0/1）
    fw_flag = int(args.use_fw)

    # eta を小数3桁で文字列化（例：0.30 → "030"）
    eta_str = f"{args.eta:.3f}".replace(".", "")

    # 保存ファイル名
    ckpt_path = os.path.join(
        SAVE_DIR,
        f"kv_{args.core_type}_S{args.S}_noise{args.bind_noise_std:.3f}_eta{eta_str}_lam{lam_str}_seed{args.seed}.pt"
    )


    # === モデル＋ヘッドをまとめて保存 ===
    torch.save(
        {
            "core_type": args.core_type,
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "mu": mu_value,
            "key_proto": key_proto,
        },
        ckpt_path
    )

    print(f"[SAVE] Trained weights saved → {ckpt_path}")

if __name__ == "__main__":
    main()
    
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from fw_kv.utils import set_seed, write_csv_row
from fw_kv.data.kv import KVConfig, KVDataset
from fw_kv.models.head import OutputHead
from fw_kv.train import TrainCfg, run_epoch_seq

# ★ コアタイプを後で選択的にインポートする
from fw_kv.models.core_config import CoreCfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_steps", type=int, default=100)
    ap.add_argument("--val_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--d_g", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=128)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--T_bind", type=int, default=5)
    ap.add_argument("--noise_std", type=float, default=0.0)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--lambda_decay", type=float, default=0.97)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--use_ln", action="store_true")
    ap.add_argument("--no_ln", dest="use_ln", action="store_false")
    ap.add_argument("--use_fw", action="store_true")
    ap.add_argument("--no_fw", dest="use_fw", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results/kv_fw_vs_rnn.csv")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--delta_wait", type=int, default=6, help="待機区間 Δ_wait")

    ap.add_argument("--core_type", type=str, default="rnn",
                    choices=["fw", "rnn", "tanh"],
                    help="fw=core_fw.py / rnn=core_rnn.py / tanh=core_tanh.py")


    ap.set_defaults(use_ln=True, use_fw=True)
    args = ap.parse_args()

    # ======================================================
    # ✅ ここに追加（AMP高速化＋精度維持）
    # ======================================================
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        print("[Info] Using float32 matmul precision = 'medium' (TF32 enabled)")

    # ======================================================
    # 乱数シードの固定
    # ======================================================
    set_seed(args.seed)
    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    # ======================================================
    # μ（クラスプロトタイプ）の生成
    # ======================================================
    if not hasattr(main, "base_mu"):
        rng = np.random.RandomState(123)
        main.base_mu = np.stack(
            [rng.randn(args.d_g).astype(np.float32) for _ in range(args.num_classes)],
            axis=0
        )
        main.base_mu /= (np.linalg.norm(main.base_mu, axis=1, keepdims=True) + 1e-8)
    base_mu = main.base_mu

    # ======================================================
    # key prototypes の生成（★追加）
    # ======================================================
    if not hasattr(main, "base_key_proto"):
        rng_k = np.random.RandomState(777)
        num_key_proto = 3 * args.num_classes
        main.base_key_proto = np.stack(
            [rng_k.randn(args.d_g).astype(np.float32) 
            for _ in range(num_key_proto)],
            axis=0
        )
        main.base_key_proto /= (
            np.linalg.norm(main.base_key_proto, axis=1, keepdims=True) + 1e-8
        )

    base_key_proto = main.base_key_proto


    # ======================================================
    # ★ 待機ベクトル（固定）の生成
    # ======================================================
    rng_wait = np.random.RandomState(999)  # 固定seed
    wait_vec = rng_wait.randn(args.d_g).astype(np.float32)
    wait_vec /= np.linalg.norm(wait_vec) + 1e-8  # unit norm
    print(f"Using fixed wait vector (Δ_wait={args.delta_wait}) | norm={np.linalg.norm(wait_vec):.3f}")

    # ======================================================
    # Dataset構築
    # ======================================================
    kv_cfg_tr = KVConfig(
        d_g=args.d_g, num_classes=args.num_classes, T_bind=args.T_bind,
        noise_std=args.noise_std, device=device, seed=123,
        mu=base_mu, beta=args.beta
    )
    kv_cfg_va = KVConfig(
        d_g=args.d_g, num_classes=args.num_classes, T_bind=args.T_bind,
        noise_std=args.noise_std, device=device, seed=456,
        mu=base_mu, beta=args.beta
    )

    # ★ KVDatasetに wait_vec と Δ_wait を渡す
    train_data = KVDataset(kv_cfg_tr, key_proto=base_key_proto, wait_vec=wait_vec, delta_wait=args.delta_wait)
    valid_data = KVDataset(kv_cfg_va, key_proto=base_key_proto, wait_vec=wait_vec, delta_wait=args.delta_wait)

    # ======================================================
    # μの同一性チェック
    # ======================================================
    cos_mean = np.mean(np.sum(train_data.mu * valid_data.mu, axis=1))
    print(f"mean cos(train_mu, valid_mu) = {cos_mean:.6f} (expected ≈ 1.0)")

    # ======================================================
    # key_proto の同一性チェック
    # ======================================================
    cos_key = np.mean(
        np.sum(train_data.key_proto * valid_data.key_proto, axis=1)
    )
    print(f"mean cos(train_key_proto, valid_key_proto) = {cos_key:.6f} (expected ≈ 1.0)")

    # ======================================================
    # Core モデル構築（fw / rnn）
    # ======================================================
    core_cfg = CoreCfg(
        glimpse_dim=args.d_g, hidden_dim=args.d_h,
        inner_steps=args.S, lambda_decay=args.lambda_decay,
        eta=args.eta, use_layernorm=args.use_ln, use_A=args.use_fw
    )

    if args.core_type == "fw":
        from fw_kv.models.core_fw import CoreRNNFW as CoreFW
        model = CoreFW(core_cfg)
        print("[Model] Using core_fw.py (step-wise Fast Weights)")

    elif args.core_type == "rnn":
        from fw_kv.models.core_rnn import CoreRNNFW as CoreRNN
        model = CoreRNN(core_cfg)
        print("[Model] Using core_rnn_fw.py (Bind/Query Fast Weights)")

    elif args.core_type == "tanh":
        from fw_kv.models.core_tanh import CoreRNNFW as CoreTanh
        model = CoreTanh(core_cfg)
        print("[Model] Using core_tanh.py (Residual + tanh Fast Weights)")

    else:
        raise ValueError(f"Unknown core_type: {args.core_type}")


    model = model.to(device)
    head = OutputHead(d_h=args.d_h, num_classes=args.num_classes).to(device)

    torch.backends.cudnn.benchmark = True      # ① CuDNN最適化ON

    opt = optim.AdamW(list(model.parameters()) + list(head.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    train_cfg = TrainCfg(
        epochs=args.epochs, train_steps=args.train_steps, val_steps=args.val_steps,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, device=device,
        # ★ 学習時は統計オフ、評価時は毎回
        stats_stride_train=0,   # 完全に統計を切る（最速）
        stats_stride_eval=1     # 評価は毎ステップ統計を取る
    )


    # ======================================================
    # CSVファイル設定
    # ======================================================
    tag = f"_beta{args.beta:.2f}_wait{args.delta_wait}"
    if args.out_csv.endswith(".csv"):
        args.out_csv = args.out_csv.replace(".csv", f"{tag}.csv")
    else:
        args.out_csv = args.out_csv + tag + ".csv"

    # ======================================================
    # 学習ループ
    # ======================================================
    for ep in range(1, args.epochs + 1):
        # ====== 学習 ======
        tr = run_epoch_seq(model, head, train_data, opt=opt, cfg=train_cfg, S=args.S)

        # ====== 評価 ======
        va = run_epoch_seq(model, head, valid_data, opt=None, cfg=train_cfg, S=args.S)

        row = {
            "epoch": ep,
            "seed": args.seed,
            "use_fw": int(args.use_fw),
            "use_ln": int(args.use_ln),
            "S": args.S,
            "lambda": args.lambda_decay,
            "eta": args.eta,
            "T_bind": args.T_bind,
            "delta_wait": args.delta_wait,
            "d_g": args.d_g,
            "d_h": args.d_h,
            "beta": args.beta,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "valid_loss": va["loss"],
            "valid_acc": va["acc"],
            "valid_acc_se": va.get("acc_se", 0.0),
            "valid_acc_lo": va.get("acc_lo", 0.0),
            "valid_acc_hi": va.get("acc_hi", 0.0),
            "specA": tr["specA"],
            "froA": tr["froA"],
            "addAh_norm": tr["addAh_norm"],
            "logit_margin": tr["logit_margin"],
            "alpha_dyn_train": tr.get("alpha_dyn", 0.0),
            "alpha_dyn_valid": va.get("alpha_dyn", 0.0),
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
            f"(S={args.S}, Δ_wait={args.delta_wait}, core={args.core_type})"
        )
        print(f"Saved CSV -> {args.out_csv}")

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
        f"kv_{args.core_type}_S{args.S}_fw{fw_flag}_eta{eta_str}_lam{lam_str}_seed{args.seed}.pt"
    )

    # === モデル＋ヘッドをまとめて保存 ===
    torch.save(
        {
            "core_type": args.core_type,
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "mu": base_mu,
            "key_proto": base_key_proto,
        },
        ckpt_path
    )

    print(f"[SAVE] Trained weights saved → {ckpt_path}")


if __name__ == "__main__":
    main()
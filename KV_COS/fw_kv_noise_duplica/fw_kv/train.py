from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import amp  # ✅ 新しいAMP API だけを使う

# from torch.cuda import amp  # ❌ 古いAPIなので削除

from .utils import spectral_norm_batch  # 使わなければ残っていてもOK


# ======================================================
# 設定クラス
# ======================================================
@dataclass
class TrainCfg:
    epochs: int = 10
    train_steps: int = 100
    val_steps: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"

    # 統計計算の間引き設定
    stats_stride_train: int = 1   # 学習時: 0=計算しない / 5=5ステップに1回など
    stats_stride_eval: int = 1    # 評価時: 1=毎ステップ / 5=5ステップに1回


# ======================================================
# エポック単位の学習・評価ループ（方向復元タスク版）
# ======================================================
def run_epoch_seq(model, head, data, opt=None, cfg=None, S=1):
    """
    1エポック分の学習または評価を実行する（方向復元タスク用）。

    Args:
        model, head: モデル本体と出力層（d_h -> d_g）
        data: データセット (KVDataset) 
              sample_batch(B) -> (z_seq, clean_vec)
        opt: Optimizer (Noneなら評価モード)
        cfg: TrainCfg 設定
        S:  内部 Fast Weights の自己整合ステップ数
    """
    train = opt is not None
    model.train(train)
    head.train(train)

    B = cfg.batch_size
    steps = cfg.train_steps if train else cfg.val_steps

    tot_loss = 0.0
    acc_list = []
    specA_vals, froA_vals, addAh_vals, margin_vals, alpha_vals = [], [], [], [], []

    # ======================================================
    # AMP スケーラの初期化（GPUのみ有効）
    # ======================================================
    scaler = amp.GradScaler("cuda", enabled=cfg.device.startswith("cuda"))

    # ======================================================
    # 学習 or 評価ループ
    # ======================================================
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for step in range(steps):
            # ---- サンプルを取得 ----
            # KVDataset: (z_seq, clean_vec) を返す版
            z_seq, clean_vec = data.sample_batch(B)
            clean_vec = clean_vec.to(cfg.device)

            # ---- 統計の計算間引き設定 ----
            if train:
                compute_stats = (cfg.stats_stride_train > 0 and (step % cfg.stats_stride_train == 0))
            else:
                compute_stats = (cfg.stats_stride_eval > 0 and (step % cfg.stats_stride_eval == 0))

            # ======================================================
            # Forward パス（AMP有効化）
            # ======================================================
            with amp.autocast("cuda", enabled=cfg.device.startswith("cuda")):
                loss, acc, stats = model.forward_episode(
                    z_seq, clean_vec, head=head, S=S, compute_stats=compute_stats
                )

            # ======================================================
            # Backward パス（AMP対応）
            # ======================================================
            if train:
                opt.zero_grad(set_to_none=True)

                # AMP対応の backward
                scaler.scale(loss).backward()

                # 勾配クリッピング（unscale後）
                if cfg.grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()), cfg.grad_clip
                    )

                # Optimizer 更新
                scaler.step(opt)
                scaler.update()

            # ======================================================
            # 統計集計
            # ======================================================
            tot_loss += float(loss.detach().cpu())
            acc_list.append(float(acc))

            if "specA" in stats:      specA_vals.append(stats["specA"])
            if "froA" in stats:       froA_vals.append(stats["froA"])
            if "addAh_norm" in stats: addAh_vals.append(stats["addAh_norm"])
            if "margin" in stats:     margin_vals.append(stats["margin"])
            if "alpha_dyn" in stats:  alpha_vals.append(stats["alpha_dyn"])

    # ======================================================
    # エポック終了後の平均統計計算
    # ======================================================
    n = max(steps, 1)
    acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
    acc_se = float(np.std(acc_list, ddof=1) / max(len(acc_list)**0.5, 1)) if len(acc_list) > 1 else 0.0
    acc_lo, acc_hi = acc_mean - 1.96 * acc_se, acc_mean + 1.96 * acc_se

    return {
        "loss": tot_loss / n,
        "acc": acc_mean,
        "specA": float(np.mean(specA_vals)) if specA_vals else 0.0,
        "froA": float(np.mean(froA_vals)) if froA_vals else 0.0,
        "addAh_norm": float(np.mean(addAh_vals)) if addAh_vals else 0.0,
        "logit_margin": float(np.mean(margin_vals)) if margin_vals else 0.0,
        "acc_se": acc_se,
        "acc_lo": acc_lo,
        "acc_hi": acc_hi,
        "alpha_dyn": float(np.mean(alpha_vals)) if alpha_vals else 0.0,
    }

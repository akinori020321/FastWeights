# -*- coding: utf-8 -*-
# fw_kv/models/core_fw.py
"""
Fast Weights Core (Step + Episode compatible version)
----------------------------------------------------
1ステップforward + 連続エピソード処理対応。
train.pyのrun_epoch_seqと完全互換。
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from fw_kv.models.core_config import CoreCfg


@torch.no_grad()
def batch_spectral_radius(A, iters=20):
    """
    A: (B, d, d)
    return: 平均スペクトル半径 (float)
    """
    B, d, _ = A.shape

    # 初期ベクトル（乱数）
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    for _ in range(iters):
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)   # (B, d)
        v = Av / (Av.norm(dim=1, keepdim=True) + 1e-8)

    # Rayleigh quotient → 最大固有値近似
    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)       # (B, d)
    lambda_max = (Av * v).sum(dim=1)                     # (B,)
    return lambda_max.mean().item()


# ==============================================================
# CoreRNNFW 本体
# ==============================================================
class CoreRNNFW(nn.Module):
    """
    RNN core with optional Fast Weights (Hebbian) and LayerNorm.

    - Input per time step: z_list (list of (B, d_g)) or (B, d_g)
    - Hidden update: h_t = ReLU(W_h h_prev + W_g g_t + A h_t(inner) + b)
    - Inner loop: update A within the loop S times
    - Hebbian update: A <- λA + η (h hᵀ) / (‖h‖² + ε)
    - Compatible with run_epoch_seq via forward_episode()
    """
    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg
        d_g, d_h = cfg.glimpse_dim, cfg.hidden_dim
        self.d_g, self.d_h = d_g, d_h
        self.S = cfg.inner_steps
        self.lambda_ = cfg.lambda_decay
        self.eta = cfg.eta
        self.eps = cfg.epsilon
        self.use_A = cfg.use_A

        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_g = nn.Linear(d_g, d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(d_h))

        self.ln_h = nn.LayerNorm(d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # ==============================================================
    # 1ステップforward
    # ==============================================================
    def forward(self, z_list, h_prev, A_prev):
        if isinstance(z_list, (list, tuple)):
            g_t = torch.stack(z_list, dim=0).mean(dim=0)  # (B, d_g)
        else:
            g_t = z_list  # (B, d_g)

        base = self.W_h(h_prev) + self.W_g(g_t) + self.b_h
        h_t = torch.relu(self.ln_h(base))
        A_k = A_prev.clone()

        if self.use_A:
            for _ in range(self.S):
                add_Ah = torch.bmm(A_k, h_t.unsqueeze(-1)).squeeze(-1)  # (B, d_h)
                h_pre = base + add_Ah
                h_t = torch.relu(self.ln_h(h_pre))
            
            # Hebbian update
            delta_A = torch.bmm(h_t.unsqueeze(2), h_t.unsqueeze(1))
            A_k = self.lambda_ * A_k + self.eta * delta_A

            # h_norm2 = (h_t**2).sum(dim=1, keepdim=True) + self.eps
            # delta_A = torch.bmm(h_t.unsqueeze(2), h_t.unsqueeze(1)) / h_norm2.unsqueeze(-1)
            # A = self.lambda_ * A + self.eta * delta_A
                
        else:
            # base にだけ LN → ReLU
            pass


        return h_t, A_k

    # ==============================================================
    # エピソード全体（Bind〜Query）処理
    # ==============================================================
    def forward_episode(self, z_seq, y, head, S=None, compute_stats=False):
        """
        学習・評価時に1エピソード全体を処理する。
        z_seq : list[(B, d_g)] or tensor (T, B, d_g)
        y     : (B,)
        head  : 出力層
        """
        if isinstance(z_seq, list):
            T_total = len(z_seq)
        else:
            T_total = z_seq.shape[0]

        B = y.size(0)
        device = y.device
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # --- シーケンス展開 ---
        for t in range(T_total):
            z_t = z_seq[t]
            h, A = self.forward(z_t, h, A)

        # --- 出力層 ---
        logits = head(h)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()

        # ==============================================================
        # 統計 (specA)
        # ==============================================================
        stats = {}
        if self.use_A:
            with torch.no_grad():
                specA = batch_spectral_radius(A)
                stats["specA"] = float(specA)

                # ★ alpha_dyn（安全版）
                if "alpha_log_this_step" in locals() and len(alpha_log_this_step) > 0:
                    stats["alpha_dyn"] = float(np.mean(alpha_log_this_step))
                else:
                    stats["alpha_dyn"] = 0.0    # or None

        return loss, acc, stats

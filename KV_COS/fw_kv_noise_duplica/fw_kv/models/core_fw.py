# -*- coding: utf-8 -*-
# fw_kv/models/core_fw.py
"""
Fast Weights Core (Step-wise Fast Weights)
------------------------------------------
方向復元タスク用に書き換え済み版

- Stepごとに: h_t = ReLU(W_h h + W_g z_t + A h + b)
- Hebbian update: A ← λA + η h h^T
- 出力: pred_vec = head(h_T)
- loss = 1 - cosine(pred_vec, clean_vec)
"""

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
# CoreRNNFW（方向復元タスク対応版）
# ==============================================================

class CoreRNNFW(nn.Module):

    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg

        d_g, d_h = cfg.glimpse_dim, cfg.hidden_dim
        self.d_g, self.d_h = d_g, d_h

        self.lambda_ = cfg.lambda_decay
        self.eta      = cfg.eta
        self.eps      = cfg.epsilon
        self.use_A    = cfg.use_A
        self.S        = cfg.inner_steps  # inner loop

        # ----------------------------
        # RNN parameters
        # ----------------------------
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_g = nn.Linear(d_g, d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(d_h))

        self.ln_h = nn.LayerNorm(d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # ==============================================================
    # 1 step forward
    # ==============================================================
    def forward_step(self, z_t, h_prev, A_prev):

        base = self.W_h(h_prev) + self.W_g(z_t) + self.b_h
        h_t = torch.relu(self.ln_h(base))

        A_k = A_prev.clone()

        if self.use_A:
            # S-loop
            for _ in range(self.S):
                add_Ah = torch.bmm(A_k, h_t.unsqueeze(-1)).squeeze(-1)
                h_pre  = base + add_Ah
                h_t    = torch.relu(self.ln_h(h_pre))

            # Hebbian update
            delta_A = torch.bmm(h_t.unsqueeze(2), h_t.unsqueeze(1))
            A_k = self.lambda_ * A_k + self.eta * delta_A
        
        else:
            h_t = torch.relu(self.ln_h(base))

        return h_t, A_k

    # ==============================================================
    # エピソード全体 (Bind〜Query)
    # ==============================================================
    def forward_episode(self, z_seq, clean_vec, head, S=None, compute_stats=False):
        """
        z_seq     : list[(B, d_g)] or (T, B, d_g)
        clean_vec : (B, d_g)  ← 正解方向ベクトル
        head      : d_h → d_g の線形層
        """

        if isinstance(z_seq, list):
            T_total = len(z_seq)
        else:
            T_total = z_seq.size(0)

        B = clean_vec.size(0)
        device = clean_vec.device

        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # ----------------------------
        # 展開
        # ----------------------------
        for t in range(T_total):
            z_t = z_seq[t].to(device)
            h, A = self.forward_step(z_t, h, A)
        
        # ----------------------------
        # 出力ベクトル
        # ----------------------------
        pred_vec = head(h)  # (B, d_g)

        # 出力側の L2 正規化（cosine 用）
        pred_norm = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + 1e-6)
        clean_norm = clean_vec / (clean_vec.norm(dim=1, keepdim=True) + 1e-6)

        # ----------------------------
        # 相対 MSE 損失（大きさも含めた復元誤差）
        #   r_b = ||P - C||^2 / (||C||^2 + eps)
        #   loss = mean_b r_b
        # ----------------------------
        diff = pred_vec - clean_vec                     # (B, d_g)
        sq_err = (diff ** 2).sum(dim=1)                 # (B,)
        target_norm_sq = (clean_vec ** 2).sum(dim=1)    # (B,)
        denom = target_norm_sq + self.eps               # (B,)
        rel_err = sq_err / denom                        # (B,)
        loss = rel_err.mean()

        # ----------------------------
        # Accuracy（cosine）
        # ----------------------------
        acc = (pred_norm * clean_norm).sum(dim=1).mean().item()

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

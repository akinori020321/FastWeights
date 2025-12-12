# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_fw_ln_residual.py
#
# Fast Weights（Residual Correction 版）
# =========================================
# 方向復元タスク（cosine reconstruction）のために修正した版：
#  - 出力: pred_vec = head(h_final) (d_h → d_g)
#  - loss = 1 - cos(pred_vec, clean_vec)
#  - acc = (cos > threshold) など


from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fw_kv.models.core_config import CoreCfg

@torch.no_grad()
def batch_spectral_radius(A, iters=20):
    B, d, _ = A.shape
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    for _ in range(iters):
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        v = Av / (Av.norm(dim=1, keepdim=True) + 1e-8)

    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    lambda_max = (Av * v).sum(dim=1)
    return lambda_max.mean().item()


# ==============================================================
# CoreRNNFW (Residual Correction)
# ==============================================================

class CoreRNNFW(nn.Module):

    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg
        self.d_g, self.d_h = cfg.glimpse_dim, cfg.hidden_dim
        self.lambda_ = cfg.lambda_decay
        self.eta = cfg.eta
        self.eps = cfg.epsilon
        self.use_A = cfg.use_A
        self.S = cfg.inner_steps

        # α_fw が凸⇔凹を司る
        self.alpha_fw = nn.Parameter(torch.tensor(cfg.alpha_fw, dtype=torch.float32))
        # Ah のスケーリング係数 β（学習可能パラメータ）
        # self.beta = nn.Parameter(torch.tensor(cfg.beta, dtype=torch.float32))

        # RNN
        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))
        self.ln_h = nn.LayerNorm(self.d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # ==============================================================
    # α_fw の符号で凸⇄凹を切り替える k(α)
    # ==============================================================
    def compute_k(self):
        alpha = self.alpha_fw

        # α>0 → k>1（上に凸）
        k_pos = 1.0 + F.softplus(alpha)

        # α<0 → 0<k<1（下に凸）
        k_neg = 1.0 / (1.0 + F.softplus(-alpha))

        # 要素ごとに切り替え（scalarなので where でOK）
        k = torch.where(alpha >= 0, k_pos, k_neg)
        return k


    # ==============================================================
    # 前向き（1エピソード）
    # ==============================================================

    def forward_episode(self, z_seq, clean_vec, head,
                        S=None, train=True, compute_stats=False):

        # z_seq: List[(B, d_g)]
        # clean_vec: (B, d_g)

        if isinstance(z_seq, list):
            T_total = len(z_seq)
        else:
            T_total = z_seq.size(0)

        B = clean_vec.size(0)
        device = clean_vec.device
        S_loop = S if S is not None else self.S

        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        alpha_log_this_step = []
        k = self.compute_k()

        # ==============================================================
        # 時系列展開
        # ==============================================================
        for t in range(T_total):
            z_t = z_seq[t].to(device)

            # --- RNN 基本更新 ---
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(self.ln_h(h_base))

            # ----------------------------------------------------------
            # ★ Query (t == T_total - 1) → Ba-style
            # ----------------------------------------------------------
            if t == T_total - 1:
                if self.use_A and S_loop > 0:
                    for _ in range(S_loop):
                        Ah = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)
                        # Ba-style: h_base + β·Ah を LN → ReLU
                        # h = torch.relu(self.ln_h(h_base + self.beta * Ah))
                        h = torch.relu(self.ln_h(h_base + Ah))
                # Query では Hebbian 更新しない
                continue

            # ----------------------------------------------------------
            # Self-consistent S-loop（Bind / Wait 用）
            # ----------------------------------------------------------
            if self.use_A and S_loop > 0:
                for _ in range(S_loop):
                    Ah = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)

                    # 整合度 cos(h, Ah)
                    dot  = (h * Ah).sum(dim=1, keepdim=True)
                    norm1 = h.norm(dim=1, keepdim=True) + 1e-6
                    norm2 = Ah.norm(dim=1, keepdim=True) + 1e-6

                    R = dot / (norm1 * norm2 + 1e-6)   # 理論上 [-1,1]
                    R_pos = torch.clamp(R, min=0.0, max=1.0)  # 負の相関は0に潰す

                    # --- 凸⇄凹切り替え可能なゲイン ---
                    alpha_dyn = 1 - (1 - R_pos) ** k   # or: alpha_dyn = R_pos ** k
                    alpha_log_this_step.append(alpha_dyn.mean().item())

                    # Residual correction
                    h = (1 - alpha_dyn**2) * h_base + alpha_dyn * Ah
                    h = torch.relu(self.ln_h(h))

            # ----------------------------------------------------------
            # Hebbian 更新 (Bind のみを想定)
            # ----------------------------------------------------------
            if self.use_A:
                h_norm2 = (h**2).sum(dim=1, keepdim=True) + self.eps
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)) #/ h_norm2.unsqueeze(-1)
                A = self.lambda_ * A + self.eta * delta_A

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



            # # --- 補正項として Fast Weights 寄与を追加（非線形自己整合型 α + Sループ）---
            # if self.use_A and self.S > 0:
            #     for _ in range(self.S):
            #         Ah = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)     # (B, d_h)
            #         ah_norm = Ah.norm(dim=1, keepdim=True)             # ||A h||
                    
            #         # --- tanh内部でスケーリングを制御 ---
            #         # α が内部に入り、非線形的に符号反転も許す
            #         alpha_dyn = torch.tanh(-self.alpha_fw * ah_norm)   # [-1, +1] の範囲

            #         # --- 自己整合的な補正を適用 ---
            #         h = h_base + alpha_dyn * Ah                        # Residual補正
            #         h = torch.relu(self.ln_h(h))



                    # === 新しい抑制ゲイン（符号反転なし）======================
                    # g(r) = alpha / (1 + r)
                    # alpha_pos = F.softplus(self.alpha_fw)
                    # alpha_dyn = alpha_pos * torch.tanh(1 / (1.0 + ah_norm))
                    # =============================================================



                    # Ah = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)    # (B, d_h)
                    # ah_norm = Ah.norm(dim=1, keepdim=True)            # ||A h||

                    # # 初期値:-0.5
                    # alpha_pos = F.softplus(self.alpha_fw) + 1              # α_pos > 0
                    # alpha_dyn = alpha_pos / torch.sqrt(1.0 + ah_norm)  # 抑制は入るが弱め


                    # alpha_log_this_step.append(alpha_dyn.mean().item())

                    # # --- 自己整合的な補正 ---
                    # h = h_base + alpha_dyn * Ah                      # Residual correction
                    # h = torch.relu(self.ln_h(h))
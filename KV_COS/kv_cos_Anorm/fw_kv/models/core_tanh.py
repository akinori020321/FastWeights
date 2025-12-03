# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_fw_ln_residual_direction.py
#
# Residual FastWeights + unified logging
# 分類版から「方向復元タスク（pred_vec の cosine/MSE）」に書き換え済み

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def batch_spectral_radius(A, iters=20):
    B, d, _ = A.shape
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    for _ in range(iters):
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        v = Av / (v.norm(dim=1, keepdim=True) + 1e-8)

    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    return (Av * v).sum(dim=1).mean().item()


class CoreRNNFW(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_g, self.d_h = cfg.glimpse_dim, cfg.hidden_dim
        self.lambda_ = cfg.lambda_decay
        self.eta = cfg.eta    
        self.eps = cfg.epsilon
        self.use_A = cfg.use_A
        self.S = cfg.inner_steps

        self.alpha_fw = nn.Parameter(torch.tensor(cfg.alpha_fw))

        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))
        self.ln_h = nn.LayerNorm(self.d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    def compute_k(self):
        alpha = self.alpha_fw
        k_pos = 1.0 + F.softplus(alpha)
        k_neg = 1.0 / (1.0 + F.softplus(-alpha))
        return torch.where(alpha >= 0, k_pos, k_neg)

    # ==============================================================
    # forward_episode（方向復元タスクに変更）
    # ==============================================================
    def forward_episode(self, z_seq, clean_vec, head,
                        event_list=None,
                        S=None,
                        train=True,
                        compute_stats=False):

        # -------------------------
        # Shape
        # -------------------------
        if isinstance(z_seq, list):
            T_total = len(z_seq)
            B = z_seq[0].size(0)
            device = z_seq[0].device
        else:
            T_total = z_seq.size(0)
            B = z_seq.size(1)
            device = z_seq.device

        S_loop = S if S is not None else self.S

        # -------------------------
        # Init states
        # -------------------------
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # -------------------------
        # Logs（unified と一致）
        # -------------------------
        self.log_froA = []
        self.log_specA = []
        self.log_hnorm = []
        self.log_Ahnorm = []
        self.log_kind = []
        self.log_class = []
        self.log_h = []
        self.log_sloop = []
        self.log_query = {}
        self.bind_h = {}  # value h の保存

        self.log_A = []

        # Gain k（そのまま）
        k = self.compute_k()

        # -------------------------
        # Logging function
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p="fro", dim=(1,2)).mean().item()
            specA = batch_spectral_radius(A_t)
            h_norm = h_t.norm(dim=1).mean().item()
            Ah = torch.bmm(A_t, h_t.unsqueeze(-1)).squeeze(-1)
            Ah_norm = Ah.norm(dim=1).mean().item()

            self.log_froA.append(froA)
            self.log_specA.append(specA)
            self.log_hnorm.append(h_norm)
            self.log_Ahnorm.append(Ah_norm)
            self.log_kind.append(kind)
            self.log_class.append(cid)
            self.log_h.append(h_t.detach().clone())

        # ==============================================================
        # Time loop
        # ==============================================================
        for t in range(T_total):

            z_t = z_seq[t].to(device)
            kind, cid_raw = event_list[t] if event_list is not None else ("-", -1)

            # base RNN update
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(h_base)

            # -------------------------
            # Query（last step）
            # -------------------------
            if t == T_total - 1:

                sloop_t = []
                if self.use_A:
                    h_s = h.clone()
                    h0 = h_s.clone()

                    for s in range(S_loop):
                        Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                        # ★ value h との cosine（分類ではなく統計）
                        if cid_raw in self.bind_h:
                            q_bind_idx = cid_raw
                            h_value = self.bind_h[q_bind_idx]
                            cos_value = F.cosine_similarity(h_s, h_value, dim=1).mean().item()
                        else:
                            cos_value = 0.0

                        sloop_t.append({
                            "s": s,
                            "h_norm": h_s.norm(dim=1).mean().item(),
                            "Ah_norm": Ah.norm(dim=1).mean().item(),
                            "cos_h0": F.cosine_similarity(h0, h_s).mean().item(),
                            "cos_value": cos_value,
                        })

                        h_s = torch.relu(self.ln_h(h_base + Ah))

                    h = h_s

                self.log_sloop.append(sloop_t)
                self.log_A.append(A.detach().cpu().clone())
                append_log(kind, cid_raw, A, h)

                # ==================================================
                # ★ 方向復元タスク（分類 → MSE/cosine に置換）
                # ==================================================
                pred_vec = head(h)   # (B, d_g)

                # ------------------------------------
                # 正規化付き 再構成誤差（相対MSE）
                #  - per_sample_se: 各サンプルごとの二乗誤差 ||p - c||^2
                #  - norm_clean: 正解ベクトルのノルム^2
                #  - rel_err: 「正解ノルムに対する相対誤差」
                # ------------------------------------
                diff = pred_vec - clean_vec                    # (B, d_g)
                per_sample_se = (diff ** 2).sum(dim=1)         # (B,)
                norm_clean    = (clean_vec ** 2).sum(dim=1) + 1e-6

                rel_err = per_sample_se / norm_clean          # (B,)

                # 大きな誤差に対して勾配が出過ぎないよう log1p で圧縮
                #   rel_err ≈ 1 → loss ≈ log(2) ≈ 0.69
                #   rel_err → 0 → loss ≈ rel_err （近傍ではほぼ線形）
                loss = torch.log1p(rel_err).mean()

                # 「1 - 相対誤差」を Accuracy 風のスカラーとしてログに出す
                acc = (1.0 - rel_err).mean().item()

                # ★ log_query（分類情報なし）
                self.log_query = {
                    "cosine": acc,
                }

                # stats
                stats = {}
                if self.use_A:
                    stats["specA"] = float(batch_spectral_radius(A))
                    stats["alpha_dyn"] = 0.0

                return loss, acc, stats

            # --------------------------------------------------------
            # Bind（最終 step 以外）
            # --------------------------------------------------------
            sloop_t = []
            if self.use_A and S_loop > 0:
                h_s = h.clone()
                h0 = h_s.clone()

                for s in range(S_loop):
                    Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                    # α_dyn（Residual FW）
                    dot = (h_s * Ah).sum(dim=1, keepdim=True)
                    norm1 = h_s.norm(dim=1, keepdim=True) + 1e-6
                    norm2 = Ah.norm(dim=1, keepdim=True) + 1e-6

                    # cos(h_s, Ah) を計算（理論上 [-1, 1]）
                    R = dot / (norm1 * norm2 + 1e-6)

                    # 負の相関は 0 に潰して「寄与なし」とみなす
                    R_pos = torch.clamp(R, min=0.0, max=1.0)

                    # 凸/凹の形は k で調整
                    alpha_dyn = 1 - (1.0 - R_pos) ** k

                    sloop_t.append({
                        "s": s,
                        "h_norm": h_s.norm(dim=1).mean().item(),
                        "Ah_norm": Ah.norm(dim=1).mean().item(),
                        "cos_h0": F.cosine_similarity(h0, h_s).mean().item(),
                        "alpha_dyn": alpha_dyn.mean().item(),
                    })

                    h_s = (1 - alpha_dyn**2) * h_base + alpha_dyn * Ah
                    h_s = torch.relu(self.ln_h(h_s))

                h = h_s

            # Hebbian update（そのまま）
            if self.use_A:
                h_norm2 = (h**2).sum(dim=1, keepdim=True) + self.eps
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)) / h_norm2.unsqueeze(-1)
                A = self.lambda_ * A + self.eta * delta_A

            if kind == "bind":
                bind_idx = cid_raw
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_sloop.append(sloop_t)

            self.log_A.append(A.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

        return None, None, {}




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
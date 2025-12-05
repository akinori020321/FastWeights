# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_fw_ln_residual.py
#
# A-dynamics + Query 判定（正解 value の内部状態との比較）を Core 内で完結する版
# ※ FastWeights / Hebbian / α_dyn / S-loop の計算処理は一切変更していません。

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
    # forward_episode（Query 判定を内部処理）
    # ==============================================================
    def forward_episode(self, z_seq, y, head,
                        event_list=None,
                        S=None,
                        class_ids=None,
                        mu_value=None,
                        train=True,
                        compute_stats=False):

        # -------------------------
        # Sequence shape
        # -------------------------
        if isinstance(z_seq, list):
            T_total = len(z_seq)
            B = z_seq[0].size(0)
        else:
            T_total = z_seq.size(0)
            B = z_seq.size(1)

        device = z_seq.device if not isinstance(z_seq, list) else z_seq[0].device
        S_loop = S if S is not None else self.S

        # -------------------------
        # Init states
        # -------------------------
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # -------------------------
        # Logs
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

        # ★ value の S-loop 後の h を保存するバッファ（bind_idx → h_value）
        self.bind_h = {}
        
        self.log_A = []

        self.log_h_full = []     # ★ h_t を保存 ← 追加

        # -------------------------
        # Gain k
        # -------------------------
        k = self.compute_k()

        # -------------------------
        # Logging function
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p="fro", dim=(1,2)).mean().item()
            specA = torch.linalg.svdvals(A_t)[..., 0].mean().item()
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

            # ★ h_t を保存 ← 追加
            self.log_h_full.append(h.detach().cpu().clone())

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

                        dot  = (h_s * Ah).sum(dim=1, keepdim=True)
                        eps = 1e-6
                        norm1 = h_s.norm(dim=1, keepdim=True).clamp_min(eps)
                        norm2 = Ah.norm(dim=1, keepdim=True).clamp_min(eps)

                        R = dot / (norm1 * norm2)
                        R_pos = R.clamp(min=0.0, max=1.0)

                        alpha_dyn = 1.0 - (1.0 - R_pos) ** k

                        sloop_t.append({
                            "s": s,
                            "h_norm": h_s.norm(dim=1).mean().item(),
                            "Ah_norm": Ah.norm(dim=1).mean().item(),
                            "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                            "alpha_dyn": alpha_dyn.mean().item(),
                        })

                        h_s = (1 - alpha_dyn**2) * h_base + alpha_dyn * Ah
                        h_s = torch.relu(self.ln_h(h_s))

                    h = h_s

                self.log_sloop.append(sloop_t)
                self.log_A.append(A.detach().cpu().clone())

                # ---- Query classification result ----
                if (head is not None) and (class_ids is not None):

                    q_bind_idx = cid_raw
                    true_class = class_ids[q_bind_idx]

                    with torch.no_grad():
                        logits = head(h)
                        pred = logits.argmax(dim=1)
                        correct = (pred == true_class).float().mean().item()

                        sorted_logits, _ = torch.sort(logits, descending=True)
                        margin = (sorted_logits[:,0] - sorted_logits[:,1]).mean().item()

                        W = head.fc.weight
                        w_true = W[true_class]
                        w_true = w_true.unsqueeze(0).expand(B, -1)

                        cos = F.cosine_similarity(h, w_true, dim=1).mean().item()

                    self.log_query = {
                        "true": true_class,
                        "pred": pred[0].item(),
                        "correct": correct,
                        "margin": margin,
                        "cosine": cos,
                    }

                append_log(kind, cid_raw, A, h)
                continue

            # --------------------------------------------------------
            # Bind
            # --------------------------------------------------------
            sloop_t = []
            if self.use_A and S_loop > 0:
                h_s = h.clone()
                h0 = h_s.clone()

                for s in range(S_loop):
                    Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                    dot = (h_s * Ah).sum(dim=1, keepdim=True)

                    eps = 1e-6
                    norm1 = h_s.norm(dim=1, keepdim=True).clamp_min(eps)
                    norm2 = Ah.norm(dim=1, keepdim=True).clamp_min(eps)

                    R = dot / (norm1 * norm2)
                    R_pos = R.clamp(min=0.0, max=1.0)

                    alpha_dyn = 1.0 - (1.0 - R_pos) ** k

                    sloop_t.append({
                        "s": s,
                        "h_norm": h_s.norm(dim=1).mean().item(),
                        "Ah_norm": Ah.norm(dim=1).mean().item(),
                        "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                        "alpha_dyn": alpha_dyn.mean().item(),
                    })

                    h_s = (1 - alpha_dyn**2) * h_base + alpha_dyn * Ah
                    h_s = torch.relu(self.ln_h(h_s))

                h = h_s

            # Hebbian update
            if self.use_A:
                h_norm2 = (h**2).sum(dim=1, keepdim=True) + self.eps
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)) / h_norm2.unsqueeze(-1)
                A = self.lambda_ * A + self.eta * delta_A

            # ★ value の内部状態を保存
            if kind == "value":
                bind_idx = t // 2
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_sloop.append(sloop_t)
            self.log_A.append(A.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

        return None, None




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
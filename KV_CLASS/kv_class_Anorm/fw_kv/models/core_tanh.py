# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_fw_ln_residual.py
#
# Fast Weights を補正項として扱う Residual 型 RNN
# α_fw の符号で「上に凸 / 下に凸」の両方を実現するゲイン関数を採用。
# Bind では Self-consistent S-loop、Query では Ba-style (h_base + βAh) を使用。
#
# ★ 本バージョンは旧 forward_episode のログ形式を完全に残しつつ、
#    計算処理だけを最新版に書き換えた「Log-compatible Modern FW Core」です。

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------------------------------------
# Spectral radius (for stats)
# -------------------------------------------------------------
@torch.no_grad()
def batch_spectral_radius(A, iters=20):
    B, d, _ = A.shape
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    for _ in range(iters):
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        v = Av / (Av.norm(dim=1, keepdim=True) + 1e-8)

    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    return (Av * v).sum(dim=1).mean().item()


# ==============================================================
# CoreRNNFW (Residual FW with SC-loop + Ba-style Query)
# ==============================================================
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

        # α_fw：凸⇄凹ゲイン
        self.alpha_fw = nn.Parameter(torch.tensor(cfg.alpha_fw, dtype=torch.float32))
        # β：Query の Ah スケール
        # self.beta = nn.Parameter(torch.tensor(cfg.beta, dtype=torch.float32))

        # RNN 本体
        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))
        self.ln_h = nn.LayerNorm(self.d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # --------------------------------------------------------------
    # α_fw の符号で凸⇄凹を切り替えるゲイン k(α)
    # --------------------------------------------------------------
    def compute_k(self):
        alpha = self.alpha_fw

        # α>0 → k>1（上に凸）
        k_pos = 1.0 + F.softplus(alpha)

        # α<0 → 0<k<1（下に凸）
        k_neg = 1.0 / (1.0 + F.softplus(-alpha))

        return torch.where(alpha >= 0, k_pos, k_neg)

    # ==============================================================
    # forward_episode（ログ互換版）
    # ==============================================================
    def forward_episode(self, z_seq, y, head,
                        event_list=None,
                        S=None,
                        class_ids=None,
                        mu_value=None,
                        train=True,
                        compute_stats=False):

        # -------------------------
        # Input shape
        # -------------------------
        if isinstance(z_seq, list):
            T_total = len(z_seq)
            B = z_seq[0].size(0)
        else:
            T_total = z_seq.size(0)
            B = z_seq.size(1)

        device = z_seq[0].device if isinstance(z_seq, list) else z_seq.device
        S_loop = S if S is not None else self.S

        # -------------------------
        # Init states
        # -------------------------
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # -------------------------
        # Reset logs（旧版と完全互換）
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
        self.bind_h = {}
        self.log_A = []
        self.log_h_full = []
        self.log_base = []
        self.log_h_sloop = []

        # -------------------------
        # Gain k(α)
        # -------------------------
        k = self.compute_k()

        # -------------------------
        # Log helper
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
        self.log_A.append(A.detach().cpu().clone())  # 初期A

        for t in range(T_total):

            z_t = z_seq[t].to(device)
            kind, cid_raw = event_list[t] if event_list is not None else ("-", -1)

            # ----------------------------------------------------------
            # Base RNN update
            # ----------------------------------------------------------
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(self.ln_h(h_base))

            self.log_base.append(h_base.detach().cpu().clone())

            # ----------------------------------------------------------
            # Query step (Ba-style)
            # ----------------------------------------------------------
            if t == T_total - 1:

                sloop_logs = []
                h_s_vecs = []

                if self.use_A and S_loop > 0:
                    h_s = h.clone()
                    h0 = h_s.clone()

                    h_s_vecs.append(h_s.detach().cpu().clone())

                    for s in range(S_loop):

                        Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                        sloop_logs.append({
                            "s": s,
                            "h_norm": h_s.norm(dim=1).mean().item(),
                            "Ah_norm": Ah.norm(dim=1).mean().item(),
                            "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                            "alpha_dyn": alpha_dyn.mean().item(),
                        })

                        # Ba-style: h ← ReLU(LN(h_base + βAh))
                        # h_s = torch.relu(self.ln_h(h_base + self.beta * Ah))
                        h_s = torch.relu(self.ln_h(h_base + Ah))

                        h_s_vecs.append(h_s.detach().cpu().clone())

                    h = h_s

                self.log_sloop.append(sloop_logs)
                self.log_h_sloop.append(h_s_vecs)
                self.log_h_full.append(h.detach().cpu().clone())

                # Query classification
                if head is not None and class_ids is not None:

                    q_bind_idx = cid_raw
                    true_class = class_ids[q_bind_idx]

                    with torch.no_grad():
                        logits = head(h)
                        pred = logits.argmax(dim=1)
                        correct = (pred == true_class).float().mean().item()

                        sorted_logits,_ = torch.sort(logits, descending=True)
                        margin = (sorted_logits[:,0] - sorted_logits[:,1]).mean().item()

                        W = head.fc.weight
                        w_true = W[true_class].unsqueeze(0).expand(B,-1)
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

            # ----------------------------------------------------------
            # Bind / Wait : Self-consistent S-loop（SC-FW）
            # ----------------------------------------------------------
            sloop_logs = []
            h_s_vecs = []

            if self.use_A and S_loop > 0:

                h_s = h.clone()
                h0 = h_s.clone()
                h_s_vecs.append(h_s.detach().cpu().clone())

                for s in range(S_loop):

                    Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                    dot = (h_s * Ah).sum(dim=1, keepdim=True)
                    norm1 = h_s.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    norm2 = Ah.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    R = dot / (norm1 * norm2)
                    R_pos = R.clamp(min=0.0, max=1.0)

                    # Gain
                    alpha_dyn = 1 - (1 - R_pos)**k

                    sloop_logs.append({
                        "s": s,
                        "h_norm": h_s.norm(dim=1).mean().item(),
                        "Ah_norm": Ah.norm(dim=1).mean().item(),
                        "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                        "alpha_dyn": alpha_dyn.mean().item(),
                    })

                    # SC-loop correction
                    h_s = (1 - alpha_dyn**2) * h_base + alpha_dyn * Ah
                    h_s = torch.relu(self.ln_h(h_s))

                    h_s_vecs.append(h_s.detach().cpu().clone())

                h = h_s

            self.log_sloop.append(sloop_logs)
            self.log_h_sloop.append(h_s_vecs)

            # ----------------------------------------------------------
            # Hebbian Update
            # ----------------------------------------------------------
            if self.use_A:
                h_norm2 = (h**2).sum(dim=1, keepdim=True) + self.eps
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1)) # / h_norm2.unsqueeze(-1)
                A = self.lambda_ * A + self.eta * delta_A
            
            if kind == "value":
                bind_idx = t // 2
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_A.append(A.detach().cpu().clone())
            self.log_h_full.append(h.detach().clone())

            append_log(kind, cid_raw, A, h)

        return None, None

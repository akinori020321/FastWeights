# -*- coding: utf-8 -*-
# fw_kv/models/core_fw_unified.py
"""
Fast Weights Core (Unified Episode Version)
-------------------------------------------
A-dynamics 対応／Residual版と同一ログ構造
tanh系 Fast Weights（軽量版）
"""

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
        v = Av / (v.norm(dim=1, keepdim=True) + 1e-8)

    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    return (Av * v).sum(dim=1).mean().item()


# ==============================================================
# CoreRNNFW (Unified tanh-FW)
# ==============================================================
class CoreRNNFW(nn.Module):

    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg
        self.d_g = cfg.glimpse_dim
        self.d_h = cfg.hidden_dim

        self.S = cfg.inner_steps
        self.lambda_ = cfg.lambda_decay
        self.eta = cfg.eta
        self.eps = cfg.epsilon
        self.use_A = cfg.use_A

        # RNN main
        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))

        self.ln_h = nn.LayerNorm(self.d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # ==============================================================
    # forward_episode
    # ==============================================================
    def forward_episode(self, z_seq, y, head,
                        event_list=None,
                        S=None,
                        class_ids=None,
                        mu_value=None,
                        train=True,
                        compute_stats=False):

        # -------------------------
        # サイズ
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
        # 初期化
        # -------------------------
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)

        # -------------------------
        # ログ（Residual版と完全一致）
        # -------------------------
        self.log_froA = []
        self.log_specA = []
        self.log_hnorm = []
        self.log_Ahnorm = []
        self.log_kind = []
        self.log_class = []
        self.log_h = []
        self.log_sloop = []
        self.log_h_sloop = []     # ★ 全 t×S の h_s ベクトル保存
        self.log_base = []         # ★ 各 t の h_base 保存
        self.log_query = {}

        # ★ value S-loop 後の h を保存
        self.bind_h = {}

        self.log_A = []

        self.log_h_full = []        # ★ フル h_t 保存用 ←追加行

        # -------------------------
        # ログ関数
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p='fro', dim=(1, 2)).mean().item()
            specA = torch.linalg.svdvals(A_t)[..., 0].mean().item()
            h_norm = h_t.norm(dim=1).mean().item()
            Ah = torch.bmm(A_t, h_t.unsqueeze(2)).squeeze(-1)
            Ah_norm = Ah.norm(dim=1).mean().item()

            self.log_froA.append(froA)
            self.log_specA.append(specA)
            self.log_hnorm.append(h_norm)
            self.log_Ahnorm.append(Ah_norm)
            self.log_kind.append(kind)
            self.log_class.append(cid)
            self.log_h.append(h_t.detach().clone())

        # ==============================================================
        # 時系列展開
        # ==============================================================
        
        self.log_A.append(A.detach().cpu().clone())

        for t in range(T_total):

            z_t = z_seq[t].to(device)
            kind, cid_raw = event_list[t] if event_list is not None else ("-", -1)

            # ===== Basic RNN =====
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(self.ln_h(h_base))

            self.log_base.append(h_base.detach().cpu().clone())

            # ==========================================================
            # Query
            # ==========================================================
            if t == T_total - 1:

                sloop_t = []
                h_s_vecs = [] 

                if self.use_A:
                    h_s = h.clone()
                    h0 = h_s.clone()

                    h_s_vecs.append(h_s.detach().cpu().clone())

                    for s in range(S_loop):
                        Ah = torch.bmm(A, h_s.unsqueeze(2)).squeeze(-1)

                        # ★ 正解 value の内部状態との cosine（cid_raw は bind_idx）
                        q_bind_idx = cid_raw   # ← Query の cid_raw は bind_idx のため正しい

                        if q_bind_idx in self.bind_h:
                            h_value = self.bind_h[q_bind_idx]    # ← 正しい bind 順の value h
                            cos_value = F.cosine_similarity(h_s, h_value, dim=1).mean().item()
                        else:
                            cos_value = 0.0

                        sloop_t.append({
                            "s": s,
                            "h_norm": h_s.norm(dim=1).mean().item(),
                            "Ah_norm": Ah.norm(dim=1).mean().item(),
                            "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                            "cos_value": cos_value,
                        })

                        h_s = torch.relu(self.ln_h(h_base + Ah))

                        h_s_vecs.append(h_s.detach().cpu().clone())

                    h = h_s
                
                self.log_h_sloop.append(h_s_vecs)

                self.log_sloop.append(sloop_t)

                self.log_h_full.append(h.detach().cpu().clone())

                # -------------------------
                # classification check
                # -------------------------
                if (head is not None) and (class_ids is not None):

                    true_class = class_ids[cid_raw]

                    with torch.no_grad():
                        logits = head(h)
                        pred = logits.argmax(dim=1)
                        correct = (pred == true_class).float().mean().item()

                        sorted_logits, _ = torch.sort(logits, descending=True)
                        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()

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

            # ==========================================================
            # Bind
            # ==========================================================
            sloop_t = []
            h_s_vecs = []   

            if self.use_A and S_loop > 0:
                h_s = h.clone()
                h0 = h_s.clone()

                h_s_vecs.append(h_s.detach().cpu().clone())

                for s in range(S_loop):
                    Ah = torch.bmm(A, h_s.unsqueeze(2)).squeeze(-1)

                    sloop_t.append({
                        "s": s,
                        "h_norm": h_s.norm(dim=1).mean().item(),
                        "Ah_norm": Ah.norm(dim=1).mean().item(),
                        "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                    })

                    h_s = torch.relu(self.ln_h(h_base + Ah))

                    h_s_vecs.append(h_s.detach().cpu().clone())

                h = h_s
            
            self.log_h_sloop.append(h_s_vecs)

            # ===== Hebbian =====
            if self.use_A:
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))
                A = self.lambda_ * A + self.eta * delta_A

            # ★ value の内部状態を保存
            if kind == "value":
                bind_idx = t // 2 
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_sloop.append(sloop_t)
            self.log_A.append(A.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

            self.log_h_full.append(h.detach().cpu().clone())

        return None, None

# -*- coding: utf-8 -*-
# fw_kv/models/core_fw_unified_direction.py
"""
Fast Weights Core (Unified Episode Version, Direction Reconstruction)
---------------------------------------------------------------------
分類タスク版から、方向復元タスク（cosine reconstruction）版へ置き換え済み

◎ 計算部分（RNN, S-loop, Hebbian update）は unified と全く同じ
◎ ログ構造も unified 版と完全一致
◎ Query 時の分類ロジックのみ方向復元ロジックに置換
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
# CoreRNNFW (Unified tanh-FW → Direction reconstruction)
# ==============================================================
class CoreRNNFW(nn.Module):

    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg
        self.d_g = cfg.glimpse_dim
        self.d_h = cfg.hidden_dim

        self.S = cfg.inner_steps
        self.lambda_  = cfg.lambda_decay
        self.eta      = cfg.eta
        self.eps      = cfg.epsilon
        self.use_A    = cfg.use_A

        # RNN main
        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))

        self.ln_h = nn.LayerNorm(self.d_h) if cfg.use_layernorm else nn.Identity()

        nn.init.kaiming_uniform_(self.W_h.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.W_g.weight)

    # ==============================================================
    # forward_episode（分類ロジック→方向復元ロジックへ置換）
    # ==============================================================
    def forward_episode(self, z_seq, clean_vec, head,
                        event_list=None,
                        S=None,
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
        # unified ログ構造
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

        # -------------------------
        # ログ関数（unified と完全一致）
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p='fro', dim=(1, 2)).mean().item()
            specA = batch_spectral_radius(A_t)
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
        for t in range(T_total):

            z_t = z_seq[t].to(device)
            kind, cid_raw = event_list[t] if event_list is not None else ("-", -1)

            # 基本 RNN
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(h_base)

            # ==========================================================
            # Query（最後のステップ）
            # ==========================================================
            if t == T_total - 1:

                sloop_t = []

                if self.use_A:
                    h_s = h.clone()
                    h0  = h_s.clone()

                    for s in range(S_loop):
                        Ah = torch.bmm(A, h_s.unsqueeze(2)).squeeze(-1)

                        # ★ value h との cosine（分類ではなく統計用）
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
                            "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                            "cos_value": cos_value,
                        })

                        h_s = torch.relu(self.ln_h(h_base + Ah))

                    h = h_s

                self.log_sloop.append(sloop_t)
                self.log_A.append(A.detach().cpu().clone())
                append_log(kind, cid_raw, A, h)

                # -------------------------
                # 方向復元タスクの最重要部分
                # -------------------------
                pred_vec = head(h)  # (B, d_g)

                pred_norm  = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + 1e-6)
                clean_norm = clean_vec / (clean_vec.norm(dim=1, keepdim=True) + 1e-6)

                loss = F.mse_loss(pred_norm, clean_norm)
                acc  = (pred_norm * clean_norm).sum(dim=1).mean().item()

                # ★ log_query（分類 → cosine のみ）
                self.log_query = {
                    "cosine": acc,
                }

                # stats
                stats = {}
                if self.use_A:
                    stats["specA"] = float(batch_spectral_radius(A))
                    stats["alpha_dyn"] = 0.0

                return loss, acc, stats

            # ==========================================================
            # Bind（最後以外のステップ）
            # ==========================================================
            sloop_t = []

            if self.use_A and S_loop > 0:
                h_s = h.clone()
                h0  = h_s.clone()

                for s in range(S_loop):
                    Ah = torch.bmm(A, h_s.unsqueeze(2)).squeeze(-1)

                    sloop_t.append({
                        "s": s,
                        "h_norm": h_s.norm(dim=1).mean().item(),
                        "Ah_norm": Ah.norm(dim=1).mean().item(),
                        "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                    })

                    h_s = torch.relu(self.ln_h(h_base + Ah))

                h = h_s

            # Hebbian update
            if self.use_A:
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))
                A = self.lambda_ * A + self.eta * delta_A

            if kind == "bind":
                bind_idx = cid_raw
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_sloop.append(sloop_t)

            self.log_A.append(A.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

        return None, None, {}

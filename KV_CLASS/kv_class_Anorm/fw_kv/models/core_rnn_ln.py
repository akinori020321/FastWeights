# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_ln.py
"""
Pure RNN + LayerNorm Core (Unified Episode Version)
---------------------------------------------------
Fast Weights を一切使わず A=0 のまま進めるが、
core_fw_unified.py と同じログ構造を維持する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fw_kv.models.core_config import CoreCfg


class CoreRNN_LN(nn.Module):

    def __init__(self, cfg: CoreCfg):
        super().__init__()
        self.cfg = cfg
        self.d_g = cfg.glimpse_dim
        self.d_h = cfg.hidden_dim

        # S-loop, FW 系は使わないがログ整合性のため保持
        self.S = cfg.inner_steps
        self.use_A = False   # ここを強制 False

        # RNN main
        self.W_h = nn.Linear(self.d_h, self.d_h, bias=False)
        self.W_g = nn.Linear(self.d_g, self.d_h, bias=False)
        self.b_h = nn.Parameter(torch.zeros(self.d_h))

        self.ln_h = nn.LayerNorm(self.d_h)

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

        # -------------------------
        # 初期化
        # -------------------------
        h = torch.zeros(B, self.d_h, device=device)
        A = torch.zeros(B, self.d_h, self.d_h, device=device)   # 常にゼロ

        # -------------------------
        # ログ（FW 版と完全一致）
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
        self.bind_h = {}       # 互換性のため保持
        self.log_A = []
        self.log_h_full = []

        # -------------------------
        # ログ関数
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p='fro', dim=(1, 2)).mean().item()
            specA = 0.0
            h_norm = h_t.norm(dim=1).mean().item()
            Ah = torch.zeros_like(h_t)
            Ah_norm = 0.0

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

            # ===== Basic RNN =====
            h_base = self.W_h(h) + self.W_g(z_t) + self.b_h
            h = torch.relu(self.ln_h(h_base))

            # ==========================================================
            # Query（最終ステップ）
            # ==========================================================
            if t == T_total - 1:

                # S-loop 互換ログ → すべてゼロで埋める
                sloop_t = []
                for s in range(self.S):
                    sloop_t.append({
                        "s": s,
                        "h_norm": h.norm(dim=1).mean().item(),
                        "Ah_norm": 0.0,
                        "cos_h0": 1.0,
                        "cos_value": 0.0,
                    })
                self.log_sloop.append(sloop_t)

                self.log_A.append(A.detach().cpu().clone())

                # 分類
                if (head is not None) and (class_ids is not None):
                    true_class = class_ids[cid_raw]
                    with torch.no_grad():
                        logits = head(h)
                        pred = logits.argmax(dim=1)
                        correct = (pred == true_class).float().mean().item()
                        sorted_logits, _ = torch.sort(logits, descending=True)
                        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()

                        W = head.fc.weight
                        w_true = W[true_class].unsqueeze(0).expand(B, -1)

                        cos = F.cosine_similarity(h, w_true, dim=1).mean().item()

                    self.log_query = {
                        "true": true_class,
                        "pred": pred[0].item(),
                        "correct": correct,
                        "margin": margin,
                        "cosine": cos,
                    }

                append_log(kind, cid_raw, A, h)
                self.log_h_full.append(h.detach().cpu().clone())
                continue

            # ==========================================================
            # Bind（FW は無いので何もしない）
            # ==========================================================
            sloop_t = []
            for s in range(self.S):
                sloop_t.append({
                    "s": s,
                    "h_norm": h.norm(dim=1).mean().item(),
                    "Ah_norm": 0.0,
                    "cos_h0": 1.0,
                })
            self.log_sloop.append(sloop_t)

            # A はゼロのまま保存
            self.log_A.append(A.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

            self.log_h_full.append(h.detach().cpu().clone())

        return None, None

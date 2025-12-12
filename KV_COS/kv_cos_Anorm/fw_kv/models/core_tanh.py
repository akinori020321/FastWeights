# -*- coding: utf-8 -*-
# fw_kv/models/core_rnn_fw_ln_residual_direction.py
#
# Residual FastWeights + unified logging (Direction Reconstruction)
# 分類版 core_rnn_fw_ln_residual.py とログ形式を完全一致させつつ，
# Query 部だけ「方向復元タスク（pred_vec の方向 ≈ clean_vec）」に置換した版。

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
# CoreRNNFW (Residual FW with SC-loop + Ba-style Query, Direction)
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
        # β（Query の Ah スケール）はここでは固定 1.0 相当（Ba-style: h_base + Ah）
        # 必要なら cfg.beta 等から追加可能

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
    # forward_episode（方向復元タスク版 / ログ互換）
    #   z_seq: (T, B, d_g)
    #   clean_vec: (B, d_g) - Query で復元したい target ベクトル
    #   head: h → pred_vec の線形変換
    # ==============================================================
    def forward_episode(self, z_seq, clean_vec, head,
                        event_list=None,
                        S=None,
                        train=True,
                        compute_stats=False):

        # -------------------------
        # Input shape
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
        # Reset logs（分類版と完全互換）
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
        # Log helper（specA は svdvals の最大特異値）
        # -------------------------
        def append_log(kind, cid, A_t, h_t):
            froA = torch.norm(A_t, p="fro", dim=(1, 2)).mean().item()
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
        # 初期 A（t = -1 相当）
        self.log_A.append(A.detach().cpu().clone())

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
            # Query step (Ba-style, Direction Reconstruction)
            # ----------------------------------------------------------
            if t == T_total - 1:

                sloop_logs = []
                h_s_vecs = []

                if self.use_A and S_loop > 0:
                    h_s = h.clone()
                    h0 = h_s.clone()

                    # s = 0（S-loop 前）
                    h_s_vecs.append(h_s.detach().cpu().clone())

                    # Ba-style: ここでは S_loop に関わらず 1 回だけ (分類版と互換)
                    for s in range(S_loop):
                        Ah = torch.bmm(A, h_s.unsqueeze(-1)).squeeze(-1)

                        # value h との cosine（方向復元タスク向け統計）
                        if cid_raw in self.bind_h:
                            q_bind_idx = cid_raw
                            h_value = self.bind_h[q_bind_idx]
                            cos_value = F.cosine_similarity(h_s, h_value, dim=1).mean().item()
                        else:
                            cos_value = 0.0

                        sloop_logs.append({
                            "s": s,
                            "h_norm": h_s.norm(dim=1).mean().item(),
                            "Ah_norm": Ah.norm(dim=1).mean().item(),
                            "cos_h0": F.cosine_similarity(h0, h_s, dim=1).mean().item(),
                            "alpha_dyn": alpha_dyn.mean().item(),
                            "cos_value": cos_value,
                        })

                        # Ba-style: h ← ReLU(LN(h_base + Ah))  （β=1 固定）
                        h_s = torch.relu(self.ln_h(h_base + Ah))
                        h_s_vecs.append(h_s.detach().cpu().clone())

                    h = h_s

                self.log_sloop.append(sloop_logs)
                self.log_h_sloop.append(h_s_vecs)
                self.log_h_full.append(h.detach().cpu().clone())

                # ==================================================
                # ★ 方向復元タスク（pred_vec の方向 ≈ clean_vec）
                # ==================================================
                loss = None
                cosine = 0.0

                if (head is not None) and (clean_vec is not None):
                    clean_vec_dev = clean_vec.to(device)  # (B, d_g) を想定

                    pred_vec = head(h)  # (B, d_g)

                    # 単位ノルムに正規化して方向のみ比較
                    pred_norm = pred_vec / (pred_vec.norm(dim=1, keepdim=True) + 1e-6)
                    target_norm = clean_vec_dev / (clean_vec_dev.norm(dim=1, keepdim=True) + 1e-6)

                    loss = F.mse_loss(pred_norm, target_norm)
                    cosine = (pred_norm * target_norm).sum(dim=1).mean().item()
                else:
                    loss = None
                    cosine = None

                # log_query は分類版と同じキー構造にそろえる
                self.log_query = {
                    "true": -1,
                    "pred": -1,
                    "correct": 0.0,
                    "margin": 0.0,
                    "cosine": float(cosine) if cosine is not None else 0.0,
                }

                append_log(kind, cid_raw, A, h)

                # stats
                stats = {}
                if compute_stats and self.use_A:
                    stats["specA"] = float(batch_spectral_radius(A))
                    # α_dyn の代表値を入れたければここで計算してもよいが，
                    # とりあえず 0.0 にしておく
                    stats["alpha_dyn"] = 0.0

                return loss, cosine, stats

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
                    alpha_dyn = 1 - (1 - R_pos) ** k

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
                h_norm2 = (h ** 2).sum(dim=1, keepdim=True) + self.eps
                # 旧版互換：正規化をコメントアウトした outer-product 版
                delta_A = torch.bmm(h.unsqueeze(2), h.unsqueeze(1))  # / h_norm2.unsqueeze(-1)
                A = self.lambda_ * A + self.eta * delta_A

            # ★ value の内部状態を保存（direction 版では "bind" で class_id を持つ想定）
            if kind == "bind":
                bind_idx = cid_raw   # class_id として使用
                self.bind_h[bind_idx] = h.detach().clone()

            self.log_A.append(A.detach().cpu().clone())
            self.log_h_full.append(h.detach().cpu().clone())

            append_log(kind, cid_raw, A, h)

        # 通常は Query で return 済み
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
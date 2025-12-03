# -*- coding: utf-8 -*-
"""
utils_dyn.py
----------------------------------------
A-dynamics 実験用の統計計算ユーティリティ。

- A の Frobenius ノルム / スペクトル半径
- h のノルム / A h のノルム
- h と「隠れ表現プロトタイプ」との cos 類似度
"""

from __future__ import annotations
import torch
from fw_kv.utils import batch_fro_norm, batch_spectral_radius


@torch.no_grad()
def compute_A_stats(A: torch.Tensor, h: torch.Tensor):
    """
    A と h から基本的な統計量を計算する。

    Parameters
    ----------
    A : (B, d_h, d_h)
    h : (B, d_h)

    Returns
    -------
    froA : float
        Frobenius norm ||A||_F のバッチ平均。
    specA : float
        平均スペクトル半径 ρ(A)。
    h_norm : float
        ||h|| のバッチ平均。
    Ah_norm : float
        ||A h|| のバッチ平均。
    """
    froA = batch_fro_norm(A)
    specA = batch_spectral_radius(A)

    h_norm = h.norm(dim=1).mean().item()

    Ah = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1)  # (B, d_h)
    Ah_norm = Ah.norm(dim=1).mean().item()

    return froA, specA, h_norm, Ah_norm


@torch.no_grad()
def mean_cos_to_protos(h: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """
    h と K 個の「隠れ表現プロトタイプ」との cos 類似度（バッチ平均）を計算。

    Parameters
    ----------
    h : (B, d_h)
        現在の隠れ状態。
    protos : (K, d_h)
        K 個の隠れ表現プロトタイプ。

    Returns
    -------
    mean_cos : torch.Tensor, shape (K,)
        各プロトタイプごとの cos(h, proto_k) をバッチ平均した値。
    """
    # 正規化
    h_normed = h / (h.norm(dim=1, keepdim=True) + 1e-8)          # (B, d_h)
    p_normed = protos / (protos.norm(dim=1, keepdim=True) + 1e-8)  # (K, d_h)

    # (B, K)
    cos_all = h_normed @ p_normed.transpose(0, 1)
    mean_cos = cos_all.mean(dim=0)  # (K,)

    return mean_cos

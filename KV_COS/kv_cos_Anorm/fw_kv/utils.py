import csv
import math
import random
from typing import Dict, Any, Iterable

import numpy as np
import torch

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==============================================================
# Fast Weights 行列 A のノルム / スペクトル半径 解析関数
# ==============================================================
@torch.no_grad()
def batch_fro_norm(A: torch.Tensor) -> float:
    """
    A: (B, d, d)
    return: 平均フロベニウスノルム ||A||_F のバッチ平均 (float)
    """
    # Frobenius norm per sample
    fro = torch.linalg.norm(A, ord="fro", dim=(1, 2))  # (B,)
    return fro.mean().item()


@torch.no_grad()
def batch_spectral_radius(A: torch.Tensor, iters: int = 20) -> float:
    """
    A: (B, d, d)
    return: 平均スペクトル半径 (最大固有値の絶対値) のバッチ平均 (float)
    
    パワーイテレーション法による近似：
        max_eig ≒ v^T A v  /  (||v||^2)
    """
    B, d, _ = A.shape

    # 初期ベクトル（ランダム）
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

    for _ in range(iters):
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)  # (B, d)
        v = Av / (Av.norm(dim=1, keepdim=True) + 1e-8)

    # Rayleigh quotient
    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    lambda_max = (Av * v).sum(dim=1)  # (B,)
    return lambda_max.abs().mean().item()
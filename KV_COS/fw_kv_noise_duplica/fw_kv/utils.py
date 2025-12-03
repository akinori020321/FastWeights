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

def write_csv_row(path: str, row: Dict[str, Any], header: bool = False) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)

@torch.no_grad()
def spectral_norm_batch(A: torch.Tensor, n_iter: int = 2) -> torch.Tensor:
    """
    Approximate the spectral norm (largest singular value) per sample.
    A: (B, d, d)
    Returns: (B,)
    """
    B, d, _ = A.shape
    # initialize v randomly
    v = torch.randn(B, d, device=A.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
    for _ in range(n_iter):
        # u = A v
        u = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        # v = A^T u
        v = torch.bmm(A.transpose(1,2), u.unsqueeze(-1)).squeeze(-1)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
    Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
    sigma = Av.norm(dim=1) / (v.norm(dim=1) + 1e-9)
    return sigma

def to_device(xs: Iterable[torch.Tensor], device: str):
    return [x.to(device) for x in xs]

def detach_all(xs: Iterable[torch.Tensor]):
    return [x.detach().cpu() for x in xs]
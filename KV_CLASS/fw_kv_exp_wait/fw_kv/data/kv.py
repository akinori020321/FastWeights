from dataclasses import dataclass
import torch
import numpy as np
from typing import List, Tuple

# ======================================================
# Utility
# ======================================================

def unit_sphere(d: int) -> np.ndarray:
    v = np.random.randn(d).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

# ======================================================
# Config
# ======================================================

@dataclass
class KVConfig:
    d_g: int = 64
    num_classes: int = 20
    T_bind: int = 5
    noise_std: float = 0.0
    device: str = "cpu"
    seed: int = 123
    mu: np.ndarray | None = None
    beta: float = 1.0

# ======================================================
# Dataset
# ======================================================

class KVDataset:
    def __init__(self, cfg: KVConfig, key_proto, wait_vec=None, delta_wait=0):
        self.cfg = cfg
        
        # ★ run.py から受け取った key_proto をそのまま使う
        self.key_proto = key_proto.astype(np.float32)
        self.num_key_proto = key_proto.shape[0]
        
        self.wait_vec = wait_vec
        self.delta_wait = delta_wait
        self.rng = np.random.RandomState(cfg.seed)

        # μ は従来通り
        if cfg.mu is None:
            self.mu = np.stack([unit_sphere(cfg.d_g) for _ in range(cfg.num_classes)], axis=0)
        else:
            self.mu = cfg.mu.astype(np.float32)

    def sample_batch(self, B: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cfg = self.cfg
        C, d, T_bind = cfg.num_classes, cfg.d_g, cfg.T_bind
        device = cfg.device
        noise_std = cfg.noise_std
        beta = cfg.beta

        # ============================================================
        # 修正：エピソード内（T_bind）で key は重複なし
        # ============================================================
        if T_bind <= self.num_key_proto:
            # key_proto から T_bind 個を重複なしで選ぶ
            key_base = self.rng.choice(self.num_key_proto, size=T_bind, replace=False)
        else:
            # key_proto の数より多く必要な場合は、全種使って残りは重複OK
            base = self.rng.permutation(self.num_key_proto)              # 全部一度ずつ
            rest = self.rng.choice(self.num_key_proto, size=T_bind - self.num_key_proto, replace=True)
            key_base = np.concatenate([base, rest])

        # バッチ方向にタイリング（B, T_bind）
        key_idx = np.tile(key_base[None, :], (B, 1))     # shape (B, T_bind)

        k = self.key_proto[key_idx]  # (B, T_bind, d_g)

        # ============================================================
        # Bind クラス列：バッチ内で全て揃える
        # ============================================================

        # ★ 1つの bind 列（長さ T_bind）だけ生成
        y_bind_single = self.rng.randint(0, C, size=(T_bind,))  # (T_bind,)

        # ★ 全バッチにコピー（B, T_bind）
        y_bind = np.tile(y_bind_single, (B, 1))  # (B, T_bind)

        # value ベクトル
        v = self.mu[y_bind]  # (B, T_bind, d_g)
        
        # === Query index ===
        q_idx = self.rng.randint(0, T_bind, size=(B,))

        # === Bind ===
        z_seq: List[np.ndarray] = []

        for t in range(T_bind):
            # -------------------------------
            # (1) key をそのまま入力
            # -------------------------------
            z_key = k[:, t, :].copy()
            z_key /= (np.linalg.norm(z_key, axis=1, keepdims=True) + 1e-8)
            z_seq.append(z_key)

            # -------------------------------
            # (2) value (= μ[class]) をそのまま入力
            # -------------------------------
            z_val = v[:, t, :].copy()
            eps = np.stack([unit_sphere(d) for _ in range(B)], axis=0)
            if noise_std > 0:
                z_val = noise_std * z_val + (1 - noise_std) * eps
            z_val /= (np.linalg.norm(z_val, axis=1, keepdims=True) + 1e-8)
            z_seq.append(z_val)

        # === Wait ===
        if self.delta_wait > 0 and self.wait_vec is not None:
            z_wait = np.tile(self.wait_vec, (B, self.delta_wait, 1))
            for t in range(self.delta_wait):
                z_seq.append(z_wait[:, t, :])

        # === Query ===
        k_q = k[np.arange(B), q_idx, :]
        eps = np.stack([unit_sphere(d) for _ in range(B)], axis=0)
        k_q_weak = beta * k_q + (1 - beta) * eps
        z_T = k_q_weak / (np.linalg.norm(k_q_weak, axis=1, keepdims=True) + 1e-8)

        y = y_bind[np.arange(B), q_idx]

        z_seq.append(z_T)
        z_seq = [torch.from_numpy(z).float().to(device) for z in z_seq]
        y = torch.from_numpy(y).long().to(device)
        
        return z_seq, y

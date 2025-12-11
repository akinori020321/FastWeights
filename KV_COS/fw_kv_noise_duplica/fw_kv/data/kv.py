# -*- coding: utf-8 -*-
"""
========================================================
方向復元タスク KVDataset
 - Clean は duplicate 回繰り返す
 - Anti は Clean に登場した class も key も一切使用しない
   （完全に異なる class×key を一度だけ使用）
 - Query は Clean で複数回登場した class から選択
========================================================
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import Optional


# -----------------------------------------------------
# Utility
# -----------------------------------------------------
def unit_sphere(d: int, rng: np.random.RandomState) -> np.ndarray:
    v = rng.randn(d).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v


# -----------------------------------------------------
# Config
# -----------------------------------------------------
@dataclass
class KVConfig:
    d_g: int = 64
    T_bind: int = 60
    duplicate: int = 3
    device: str = "cpu"
    seed: int = 123
    beta: float = 1.0
    bind_noise_std: float = 0.0    # 追加


# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class KVDataset(IterableDataset):

    def __init__(
        self,
        cfg: KVConfig,
        mu_value: np.ndarray,
        key_proto: np.ndarray,
        wait_vec: Optional[np.ndarray] = None,
        delta_wait: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.mu_value = mu_value
        self.key_proto = key_proto
        self.wait_vec = wait_vec
        self.delta_wait = delta_wait
        self.rng = np.random.RandomState(cfg.seed)

        self.num_classes = mu_value.shape[0]
        self.num_keys = key_proto.shape[0]

        # ===== Projection matrix W =====
        d = cfg.d_g
        rng = np.random.RandomState(999)
        self.W = rng.randn(d * d, d).astype(np.float32)
        self.W /= (np.linalg.norm(self.W, axis=0, keepdims=True) + 1e-8)

    # -------------------------------------------------
    # sample_batch
    # -------------------------------------------------
    def sample_batch(self, B: int):

        cfg, d, device = self.cfg, self.cfg.d_g, self.cfg.device

        assert cfg.T_bind % cfg.duplicate == 0
        num_items = cfg.T_bind // cfg.duplicate

        # ============================================================
        # ★ このバッチ専用の key / value テーブルをシャッフル
        #    （バッチ内では固定，バッチが変わると perm も変わる）
        # ============================================================
        perm_c = self.rng.permutation(self.num_classes)
        perm_k = self.rng.permutation(self.num_keys)
        mu_ep = self.mu_value[perm_c]      # shape: (num_classes, d_g)
        key_ep = self.key_proto[perm_k]    # shape: (num_keys, d_g)

        # ============================================================
        # 1. Clean（duplicate 回登場）
        # ============================================================
        class_ids_base = np.arange(num_items) % self.num_classes
        key_ids_base   = np.arange(num_items) % self.num_keys

        class_ids_clean = np.repeat(class_ids_base, cfg.duplicate)
        key_ids_clean   = np.repeat(key_ids_base,   cfg.duplicate)

        # Clean の class/key セット
        clean_classes = set(class_ids_clean.tolist())
        clean_keys    = set(key_ids_clean.tolist())

        # Clean ペア（旧仕様残し）
        clean_pairs = set(zip(class_ids_clean, key_ids_clean))

        # ============================================================
        # 2. Anti（Clean の class も key も一切使用しない）
        # ============================================================
        num_anti = 0  # cfg.T_bind

        # Clean に出ていない class / key を抽出
        anti_classes = [c for c in range(self.num_classes)
                        if c not in clean_classes]

        anti_keys = [k for k in range(self.num_keys)
                     if k not in clean_keys]

        if len(anti_classes) == 0 or len(anti_keys) == 0:
            raise ValueError("Anti に使える class または key がありません。")

        # Clean と完全に分離された Anti 用 (class,key)
        all_pairs_anti = [(c, k) for c in anti_classes for k in anti_keys]

        if num_anti > len(all_pairs_anti):
            raise ValueError("num_anti が候補より大きい（class/key 丸ごと除外のため）。")

        pairs_idx = self.rng.choice(len(all_pairs_anti), size=num_anti, replace=False)
        pairs = [all_pairs_anti[i] for i in pairs_idx]

        class_ids_anti = np.array([p[0] for p in pairs], dtype=int)
        key_ids_anti   = np.array([p[1] for p in pairs], dtype=int)

        # ============================================================
        # 3. Merge → Shuffle
        # ============================================================
        class_ids = np.concatenate([class_ids_clean, class_ids_anti], axis=0)
        key_ids   = np.concatenate([key_ids_clean,   key_ids_anti],   axis=0)

        perm = self.rng.permutation(len(class_ids))
        class_ids = class_ids[perm]
        key_ids   = key_ids[perm]

        z_list = []

        # ============================================================
        # 4. Bind sequence（Clean と Anti が完全に disjoint）
        # ============================================================

        for cls_t, key_t in zip(class_ids, key_ids):

            # ★ このバッチ専用にシャッフルされた key_ep / mu_ep を使用
            key = key_ep[key_t]
            val = mu_ep[cls_t]

            outer = np.outer(key, val).astype(np.float32)
            flat = outer.reshape(-1)
            clean_vec = flat @ self.W
            clean_vec /= (np.linalg.norm(clean_vec) + 1e-8)

            if (cls_t in clean_classes) and (key_t in clean_keys):
                # Clean
                eps = unit_sphere(d, self.rng)
                r = cfg.bind_noise_std
                mixed = r * clean_vec + (1 - r) * eps
                mixed /= (np.linalg.norm(mixed) + 1e-8)
            else:
                # Clean
                eps = unit_sphere(d, self.rng)
                r = 0.0
                mixed = r * clean_vec + (1 - r) * eps
                mixed /= (np.linalg.norm(mixed) + 1e-8)

            mixed_batch = np.stack([mixed for _ in range(B)], axis=0)
            mixed_batch /= (np.linalg.norm(mixed_batch, axis=1, keepdims=True) + 1e-8)

            z_list.append(torch.from_numpy(mixed_batch).float().to(device))

        # ============================================================
        # 5. Query（Clean の複数回登場 class から選択）
        # ============================================================
        unique, counts = np.unique(class_ids_clean, return_counts=True)
        multi_classes = [c for c, cnt in zip(unique, counts) if cnt > 1]

        target_c = self.rng.choice(multi_classes)

        valid_keys = key_ids_base[class_ids_base == target_c]
        target_k = self.rng.choice(valid_keys)

        # ★ Query でも同じ mu_ep / key_ep を使用（バッチ内で一貫）
        key = key_ep[target_k]
        val = mu_ep[target_c]

        outer = np.outer(key, val).astype(np.float32)
        clean_target = outer.reshape(-1) @ self.W
        clean_target /= (np.linalg.norm(clean_target) + 1e-8)

        clean_vec = torch.from_numpy(
            np.tile(clean_target, (B, 1))
        ).float().to(device)

        # ----- Query vector -----
        key_q = np.tile(key_ep[target_k], (B, 1))
        noise_q = np.stack([unit_sphere(d, self.rng) for _ in range(B)], axis=0)

        z_q = cfg.beta * key_q + (1 - cfg.beta) * noise_q
        z_q /= (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)

        z_list.append(torch.from_numpy(z_q).float().to(device))

        return z_list, clean_vec

    def __iter__(self):
        while False:
            yield None

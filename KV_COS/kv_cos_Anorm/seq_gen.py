# -*- coding: utf-8 -*-
"""
seq_gen.py  — outer product + random projection + anti-noise 対応版（修正版）
"""

from __future__ import annotations
import numpy as np
import torch


# ======================================================
# Utility
# ======================================================

def unit_sphere(d: int, rng: np.random.RandomState) -> np.ndarray:
    v = rng.randn(d).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v



def make_keyvalue_sequence_direction(
    d_g: int,
    batch_size: int,
    T_bind: int,
    duplicate: int,
    beta: float = 1.0,
    bind_noise_std: float = 0.0,
    query_noise_std: float = 0.0,
    device: str = "cpu",
    seed: int = 123,

    mu_value: np.ndarray | None = None,
    key_proto: np.ndarray | None = None,
):
    """
    方向復元タスクの KV シーケンス生成
    （outer product + random projection + anti-noise 対応）
    """
    rng = np.random.RandomState(seed)
    assert T_bind % duplicate == 0

    assert key_proto is not None
    assert mu_value is not None

    num_keys = key_proto.shape[0]
    num_classes = mu_value.shape[0]
    num_items = T_bind // duplicate

    # ------------------------------------------------------
    # outer product 用ランダム射影行列 W（固定）
    # ------------------------------------------------------
    proj_rng = np.random.RandomState(999)
    W = proj_rng.randn(d_g * d_g, d_g).astype(np.float32)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

    # ------------------------------------------------------
    # ★ このバッチ専用の key / value テーブルをシャッフル
    # ------------------------------------------------------
    perm_c = rng.permutation(num_classes)
    perm_k = rng.permutation(num_keys)
    mu_ep  = mu_value[perm_c]      # (num_classes, d_g)
    key_ep = key_proto[perm_k]     # (num_keys, d_g)

    # ------------------------------------------------------
    # 基本 class_ids / key_ids
    # ------------------------------------------------------
    class_ids_base = np.arange(num_items) % num_classes
    key_ids_base   = np.arange(num_items) % num_keys

    class_ids_clean = np.repeat(class_ids_base, duplicate)   # (T_bind,)
    key_ids_clean   = np.repeat(key_ids_base,   duplicate)

    assert len(class_ids_clean) == T_bind

    # ------------------------------------------------------
    # ★ Anti-noise の数を T_bind に変更（KVDataset に合わせる）
    # ------------------------------------------------------
    num_anti = T_bind                            # ← ここが変更点

    class_ids_noise = np.full(num_anti, -1)
    key_ids_noise   = np.full(num_anti, -1)

    # clean + anti を連結 → シャッフル
    class_ids = np.concatenate([class_ids_clean, class_ids_noise], axis=0)
    key_ids   = np.concatenate([key_ids_clean,   key_ids_noise],   axis=0)

    perm = rng.permutation(len(class_ids))
    class_ids = class_ids[perm]
    key_ids   = key_ids[perm]

    # ------------------------------------------------------
    # Bind（clean / anti）
    # ------------------------------------------------------
    z_list = []
    event_list = []

    for cls_t, key_t in zip(class_ids, key_ids):

        # ------------------------------------------------------
        # ★ Anti-noise（KVDataset 完全一致版）
        # ------------------------------------------------------
        if cls_t == -1:

            # ランダム class/key
            anti_cid = rng.randint(num_classes)
            anti_kid = rng.randint(num_keys)

            # ★ シャッフル済みテーブルから取得
            key_a = key_ep[anti_kid]
            val_a = mu_ep[anti_cid]

            # outer → proj
            outer = np.outer(key_a, val_a).astype(np.float32)
            flat = outer.reshape(-1)
            clean_a = flat @ W
            clean_a /= (np.linalg.norm(clean_a) + 1e-8)

            # clean を反転
            anti_vec = -clean_a

            # pure noise
            noise = np.stack([unit_sphere(d_g, rng) for _ in range(batch_size)], axis=0)

            r = 0.0  # pure noise
            z_t = r * np.tile(anti_vec, (batch_size, 1)) + (1 - r) * noise
            z_t /= (np.linalg.norm(z_t, axis=1, keepdims=True) + 1e-8)

            z_list.append(torch.from_numpy(z_t).float().to(device))
            event_list.append(("bind_noise", -1))
            continue

        # ------------------------------------------------------
        # Clean（元のままだが mu_ep/key_ep を使用）
        # ------------------------------------------------------
        key = key_ep[key_t]
        val = mu_ep[cls_t]

        outer = np.outer(key, val).astype(np.float32)
        flat = outer.reshape(-1)
        clean_t = flat @ W
        clean_t /= (np.linalg.norm(clean_t) + 1e-8)

        clean_tile = np.tile(clean_t, (batch_size, 1))
        noise = np.stack([unit_sphere(d_g, rng) for _ in range(batch_size)], axis=0)

        r = 0.3   # ★ Clean 部分は「そのまま」にする指示
        z_t = r * clean_tile + (1 - r) * noise
        z_t /= (np.linalg.norm(z_t, axis=1, keepdims=True) + 1e-8)

        z_list.append(torch.from_numpy(z_t).float().to(device))
        event_list.append(("bind", int(cls_t)))

    # ------------------------------------------------------
    # target clean（元のままだが mu_ep/key_ep を使用）
    # ------------------------------------------------------
    target = rng.randint(num_items)

    key = key_ep[key_ids_base[target]]
    val = mu_ep[class_ids_base[target]]

    outer = np.outer(key, val).astype(np.float32)
    flat = outer.reshape(-1)
    clean_target_np = flat @ W
    clean_target_np /= (np.linalg.norm(clean_target_np) + 1e-8)

    clean_target = np.tile(clean_target_np, (batch_size, 1))
    clean_target = torch.from_numpy(clean_target).float().to(device)

    # ------------------------------------------------------
    # Query（元のままだが key_ep を使用）
    # ------------------------------------------------------
    key_q = np.tile(key_ep[key_ids_base[target]], (batch_size, 1))
    noise_q = np.stack([unit_sphere(d_g, rng) for _ in range(batch_size)], axis=0)

    z_q = beta * key_q + (1 - beta) * noise_q
    z_q /= (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)

    z_list.append(torch.from_numpy(z_q).float().to(device))
    event_list.append(("query", int(class_ids_base[target])))

    # ------------------------------------------------------
    # Stack（元のまま）
    # ------------------------------------------------------
    z_seq = torch.stack(z_list, dim=0)

    return z_seq, event_list, clean_target

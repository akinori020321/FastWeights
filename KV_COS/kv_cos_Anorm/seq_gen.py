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
    query_noise_std: float = 0.0,   # 今は未使用（KVDataset と揃えるなら 0.0 前提）
    device: str = "cpu",
    seed: int = 123,

    mu_value: np.ndarray | None = None,
    key_proto: np.ndarray | None = None,

    # ★ 追加：Wait ステップ用
    num_wait: int = 0,
    wait_vec: np.ndarray | None = None,
):
    """
    方向復元タスクの KV シーケンス生成

    KVDataset(sample_batch) と同じロジック：
      - Clean は duplicate 回繰り返し（class_ids_clean, key_ids_clean）
      - Anti は Clean に登場した class も key も一切使用しない組み合わせ
        （ただし現状 num_anti=0 で無効化）
      - Query は Clean で複数回登場した class から選択
      - outer product + 固定ランダム射影 W (seed=999) で Bind 入力を作る

    さらに:
      - Bind のあとに num_wait ステップだけ wait_vec を挿入（あれば）
      - event_list を返す: "bind" / "bind_noise" / "wait" / "query"
      - ★ clean_target は「Query で参照される clean パターン」（outer(key, val)→W）
    """
    assert mu_value is not None and key_proto is not None, \
        "mu_value と key_proto を必ず渡してください。"

    rng = np.random.RandomState(seed)
    d = d_g

    num_classes = mu_value.shape[0]
    num_keys = key_proto.shape[0]

    assert T_bind % duplicate == 0
    num_items = T_bind // duplicate

    # ------------------------------------------------------
    # outer product 用ランダム射影行列 W（KVDataset と同じ）
    # ------------------------------------------------------
    proj_rng = np.random.RandomState(999)
    W = proj_rng.randn(d * d, d).astype(np.float32)
    W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

    # ------------------------------------------------------
    # ★ このバッチ専用の key / value テーブルをシャッフル
    # ------------------------------------------------------
    perm_c = rng.permutation(num_classes)
    perm_k = rng.permutation(num_keys)
    mu_ep  = mu_value[perm_c]      # (num_classes, d_g)
    key_ep = key_proto[perm_k]     # (num_keys, d_g)

    # ------------------------------------------------------
    # 1. Clean（duplicate 回登場）
    # ------------------------------------------------------
    class_ids_base = np.arange(num_items) % num_classes
    key_ids_base   = np.arange(num_items) % num_keys

    class_ids_clean = np.repeat(class_ids_base, duplicate)   # (T_bind,)
    key_ids_clean   = np.repeat(key_ids_base,   duplicate)

    # Clean に出てくる class / key
    clean_classes = set(class_ids_clean.tolist())
    clean_keys    = set(key_ids_clean.tolist())

    # Clean ペア（旧仕様残し）
    clean_pairs = set(zip(class_ids_clean, key_ids_clean))

    # ------------------------------------------------------
    # 2. Anti（Clean の class も key も一切使用しない）
    #    KVDataset の現行コードでは num_anti = 0 に固定
    # ------------------------------------------------------
    num_anti = 0  # KVDataset の実装に合わせる

    anti_classes = [c for c in range(num_classes) if c not in clean_classes]
    anti_keys    = [k for k in range(num_keys)    if k not in clean_keys]

    if num_anti > 0:
        if len(anti_classes) == 0 or len(anti_keys) == 0:
            raise ValueError("Anti に使える class または key がありません。")

        all_pairs_anti = [(c, k) for c in anti_classes for k in anti_keys]
        if num_anti > len(all_pairs_anti):
            raise ValueError("num_anti が候補より大きい（class/key 丸ごと除外のため）。")

        pairs_idx = rng.choice(len(all_pairs_anti), size=num_anti, replace=False)
        pairs = [all_pairs_anti[i] for i in pairs_idx]

        class_ids_anti = np.array([p[0] for p in pairs], dtype=int)
        key_ids_anti   = np.array([p[1] for p in pairs], dtype=int)
    else:
        class_ids_anti = np.array([], dtype=int)
        key_ids_anti   = np.array([], dtype=int)

    # ------------------------------------------------------
    # 3. Merge → Shuffle
    # ------------------------------------------------------
    class_ids = np.concatenate([class_ids_clean, class_ids_anti], axis=0)
    key_ids   = np.concatenate([key_ids_clean,   key_ids_anti],   axis=0)

    perm = rng.permutation(len(class_ids))
    class_ids = class_ids[perm]
    key_ids   = key_ids[perm]

    z_list = []
    event_list = []

    B = batch_size
    torch_device = device

    # ------------------------------------------------------
    # 4. Bind sequence（Clean と Anti が完全に disjoint）
    # ------------------------------------------------------
    for cls_t, key_t in zip(class_ids, key_ids):

        # このバッチ専用にシャッフルされた key_ep / mu_ep を使用
        key = key_ep[key_t]
        val = mu_ep[cls_t]

        outer = np.outer(key, val).astype(np.float32)  # (d, d)
        flat = outer.reshape(-1)                       # (d*d,)
        clean_vec = flat @ W                           # (d,)
        clean_vec /= (np.linalg.norm(clean_vec) + 1e-8)

        # Clean or Anti の判定 （KVDataset と同じロジック）
        if (cls_t in clean_classes) and (key_t in clean_keys):
            # ----- Clean -----
            eps = unit_sphere(d, rng)
            r = bind_noise_std
            mixed = r * clean_vec + (1 - r) * eps
            mixed /= (np.linalg.norm(mixed) + 1e-8)

            kind = "bind"
            cid  = int(cls_t)

        else:
            # ----- Anti -----
            eps = unit_sphere(d, rng)
            r = 0.0
            mixed = r * clean_vec + (1 - r) * eps
            mixed /= (np.linalg.norm(mixed) + 1e-8)

            # 「純ノイズ」扱いにしたいので class_id は -1 にして bind_noise に
            kind = "bind_noise"
            cid  = -1

        mixed_batch = np.stack([mixed for _ in range(B)], axis=0)
        mixed_batch /= (np.linalg.norm(mixed_batch, axis=1, keepdims=True) + 1e-8)

        z_list.append(torch.from_numpy(mixed_batch).float().to(torch_device))
        event_list.append((kind, cid))

    # ------------------------------------------------------
    # 5. Query ターゲットの決定
    #    「Clean で複数回登場した class から選択」
    # ------------------------------------------------------
    unique, counts = np.unique(class_ids_clean, return_counts=True)
    multi_classes = [c for c, cnt in zip(unique, counts) if cnt > 1]

    if len(multi_classes) == 0:
        raise ValueError("複数回登場するクラスがありません（duplicate の設定を確認）。")

    target_c = int(rng.choice(multi_classes))

    # このクラスに対応する key_ids_base から target_k を選ぶ
    valid_keys = key_ids_base[class_ids_base == target_c]
    target_k = int(rng.choice(valid_keys))

    # ------------------------------------------------------
    # 6. clean_target（方向復元の「正解ベクトル」＝ clean パターン）
    # ------------------------------------------------------
    key = key_ep[target_k]
    val = mu_ep[target_c]  # (d_g,)

    outer_q = np.outer(key, val).astype(np.float32)   # (d_g, d_g)
    flat_q  = outer_q.reshape(-1)                     # (d_g * d_g,)
    target_clean = flat_q @ W                         # (d_g,)
    target_clean /= (np.linalg.norm(target_clean) + 1e-8)

    clean_target_np = np.tile(target_clean, (B, 1))   # (B, d_g)
    clean_target = torch.from_numpy(clean_target_np).float().to(torch_device)

    # ------------------------------------------------------
    # 7. Wait ステップ（あれば）
    # ------------------------------------------------------
    if num_wait > 0:
        if wait_vec is None:
            raise ValueError("num_wait > 0 なのに wait_vec が None です。")

        w = wait_vec.astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-8)

        wait_batch = np.stack([w for _ in range(B)], axis=0)
        wait_batch /= (np.linalg.norm(wait_batch, axis=1, keepdims=True) + 1e-8)

        wait_tensor = torch.from_numpy(wait_batch).float().to(torch_device)

        for _ in range(num_wait):
            z_list.append(wait_tensor)
            event_list.append(("wait", -1))

    # ------------------------------------------------------
    # 8. Query ベクトル
    # ------------------------------------------------------
    key_q = np.tile(key_ep[target_k], (B, 1))
    noise_q = np.stack([unit_sphere(d, rng) for _ in range(B)], axis=0)

    z_q = beta * key_q + (1 - beta) * noise_q
    z_q /= (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)

    z_list.append(torch.from_numpy(z_q).float().to(torch_device))
    event_list.append(("query", target_c))

    # ------------------------------------------------------
    # 9. stack & return
    # ------------------------------------------------------
    z_seq = torch.stack(z_list, dim=0)  # (T_total, B, d_g)

    return z_seq, event_list, clean_target

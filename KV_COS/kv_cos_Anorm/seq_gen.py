# -*- coding: utf-8 -*-
"""
seq_gen.py  — outer product + random projection + anti-noise 対応版（修正版）
"""

from __future__ import annotations
import numpy as np
import torch
import os          # ★追加
import csv         # ★追加


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

    # ★追加：clean 保存先（毎回上書き）
    clean_csv_path: str | None = None,
):
    """
    方向復元タスクの KV シーケンス生成
    （中略：あなたの元コメントそのまま）
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

    clean_classes = set(class_ids_clean.tolist())
    clean_keys    = set(key_ids_clean.tolist())

    clean_pairs = set(zip(class_ids_clean, key_ids_clean))

    # ------------------------------------------------------
    # 2. Anti（現状 num_anti=0）
    # ------------------------------------------------------
    num_anti = 0

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

    # ★追加：各ステップの clean ベクトル（d_g）を保持
    clean_step_list = []

    B = batch_size
    torch_device = device

    # ------------------------------------------------------
    # 4. Bind sequence
    # ------------------------------------------------------
    for cls_t, key_t in zip(class_ids, key_ids):

        key = key_ep[key_t]
        val = mu_ep[cls_t]

        outer = np.outer(key, val).astype(np.float32)
        flat = outer.reshape(-1)
        clean_vec = flat @ W
        clean_vec /= (np.linalg.norm(clean_vec) + 1e-8)

        if (cls_t in clean_classes) and (key_t in clean_keys):
            eps = unit_sphere(d, rng)
            r = bind_noise_std
            mixed = r * clean_vec + (1 - r) * eps
            mixed /= (np.linalg.norm(mixed) + 1e-8)

            kind = "bind"
            cid  = int(cls_t)

        else:
            eps = unit_sphere(d, rng)
            r = 0.0
            mixed = r * clean_vec + (1 - r) * eps
            mixed /= (np.linalg.norm(mixed) + 1e-8)

            kind = "bind_noise"
            cid  = -1

        mixed_batch = np.stack([mixed for _ in range(B)], axis=0)
        mixed_batch /= (np.linalg.norm(mixed_batch, axis=1, keepdims=True) + 1e-8)

        z_list.append(torch.from_numpy(mixed_batch).float().to(torch_device))
        event_list.append((kind, cid))

        # ★追加：Bind ステップの clean を保存（ノイズ混入前）
        clean_step_list.append(clean_vec.copy())

    # ------------------------------------------------------
    # 5. Query ターゲット決定
    # ------------------------------------------------------
    unique, counts = np.unique(class_ids_clean, return_counts=True)
    multi_classes = [c for c, cnt in zip(unique, counts) if cnt > 1]

    if len(multi_classes) == 0:
        raise ValueError("複数回登場するクラスがありません（duplicate の設定を確認）。")

    target_c = int(rng.choice(multi_classes))

    valid_keys = key_ids_base[class_ids_base == target_c]
    target_k = int(rng.choice(valid_keys))

    # ------------------------------------------------------
    # 6. clean_target（正解ベクトル）
    # ------------------------------------------------------
    key = key_ep[target_k]
    val = mu_ep[target_c]

    outer_q = np.outer(key, val).astype(np.float32)
    flat_q  = outer_q.reshape(-1)
    target_clean = flat_q @ W
    target_clean /= (np.linalg.norm(target_clean) + 1e-8)

    clean_target_np = np.tile(target_clean, (B, 1))
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

            # ★追加：Wait は参照用に wait_vec を入れておく（cos 計算で 0 除算を避ける）
            clean_step_list.append(w.copy())

    # ------------------------------------------------------
    # 8. Query ベクトル（入力）
    # ------------------------------------------------------
    key_q = np.tile(key_ep[target_k], (B, 1))
    noise_q = np.stack([unit_sphere(d, rng) for _ in range(B)], axis=0)

    z_q = beta * key_q + (1 - beta) * noise_q
    z_q /= (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)

    z_list.append(torch.from_numpy(z_q).float().to(torch_device))
    event_list.append(("query", target_c))

    # ★追加：Query ステップの clean（正解）を入れる
    clean_step_list.append(target_clean.copy())

    # ------------------------------------------------------
    # 9. stack & return
    # ------------------------------------------------------
    z_seq = torch.stack(z_list, dim=0)  # (T_total, B, d_g)

    # ★追加：clean CSV を毎回上書き保存
    if clean_csv_path is not None:
        out_dir = os.path.dirname(clean_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(clean_csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["step", "kind", "class_id"] + [f"clean[{i}]" for i in range(d_g)])

            for t, ((kind, cid), vec) in enumerate(zip(event_list, clean_step_list)):
                wcsv.writerow([t, kind, int(cid)] + vec.astype(float).tolist())

    return z_seq, event_list, clean_target

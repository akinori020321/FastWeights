# -*- coding: utf-8 -*-
"""
seq_gen.py
----------------------------------------
A-dynamics 実験用の入力シーケンス生成ユーティリティ。

- make_keyvalue_sequence:
    学習時と同じ KVTask の Bind 仕様に従い、
    key → value ペアを重複なしで生成する。
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


# ======================================================
# Key–Value sequence generator
# ======================================================

def make_keyvalue_sequence(
    mu_value: np.ndarray | torch.Tensor,        # (num_classes, d_g)
    key_proto: np.ndarray | torch.Tensor,       # (num_key_proto, d_g)
    batch_size: int,
    class_ids: list[int] | None,                # value側のクラス列
    device: str,
    seed: int = 123,
):
    """
    KVTask の Bind 仕様に合わせたシーケンス生成：

    - class_ids が None の場合は自動生成（保険）
    - key: key_proto から T_bind 本を重複なしで選ぶ
    - 出力: key → value → key → value → ... の時系列
    """

    # ------------------------------
    # torch 化
    # ------------------------------
    if isinstance(mu_value, np.ndarray):
        mu_value = torch.from_numpy(mu_value).float().to(device)
    if isinstance(key_proto, np.ndarray):
        key_proto = torch.from_numpy(key_proto).float().to(device)

    num_key_proto = key_proto.size(0)
    d_g = key_proto.size(1)

    rng = np.random.RandomState(seed)

    # ------------------------------
    # class_ids が None のとき自動生成
    # ------------------------------
    if class_ids is None:
        T_bind = min(5, mu_value.size(0))
        class_ids = rng.randint(0, mu_value.size(0), size=T_bind).tolist()
    else:
        T_bind = len(class_ids)

    # ------------------------------
    # key を重複なしで選ぶ
    # ------------------------------
    if T_bind <= num_key_proto:
        key_indices = rng.choice(num_key_proto, size=T_bind, replace=False)
    else:
        base = rng.permutation(num_key_proto)
        rest = rng.choice(num_key_proto, size=T_bind - num_key_proto, replace=True)
        key_indices = np.concatenate([base, rest])

    # ------------------------------
    # key_proto → (T_bind, B, d_g)
    # value     → (T_bind, B, d_g)
    # ------------------------------
    z_list = []
    event_list = []

    for i, cid in enumerate(class_ids):

        # key
        key_vec = key_proto[key_indices[i]].unsqueeze(0).expand(batch_size, -1)
        z_list.append(key_vec)
        event_list.append(("key", cid))

        # value
        value_vec = mu_value[cid].unsqueeze(0).expand(batch_size, -1)
        z_list.append(value_vec)
        event_list.append(("value", cid))

    # ------------------------------
    # Query: Bind 内の任意の key を使う
    # ------------------------------
    q_bind_idx = rng.randint(T_bind)
    query_key = key_proto[key_indices[q_bind_idx]].unsqueeze(0).expand(batch_size, -1)
    z_list.append(query_key)
    event_list.append(("query", q_bind_idx))

    z_seq = torch.stack(z_list, dim=0)

    return z_seq, event_list

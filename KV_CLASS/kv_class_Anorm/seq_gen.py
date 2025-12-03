# -*- coding: utf-8 -*-
"""
seq_gen.py
----------------------------------------
A-dynamics 実験用の入力シーケンス生成ユーティリティ。

- Single-clean:
    同じ clean ベクトルを T_long ステップ繰り返し入力するシーケンス

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
    """
    単位球面上からランダムベクトルを1本サンプル。
    """
    v = rng.randn(d).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v


# # ======================================================
# # Single-clean（そのまま）
# # ======================================================

# def make_single_clean_sequence(
#     d_g: int,
#     batch_size: int,
#     T_long: int,
#     device: str,
#     seed: int = 123,
# ):
#     """
#     同じ clean (=value) を T_long 繰り返すシーケンス。
#     """
#     rng = np.random.RandomState(seed)
#     clean_np = unit_sphere(d_g, rng)
#     clean_vec = torch.from_numpy(clean_np).float().to(device)

#     z_step = clean_vec.unsqueeze(0).expand(batch_size, -1)
#     z_seq = z_step.unsqueeze(0).repeat(T_long, 1, 1).contiguous()

#     return z_seq, clean_vec



def make_keyvalue_sequence(
    mu_value: np.ndarray | torch.Tensor,        # (num_classes, d_g)
    key_proto: np.ndarray | torch.Tensor,       # (num_key_proto, d_g)
    batch_size: int,
    class_ids: list[int],                       # value側のクラス列（重複なし）
    device: str,
    seed: int = 123,
):
    """
    KVTask の Bind 仕様に合わせたシーケンス生成：

    - value: μ[class_ids[j]]
    - key:   key_proto から T_bind 本を重複なしで選ぶ
    - 出力: key → value → key → value → ... の時系列
    """

    # ------------------------------
    # torch 化
    # ------------------------------
    if isinstance(mu_value, np.ndarray):
        mu_value = torch.from_numpy(mu_value).float().to(device)
    if isinstance(key_proto, np.ndarray):
        key_proto = torch.from_numpy(key_proto).float().to(device)

    T_bind = len(class_ids)
    num_key_proto = key_proto.size(0)

    rng = np.random.RandomState(seed)

    # ------------------------------
    # ★ key は重複なしで T_bind 個選ぶ
    # ------------------------------
    if T_bind <= num_key_proto:
        key_indices = rng.choice(num_key_proto, size=T_bind, replace=False)
    else:
        base = rng.permutation(num_key_proto)
        rest = rng.choice(num_key_proto, size=T_bind - num_key_proto, replace=True)
        key_indices = np.concatenate([base, rest])

    # ------------------------------
    # 時系列作成: key → value
    # ------------------------------
    z_list = []
    event_list = []   # ("key", None) / ("value", cid)

    for i, cid in enumerate(class_ids):
        key_vec = key_proto[key_indices[i]].unsqueeze(0).expand(batch_size, -1)
        value_vec = mu_value[cid].unsqueeze(0).expand(batch_size, -1)

        z_list.append(key_vec)
        event_list.append(("key", cid))

        z_list.append(value_vec)
        event_list.append(("value", cid))
    
    # ------------------------------
    # ★ 追加: Query 用 key を1本サンプル（ランダム）
    #      ランダムに選ぶのは「Bind ステップ j」
    # ------------------------------
    # 0〜T_bind-1 の中からランダムに 1 つ選ぶ
    q_bind_idx = rng.randint(T_bind)
    # そのときに使った key をそのまま Query に使う
    query_key = key_proto[key_indices[q_bind_idx]].unsqueeze(0).expand(batch_size, -1)

    z_list.append(query_key)
    event_list.append(("query", q_bind_idx)) 

    # (T_total, B, d_g)
    z_seq = torch.stack(z_list, dim=0)

    return z_seq, event_list

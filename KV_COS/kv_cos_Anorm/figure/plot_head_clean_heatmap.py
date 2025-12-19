#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_head_clean_heatmap.py  (Class-wise view)
-------------------------------------------------------------
- results_A_kv/ の
    H_kv_*.csv（各時刻の h 平均）
    Clean_kv_*.csv（各時刻の clean ベクトル）
  を読み込み、
- checkpoints/*.pt から head 重みを読み込んで、
- cos( head(h_t), clean_{t'} ) を計算し、

★クラスごとの表示（見やすさ優先）:
  (A) T×Class heatmap:
      score[t, c] = max_{t' in class c} cos(head(h_t), clean_{t'})
  (B) Query row の class bar plot（数値がわかる）
  (C) margin over time（「上がってる」がわかる）
      margin[t] = score[t, y(t)] - max_{c≠y(t)} score[t, c]

★修正点（今回 / A）:
  - T×Class 図において，各時刻 t の class_id と同じクラス列のセル (t, class[t]) を枠で囲む
  - 枠色は bvq 初出順ルール（PCA / h_heatmap(all) と統一）
    kind in ("bind","value","query") かつ class_id>=0 を t順に走査し、
    初出 class_id に順番に色を割り当てる。
  - ★query 行の追加マーカー（青線）は描かない（他と同じく余計な装飾なし）

出力:
  figure/plots/clean_heatmap/<core>/ にモデルごとにPNG保存（最新ペアを採用）

実行:
  python3 figure/plot_head_clean_heatmap.py
  (引数なしでOK)
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# ★ head を直書き（fw_kv/models/head.py 相当）
# ============================================================
class OutputHead(nn.Module):
    def __init__(self, d_h: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(d_h, out_dim)

    def forward(self, h):
        return self.fc(h)


# ============================================================
# パス（スクリプト位置基準）
#   figure/plot_head_clean_heatmap.py
#   checkpoints/*.pt
#   results_A_kv/*.csv
#   figure/plots/clean_heatmap/*.png
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_DEFAULT = os.path.join(THIS_DIR, "..", "results_A_kv")
CKPT_DIR_DEFAULT = os.path.join(THIS_DIR, "..", "checkpoints")
OUT_ROOT_DEFAULT = os.path.join(THIS_DIR, "plots", "clean_heatmap")


# ============================================================
# 共通: cosine
# ============================================================
def row_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def cosine_pred_clean(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    Pn = row_normalize(P)
    Cn = row_normalize(C)
    return Pn @ Cn.T


# ============================================================
# CSV ロード
# ============================================================
def load_h_csv(path: str):
    df = pd.read_csv(path)
    h_cols = [c for c in df.columns if c.startswith("h[")]
    H = df[h_cols].values.astype(np.float32)
    kinds = df["kind"].astype(str).tolist()
    class_ids = df["class_id"].astype(int).tolist()
    steps = df["step"].astype(int).tolist() if "step" in df.columns else list(range(len(df)))
    return H, kinds, class_ids, steps


def load_clean_csv(path: str):
    df = pd.read_csv(path)
    c_cols = [c for c in df.columns if c.startswith("clean[")]
    C = df[c_cols].values.astype(np.float32)
    kinds = df["kind"].astype(str).tolist()
    class_ids = df["class_id"].astype(int).tolist()
    steps = df["step"].astype(int).tolist() if "step" in df.columns else list(range(len(df)))
    return C, kinds, class_ids, steps


# ============================================================
# ファイル名パース（H/Clean 共通）
# ============================================================
@dataclass(frozen=True)
class RunID:
    core: str
    S: int
    eta: str
    lam: str
    seed: str
    noise: str | None = None


def parse_kv_csv_id(fname: str) -> RunID | None:
    base = os.path.basename(fname)
    pat = (
        r"^(?:H_kv_|Clean_kv_|clean_kv_|CLEAN_kv_)(fw|tanh|rnn)_S([0-9]+)"
        r"(?:_noise([0-9]+))?"
        r"_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)\.csv$"
    )
    m = re.match(pat, base)
    if m is None:
        return None
    core = m.group(1)
    S = int(m.group(2))
    noise = m.group(3)
    eta = m.group(4)
    lam = m.group(5)
    seed = m.group(6)
    return RunID(core=core, S=S, eta=eta, lam=lam, seed=seed, noise=noise)


def parse_ckpt_id(fname: str) -> RunID | None:
    base = os.path.basename(fname)
    pat = r"^kv_(fw|tanh|rnn)_S([0-9]+).*_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)\.pt$"
    m = re.match(pat, base)
    if m is None:
        return None
    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)
    return RunID(core=core, S=S, eta=eta, lam=lam, seed=seed, noise=None)


# ============================================================
# head_state ロード（互換）
# ============================================================
def load_head_from_ckpt(ckpt_path: str, d_h: int, out_dim: int, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "head_state" not in state:
        raise KeyError("Checkpoint does not contain 'head_state'.")

    raw_head = state["head_state"]

    if ("weight" in raw_head) and ("bias" in raw_head):
        head_state = {"fc.weight": raw_head["weight"], "fc.bias": raw_head["bias"]}
    else:
        head_state = raw_head

    head = OutputHead(d_h, out_dim).to(device)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    return head


# ============================================================
# ckpt 自動選択
# ============================================================
def pick_ckpt_for_run(ckpt_dir: str, rid: RunID) -> str:
    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not ckpt_list:
        raise FileNotFoundError(f"*.pt not found in: {ckpt_dir}")

    ckpt_map = {}
    for p in ckpt_list:
        cid = parse_ckpt_id(p)
        if cid is not None:
            ckpt_map[cid] = p

    for cid, p in ckpt_map.items():
        if (cid.core == rid.core and cid.S == rid.S and cid.eta == rid.eta and cid.lam == rid.lam and cid.seed == rid.seed):
            return p

    core_ckpts = [p for cid, p in ckpt_map.items() if cid.core == rid.core]
    if core_ckpts:
        return max(core_ckpts, key=os.path.getmtime)

    return max(ckpt_list, key=os.path.getmtime)


# ============================================================
# Class-wise 集約
# ============================================================
def build_class_groups(col_kinds, col_cids):
    """
    clean 側の (kind, class_id) からクラス列を作る。
    wait(-1) は除外。kind も一応 bind/value/query 以外は除外。
    """
    idx_by_class = {}
    for j, (k, cid) in enumerate(zip(col_kinds, col_cids)):
        cid = int(cid)
        if cid < 0:
            continue
        if k not in ("bind", "value", "query"):
            continue
        idx_by_class.setdefault(cid, []).append(j)

    classes = sorted(idx_by_class.keys())
    return classes, idx_by_class


def collapse_M_to_time_class(M: np.ndarray, classes, idx_by_class, reduce: str = "max"):
    """
    M: (T_row, T_col)
    score[t,c] = max or mean over t' belonging to class c
    """
    T_row = M.shape[0]
    K = len(classes)
    S = np.full((T_row, K), np.nan, dtype=np.float32)

    for k, cid in enumerate(classes):
        idx = idx_by_class[cid]
        sub = M[:, idx]  # (T_row, n_idx)
        if reduce == "mean":
            S[:, k] = np.mean(sub, axis=1)
        else:
            S[:, k] = np.max(sub, axis=1)
    return S


# ============================================================
# ★ class 枠線色 & cid->color（参照コードと同じ）
# ============================================================
CLASS_OUTLINE_COLORS = [
    "#ff4fa3",  # vivid pink
    "#ff9f1a",  # vivid orange
    "#4dd9ff",  # bright cyan-blue
    "#6dff6d",  # bright green
]


def build_cid2color_bvq(kinds, class_ids):
    """
    kind in ("bind","value","query") かつ class_id>=0 を t順に走査し、
    初出 class_id に順番に色を割り当てる。
    戻り値: (cid2color, ordered_cids)
    """
    cid2color = {}
    ordered_cids = []
    color_idx = 0

    for k, cid in zip(kinds, class_ids):
        cid = int(cid)
        if cid < 0:
            continue
        if k not in ("bind", "value", "query"):
            continue
        if cid in cid2color:
            continue

        cid2color[cid] = CLASS_OUTLINE_COLORS[color_idx % len(CLASS_OUTLINE_COLORS)]
        ordered_cids.append(cid)
        color_idx += 1

    return cid2color, ordered_cids


# ============================================================
# 描画（3種）
# ============================================================
CORE_NAME = {"fw": "Ba-FW", "tanh": "SC-FW", "rnn": "RNN-LN"}


def plot_time_class_heatmap(S_tc: np.ndarray, classes, row_kinds, row_cids, title: str, out_path: str):
    """
    T×Class heatmap に対して、
    各時刻 t の class_id と同じクラス列のセル (t, class[t]) を枠で囲む。
    （★query 行の追加マーカー等は一切描かない）
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    im = ax.imshow(S_tc, aspect="auto", vmin=-1.0, vmax=1.0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- class -> col index（横軸は class）
    cid_to_col = {cid: k for k, cid in enumerate(classes)}

    # ---- class_to_steps（縦軸は t）
    class_to_steps = defaultdict(list)
    for t, cid in enumerate(row_cids):
        cid = int(cid)
        if cid < 0:
            continue
        class_to_steps[cid].append(t)

    # ---- 出現順の cid2color（bvq初出順）
    cid2color, ordered = build_cid2color_bvq(row_kinds, row_cids)

    # ---- 枠線描画（T×Class 版）
    for cid in ordered:
        if cid not in cid_to_col:
            continue
        color = cid2color[cid]
        col = cid_to_col[cid]
        steps = class_to_steps.get(cid, [])
        for t in steps:
            ax.add_patch(
                patches.Rectangle(
                    (col - 0.5, t - 0.5), 1, 1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                    zorder=10
                )
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("class id")
    ax.set_ylabel("t (head(h) step)")

    # x軸：クラスID（多いと潰れるので間引き）
    K = len(classes)
    if K <= 30:
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([str(c) for c in classes], rotation=0)
    else:
        step = max(1, K // 20)
        ticks = list(range(0, K, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(classes[i]) for i in ticks], rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[SAVE] {out_path}")
    out_path_eps = os.path.splitext(out_path)[0] + ".eps"
    plt.savefig(out_path_eps, format="eps")
    print(f"[SAVE] {out_path_eps}")
    plt.close()


def plot_query_bar(S_tc: np.ndarray, classes, row_kinds, title: str, out_path: str):
    if "query" not in row_kinds:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    q = row_kinds.index("query")
    scores = S_tc[q].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))
    x = np.arange(len(classes))
    ax.bar(x, scores)

    # 数値ラベル（見える範囲だけ）
    for i, v in enumerate(scores):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title + "  [Query row scores]", fontsize=11)
    ax.set_xlabel("class id")
    ax.set_ylabel("max cos over clean steps in class")

    if len(classes) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in classes], rotation=0)
    else:
        step = max(1, len(classes) // 20)
        ticks = list(range(0, len(classes), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(classes[i]) for i in ticks], rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")
    out_path_eps = os.path.splitext(out_path)[0] + ".eps"
    plt.savefig(out_path_eps, format="eps")
    print(f"[SAVE] {out_path_eps}")


def plot_margin_curve(S_tc: np.ndarray, classes, row_cids, row_kinds, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cid_to_k = {cid: k for k, cid in enumerate(classes)}
    T = S_tc.shape[0]
    t = np.arange(T)

    margin = np.full(T, np.nan, dtype=np.float32)
    correct = np.full(T, np.nan, dtype=np.float32)
    other = np.full(T, np.nan, dtype=np.float32)

    for i in range(T):
        y = int(row_cids[i])
        if y < 0:
            continue
        if y not in cid_to_k:
            continue
        ky = cid_to_k[y]
        s = S_tc[i].copy()
        cy = s[ky]
        s[ky] = -np.inf
        oy = np.max(s)
        correct[i] = cy
        other[i] = oy
        margin[i] = cy - oy

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    ax.plot(t, correct, marker="o", linewidth=1.2, markersize=3.5, label="correct-class score")
    ax.plot(t, other, marker="o", linewidth=1.2, markersize=3.5, label="best-other score")
    ax.plot(t, margin, marker="o", linewidth=1.2, markersize=3.5, label="margin (correct - other)")

    # ★ここは heatmap と違って、縦線は「他と同じく余計な装飾なし」にしたいなら消してOK
    # if "query" in row_kinds:
    #     q = row_kinds.index("query")
    #     ax.axvline(q, linestyle="--", linewidth=1.2)

    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title + "  [margin over time]", fontsize=11)
    ax.set_xlabel("t (head(h) step)")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")
    out_path_eps = os.path.splitext(out_path)[0] + ".eps"
    plt.savefig(out_path_eps, format="eps")
    print(f"[SAVE] {out_path_eps}")


# ============================================================
# results から「coreごとの最新ペア」を拾う
# ============================================================
def build_pair_maps(results_dir: str):
    h_list = glob.glob(os.path.join(results_dir, "H_kv_*.csv"))
    c_list = []
    c_list += glob.glob(os.path.join(results_dir, "Clean_kv_*.csv"))
    c_list += glob.glob(os.path.join(results_dir, "clean_kv_*.csv"))
    c_list += glob.glob(os.path.join(results_dir, "CLEAN_kv_*.csv"))

    h_map = {}
    for p in h_list:
        rid = parse_kv_csv_id(os.path.basename(p))
        if rid is not None:
            h_map[rid] = p

    c_map = {}
    for p in c_list:
        rid = parse_kv_csv_id(os.path.basename(p))
        if rid is not None:
            c_map[rid] = p

    return h_map, c_map


def pick_latest_run_per_core(h_map: dict, c_map: dict):
    common = list(set(h_map.keys()) & set(c_map.keys()))
    by_core = {"fw": [], "tanh": [], "rnn": []}
    for rid in common:
        if rid.core in by_core:
            by_core[rid.core].append(rid)

    latest = {}
    for core, rids in by_core.items():
        if not rids:
            continue
        rids_sorted = sorted(
            rids,
            key=lambda rid: max(os.path.getmtime(h_map[rid]), os.path.getmtime(c_map[rid])),
            reverse=True
        )
        latest[core] = rids_sorted[0]
    return latest


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--ckpt_dir", type=str, default=CKPT_DIR_DEFAULT)
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    ap.add_argument("--reduce", type=str, default="max", choices=["max", "mean"])
    args = ap.parse_args()

    # device 自動
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] --device cuda だが cuda が無いので cpu にします")
            device = "cpu"

    os.makedirs(OUT_ROOT_DEFAULT, exist_ok=True)

    h_map, c_map = build_pair_maps(args.results_dir)
    if not h_map:
        raise FileNotFoundError(f"H_kv_*.csv not found in: {args.results_dir}")
    if not c_map:
        raise FileNotFoundError(f"Clean_kv_*.csv not found in: {args.results_dir}")

    latest = pick_latest_run_per_core(h_map, c_map)
    if not latest:
        raise RuntimeError("H_kv と Clean_kv の RunID が一致するペアが見つかりません。ファイル名規則を確認してください。")

    for core in ["fw", "tanh", "rnn"]:
        if core not in latest:
            continue

        rid = latest[core]
        h_csv = h_map[rid]
        clean_csv = c_map[rid]
        ckpt = pick_ckpt_for_run(args.ckpt_dir, rid)

        print("===========================================")
        print(f"[RUN] core={core}")
        print(f"  H     : {h_csv}")
        print(f"  Clean : {clean_csv}")
        print(f"  ckpt  : {ckpt}")
        print("===========================================")

        out_dir = os.path.join(OUT_ROOT_DEFAULT, core)
        os.makedirs(out_dir, exist_ok=True)

        H, h_kinds, h_cids, _ = load_h_csv(h_csv)
        C, c_kinds, c_cids, _ = load_clean_csv(clean_csv)

        T = min(H.shape[0], C.shape[0])
        if H.shape[0] != C.shape[0]:
            print(f"[WARN] length mismatch: H={H.shape[0]} rows, Clean={C.shape[0]} rows. Using T={T}.")
            H, h_kinds, h_cids = H[:T], h_kinds[:T], h_cids[:T]
            C, c_kinds, c_cids = C[:T], c_kinds[:T], c_cids[:T]

        d_h = H.shape[1]
        d_out = C.shape[1]

        head = load_head_from_ckpt(ckpt, d_h=d_h, out_dim=d_out, device=device)

        with torch.no_grad():
            pred = head(torch.from_numpy(H).to(device)).detach().cpu().numpy()

        M = cosine_pred_clean(pred, C)  # (T, T)

        classes, idx_by_class = build_class_groups(c_kinds, c_cids)
        if not classes:
            print("[WARN] No valid classes found in clean CSV (class_id>=0). Skip.")
            continue

        S_tc = collapse_M_to_time_class(M, classes, idx_by_class, reduce=args.reduce)

        core_name = CORE_NAME.get(core, core)
        title = f"{core_name}, S={rid.S}, η={int(rid.eta)/1000.0:g}, λ={int(rid.lam)/1000.0:g} (seed={rid.seed}), reduce={args.reduce}"

        noise_tag = f"_noise{rid.noise}" if rid.noise is not None else ""
        base = f"{core}_S{rid.S}{noise_tag}_eta{rid.eta}_lam{rid.lam}_seed{rid.seed}"

        plot_time_class_heatmap(
            S_tc, classes, h_kinds, h_cids,
            title=title,
            out_path=os.path.join(out_dir, f"time_class_{base}.png")
        )

        plot_query_bar(
            S_tc, classes, h_kinds,
            title=title,
            out_path=os.path.join(out_dir, f"query_bar_{base}.png")
        )

        plot_margin_curve(
            S_tc, classes, h_cids, h_kinds,
            title=title,
            out_path=os.path.join(out_dir, f"margin_{base}.png")
        )


if __name__ == "__main__":
    main()

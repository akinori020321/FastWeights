# -*- coding: utf-8 -*-
"""
plot_A_kv.py（S-loop追加版）
----------------------------------------
KV Bind/Query A-dynamics 可視化
 + S-loop 可視化 (A形式CSV)
"""

from __future__ import annotations
import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


# ========================================================
# A-dynamics CSV 読み込み
# ========================================================
def load_csv(path):
    steps = []
    kind = []
    cid = []
    froA = []
    specA = []
    h_norm = []
    Ah_norm = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            kind.append(row["kind"])
            cid.append(int(row["class_id"]))
            froA.append(float(row["froA"]))
            specA.append(float(row["specA"]))
            h_norm.append(float(row["h_norm"]))
            Ah_norm.append(float(row["Ah_norm"]))

    return steps, kind, cid, froA, specA, h_norm, Ah_norm


# ========================================================
# S-loop CSV 読み込み  (A形式)
# ========================================================
def load_sloop_csv(path):
    t_list = []
    s_list = []
    hnorm = []
    Ah_norm = []
    cos_h0 = []

    if path is None:
        return None

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(int(row["t"]))
            s_list.append(int(row["s"]))
            hnorm.append(float(row["h_norm"]))
            Ah_norm.append(float(row["Ah_norm"]))
            cos_h0.append(float(row["cos_h0"]))

    return t_list, s_list, hnorm, Ah_norm, cos_h0


# ========================================================
# key の class_id 補完
# ========================================================
def assign_key_class_ids(kind, cid):
    cid2 = cid.copy()
    last_value = None

    for i in range(len(kind)):
        if kind[i] == "value":
            last_value = cid[i]
        elif kind[i] == "key" and last_value is not None:
            cid2[i] = last_value

    return cid2


# ========================================================
# メイン
# ========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, default="A_kv.png")

    # ★ S-loop CSV
    ap.add_argument("--sloop_csv", type=str, default=None)
    # ★ S-loop を描く t（例：--sloop_t 12）
    ap.add_argument("--sloop_t", type=int, default=None)

    ap.add_argument("--s_max", type=int, default=5)

    args = ap.parse_args()

    # ========================================================
    # ★★★ MODE 1: S-loop GRID + t=acc subplot ★★★
    # ========================================================
    if args.sloop_csv is not None:
        print("[MODE] S-loop GRID only")

        # --- load main S-loop CSV ---
        t_list, s_list, h_list, a_list, c_list = load_sloop_csv(args.sloop_csv)
        unique_t = sorted(set(t_list))

        # --- load CosValue CSV ---
        cos_csv = args.csv.replace("A_kv_", "CosValue_kv_")
        cos_available = os.path.exists(cos_csv)

        if cos_available:
            t_cos = []
            s_cos = []
            v_cos = []
            with open(cos_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_cos.append(int(row["t"]))
                    s_cos.append(int(row["s"]))
                    v_cos.append(float(row["cos_value"]))

        # --- load Query CSV ---
        query_csv = args.csv.replace("A_kv_", "Query_kv_")
        qinfo = None

        if os.path.exists(query_csv):
            with open(query_csv, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > 0:
                    qinfo = rows[0]   # 1 行目を採用

        # =====================================================
        # ★ layout: 元の grid ＋ 1 パネル（t=acc）
        #   → unique_t の数に +1 をして総パネル数を作る
        # =====================================================
        TOTAL = len(unique_t) + 1  # 最後の１つが t="acc"

        rows = 4
        cols = 4
        assert TOTAL <= rows * cols, "パネル数が grid を超えています"

        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

        # ------------------------
        # S-loop t-grid を描画
        # ------------------------
        idx = 0
        for t in unique_t:
            r, c = divmod(idx, cols)
            ax = axes[r][c]

            xs = [s + 1 for tt, s in zip(t_list, s_list) if tt == t]
            hn = [v for tt, v in zip(t_list, h_list) if tt == t]
            an = [v for tt, v in zip(t_list, a_list) if tt == t]
            cs = [v for tt, v in zip(t_list, c_list) if tt == t]

            if len(xs) > 0:
                ax.plot(xs, hn, "-o", markersize=3, label="||h||")
                ax.plot(xs, an, "-o", markersize=3, label="||Ah||")
                ax.plot(xs, cs, "-o", markersize=3, label="cos(h0,h)")

            ax.set_xlim(0.5, args.s_max + 0.5)
            ax.set_xticks(list(range(1, args.s_max + 1)))
            ax.set_title(f"t={t}", fontsize=10)
            ax.grid(True)

            idx += 1

        # ------------------------
        # ★ t="acc" の CosValue サブグラフ追加
        # ------------------------
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_title("t=acc")

        if cos_available:
            # CosValue を線グラフで描く
            xcos = [s + 1 for s in s_cos]
            ax.plot(xcos, v_cos, "-o", label="cos_value")
            ax.set_xlim(0.5, args.s_max + 0.5)
            ax.set_xticks(list(range(1, args.s_max + 1)))
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No CosValue CSV", ha="center", va="center")
            ax.axis("off")

        # ================================
        # ★ 図全体の下に Query 情報を描画
        # ================================
        if qinfo is not None:
            txt = (
                f"true={qinfo['true']}, "
                f"pred={qinfo['pred']}, "
                f"acc={qinfo['correct']}, "
                f"margin={float(qinfo['margin']):.3f}, "
                f"cos(W)={float(qinfo['cosine']):.3f}"
            )
            fig.text(
                0.02,          # x 位置（左寄せ）
                0.01,          # y 位置（下に配置）
                txt,
                fontsize=11,
                ha="left",
                va="bottom"
            )
        
        # ------------------------
        # 残りパネルは空白
        # ------------------------
        for k in range(idx+1, rows * cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")

        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"[DONE] Saved S-loop GRID with CosValue+Query → {args.out_png}")
        return


    # ========================================================
    # ★★★ MODE 2: A-dynamics（S-loop を描かない）★★★
    # ========================================================
    steps, kind, cid_raw, froA, specA, h_norm, Ah_norm = load_csv(args.csv)

    # key の class_id 補完
    cid = assign_key_class_ids(kind, cid_raw)

    # -------------------------------
    # Query 情報
    # -------------------------------
    try:
        query_idx = kind.index("query")
        query_class = cid[query_idx]
    except ValueError:
        query_idx = None
        query_class = None

    # -------------------------------
    # ★ 星マーカーにすべき index
    # -------------------------------
    star_indices = set()

    if query_idx is not None:
        # Query 自身
        star_indices.add(query_idx)

        # Query class に一致する key（★この key も星にする）
        for i in range(len(kind)):
            if kind[i] == "key":
                bind_idx = i // 2   # ★ key の bind 番号
                if bind_idx == query_class:
                    star_indices.add(i)


    # -------------------------------
    # 色 & マーカー
    # -------------------------------

    # 10 色パレット
    cmap10 = plt.cm.tab10(np.linspace(0, 1, 10))

    colors = []
    markers = []

    for i in range(len(steps)):
        k = kind[i]

        # ===========================
        # 色の決定
        # ===========================
        if k == "value":
            # value → class ごとの色
            cls = cid[i] % 10
            c = cmap10[cls]

        elif k == "query":
            # query_class には bind 番号（q_bind_idx）が入っている
            q_bind_idx = query_class

            # query の本当の class は class_ids[q_bind_idx]
            true_q_class = cid[2 * q_bind_idx + 1]

            cls = true_q_class % 10
            c = cmap10[cls]

        else:
            # key / init / その他 → 灰色
            c = "#000000"

        colors.append(c)

        # ===========================
        # マーカーの決定
        # ===========================
        if i in star_indices:
            # query と対応 key は星（色は上で決まる）
            m = "*"
        else:
            if k == "key":
                m = "s"   # key は四角
            else:
                m = "o"   # value/init は丸

        markers.append(m)

    # -------------------------------
    # A-dynamics プロット
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    for i in range(len(steps)):
        ax1.scatter(steps[i], froA[i], color=colors[i], marker=markers[i], s=80)
        ax2.scatter(steps[i], specA[i], color=colors[i], marker=markers[i], s=80)
        ax3.scatter(steps[i], h_norm[i], color=colors[i], marker=markers[i], s=80)
        ax4.scatter(steps[i], Ah_norm[i], color=colors[i], marker=markers[i], s=80)

    ax1.set_title("Frobenius norm ||A_t||_F")
    ax2.set_title("Spectral radius ρ(A_t)")
    ax3.set_title("||h_t||")
    ax4.set_title("||A_t h_t||")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("step")
        ax.grid(True)

    fig.tight_layout()

    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out_png, dpi=200)
    print(f"[DONE] A-dynamics Saved → {args.out_png}")

if __name__ == "__main__":
    main()

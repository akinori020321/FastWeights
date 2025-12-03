# -*- coding: utf-8 -*-
"""
plot_A_kv.py（方向復元タスク版）
----------------------------------------
Bind / Query A-dynamics 可視化
 + S-loop（grid 版にも対応）
"""

from __future__ import annotations
import argparse
import csv
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ========================================================
# safe float loader（None / 空欄対策）
# ========================================================
def safe_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return float("nan")
    try:
        return float(s)
    except:
        return float("nan")


# ========================================================
# A-dynamics CSV 読み込み
# ========================================================
def load_csv(path):
    steps, kind, cid = [], [], []
    froA, specA = [], []
    h_norm, Ah_norm = [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            kind.append(row["kind"])
            cid.append(safe_float(row["cid"]))
            froA.append(safe_float(row["froA"]))
            specA.append(safe_float(row["specA"]))
            h_norm.append(safe_float(row["h_norm"]))
            Ah_norm.append(safe_float(row["Ah_norm"]))

    return steps, kind, cid, froA, specA, h_norm, Ah_norm


# ========================================================
# S-loop CSV 読み込み
# ========================================================
def load_sloop_csv(path):
    t_list, s_list = [], []
    hnorm, Ah_norm = [], []
    cos_h0, cos_value = [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(int(row["t"]))
            s_list.append(int(row["s"]))
            hnorm.append(safe_float(row["h_norm"]))
            Ah_norm.append(safe_float(row["Ah_norm"]))
            cos_h0.append(safe_float(row["cos_h0"]))
            cos_value.append(safe_float(row["cos_value"]))

    return t_list, s_list, hnorm, Ah_norm, cos_h0, cos_value


# ========================================================
# Main
# ========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--sloop_csv", type=str, default=None)
    ap.add_argument("--out_png", type=str, default="A_kv.png")
    ap.add_argument("--s_max", type=int, default=5)
    args = ap.parse_args()

    # ========================================================
    # MODE 1: S-loop GRID only
    # ========================================================
    if args.sloop_csv is not None:
        print("[MODE] S-loop GRID only")

        # S-loop
        t_list, s_list, h_list, a_list, c_list, v_list = load_sloop_csv(args.sloop_csv)
        unique_t = sorted(set(t_list))

        # CosValue CSV
        cos_csv = args.csv.replace("A_kv_", "CosValue_kv_")
        cos_available = os.path.exists(cos_csv)

        if cos_available:
            t_cos, s_cos, v_cos = [], [], []
            with open(cos_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_cos.append(int(row["t"]))
                    s_cos.append(int(row["s"]))
                    v_cos.append(safe_float(row["cos_value"]))

        # grid
        rows, cols = 6, 6
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

        idx = 0
        max_cells = rows * cols

        # t ごとの GRID
        for t in unique_t:
            if idx >= max_cells:
                break

            r, c = divmod(idx, cols)
            ax = axes[r][c]

            xs = [s + 1 for tt, s in zip(t_list, s_list) if tt == t]
            hn = [v for tt, v in zip(t_list, h_list) if tt == t]
            an = [v for tt, v in zip(t_list, a_list) if tt == t]
            cs = [v for tt, v in zip(t_list, c_list) if tt == t]

            if xs:
                ax.plot(xs, hn, "-o", markersize=3, label="||h||")
                ax.plot(xs, an, "-o", markersize=3, label="||Ah||")
                ax.plot(xs, cs, "-o", markersize=3, label="cos(h0,h)")

            ax.set_xlim(0.5, args.s_max + 0.5)
            ax.set_xticks(list(range(1, args.s_max + 1)))
            ax.set_title(f"t={t}", fontsize=10)
            ax.grid(True)

            idx += 1

        # ----------------------------------------------------
        # t=acc パネル
        # ----------------------------------------------------
        if idx < max_cells:
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            ax.set_title("t=acc")

            if cos_available:
                xcos = [s + 1 for s in s_cos]
                ax.plot(xcos, v_cos, "-o")
                ax.set_xlim(0.5, args.s_max + 0.5)
                ax.set_xticks(list(range(1, args.s_max + 1)))
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "No CosValue CSV", ha="center", va="center")
                ax.axis("off")

            idx += 1

        # ----------------------------------------------------
        # Query 情報（左下に cosine のみ表示）
        # ----------------------------------------------------
        query_csv = args.csv.replace("A_kv_", "Query_kv_")
        qinfo = None

        if os.path.exists(query_csv):
            with open(query_csv, "r") as f:
                reader = list(csv.DictReader(f))
                if len(reader) > 0:
                    qinfo = reader[0]

        if qinfo is not None:
            # cosine 値のみ取り出す（float に変換、存在しない場合は nan）
            cos_val = safe_float(qinfo.get("cosine", "nan"))

            txt = f"Query Cosine = {cos_val:.3f}"

            fig.text(
                0.02, 0.01,
                txt,
                fontsize=11,
                ha="left",
                va="bottom"
            )


        # 空白パネル OFF
        for k in range(idx, max_cells):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")

        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"[DONE] S-loop GRID → {args.out_png}")
        return

    # ========================================================
    # MODE 2: A-dynamics（Bind / Wait / Query）
    # ========================================================
    steps, kind, cid, froA, specA, h_norm, Ah_norm = load_csv(args.csv)

    # Query index
    try:
        query_idx = kind.index("query")
        query_cid = cid[query_idx]
    except ValueError:
        query_idx, query_cid = None, None

    # カラー設定
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    unique_cids = sorted(set(c for c in cid if not math.isnan(c)))

    color_map = {}
    color_i = 0
    for c in unique_cids:
        if c == query_cid:
            continue
        color_map[c] = base_colors[color_i % len(base_colors)]
        color_i += 1

    query_color = "black"

    colors, markers = [], []
    for i in range(len(steps)):
        k = kind[i]
        c = cid[i]

        if i == query_idx:
            colors.append(query_color)
            markers.append("*")
        elif k == "bind" and c == query_cid:
            colors.append(query_color)
            markers.append("*")
        elif k == "bind":
            colors.append(color_map.get(c, "gray"))
            markers.append("o")
        else:
            colors.append("gray")
            markers.append("o")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, data, title in [
        (axes[0][0], froA, "||A_t||_F"),
        (axes[0][1], specA, "ρ(A_t)"),
        (axes[1][0], h_norm, "||h_t||"),
        (axes[1][1], Ah_norm, "||A_t h_t||"),
    ]:
        for i in range(len(steps)):
            if not math.isnan(data[i]):
                ax.scatter(steps[i], data[i], c=colors[i], marker=markers[i], s=80)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    print(f"[DONE] A-dynamics Saved → {args.out_png}")


if __name__ == "__main__":
    main()


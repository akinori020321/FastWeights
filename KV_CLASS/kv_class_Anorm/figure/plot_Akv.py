#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_Akv.py
----------------------------------------
A-dynamics 実験結果をプロットするツール（色自動生成＋線種＝正誤対応）
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import colorsys

# --------------------------------------------------
# ディレクトリ設定
# --------------------------------------------------
RESULT_DIR = "../results_A_kv"
SAVE_DIR = "plots/figure_norm"
os.makedirs(SAVE_DIR, exist_ok=True)


# --------------------------------------------------
# CSV 探索
# --------------------------------------------------
def scan_all_csv():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "A_kv_*.csv")))
    print(f"[INFO] Found {len(files)} A-dynamics files.")
    return files


# --------------------------------------------------
# Query CSV correct を取得
# --------------------------------------------------
def load_correct(csv_path):
    query_csv = csv_path.replace("A_kv_", "Query_kv_")
    if not os.path.exists(query_csv):
        print(f"[WARN] Query CSV not found: {query_csv}")
        return None

    df = pd.read_csv(query_csv)
    return float(df["correct"].iloc[0])


# --------------------------------------------------
# ファイル名パーサー
# --------------------------------------------------
def parse_core_S(filename):
    base = os.path.basename(filename)

    pattern = r"A_kv_(fw|tanh|rnn)_S([0-9]+)_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match new A_kv_ pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)

    return core, S, eta, lam, seed


# --------------------------------------------------
# ★ signature（パラメータが1つでも違えば別物）
# --------------------------------------------------
def make_signature(item):
    return f"{item['core']}_S{item['S']}_eta{item['eta']}_lam{item['lam']}_seed{item['seed']}"


# --------------------------------------------------
# ★ 色生成：色相固定、明度変化で複数色を生成
# --------------------------------------------------
def generate_colors(base_hue, n):
    colors = []
    for i in range(n):
        # 明度 0.30〜0.85 で変化（視認性が高く揃う）
        l = 0.30 + 0.55 * (i / max(1, n - 1))
        s = 0.95
        r, g, b = colorsys.hls_to_rgb(base_hue, l, s)
        colors.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
    return colors


# --------------------------------------------------
# ★ signature ごとに色を割り当てる
# --------------------------------------------------
def assign_colors(all_data):

    groups = {"fw": [], "tanh": [], "rnn": []}

    # core ごとに signature を分類
    for item in all_data:
        sig = make_signature(item)
        groups[item["core"]].append(sig)

    # 重複排除
    for c in groups:
        groups[c] = sorted(list(set(groups[c])))

    color_map = {}

    for core, sigs in groups.items():

        if core == "fw":
            base_hue = 0.0    # 赤
        elif core == "tanh":
            base_hue = 0.60   # 青
        else:
            base_hue = 0.33   # 緑

        palette = generate_colors(base_hue, len(sigs))

        for sig, col in zip(sigs, palette):
            color_map[sig] = col

    return color_map


# --------------------------------------------------
# h_norm plot → ★線種は正誤で決定
# --------------------------------------------------
def plot_h_norm(all_data, color_map):
    plt.figure(figsize=(11, 7))

    for item in all_data:
        df = item["df"]
        sig = make_signature(item)
        color = color_map[sig]

        # ★ correct → 実線、不正解 → 破線
        linestyle = "-" if item["correct"] >= 0.5 else "--"
        lw = 2.2 if item["correct"] >= 0.5 else 1.4

        label = sig

        plt.plot(df["step"], df["h_norm"],
                 label=label,
                 color=color,
                 linestyle=linestyle,
                 linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("||h_t||")
    plt.title("h_norm over time")
    plt.legend(fontsize=8)
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, "h_norm_all.png")
    plt.savefig(save_path, dpi=200)
    print(f"[SAVE] {save_path}")


# --------------------------------------------------
# specA plot（RNN の線は描かない）
# --------------------------------------------------
def plot_specA(all_data, color_map):
    plt.figure(figsize=(11, 7))

    for item in all_data:
        if item["core"] == "rnn":
            continue  # A=0 のためスキップ

        df = item["df"]
        sig = make_signature(item)
        color = color_map[sig]

        # 正誤で線種を決定
        linestyle = "-" if item["correct"] >= 0.5 else "--"
        lw = 2.2 if item["correct"] >= 0.5 else 1.4

        label = sig

        plt.plot(df["step"], df["specA"],
                 label=label,
                 color=color,
                 linestyle=linestyle,
                 linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("spec(A_t)")
    plt.title("specA over time")
    plt.legend(fontsize=8)
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, "specA_all.png")
    plt.savefig(save_path, dpi=200)
    print(f"[SAVE] {save_path}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():

    csv_files = scan_all_csv()
    all_data = []

    # CSV 読み込み
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        core, S, eta, lam, seed = parse_core_S(csv_path)
        correct = load_correct(csv_path) or 0.0

        all_data.append({
            "df": df,
            "core": core,
            "S": S,
            "eta": eta,
            "lam": lam,
            "seed": seed,
            "correct": correct,
        })

        print(f"[LOAD] {csv_path} | core={core} | S={S} | eta={eta} | lam={lam} | seed={seed} | correct={correct}")

    # ★ signature ごとに色割り当て
    color_map = assign_colors(all_data)

    plot_h_norm(all_data, color_map)
    plot_specA(all_data, color_map)

    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()

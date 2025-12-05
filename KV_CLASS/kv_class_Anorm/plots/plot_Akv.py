#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_Akv.py
----------------------------------------
A-dynamics 実験結果をプロットするツール
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

# --------------------------------------------------
# 設定
# --------------------------------------------------
RESULT_DIR = "../results_A_kv"
SAVE_DIR = "plots/figure_norm"
os.makedirs(SAVE_DIR, exist_ok=True)

COLOR_FW = {
    1: "#ff0000",
    3: "#cc0000",
    5: "#ff6666",
}

COLOR_TANH = {
    1: "#0066ff",
    3: "#003399",
    5: "#66aaff",
}

LINESTYLE_CORRECT = {
    1: "-",
    3: "-",
    5: "-",
}

LINESTYLE_WRONG = {
    1: "--",
    3: ":",
    5: "-.",
}

# --------------------------------------------------
# CSV 自動探索
# --------------------------------------------------
def scan_all_csv():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "A_kv_*.csv")))
    print(f"[INFO] Found {len(files)} A-dynamics files.")
    return files


# --------------------------------------------------
# Query CSV の correct を読む
# --------------------------------------------------
def load_correct(csv_path):
    query_csv = csv_path.replace("A_kv_", "Query_kv_")
    if not os.path.exists(query_csv):
        print(f"[WARN] Query CSV not found: {query_csv}")
        return None

    df = pd.read_csv(query_csv)
    return float(df["correct"].iloc[0])


# --------------------------------------------------
# ★ 新しい形式のファイル名に完全対応
# A_kv_fw_S1_eta0300_lam0850_seed0.csv
# --------------------------------------------------
def parse_core_S(filename):
    base = os.path.basename(filename)

    pattern = r"A_kv_(fw|tanh)_S([0-9]+)_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match new A_kv_ pattern: {base}")

    core = m.group(1)     # fw / tanh
    S = int(m.group(2))   # S
    eta = m.group(3)      # 0300 の string
    lam = m.group(4)      # 0850 の string
    seed = m.group(5)     # seed

    return core, S, eta, lam, seed


# --------------------------------------------------
# style 設定
# --------------------------------------------------
def get_style(core, S, correct):

    if core == "fw":
        color = COLOR_FW.get(S, "#ff0000")
    else:
        color = COLOR_TANH.get(S, "#0066ff")

    if correct >= 0.5:
        linestyle = LINESTYLE_CORRECT.get(S, "-")
        lw = 2.0
    else:
        linestyle = LINESTYLE_WRONG.get(S, "--")
        lw = 1.3

    return color, linestyle, lw


# --------------------------------------------------
# h_norm plot
# --------------------------------------------------
def plot_h_norm(all_data):
    plt.figure(figsize=(10, 6))

    for item in all_data:
        df = item["df"]
        core = item["core"]
        S = item["S"]
        eta = item["eta"]
        lam = item["lam"]
        correct = item["correct"]

        color, ls, lw = get_style(core, S, correct)

        # ★ 修正：eta と lam を label に追加
        label = f"{core}_S{S}_eta{eta}_lam{lam} (correct={int(correct)})"

        plt.plot(df["step"], df["h_norm"],
                 label=label,
                 color=color, linestyle=ls, linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("||h_t||")
    plt.title("h_norm over time")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, "h_norm_all.png")
    plt.savefig(save_path, dpi=200)
    print(f"[SAVE] {save_path}")


# --------------------------------------------------
# specA plot
# --------------------------------------------------
def plot_specA(all_data):
    plt.figure(figsize=(10, 6))

    for item in all_data:
        df = item["df"]
        core = item["core"]
        S = item["S"]
        eta = item["eta"]
        lam = item["lam"]
        correct = item["correct"]

        color, ls, lw = get_style(core, S, correct)

        # ★ 修正：eta と lam を label に追加
        label = f"{core}_S{S}_eta{eta}_lam{lam} (correct={int(correct)})"

        plt.plot(df["step"], df["specA"],
                 label=label,
                 color=color, linestyle=ls, linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("spec(A_t)")
    plt.title("specA over time")
    plt.legend()
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

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        core, S, eta, lam, seed = parse_core_S(csv_path)
        correct = load_correct(csv_path)
        correct = 0.0 if correct is None else correct

        all_data.append({
            "df": df,
            "core": core,
            "S": S,
            "eta": eta,
            "lam": lam,
            "seed": seed,
            "correct": correct,
        })

        print(f"[LOAD] {csv_path} | core={core} | S={S} | eta={eta} | lam={lam} | correct={correct}")

    plot_h_norm(all_data)
    plot_specA(all_data)

    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()

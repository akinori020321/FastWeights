#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_Akv.py
----------------------------------------
A-dynamics 実験結果をプロットするツール
 - 色：core & ハイパラごとに自動生成
 - 線種：cosine (or correct) が閾値以上なら実線、それ以外は破線
"""

import os
import glob
import re
import colorsys

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# ディレクトリ設定
# --------------------------------------------------
RESULT_DIR = "../results_A_kv"
SAVE_DIR = "plots/figure_norm"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# core 名表示変換
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN-LN",
}

# --------------------------------------------------
# CSV 探索
# --------------------------------------------------
def scan_all_csv():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "A_kv_*.csv")))
    print(f"[INFO] Found {len(files)} A-dynamics files.")
    return files

# --------------------------------------------------
# Query CSV から「線種を決めるスコア」を取得
#  - cosine 列があれば cosine を優先
#  - なければ correct を使う（古い分類実験用）
# --------------------------------------------------
def load_score(csv_path, cos_thresh=0.8):
    query_csv = csv_path.replace("A_kv_", "Query_kv_")
    if not os.path.exists(query_csv):
        print(f"[WARN] Query CSV not found: {query_csv}")
        return 0.0, None  # score, raw_cosine

    df = pd.read_csv(query_csv)

    cosine = None
    if "cosine" in df.columns:
        try:
            cosine = float(df["cosine"].iloc[0])
        except Exception:
            cosine = None

    if cosine is not None:
        # 方向復元タスク：cosine をそのまま score として使う
        score = cosine
    elif "correct" in df.columns:
        # 分類タスク：correct を score とする
        score = float(df["correct"].iloc[0])
    else:
        score = 0.0

    return score, cosine

# --------------------------------------------------
# ファイル名パーサー（noise あり / なし両対応）
# --------------------------------------------------
def parse_core_S(filename):
    base = os.path.basename(filename)

    pattern = (
        r"A_kv_(fw|tanh|rnn)_S([0-9]+)"
        r"(?:_noise([0-9]+))?"
        r"_eta([0-9]+)_lam([0-9]+)_seed([0-9]+)"
    )
    m = re.match(pattern, base)

    if m is None:
        raise ValueError(f"Filename does not match A_kv_ pattern: {base}")

    core = m.group(1)
    S    = int(m.group(2))
    # noise = m.group(3)  # 使わないので今回は無視
    eta  = m.group(4)
    lam  = m.group(5)
    seed = m.group(6)

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

    for item in all_data:
        sig = make_signature(item)
        groups[item["core"]].append(sig)

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
# h_norm plot → ★線種は score>=0.8 で決定
#   - direction: score = cosine
#   - classification: score = correct
# --------------------------------------------------
def plot_h_norm(all_data, color_map, thresh=0.8):
    plt.figure(figsize=(11, 7))

    for item in all_data:
        df = item["df"]
        sig = make_signature(item)
        color = color_map[sig]

        score = item["score"]
        linestyle = "-" if score >= thresh else "--"
        lw = 2.2 if score >= thresh else 1.4

        core_name = CORE_NAME[item["core"]]
        label = f"{core_name}_S{item['S']}_eta{item['eta']}_lam{item['lam']}_seed{item['seed']}"

        plt.plot(df["step"], df["h_norm"],
                 label=label,
                 color=color,
                 linestyle=linestyle,
                 linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("||h_t||")
    plt.title(f"h_norm over time (solid if score ≥ {thresh})")
    plt.legend(fontsize=8)
    plt.grid(True)

    save_path = os.path.join(SAVE_DIR, "h_norm_all.png")
    plt.savefig(save_path, dpi=200)
    print(f"[SAVE] {save_path}")

# --------------------------------------------------
# specA plot（RNN の線は描かない）
# --------------------------------------------------
def plot_specA(all_data, color_map, thresh=0.8):
    plt.figure(figsize=(11, 7))

    for item in all_data:
        if item["core"] == "rnn":
            continue  # A=0 のためスキップ

        df = item["df"]
        sig = make_signature(item)
        color = color_map[sig]

        score = item["score"]
        linestyle = "-" if score >= thresh else "--"
        lw = 2.2 if score >= thresh else 1.4

        core_name = CORE_NAME[item["core"]]
        label = f"{core_name}_S{item['S']}_eta{item['eta']}_lam{item['lam']}_seed{item['seed']}"

        plt.plot(df["step"], df["specA"],
                 label=label,
                 color=color,
                 linestyle=linestyle,
                 linewidth=lw)

    plt.xlabel("t (step)")
    plt.ylabel("spec(A_t)")
    plt.title(f"specA over time (solid if score ≥ {thresh})")
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

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        core, S, eta, lam, seed = parse_core_S(csv_path)
        score, cosine = load_score(csv_path, cos_thresh=0.8)

        all_data.append({
            "df": df,
            "core": core,
            "S": S,
            "eta": eta,
            "lam": lam,
            "seed": seed,
            "score": score,     # 線種判定に使うスカラー（direction: cosine, classification: correct）
            "cosine": cosine,   # 方向復元タスクであればその値、それ以外は None
        })

        print(f"[LOAD] {csv_path} | core={core} | S={S} | eta={eta} | lam={lam} | seed={seed} | "
              f"score={score:.4f} | cosine={cosine}")

    color_map = assign_colors(all_data)

    thresh = 0.8
    plot_h_norm(all_data, color_map, thresh=thresh)
    plot_specA(all_data, color_map, thresh=thresh)

    print("[DONE] All plots generated.")

if __name__ == "__main__":
    main()

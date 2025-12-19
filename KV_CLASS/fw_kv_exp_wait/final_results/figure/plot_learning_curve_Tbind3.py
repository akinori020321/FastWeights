#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_learning_curve_Tbind3.py
--------------------------------------------
T_bind3/ 配下の
  - fw_fw0_*.csv  (Ba-FW)
  - fw_fw1_*.csv  (RNN+LN)
  - tanh_*.csv    (SC-FW)
を読み取り，横軸 epoch / 縦軸 acc の学習曲線を 1 枚にまとめて描画する。

出力:
  T_bind3_fig/learning_curve_Tbind3.png

使い方:
  python3 plot_learning_curve_Tbind3.py
  python3 plot_learning_curve_Tbind3.py --csv_dir /path/to/T_bind3 --out_name my.png
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# 色・ラベル（ユーザー指定）
#   fw   : fw_fw0_*.csv
#   rnn  : fw_fw1_*.csv
#   tanh : tanh_*.csv
# ======================================================
COLOR_MAP = {
    "fw":   "red",
    "rnn":  "blue",
    "tanh": "green",
}

LABEL_MAP = {
    "rnn":  "RNN+LN",
    "fw":   "Ba-FW",
    "tanh": "SC-FW",
}

PATTERNS = {
    "fw":   "fw_fw0_*.csv",
    "rnn":  "fw_fw1_*.csv",
    "tanh": "tanh_*.csv",
}


# ======================================================
# CSV から (epoch, acc) を読む
# ======================================================
def read_curve(path: str):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read: {path} ({e})")
        return None, None

    if "epoch" not in df.columns:
        print(f"[WARN] 'epoch' column not found: {path}")
        return None, None

    # acc カラム候補（よくある順）
    acc_col = None
    for c in ["valid_acc", "val_acc", "acc", "accuracy"]:
        if c in df.columns:
            acc_col = c
            break

    if acc_col is None:
        print(f"[WARN] acc column not found (valid_acc/val_acc/acc/accuracy): {path}")
        return None, None

    epochs = df["epoch"].to_numpy()
    acc = df[acc_col].to_numpy()

    # 念のため float 化
    epochs = epochs.astype(float)
    acc = acc.astype(float)
    return epochs, acc


def pick_latest(paths):
    """複数ある場合は更新時刻が新しいものを採用"""
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="T_bind3 ディレクトリ（未指定ならスクリプトと同じ階層の T_bind3/）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="出力先ディレクトリ（未指定なら T_bind3_fig/）",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="learning_curve_Tbind3.png",
        help="出力ファイル名",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="latest",
        choices=["latest", "all"],
        help="同種のCSVが複数ある場合の扱い：latest=最新のみ / all=全て描画",
    )
    args = ap.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = args.csv_dir if args.csv_dir is not None else os.path.join(this_dir, "T_bind3")
    out_dir = args.out_dir if args.out_dir is not None else os.path.join(this_dir, "T_bind3_fig")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(csv_dir):
        print(f"[ERROR] csv_dir not found: {csv_dir}")
        return

    # --------------------------------------------------
    # ファイル収集
    # --------------------------------------------------
    selected = {}  # kind -> list[str]
    for kind, pat in PATTERNS.items():
        paths = glob.glob(os.path.join(csv_dir, pat))
        if not paths:
            print(f"[WARN] No files for {kind}: pattern={pat}")
            selected[kind] = []
            continue

        if args.mode == "latest":
            latest = pick_latest(paths)
            selected[kind] = [latest] if latest else []
        else:
            # all
            selected[kind] = sorted(paths)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    any_plotted = False
    for kind in ["rnn", "fw", "tanh"]:  # 凡例の順序を固定したい場合
        paths = selected.get(kind, [])
        if not paths:
            continue

        color = COLOR_MAP[kind]
        label = LABEL_MAP[kind]

        for i, path in enumerate(paths):
            epochs, acc = read_curve(path)
            if epochs is None:
                continue

            # all モードでは同じ label が複数回出るので最初の1本だけ凡例に載せる
            show_label = label if (args.mode == "latest" or i == 0) else None
            lw = 2 if args.mode == "latest" else 1.5
            alpha = 1.0 if args.mode == "latest" else 0.6

            plt.plot(epochs, acc, color=color, linewidth=lw, alpha=alpha, label=show_label)
            any_plotted = True

    if not any_plotted:
        print("[ERROR] No usable curves plotted. Check CSV contents/columns.")
        return

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.title("Learning Curves (T_bind=3)")

    out_path = os.path.join(out_dir, args.out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")


if __name__ == "__main__":
    main()

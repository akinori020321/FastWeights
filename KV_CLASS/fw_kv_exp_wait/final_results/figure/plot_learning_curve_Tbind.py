#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_learning_curve_Tbind.py
--------------------------------------------
T_bind{tbind}/ 配下の
- fw_fw0_*.csv  (Ba-FW)
- fw_fw1_*.csv  (RNN+LN)
- tanh_*.csv    (SC-FW)
を読み取り，横軸 epoch / 縦軸 acc の学習曲線を 1 枚にまとめて描画する。

★追加：固定 epoch のみ表示
  --epoch_list "0,1,2,5,10,20,50,100"
  --epoch_step 5
  --epoch_min 0 --epoch_max 300   ← ★ default=300

出力:
T_bind{tbind}_fig/learning_curve_Tbind{tbind}.png (+ .eps)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# 色・ラベル（ユーザー指定）
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

    acc_col = None
    for c in ["valid_acc", "val_acc", "acc", "accuracy"]:
        if c in df.columns:
            acc_col = c
            break

    if acc_col is None:
        print(f"[WARN] acc column not found (valid_acc/val_acc/acc/accuracy): {path}")
        return None, None

    epochs = df["epoch"].to_numpy(dtype=float)
    acc = df[acc_col].to_numpy(dtype=float)
    return epochs, acc


def pick_latest(paths):
    """複数ある場合は更新時刻が新しいものを採用"""
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


# ======================================================
# epoch フィルタ（固定epochのみ表示）
# ======================================================
def parse_epoch_list(s: str):
    # "0,1,2,5,10" -> [0.0, 1.0, ...]
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        vals.append(float(tok))
    return vals


def filter_curve_by_epoch(epochs, acc, epoch_list=None, epoch_step=None, epoch_min=None, epoch_max=None):
    e = epochs
    a = acc

    mask = np.ones_like(e, dtype=bool)
    if epoch_min is not None:
        mask &= (e >= float(epoch_min))
    if epoch_max is not None:
        mask &= (e <= float(epoch_max))
    e = e[mask]
    a = a[mask]

    # 指定epochだけ残す
    if epoch_list is not None and len(epoch_list) > 0:
        keep = np.zeros_like(e, dtype=bool)
        for v in epoch_list:
            keep |= np.isclose(e, v, rtol=0.0, atol=1e-9)
        e = e[keep]
        a = a[keep]

    # step ごと（例：5epochごと）
    if epoch_step is not None:
        step = float(epoch_step)
        if step > 0:
            keep = np.isclose(np.mod(e, step), 0.0, rtol=0.0, atol=1e-9)
            e = e[keep]
            a = a[keep]

    # epoch 昇順保証
    if len(e) > 0:
        idx = np.argsort(e)
        e = e[idx]
        a = a[idx]

    return e, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbind", type=int, default=3, help="T_bind の値（例: 3, 5, ...）")
    ap.add_argument("--csv_dir", type=str, default=None,
                    help="T_bind{tbind} ディレクトリ（未指定なら <project_root>/T_bind{tbind}/）")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="出力先ディレクトリ（未指定なら <script_dir>/T_bind{tbind}_fig/）")
    ap.add_argument("--out_name", type=str, default=None,
                    help="出力ファイル名（未指定なら learning_curve_Tbind{tbind}.png）")
    ap.add_argument("--mode", type=str, default="latest", choices=["latest", "all"],
                    help="同種のCSVが複数ある場合の扱い：latest=最新のみ / all=全て描画")

    # ★追加：epoch固定表示
    ap.add_argument("--epoch_list", type=str, default=None,
                    help='固定で表示したい epoch（カンマ区切り）。例: "0,1,2,5,10,20,50,100"')
    ap.add_argument("--epoch_step", type=float, default=None,
                    help="N epoch ごとに間引いて表示（例: 5 → 0,5,10,...）。epoch_list と併用可")
    ap.add_argument("--epoch_min", type=float, default=None, help="表示する最小 epoch（範囲制限）")
    ap.add_argument("--epoch_max", type=float, default=600, help="表示する最大 epoch（範囲制限）")

    args = ap.parse_args()

    tbind = args.tbind

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)

    default_csv_dir = os.path.join(proj_dir, f"T_bind{tbind}")
    default_out_dir = os.path.join(this_dir, f"T_bind{tbind}_fig")
    default_out_name = f"learning_curve_Tbind{tbind}.png"

    csv_dir = args.csv_dir if args.csv_dir is not None else default_csv_dir
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir
    out_name = args.out_name if args.out_name is not None else default_out_name

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(csv_dir):
        print(f"[ERROR] csv_dir not found: {csv_dir}")
        return

    epoch_list = parse_epoch_list(args.epoch_list) if args.epoch_list else None

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
            selected[kind] = sorted(paths)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    any_plotted = False
    for kind in ["rnn", "fw", "tanh"]:  # 凡例の順序固定
        paths = selected.get(kind, [])
        if not paths:
            continue

        color = COLOR_MAP[kind]
        label = LABEL_MAP[kind]

        for i, path in enumerate(paths):
            epochs, acc = read_curve(path)
            if epochs is None:
                continue

            # ★ここで epoch を固定/間引き/範囲制限
            epochs, acc = filter_curve_by_epoch(
                epochs, acc,
                epoch_list=epoch_list,
                epoch_step=args.epoch_step,
                epoch_min=args.epoch_min,
                epoch_max=args.epoch_max,
            )
            if epochs is None or len(epochs) == 0:
                continue

            show_label = label if (args.mode == "latest" or i == 0) else None
            lw = 2 if args.mode == "latest" else 1.5
            alpha = 1.0 if args.mode == "latest" else 0.6

            plt.plot(epochs, acc, color=color, linewidth=lw, alpha=alpha, label=show_label)
            any_plotted = True

    if not any_plotted:
        print("[ERROR] No usable curves plotted. Check CSV contents/columns or epoch filters.")
        return

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.title(f"Learning Curves (T_bind={tbind})")

    # x軸の目盛も固定したい場合（epoch_list 指定時のみ）
    if epoch_list is not None and len(epoch_list) > 0:
        plt.xticks(sorted(epoch_list))

    out_path = os.path.join(out_dir, out_name)
    base, _ = os.path.splitext(out_path)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(base + ".eps", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")


if __name__ == "__main__":
    main()

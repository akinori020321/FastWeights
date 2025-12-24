#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_tbind_sweep_and_curve.py
========================================================
(1) T_bind sweep（RNN+LN と Ba-FW のみ）
    - ../T_bind/results_Tbind_fw/
    - ../T_bind/results_Tbind_rnn/
    から最終 epoch の acc を集計し，T_bind vs Acc（mean±std）を描画

(2) Bind=4 学習曲線（Acc のみ：FW seed曲線 + FW mean±std + RNN mean）
    - ../T_bind/results_Tbind_fw4/ から
        fw_fw1_T4_S1_seed*_beta1.00_wait2.csv
      -> seed曲線 + mean±std + mean（黒破線）
    - ../T_bind/results_Tbind_rnn/ から bind=4（T4）のCSVを探索
      -> mean（赤実線）のみ

出力（★両方同じディレクトリ）:
  out_dir/ に
    - tbind_sweep.(png|eps)
    - learning_curve_acc_fw_vs_rnn_T4_wait2_beta1.00.(png|eps)
========================================================
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# スクリプト自身のディレクトリ
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# 入力パス
# ======================================================
# (1) T_bind sweep 用
CSV_ROOT_SWEEP = os.path.join(THIS_DIR, "..", "T_bind")

# (2) Bind=4 learning curve 用
FW_CSV_DIR_DEFAULT  = os.path.join(THIS_DIR, "..", "T_bind", "results_Tbind_fw4")
RNN_CSV_DIR_DEFAULT = os.path.join(THIS_DIR, "..", "T_bind", "results_Tbind_rnn")

# ======================================================
# ★出力先（両方同じ）
# ======================================================
OUT_DIR_DEFAULT = os.path.join(THIS_DIR, "Tbind_and_curve_fig")
os.makedirs(OUT_DIR_DEFAULT, exist_ok=True)

# sweep: 横軸に出したい最大 T_bind
TBIND_MAX_DEFAULT = 18

# ★ learning curve 図タイトル（デフォルト）
CURVE_TITLE_DEFAULT = "Bind=4: Learning curve (Acc)  (FW seed + RNN mean)"

# ======================================================
# (1) sweep 用: モデル名と色/ラベル
# ======================================================
COLOR_MAP = {
    "fw":  "red",
    "rnn": "blue",
}

LABEL_MAP = {
    "rnn": "RNN+LN",
    "fw":  "Ba-FW",
}

# ファイル名例：
# fw_fw0_T8_S1_seed0_beta1.00_wait0.csv
PATTERN_TBIND = re.compile(r"_T(?P<T>\d+)_S(?P<S>\d+)_seed(?P<seed>\d+)")


# ======================================================
# (1) T_bind sweep: CSV 読み取り（最終 epoch の acc）
# ======================================================
def read_final_acc(path: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    for key in ["valid_acc", "val_acc", "acc_val", "acc"]:
        if key in df.columns:
            return float(df[key].iloc[-1])
    return None


def load_model_stats_sweep(model_dir: str):
    acc_by_T = {}

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN_TBIND.search(fname)
        if m is None:
            continue

        T = int(m.group("T"))
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_T.setdefault(T, []).append(acc)

    T_list, acc_mean, acc_std = [], [], []

    for T in sorted(acc_by_T.keys()):
        vals = acc_by_T[T]
        T_list.append(T)
        acc_mean.append(float(np.mean(vals)))
        acc_std.append(float(np.std(vals)))

    return T_list, acc_mean, acc_std


def plot_tbind_sweep(tbind_max: int, out_dir: str):
    # ★RNN と Ba-FW のみ
    model_dirs = {
        "fw":  os.path.join(CSV_ROOT_SWEEP, "results_Tbind_fw"),
        "rnn": os.path.join(CSV_ROOT_SWEEP, "results_Tbind_rnn"),
    }

    model_data = {}
    any_data = False

    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing model dir: {path}")
            continue

        T_list, mean_list, std_list = load_model_stats_sweep(path)
        model_data[model] = (T_list, mean_list, std_list)

        if len(T_list) > 0:
            any_data = True

    if not any_data:
        print("[ERROR] No T_bind data found for sweep.")
        return

    plt.figure(figsize=(8, 5))

    for model, (T_list, mean_list, std_list) in model_data.items():
        if len(T_list) == 0:
            continue

        # 薄いガイド線（点を結ぶ）
        plt.plot(
            T_list,
            mean_list,
            color=COLOR_MAP[model],
            linewidth=1.5,
            alpha=0.35,
            zorder=1,
        )

        # 点＋エラーバー
        plt.errorbar(
            T_list,
            mean_list,
            yerr=std_list,
            fmt="o",
            markersize=6,
            capsize=3,
            capthick=2.0,
            elinewidth=2.0,
            color=COLOR_MAP[model],
            label=LABEL_MAP[model],
            linestyle="none",
            zorder=2,
        )

    plt.xlabel("T_bind")
    plt.ylabel("Accuracy")
    plt.title("Effect of Bind Length on Accuracy")
    plt.ylim(0, 1.03)
    plt.xlim(1, tbind_max + 1)

    tick_T = sorted(set().union(*[model_data[m][0] for m in model_data if m in model_data]))
    plt.xticks(tick_T, [str(t) for t in tick_T])

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(out_dir, "tbind_sweep.png")
    out_eps = os.path.join(out_dir, "tbind_sweep.eps")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_eps, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_png}")
    print(f"[INFO] Saved → {out_eps}")


# ======================================================
# (2) Bind=4 learning curve: 共通ユーティリティ
# ======================================================
def pick_col(df: pd.DataFrame, candidates, kind: str):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(
        f"[Error] {kind} column not found. candidates={candidates}, available={list(df.columns)}"
    )


def load_one_csv(path: str):
    df = pd.read_csv(path)

    epoch_col = pick_col(df, ["epoch", "ep", "step"], "epoch")
    acc_col   = pick_col(df, ["val_acc", "valid_acc", "acc_val", "acc"], "val_acc")
    loss_col  = pick_col(df, ["val_loss", "valid_loss", "loss_val", "loss", "ce", "val_ce"], "val_loss")

    ep   = df[epoch_col].astype(float).to_numpy()
    acc  = df[acc_col].astype(float).to_numpy()
    loss = df[loss_col].astype(float).to_numpy()
    return ep, acc, loss


def extract_seed(path: str) -> int:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def aggregate(files):
    if len(files) == 0:
        raise FileNotFoundError("No CSV files found.")

    frames_acc = []
    frames_loss = []

    for f in files:
        ep, acc, loss = load_one_csv(f)
        seed = extract_seed(f)

        s_acc = pd.Series(acc, index=ep, name=f"seed{seed}")
        s_loss = pd.Series(loss, index=ep, name=f"seed{seed}")

        frames_acc.append(s_acc)
        frames_loss.append(s_loss)

    acc_df = pd.concat(frames_acc, axis=1).sort_index()
    loss_df = pd.concat(frames_loss, axis=1).sort_index()

    acc_mean = acc_df.mean(axis=1, skipna=True).to_numpy()
    acc_std  = acc_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    loss_mean = loss_df.mean(axis=1, skipna=True).to_numpy()
    loss_std  = loss_df.std(axis=1, ddof=0, skipna=True).to_numpy()

    epochs = acc_df.index.to_numpy(dtype=float)

    return epochs, acc_mean, acc_std, loss_mean, loss_std, acc_df, loss_df


def moving_average(y, w: int):
    if w <= 1:
        return y
    return pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()


def find_files_with_fallback(csv_dir: str, patterns):
    for pat in patterns:
        glob_pat = os.path.join(csv_dir, pat)
        files = sorted(glob.glob(glob_pat), key=extract_seed)
        if len(files) > 0:
            return files, pat
    return [], None


def plot_learning_curve_acc_fw_vs_rnn(
    fw_dir: str,
    rnn_dir: str,
    fw_pattern: str,
    rnn_pattern: str,
    smooth: int,
    out_dir: str,
    title: str,
    no_rnn: bool,
):
    out_prefix = os.path.join(out_dir, "learning_curve_acc_fw_vs_rnn_T4_wait2_beta1.00")

    # ======================================================
    # FW files
    # ======================================================
    fw_glob = os.path.join(fw_dir, fw_pattern)
    fw_files = sorted(glob.glob(fw_glob), key=extract_seed)
    if len(fw_files) == 0:
        raise FileNotFoundError(f"[FW] No files matched: {fw_glob}")

    fw_epochs, fw_acc_m, fw_acc_s, fw_loss_m, fw_loss_s, fw_acc_df, fw_loss_df = aggregate(fw_files)
    fw_acc_m_s  = moving_average(fw_acc_m, smooth)

    # ======================================================
    # RNN files (mean only)
    # ======================================================
    rnn_ok = False
    rnn_files = []
    used_pat = None

    if not no_rnn:
        rnn_patterns = [
            rnn_pattern,
            "fw_fw0_T4_S1_seed*_beta1.00_wait2.csv",
            "rnn*_T4*_seed*_beta1.00_wait2.csv",
            "rnn*_T4*_seed*_wait2*_beta1.00*.csv",
            "rnn*_T4*_seed*.csv",
        ]
        rnn_files, used_pat = find_files_with_fallback(rnn_dir, rnn_patterns)
        if len(rnn_files) > 0:
            rnn_epochs, rnn_acc_m, rnn_acc_s, rnn_loss_m, rnn_loss_s, _, _ = aggregate(rnn_files)
            rnn_acc_m_s  = moving_average(rnn_acc_m, smooth)
            rnn_ok = True
        else:
            print(f"[Warn] RNN files not found under: {rnn_dir}")
            print(f"[Warn] tried patterns: {rnn_patterns}")
            print("[Warn] continue plotting FW only.")

    # ======================================================
    # plot (ACC only)
    # ======================================================
    fig, ax1 = plt.subplots(1, 1, figsize=(8.6, 4.2))

    SEED_LW = 1.4
    SEED_ALPHA = 0.55
    MEAN_LW = 1.6
    BAND_ALPHA_FW = 0.08

    ax1.fill_between(
        fw_epochs, fw_acc_m - fw_acc_s, fw_acc_m + fw_acc_s,
        alpha=BAND_ALPHA_FW, zorder=1, label="FW ±1 std"
    )

    for col in fw_acc_df.columns:
        ax1.plot(
            fw_acc_df.index.values, fw_acc_df[col].values,
            linewidth=SEED_LW, alpha=SEED_ALPHA, zorder=2
        )

    ax1.plot(
        fw_epochs, fw_acc_m_s,
        color="black", linestyle="--", linewidth=MEAN_LW,
        zorder=3, label="FW mean"
    )

    if rnn_ok:
        ax1.plot(
            rnn_epochs, rnn_acc_m_s,
            color="red", linestyle="-", linewidth=MEAN_LW,
            zorder=4, label="RNN mean"
        )

    ax1.set_ylabel("Validation Acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", frameon=True)

    # ★ learning curve のタイトル（常に表示）
    ax1.set_title(title)

    fig.tight_layout()

    out_png = out_prefix + ".png"
    out_eps = out_prefix + ".eps"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_eps, bbox_inches="tight")

    print(f"[OK] FW matched {len(fw_files)} files: dir={fw_dir} pattern={fw_pattern}")
    if rnn_ok:
        print(f"[OK] RNN matched {len(rnn_files)} files: dir={rnn_dir} pattern(used)={used_pat}")
    else:
        print("[OK] RNN not added.")
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_eps}")


# ======================================================
# main
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    # ★共通出力先
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="common output directory for both figures")

    # --- sweep ---
    ap.add_argument("--tbind_max", type=int, default=TBIND_MAX_DEFAULT)
    ap.add_argument("--skip_sweep", action="store_true", help="skip T_bind sweep plot")

    # --- curve ---
    ap.add_argument("--fw_dir", default=FW_CSV_DIR_DEFAULT)
    ap.add_argument("--fw_pattern", default="fw_fw1_T4_S1_seed*_beta1.00_wait2.csv")

    ap.add_argument("--rnn_dir", default=RNN_CSV_DIR_DEFAULT)
    ap.add_argument("--rnn_pattern", default="rnn_T4_seed*_beta1.00_wait2.csv")
    ap.add_argument("--no_rnn", action="store_true", help="disable adding RNN mean")

    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--skip_curve", action="store_true", help="skip bind4 learning curve plot")

    # ★ここをデフォルトでタイトル付きに
    ap.add_argument("--curve_title", default=CURVE_TITLE_DEFAULT)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_sweep:
        plot_tbind_sweep(tbind_max=args.tbind_max, out_dir=args.out_dir)

    if not args.skip_curve:
        plot_learning_curve_acc_fw_vs_rnn(
            fw_dir=args.fw_dir,
            rnn_dir=args.rnn_dir,
            fw_pattern=args.fw_pattern,
            rnn_pattern=args.rnn_pattern,
            smooth=args.smooth,
            out_dir=args.out_dir,     # ★同じ out_dir
            title=args.curve_title,
            no_rnn=args.no_rnn,
        )


if __name__ == "__main__":
    main()

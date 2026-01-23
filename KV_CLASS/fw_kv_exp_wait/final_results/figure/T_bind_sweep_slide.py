import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# スクリプト自身のディレクトリ
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ あなたの構造に対応：T_bind ディレクトリを参照
CSV_ROOT = os.path.join(THIS_DIR, "..", "T_bind")

# 出力先
OUT_DIR = os.path.join(THIS_DIR, "Tbind_sweep_fig")
os.makedirs(OUT_DIR, exist_ok=True)

# ★ 横軸に出したい最大 T_bind（固定表示したい最大値）
TBIND_MAX = 18

# ======================================================
# モデル名と色
# ======================================================
COLOR_MAP = {
    "fw":   "red",
    "rnn":  "blue",
    "tanh": "green",
}

# ★ 表示用ラベル（論文用）
LABEL_MAP = {
    "rnn":  "RNN+LN",
    "fw":   "Ba-FW",
    "tanh": "SC-FW",
}

# ======================================================
# ファイル名例：
# fw_fw0_T8_S1_seed0_beta1.00_wait0.csv
# T / seed を抽出（Wait版と同じく末尾まで縛る）
# ======================================================
PATTERN = re.compile(
    r"_T(?P<T>\d+)_S(?P<S>\d+)_seed(?P<seed>\d+)_beta[0-9.]+_wait(?P<W>\d+)\.csv$"
)

# ======================================================
# CSV 読み取り：最終 epoch の acc を取得
# ======================================================
def read_final_acc(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "valid_acc" in df.columns:
        return float(df["valid_acc"].iloc[-1])
    if "acc" in df.columns:
        return float(df["acc"].iloc[-1])
    return None

# ======================================================
# 各モデルの T_bind → acc(list) を作る（seedごとの値を保持）
# ======================================================
def load_model_raw(model_dir):
    acc_by_T = {}  # T -> [acc1, acc2, ...]

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN.search(fname)
        if m is None:
            continue

        T = int(m.group("T"))
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_T.setdefault(T, []).append(acc)

    return acc_by_T

def summarize(acc_by_T):
    T_list, mean_list, std_list = [], [], []
    for T in sorted(acc_by_T.keys()):
        vals = np.array(acc_by_T[T], dtype=float)
        T_list.append(T)
        mean_list.append(float(vals.mean()))
        std_list.append(float(vals.std()))
    return T_list, mean_list, std_list

# ======================================================
# 1モデル分を描画（背景に他モデルの平均線も入れる）
# ======================================================
def plot_single_model(
    target_key,
    raw_by_model,
    summary_by_model,
    tick_T,
    x_label,
    y_label,
    style,
):
    target_label = LABEL_MAP[target_key]
    target_color = COLOR_MAP[target_key]

    T_t, mean_t, std_t = summary_by_model[target_key]
    acc_by_T_t = raw_by_model[target_key]

    plt.figure(figsize=(8, 5))

    # ----------------------------
    # 背景：対象以外の「平均線」を薄く
    # ----------------------------
    for other_key, (T_o, mean_o, _std_o) in summary_by_model.items():
        if other_key == target_key:
            continue
        plt.plot(
            T_o, mean_o,
            color=COLOR_MAP[other_key],
            linewidth=style["BG_MEAN_LW"],
            alpha=style["BG_ALPHA"],
            zorder=1,
            label=LABEL_MAP[other_key],
        )

    # ======================================================
    # ★追加：SC-FW（tanh）表示のときだけ Ba-FW の「seed点」も薄く出す
    # ======================================================
    if target_key == "tanh" and "fw" in raw_by_model:
        acc_by_T_fw = raw_by_model["fw"]
        T_fw = summary_by_model["fw"][0]  # fwのTリスト

        xs_fw, ys_fw = [], []
        for T in T_fw:
            vals = acc_by_T_fw.get(T, [])
            n = len(vals)
            if n == 0:
                continue

            if n == 1:
                offsets = [0.0]
            else:
                offsets = np.linspace(-style["JITTER_WIDTH"], style["JITTER_WIDTH"], n)

            for off, v in zip(offsets, vals):
                xs_fw.append(T + off)
                ys_fw.append(v)

        plt.scatter(
            xs_fw, ys_fw,
            s=style["SEED_S"],
            color=COLOR_MAP["fw"],
            alpha=style["BG_ALPHA"],     # 背景と同じ薄さ
            linewidths=0.0,
            zorder=1.6,
            label="_nolegend_",
        )

    # ----------------------------
    # 対象：各seed点（濃く）
    # ----------------------------
    xs, ys = [], []
    for T in T_t:
        vals = acc_by_T_t[T]
        n = len(vals)
        if n == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-style["JITTER_WIDTH"], style["JITTER_WIDTH"], n)

        for off, v in zip(offsets, vals):
            xs.append(T + off)
            ys.append(v)

    plt.scatter(
        xs, ys,
        s=style["SEED_S"],
        color=target_color,
        alpha=style["SEED_ALPHA"],
        linewidths=0.0,
        zorder=2,
    )

    # ----------------------------
    # 対象：平均線（濃く）
    # ----------------------------
    plt.plot(
        T_t, mean_t,
        color=target_color,
        linewidth=style["MEAN_LW"],
        alpha=style["MEAN_ALPHA"],
        zorder=3,
        label=target_label,
    )

    # ----------------------------
    # 対象：平均±std（エラーバー）
    # ----------------------------
    plt.errorbar(
        T_t,
        mean_t,
        yerr=std_t,
        fmt="none",
        capsize=style["CAPSIZE"],
        capthick=style["ERR_CAPTHICK"],
        elinewidth=style["ERR_ELINEWIDTH"],
        ecolor=target_color,
        alpha=style["ERR_ALPHA"],
        linestyle="none",
        zorder=4,
        label="_nolegend_",
    )

    # ======================================================
    # 軸ラベル・tickサイズ・タイトル無し
    # ======================================================
    plt.xlabel(x_label, fontsize=style["AXIS_LABEL_FONTSIZE"])
    plt.ylabel(y_label, fontsize=style["AXIS_LABEL_FONTSIZE"])

    plt.xticks(tick_T, [str(t) for t in tick_T], fontsize=style["TICK_FONTSIZE"])
    plt.yticks(fontsize=style["TICK_FONTSIZE"])

    plt.ylim(0, 1.03)
    plt.xlim(1, TBIND_MAX + 1)

    plt.grid(True, alpha=style["GRID_ALPHA"])

    if style["SHOW_LEGEND"]:
        plt.legend(fontsize=style["LEGEND_FONTSIZE"])

    plt.tight_layout()

    out_base = os.path.join(OUT_DIR, f"tbind_sweep_{target_key}")
    plt.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(out_base + ".eps", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_base}.png / .eps")

# ======================================================
# メイン：全モデル読み込み→モデル別に3枚出す
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    # axis label
    ap.add_argument("--xlabel", default=r"$N_{\mathrm{pair}}$", help="x-axis label")
    ap.add_argument("--ylabel", default="Accuracy", help="y-axis label")

    # style controls（学習曲線スクリプトと同じノリ）
    ap.add_argument("--axis_label_fs", type=int, default=15)
    ap.add_argument("--tick_fs", type=int, default=13)
    ap.add_argument("--legend_fs", type=int, default=11)

    ap.add_argument("--bg_alpha", type=float, default=0.20)
    ap.add_argument("--bg_mean_lw", type=float, default=1.5)

    ap.add_argument("--seed_s", type=float, default=22.0)
    ap.add_argument("--seed_alpha", type=float, default=0.85)

    ap.add_argument("--mean_lw", type=float, default=1.5)
    ap.add_argument("--mean_alpha", type=float, default=0.85)

    ap.add_argument("--err_alpha", type=float, default=0.45)
    ap.add_argument("--err_elinewidth", type=float, default=1.0)
    ap.add_argument("--err_capthick", type=float, default=1.0)
    ap.add_argument("--capsize", type=float, default=3.0)

    ap.add_argument("--grid_alpha", type=float, default=0.25)

    ap.add_argument("--jitter_width", type=float, default=0.18)

    ap.add_argument("--no_legend", action="store_true", help="disable legend")

    args = ap.parse_args()

    model_dirs = {
        "fw":   os.path.join(CSV_ROOT, "results_Tbind_fw"),
        "rnn":  os.path.join(CSV_ROOT, "results_Tbind_rnn"),
        "tanh": os.path.join(CSV_ROOT, "results_Tbind_tanh"),
    }

    raw_by_model = {}
    summary_by_model = {}

    # 読み込み
    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing model dir: {path}")
            continue

        acc_by_T = load_model_raw(path)
        if len(acc_by_T) == 0:
            print(f"[WARN] No CSV matched in: {path}")
            continue

        raw_by_model[model] = acc_by_T
        summary_by_model[model] = summarize(acc_by_T)

    if len(summary_by_model) == 0:
        print("[ERROR] No T_bind data found.")
        return

    # x軸 tick を統一（全モデルの T_bind の和集合）
    tick_T = sorted(set().union(*[set(summary_by_model[m][0]) for m in summary_by_model]))

    # style dict
    style = {
        "AXIS_LABEL_FONTSIZE": args.axis_label_fs,
        "TICK_FONTSIZE": args.tick_fs,
        "LEGEND_FONTSIZE": args.legend_fs,

        "BG_ALPHA": args.bg_alpha,
        "BG_MEAN_LW": args.bg_mean_lw,

        "SEED_S": args.seed_s,
        "SEED_ALPHA": args.seed_alpha,

        "MEAN_LW": args.mean_lw,
        "MEAN_ALPHA": args.mean_alpha,

        "ERR_ALPHA": args.err_alpha,
        "ERR_ELINEWIDTH": args.err_elinewidth,
        "ERR_CAPTHICK": args.err_capthick,
        "CAPSIZE": args.capsize,

        "GRID_ALPHA": args.grid_alpha,

        "JITTER_WIDTH": args.jitter_width,

        "SHOW_LEGEND": (not args.no_legend),
    }

    # 図を出す（存在するモデルだけ）
    for target_key in ["fw", "rnn", "tanh"]:
        if target_key not in summary_by_model:
            continue
        plot_single_model(
            target_key,
            raw_by_model,
            summary_by_model,
            tick_T,
            args.xlabel,
            args.ylabel,
            style,
        )

if __name__ == "__main__":
    main()

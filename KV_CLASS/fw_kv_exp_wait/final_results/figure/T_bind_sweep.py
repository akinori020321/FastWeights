import os
import re
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
def plot_single_model(target_key, raw_by_model, summary_by_model, tick_T):
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
            linewidth=1.5,
            alpha=0.20,      # ★薄い
            zorder=1,
            label=LABEL_MAP[other_key],  # 凡例に出す
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
            jitter_width = 0.18
            offsets = np.linspace(-jitter_width, jitter_width, n)
        for off, v in zip(offsets, vals):
            xs.append(T + off)
            ys.append(v)

    plt.scatter(
        xs, ys,
        s=22,
        color=target_color,
        alpha=0.85,
        linewidths=0.0,
        zorder=2,
    )

    # ----------------------------
    # 対象：平均線（濃く）
    # ----------------------------
    plt.plot(
        T_t, mean_t,
        color=target_color,
        linewidth=1.5,
        alpha=0.85,
        zorder=3,
        label=target_label,
    )

    # ----------------------------
    # 対象：平均±std（エラーバーのみ：平均丸は消す）
    # ----------------------------
    plt.errorbar(
        T_t,
        mean_t,
        yerr=std_t,
        fmt="none",          # ★平均の丸を消す
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=target_color,
        markeredgewidth=1.2,
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        ecolor=target_color,
        alpha=0.45,
        linestyle="none",
        zorder=4,
        label="_nolegend_",
    )

    plt.xlabel("T_bind")
    plt.ylabel("Accuracy")
    plt.title(f"Effect of Bind Length on Accuracy ({target_label})")

    plt.ylim(0, 1.03)
    plt.xlim(1, TBIND_MAX + 1)

    plt.xticks(tick_T, [str(t) for t in tick_T])

    plt.grid(True, alpha=0.25)
    plt.legend()
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

    # 図を出す（存在するモデルだけ）
    for target_key in ["fw", "rnn", "tanh"]:
        if target_key not in summary_by_model:
            continue
        plot_single_model(target_key, raw_by_model, summary_by_model, tick_T)

if __name__ == "__main__":
    main()

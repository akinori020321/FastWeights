import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# スクリプト自身のディレクトリ
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ Wait ディレクトリ（あなたの構造に合わせる）
CSV_ROOT = os.path.join(THIS_DIR, "..", "Wait")

# 出力先
OUT_DIR = os.path.join(THIS_DIR, "Wait_sweep_fig")
os.makedirs(OUT_DIR, exist_ok=True)

# ★ 横軸に出したい最大 wait（固定）
WAIT_MAX = 18

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
# fw0_sigma0.00_S1_seed0_beta1.00_wait6.csv
# Seed と wait を正しく抽出する regex
# ======================================================
PATTERN = re.compile(
    r"_S(?P<S>\d+)_seed(?P<seed>\d+)_beta[0-9.]+_wait(?P<W>\d+)\.csv$"
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
# 各モデルの wait → acc の dict を作る
# ======================================================
def load_model_stats(model_dir):
    acc_by_W = {}

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN.search(fname)
        if m is None:
            continue

        W = int(m.group("W"))
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_W.setdefault(W, []).append(acc)

    W_list = []
    acc_mean_list = []
    acc_std_list = []

    for W in sorted(acc_by_W.keys()):
        vals = acc_by_W[W]
        W_list.append(W)
        acc_mean_list.append(float(np.mean(vals)))
        acc_std_list.append(float(np.std(vals)))

    return W_list, acc_mean_list, acc_std_list

# ======================================================
# メイン：点＋エラーバー + 薄いガイド線
# ======================================================
def main():

    model_dirs = {
        "fw":   os.path.join(CSV_ROOT, "results_wait_fw"),
        "rnn":  os.path.join(CSV_ROOT, "results_wait_rnn"),
        "tanh": os.path.join(CSV_ROOT, "results_wait_tanh"),
    }

    model_data = {}
    any_data = False

    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing model dir: {path}")
            continue

        W_list, mean_list, std_list = load_model_stats(path)
        model_data[model] = (W_list, mean_list, std_list)

        if len(W_list) > 0:
            any_data = True

    if not any_data:
        print("[ERROR] No wait data found.")
        return

    # ======================================================
    # Plot
    # ======================================================
    plt.figure(figsize=(8, 5))  # Tbind と一致

    for model, (W_list, mean_list, std_list) in model_data.items():
        if len(W_list) == 0:
            continue

        # ★ 薄いガイド線（点を結ぶ）
        plt.plot(
            W_list,
            mean_list,
            color=COLOR_MAP[model],
            linewidth=1.5,
            alpha=0.35,
            zorder=1,
        )

        plt.errorbar(
            W_list,
            mean_list,
            yerr=std_list,
            fmt="o",
            markersize=6,
            capsize=3,        # ← capを少し長く
            capthick=2.0,     # ← cap線を太く
            elinewidth=2.0,   # ← 誤差バー本体を太く
            color=COLOR_MAP[model],
            label=LABEL_MAP[model],   # ★ ここだけ変更
            linestyle="none",
            zorder=2,
        )

    plt.xlabel("Wait")
    plt.ylabel("Accuracy")
    plt.title("Effect of Wait Length on Accuracy")
    plt.ylim(0, 1.03)  # ★ 1.0の上に余白
    plt.xlim(-1, WAIT_MAX + 1)

    tick_W = sorted(set().union(*[model_data[m][0] for m in model_data]))
    plt.xticks(tick_W, [str(w) for w in tick_W])

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "wait_sweep.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path[:-4] + ".eps", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")

if __name__ == "__main__":
    main()

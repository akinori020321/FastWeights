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

# ファイル名例：
# fw_fw0_T8_S1_seed0_beta1.00_wait0.csv
PATTERN = re.compile(
    r"_T(?P<T>\d+)_S(?P<S>\d+)_seed(?P<seed>\d+)"
)


# ======================================================
# CSV 読み取り：最終 epoch の acc を取得
# ======================================================
def read_final_acc(path):
    try:
        df = pd.read_csv(path)
    except:
        return None
    if "valid_acc" in df.columns:
        return float(df["valid_acc"].iloc[-1])
    elif "acc" in df.columns:
        return float(df["acc"].iloc[-1])
    return None


# ======================================================
# 各モデルの T_bind → acc の dict を作る
# ======================================================
def load_model_stats(model_dir):
    acc_by_T = {}

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

    # ソートして返す
    T_list = []
    acc_mean = []
    acc_std = []

    for T in sorted(acc_by_T.keys()):
        vals = acc_by_T[T]
        T_list.append(T)
        acc_mean.append(np.mean(vals))
        acc_std.append(np.std(vals))

    return T_list, acc_mean, acc_std


# ======================================================
# メイン：点＋エラーバー
# ======================================================
def main():

    model_dirs = {
        "fw":   os.path.join(CSV_ROOT, "results_Tbind_fw"),
        "rnn":  os.path.join(CSV_ROOT, "results_Tbind_rnn"),
        "tanh": os.path.join(CSV_ROOT, "results_Tbind_tanh"),
    }

    model_data = {}
    any_data = False

    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing model dir: {path}")
            continue

        T_list, mean_list, std_list = load_model_stats(path)
        model_data[model] = (T_list, mean_list, std_list)

        if len(T_list) > 0:
            any_data = True

    if not any_data:
        print("[ERROR] No T_bind data found.")
        return

    # ======================================================
    # Plot
    # ======================================================
    plt.figure(figsize=(8, 5))  # ← 横幅 8 インチ固定

    for model, (T_list, mean_list, std_list) in model_data.items():

        if len(T_list) == 0:
            continue

        # bind の値そのものでプロット（横軸の幅は 1〜18 で固定される）
        plt.errorbar(
            T_list,
            mean_list,
            yerr=std_list,
            fmt="o",
            markersize=6,
            capsize=4,
            color=COLOR_MAP[model],
            label=model.upper(),
            linestyle="none"
        )

    plt.xlabel("T_bind")
    plt.ylabel("Accuracy")
    plt.title("T_bind Sweep (points + error bars)")
    plt.ylim(0, 1.0)

    # === ★ 横軸は 1〜TBIND_MAX のまま ===
    plt.xlim(1, TBIND_MAX+1)

    # === ★ tick は “データがある T_bind のみ” ===
    tick_T = sorted(set().union(*[model_data[m][0] for m in model_data]))
    plt.xticks(tick_T, [str(t) for t in tick_T])

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "tbind_sweep.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")


if __name__ == "__main__":
    main()

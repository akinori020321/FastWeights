import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# スクリプト自身のディレクトリ
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ ノイズスイープ CSV のルート
CSV_ROOT = os.path.join(THIS_DIR, "..", "noise_sweep")

# 出力先
OUT_DIR = os.path.join(THIS_DIR, "noise_fig")
os.makedirs(OUT_DIR, exist_ok=True)

# ★ 横軸に出したい clean_rate（= sigma）
CLEAN_RATE_LIST = [1.0, 0.8, 0.6, 0.4, 0.2]

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
# recon_fw_fw0_beta0.0_dup4_sigma0.2_S0_seed0_fw_recon_beta0.00_wait0_dup4.csv
# → sigma を抽出する regex
# ======================================================
PATTERN = re.compile(r"_sigma(?P<sigma>[0-9.]+)_S(?P<S>\d+)_seed(?P<seed>\d+)_")

# ======================================================
# CSV から最終 valid_acc を取得
# ======================================================
def read_final_acc(path):
    try:
        df = pd.read_csv(path)
    except:
        return None

    if "valid_acc" in df.columns:
        return float(df["valid_acc"].iloc[-1])

    return None

# ======================================================
# モデルごとに clean_rate → acc を集計
# ======================================================
def load_model_stats(model_dir):

    acc_by_rate = {}  # sigma → [acc]

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN.search(fname)
        if m is None:
            continue

        sigma = float(m.group("sigma"))        # clean_rate = sigma
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_rate.setdefault(sigma, []).append(acc)

    # ソート済みリスト作成
    rate_list = []
    mean_list = []
    std_list = []

    for rate in sorted(acc_by_rate.keys()):
        vals = acc_by_rate[rate]
        rate_list.append(rate)
        mean_list.append(np.mean(vals))
        std_list.append(np.std(vals))

    return rate_list, mean_list, std_list


# ======================================================
# メイン：点＋エラーバー
# ======================================================
def main():

    model_dirs = {
        "fw":   os.path.join(CSV_ROOT, "results_recon_fw"),
        "rnn":  os.path.join(CSV_ROOT, "results_recon_rnn"),
        "tanh": os.path.join(CSV_ROOT, "results_recon_tanh"),
    }

    model_data = {}
    any_data = False

    # データ読み込み
    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing dir: {path}")
            continue

        rate_list, mean_list, std_list = load_model_stats(path)
        model_data[model] = (rate_list, mean_list, std_list)

        if len(rate_list) > 0:
            any_data = True

    if not any_data:
        print("[ERROR] No data found.")
        return

    # ======================================================
    # Plot
    # ======================================================
    plt.figure(figsize=(8, 5))  # Wait_sweep と完全一致のサイズ

    for model, (rate_list, mean_list, std_list) in model_data.items():

        if len(rate_list) == 0:
            continue

        # ★ 薄いガイド線（点を結ぶ）
        plt.plot(
            rate_list,
            mean_list,
            color=COLOR_MAP[model],
            linewidth=1.5,
            alpha=0.35,
            zorder=1,
        )

        plt.errorbar(
            rate_list,
            mean_list,
            yerr=std_list,
            fmt="o",
            markersize=6,
            capsize=3,        # ← capを少し長く
            capthick=2.0,     # ← cap線を太く
            elinewidth=2.0,   # ← 誤差バー本体を太く
            color=COLOR_MAP[model],
            label=LABEL_MAP[model],
            linestyle="none",
            zorder=2,
        )

    plt.xlabel("Clean Rate")
    plt.ylabel("Accuracy (cosine)")
    plt.title("Effect of Clean Rate on Accuracy")
    plt.ylim(0, 1.03)  # ★ 1.0の上に余白

    # ★ 横軸は固定レンジ（1.0 → 0.2）
    plt.xlim(1.03, 0.17)

    # ★ tick は実際にある sigma のみ
    tick_vals = sorted(
        set().union(*[model_data[m][0] for m in model_data]),
        reverse=True
    )
    plt.xticks(tick_vals, [str(v) for v in tick_vals])

    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "clean_rate_sweep.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    out_path_eps = os.path.join(OUT_DIR, "clean_rate_sweep.eps")
    plt.savefig(out_path_eps, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Saved → {out_path_eps}")


if __name__ == "__main__":
    main()

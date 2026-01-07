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
def read_final_acc(path: str):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "valid_acc" in df.columns and len(df) > 0:
        return float(df["valid_acc"].iloc[-1])

    return None

# ======================================================
# モデルごとに clean_rate → acc（seedの生値）を集計
# ======================================================
def load_model_raw(model_dir: str):
    acc_by_rate = {}  # sigma -> [acc]

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN.search(fname)
        if m is None:
            continue

        sigma = float(m.group("sigma"))
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_rate.setdefault(sigma, []).append(acc)

    return acc_by_rate

# ======================================================
# CLEAN_RATE_LIST に揃えて mean/std と raw を返す
# ======================================================
def align_stats(acc_by_rate: dict, rate_list: list[float]):
    mean_arr = np.full(len(rate_list), np.nan, dtype=float)
    std_arr  = np.full(len(rate_list), np.nan, dtype=float)
    raw_list = []

    for i, r in enumerate(rate_list):
        vals = acc_by_rate.get(r, [])
        raw_list.append(vals)
        if len(vals) == 0:
            continue
        mean_arr[i] = float(np.mean(vals))
        std_arr[i]  = float(np.std(vals))

    return mean_arr, std_arr, raw_list

# ======================================================
# 1パネル描画（平均線＋エラーバー＋薄い生点）
# ======================================================
def plot_one_panel(ax, rate_list, mean_arr, std_arr, raw_list, color, title, show_xlabel=False):
    x = np.array(rate_list, dtype=float)

    # ---- seed生点（かなり薄く）----
    for xi, vals in zip(x, raw_list):
        if len(vals) == 0:
            continue
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.015
        ax.scatter(
            np.full(len(vals), xi) + jitter,
            vals,
            s=16,
            color=color,
            alpha=0.10,
            edgecolors="none",
            zorder=1,
        )

    # ---- 平均を結ぶ線（薄く）----
    ax.plot(
        x,
        mean_arr,
        color=color,
        linewidth=1.8,
        alpha=0.22,
        zorder=2,
    )

    # ---- エラーバー（主役：前面）----
    ax.errorbar(
        x,
        mean_arr,
        yerr=std_arr,
        fmt="none",
        ecolor=color,
        elinewidth=2.3,
        capsize=4,
        capthick=2.3,
        alpha=0.95,
        zorder=5,
    )

    # ---- 平均点（小さく・エラーバーより奥）----
    ax.scatter(
        x,
        mean_arr,
        s=18,
        color=color,
        alpha=0.95,
        edgecolors="none",
        zorder=4,
    )

    ax.set_title(title)
    ax.set_ylim(0, 1.03)

    # x軸は 1.0 -> 0.2（逆向き）
    ax.set_xlim(1.03, 0.17)
    ax.set_xticks(rate_list)
    ax.set_xticklabels([str(v) for v in rate_list])

    if not show_xlabel:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)

    ax.grid(True, alpha=0.35)

# ======================================================
# メイン：3つを縦に並べて1枚にまとめる
# ======================================================
def main():
    np.random.seed(0)  # jitter 再現性（不要なら消してOK）

    model_dirs = {
        "fw":   os.path.join(CSV_ROOT, "results_recon_fw"),
        "rnn":  os.path.join(CSV_ROOT, "results_recon_rnn"),
        "tanh": os.path.join(CSV_ROOT, "results_recon_tanh"),
    }

    # ---- データ読み込み（raw）----
    raw_by_model = {}
    any_data = False
    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing dir: {path}")
            continue

        acc_by_rate = load_model_raw(path)
        raw_by_model[model] = acc_by_rate
        if any(len(v) > 0 for v in acc_by_rate.values()):
            any_data = True

    if not any_data:
        print("[ERROR] No data found.")
        return

    # ======================================================
    # Plot（縦3段：RNN+LN / Ba-FW / SC-FW）
    # ======================================================
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 9.0),
        sharex=True, sharey=True
    )

    # ★順番を RNN -> Ba -> SC に変更
    order = ["rnn", "fw", "tanh"]

    for i, (ax, model) in enumerate(zip(axes, order)):
        if model not in raw_by_model:
            ax.axis("off")
            continue

        mean_arr, std_arr, raw_list = align_stats(raw_by_model[model], CLEAN_RATE_LIST)

        plot_one_panel(
            ax=ax,
            rate_list=CLEAN_RATE_LIST,
            mean_arr=mean_arr,
            std_arr=std_arr,
            raw_list=raw_list,
            color=COLOR_MAP[model],
            title=LABEL_MAP[model],
            show_xlabel=(i == 2),
        )

    axes[1].set_ylabel("Accuracy (cosine)")
    axes[2].set_xlabel("Clean Rate")
    axes[2].tick_params(axis="x", which="both", labelbottom=True)

    fig.suptitle("Effect of Clean Rate on Accuracy", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=1.0)

    out_path = os.path.join(OUT_DIR, "clean_rate_sweep_smallmultiples_vertical.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    out_path_eps = os.path.join(OUT_DIR, "clean_rate_sweep_smallmultiples_vertical.eps")
    fig.savefig(out_path_eps, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved → {out_path}")
    print(f"[INFO] Saved → {out_path_eps}")

if __name__ == "__main__":
    main()

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
#   ※表示はこの順（左がきれい）
CLEAN_RATE_LIST = [1.0, 0.8, 0.6, 0.4, 0.2]

# ======================================================
# モデル名と色
# ======================================================
COLOR_MAP = {
    "fw":    "red",
    "rnn":   "blue",
    "tanh":  "green",
    "tanh3": "green",   # ★追加（S=3）
}

# ★ 表示用ラベル（論文用）
LABEL_MAP = {
    "rnn":   "RNN+LN",
    "fw":    "Ba-FW",
    "tanh":  "SC-FW",
    "tanh3": "SC-FW (S=3)",  # ★追加（S=3）
}

# ======================================================
# ファイル名例：
# recon_fw_fw0_beta0.0_dup4_sigma0.2_S0_seed0_fw_recon_beta0.00_wait0_dup4.csv
# → sigma を抽出する regex
# ======================================================
PATTERN = re.compile(r"_sigma(?P<sigma>[0-9.]+)_S(?P<S>\d+)_seed(?P<seed>\d+)_")

# ======================================================
# float key の丸め（0.6000000001 対策）
# ======================================================
def norm_key(x: float, ndigits: int = 3) -> float:
    return float(np.round(float(x), ndigits))

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
# モデルごとに clean_rate(sigma) → acc（seedの生値）を集計
# ======================================================
def load_model_raw(model_dir: str):
    acc_by_rate = {}  # sigma -> [acc]

    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv"):
            continue

        m = PATTERN.search(fname)
        if m is None:
            continue

        sigma = norm_key(float(m.group("sigma")))
        acc = read_final_acc(os.path.join(model_dir, fname))
        if acc is None:
            continue

        acc_by_rate.setdefault(sigma, []).append(acc)

    return acc_by_rate

# ======================================================
# CLEAN_RATE_LIST に揃えて mean/std と raw を返す（NaNで欠損許容）
# ======================================================
def align_stats(acc_by_rate: dict, rate_list):
    rate_list = [norm_key(r) for r in rate_list]
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

    return rate_list, mean_arr, std_arr, raw_list

# ======================================================
# 1モデル分を描画（背景に他モデルの平均線も入れる）
# ======================================================
def plot_single_model(target_key, raw_by_model, aligned_by_model):
    target_label = LABEL_MAP[target_key]
    target_color = COLOR_MAP[target_key]

    rate_list, mean_t, std_t, raw_t = aligned_by_model[target_key]
    x = np.array(rate_list, dtype=float)

    plt.figure(figsize=(8, 5))

    # ----------------------------
    # 背景：対象以外の「平均線」を薄く
    # ----------------------------
    for other_key, (rate_o, mean_o, _std_o, _raw_o) in aligned_by_model.items():
        if other_key == target_key:
            continue

        # ★ S=3以外の図にはS=3を表示しない（ここだけ追加）
        if other_key == "tanh3" and target_key != "tanh3":
            continue

        # ★ S=3（tanh3）の図では、S=1（tanh）の薄い線だけを表示
        if target_key == "tanh3" and other_key != "tanh":
            continue

        xo = np.array(rate_o, dtype=float)

        # ★ tanh3 図のとき、背景ラベルだけ SC-FW (S=1) にする（他はそのまま）
        bg_label = LABEL_MAP[other_key]
        if target_key == "tanh3" and other_key == "tanh":
            bg_label = "SC-FW (S=1)"

        plt.plot(
            xo, mean_o,
            color=COLOR_MAP[other_key],
            linewidth=1.5,
            alpha=0.20,  # ★薄い（Wait版と同じ）
            zorder=1,
            label=bg_label,
        )

    # ----------------------------
    # 対象：各seed点（濃く）
    # ----------------------------
    xs, ys = [], []
    for xi, vals in zip(x, raw_t):
        n = len(vals)
        if n == 0:
            continue
        if n == 1:
            offsets = [0.0]
        else:
            jitter_width = 0.03  # 0.2刻みなのでこれくらいが見やすい
            offsets = np.linspace(-jitter_width, jitter_width, n)
        for off, v in zip(offsets, vals):
            xs.append(xi + off)
            ys.append(v)

    plt.scatter(
        xs, ys,
        s=22,
        color=target_color,
        alpha=0.85,      # ★点を濃く
        linewidths=0.0,
        zorder=2,
    )

    # ----------------------------
    # 対象：平均線（濃く）
    # ----------------------------
    plt.plot(
        x, mean_t,
        color=target_color,
        linewidth=1.5,
        alpha=0.85,      # ★濃い
        zorder=3,
        label=target_label,
    )

    # ----------------------------
    # 対象：平均±std（エラーバーのみ：平均丸なし）
    # ----------------------------
    if target_key != "tanh3":
        plt.errorbar(
            x,
            mean_t,
            yerr=std_t,
            fmt="none",
            ecolor=target_color,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            alpha=0.45,
            linestyle="none",
            zorder=4,
            label="_nolegend_",
        )

    plt.xlabel("Clean Rate")
    plt.ylabel("Accuracy (cosine)")
    plt.title(f"Effect of Clean Rate on Accuracy ({target_label})")

    plt.ylim(0, 1.03)

    # x軸は 1.0 -> 0.2（逆向き）
    plt.xlim(1.06, 0.14)  # ★端が切れないように余白を追加（ここ以外は変更なし）
    plt.xticks(rate_list, [str(v) for v in rate_list])

    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_base = os.path.join(OUT_DIR, f"clean_rate_sweep_{target_key}")
    plt.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(out_base + ".eps", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_base}.png / .eps")

# ======================================================
# メイン：全モデル読み込み→モデル別に3枚出す
# ======================================================
def main():
    model_dirs = {
        "fw":    os.path.join(CSV_ROOT, "results_recon_fw"),
        "rnn":   os.path.join(CSV_ROOT, "results_recon_rnn"),
        "tanh":  os.path.join(CSV_ROOT, "results_recon_tanh"),
        "tanh3": os.path.join(CSV_ROOT, "results_recon_tanh3"),  # ★追加（S=3）
    }

    raw_by_model = {}
    aligned_by_model = {}
    any_data = False

    # 読み込み & 整列
    for model, path in model_dirs.items():
        if not os.path.isdir(path):
            print(f"[WARN] Missing dir: {path}")
            continue

        acc_by_rate = load_model_raw(path)
        raw_by_model[model] = acc_by_rate

        rate_list, mean_arr, std_arr, raw_list = align_stats(acc_by_rate, CLEAN_RATE_LIST)
        aligned_by_model[model] = (rate_list, mean_arr, std_arr, raw_list)

        if any(len(v) > 0 for v in acc_by_rate.values()):
            any_data = True

    if not any_data:
        print("[ERROR] No data found.")
        return

    # 図を出す（存在するモデルだけ）
    for target_key in ["fw", "rnn", "tanh", "tanh3"]:
        if target_key not in aligned_by_model:
            continue
        plot_single_model(target_key, raw_by_model, aligned_by_model)

if __name__ == "__main__":
    main()

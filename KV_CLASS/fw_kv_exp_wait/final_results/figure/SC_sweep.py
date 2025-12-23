import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# スクリプト自身の場所を基準にパスを決定
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV のあるディレクトリ（スクリプトの１つ上）
CSV_ROOT = os.path.join(THIS_DIR, "..", "SC_sweep")

# 出力先ディレクトリ
OUT_DIR = os.path.join(THIS_DIR, "SC_sweep_fig")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# 固定パラメータ
# ======================================================
LAM_LIST = [0.65, 0.80, 0.95]
ETA_LIST = [0.30, 0.50, 0.70]

# lam, eta, S をファイル名から抜き出す正規表現
FILE_PATTERN = re.compile(
    r"lam(?P<lam>[0-9.]+)_eta(?P<eta>[0-9.]+)_S(?P<S>\d+)_"
)

# float比較のブレ対策
def match_index(x, candidates, tol=1e-9):
    for i, c in enumerate(candidates):
        if abs(x - c) <= tol:
            return i
    return None


# ======================================================
# results_SC*_lambda_eta_sweep ディレクトリを探す
# ======================================================
def find_s_dirs(root):
    dirs = []
    for name in os.listdir(root):
        if name.startswith("results_SC") and name.endswith("_lambda_eta_sweep"):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                dirs.append(full)
    return sorted(dirs)


# ======================================================
# CSV から valid_acc を読み取る
# ======================================================
def read_valid_acc(path):
    try:
        df = pd.read_csv(path)
        if "valid_acc" not in df.columns or len(df) == 0:
            return np.nan
        return float(df["valid_acc"].iloc[-1])
    except Exception:
        return np.nan


# ======================================================
# 個々の S ディレクトリを処理してヒートマップ生成（seed平均）
# ======================================================
def process_s_dir(s_path):
    print(f"[INFO] Processing {s_path}")

    # S 値をディレクトリ名から抽出（SCバージョン）
    m = re.search(r"results_SC(\d+)_", s_path)
    S_value = int(m.group(1)) if m else None

    # (lam, eta) ごとに acc をリストで貯める
    acc_lists = [[[] for _ in ETA_LIST] for _ in LAM_LIST]

    # CSV を読み込む（順序固定）
    for fname in sorted(os.listdir(s_path)):
        if not fname.endswith(".csv"):
            continue
        m = FILE_PATTERN.search(fname)
        if m is None:
            continue

        lam = float(m.group("lam"))
        eta = float(m.group("eta"))

        lam_i = match_index(lam, LAM_LIST)
        eta_i = match_index(eta, ETA_LIST)
        if lam_i is None or eta_i is None:
            continue

        acc = read_valid_acc(os.path.join(s_path, fname))
        if np.isfinite(acc):
            acc_lists[lam_i][eta_i].append(acc)

    # mean/std/n 行列を作る
    mat_mean = np.full((len(LAM_LIST), len(ETA_LIST)), np.nan)
    mat_std  = np.full((len(LAM_LIST), len(ETA_LIST)), np.nan)
    mat_n    = np.zeros((len(LAM_LIST), len(ETA_LIST)), dtype=int)

    for i in range(len(LAM_LIST)):
        for j in range(len(ETA_LIST)):
            vals = np.array(acc_lists[i][j], dtype=float)
            vals = vals[np.isfinite(vals)]
            mat_n[i, j] = len(vals)
            if len(vals) > 0:
                mat_mean[i, j] = float(vals.mean())
                mat_std[i, j]  = float(vals.std(ddof=0))  # population std

    # 有効データが無い場合スキップ
    if np.isnan(mat_mean).all():
        print(f"[WARN] No usable CSV inside {s_path}, skip")
        return

    # 表示用DF（平均値）
    df_mean = pd.DataFrame(mat_mean, index=LAM_LIST, columns=ETA_LIST)

    # annot を「平均値 + (n=◯)」にする
    annot = np.empty_like(mat_mean, dtype=object)
    for i in range(len(LAM_LIST)):
        for j in range(len(ETA_LIST)):
            if mat_n[i, j] == 0 or not np.isfinite(mat_mean[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{mat_mean[i,j]:.2f}"

    # ======================================================
    #   ヒートマップ描画 (低いと赤 / 高いと緑)
    # ======================================================
    sns.set(font_scale=1.3)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df_mean,
        annot=annot,
        fmt="",          # annotが文字列なのでfmt不要
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0
    )

    plt.title(f"Accuracy Heatmap (mean over seeds) of SC-FW (S = {S_value})")
    plt.xlabel("eta")
    plt.ylabel("lambda")

    # 保存先：SC_sweep_fig/
    outname = f"heatmap_mean_S{S_value}.png"
    outpath = os.path.join(OUT_DIR, outname)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath[:-4] + ".eps", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {outpath}")

    # 集計CSVも保存（平均・標準偏差・n）
    rows = []
    for i, lam in enumerate(LAM_LIST):
        for j, eta in enumerate(ETA_LIST):
            rows.append({
                "lambda": lam,
                "eta": eta,
                "mean_valid_acc": mat_mean[i, j],
                "std_valid_acc": mat_std[i, j],
                "n": mat_n[i, j],
            })
    df_out = pd.DataFrame(rows)
    sum_path = os.path.join(OUT_DIR, f"summary_mean_std_n_S{S_value}.csv")
    df_out.to_csv(sum_path, index=False)
    print(f"[INFO] Saved summary → {sum_path}")


# ======================================================
# メイン処理
# ======================================================
def main():
    s_dirs = find_s_dirs(CSV_ROOT)

    if not s_dirs:
        print("[ERROR] No results_SC*_lambda_eta_sweep found.")
        return

    for s_dir in s_dirs:
        process_s_dir(s_dir)


if __name__ == "__main__":
    main()

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
CSV_ROOT = os.path.join(THIS_DIR, "..", "FW_sweep")

# 出力先ディレクトリ
OUT_DIR = os.path.join(THIS_DIR, "FW_sweep_fig")
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


# ======================================================
# results_fw*_lambda_eta_sweep ディレクトリを探す
# ======================================================
def find_s_dirs(root):
    dirs = []
    for name in os.listdir(root):
        if name.startswith("results_fw") and name.endswith("_lambda_eta_sweep"):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                dirs.append(full)
    return dirs


# ======================================================
# CSV から valid_acc を読み取る
# ======================================================
def read_valid_acc(path):
    try:
        df = pd.read_csv(path)
        if "valid_acc" not in df.columns:
            return np.nan
        return float(df["valid_acc"].iloc[-1])
    except Exception:
        return np.nan


# ======================================================
# 個々の S ディレクトリを処理してヒートマップ生成
# ======================================================
def process_s_dir(s_path):
    print(f"[INFO] Processing {s_path}")

    # Heatmap 用 4x4 行列
    mat = np.full((len(LAM_LIST), len(ETA_LIST)), np.nan)

    # S 値をディレクトリ名から抽出
    m = re.search(r"results_fw(\d+)_", s_path)
    S_value = int(m.group(1)) if m else None

    # CSV を読み込む
    for fname in os.listdir(s_path):
        if not fname.endswith(".csv"):
            continue
        m = FILE_PATTERN.search(fname)
        if m is None:
            continue

        lam = float(m.group("lam"))
        eta = float(m.group("eta"))

        if lam not in LAM_LIST or eta not in ETA_LIST:
            continue

        lam_i = LAM_LIST.index(lam)
        eta_i = ETA_LIST.index(eta)

        acc = read_valid_acc(os.path.join(s_path, fname))
        mat[lam_i, eta_i] = acc

    # 有効データが無い場合スキップ
    if np.isnan(mat).all():
        print(f"[WARN] No usable CSV inside {s_path}, skip")
        return

    # ======================================================
    #   ヒートマップ描画 (赤→黄→緑)
    # ======================================================
    sns.set(font_scale=1.3)
    plt.figure(figsize=(8, 6))

    df = pd.DataFrame(mat, index=LAM_LIST, columns=ETA_LIST)

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",   # ★ ここを RdYlGn_r → RdYlGn に変更
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0,
        vmax=1.0
    )


    plt.title(f"Accuracy Heatmap of Ba-FW (S = {S_value})")
    plt.xlabel("eta")
    plt.ylabel("lambda")

    # 保存先：FW_sweep_fig/
    outname = f"heatmap_S{S_value}.png"
    outpath = os.path.join(OUT_DIR, outname)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {outpath}")


# ======================================================
# メイン処理
# ======================================================
def main():
    s_dirs = find_s_dirs(CSV_ROOT)

    if not s_dirs:
        print("[ERROR] No results_fw*_lambda_eta_sweep found.")
        return

    for s_dir in s_dirs:
        process_s_dir(s_dir)


if __name__ == "__main__":
    main()

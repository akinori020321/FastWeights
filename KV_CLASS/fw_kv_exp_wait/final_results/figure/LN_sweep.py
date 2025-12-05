import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# スクリプト自身の場所
# ======================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV ルート（この一つ上に LN_sweep/ がある想定）
CSV_ROOT = os.path.join(THIS_DIR, "..", "LN_sweep")

# 出力ディレクトリ
OUT_DIR = os.path.join(THIS_DIR, "LN_sweep_fig")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# 色設定
#   FW（fw1） LN0 → 青
#   FW（fw1） LN1 → 赤
#   RNN（fw0）LN0 → 緑
#   RNN（fw0）LN1 → 水色
# ======================================================
COLOR_MAP = {
    ("1", "0"): "blue",   # FW, LN0
    ("1", "1"): "red",    # FW, LN1
    ("0", "0"): "green",  # RNN, LN0
    ("0", "1"): "cyan",   # RNN, LN1
}

LABEL_MAP = {
    ("1", "0"): "FW (LN=0)",
    ("1", "1"): "FW (LN=1)",
    ("0", "0"): "RNN (LN=0)",
    ("0", "1"): "RNN (LN=1)",
}

# ファイル名から fw/ln を抜き出す用
FNAME_PATTERN = re.compile(r"fw(?P<fw>[01])_ln(?P<ln>[01])_S(?P<S>\d+)")


# ======================================================
# CSV 探し（LN_sweep 配下のサブディレクトリを全部見る）
# ======================================================
def find_all_csv(root):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, fname))
    return csv_files


# ======================================================
# CSV から曲線を読む
# ======================================================
def read_curve(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None

    if "epoch" not in df.columns:
        return None, None

    # 精度カラム優先順位：valid_acc → acc
    if "valid_acc" in df.columns:
        acc = df["valid_acc"].values
    elif "acc" in df.columns:
        acc = df["acc"].values
    else:
        return None, None

    epochs = df["epoch"].values
    return epochs, acc


# ======================================================
# メイン：線4本の図を 1枚だけ作る
# ======================================================
def main():
    csv_paths = find_all_csv(CSV_ROOT)
    if not csv_paths:
        print(f"[ERROR] No csv files found under {CSV_ROOT}")
        return

    curves = []  # (epochs, acc, label, color)

    for path in csv_paths:
        fname = os.path.basename(path)
        m = FNAME_PATTERN.search(fname)
        if m is None:
            continue

        fw = m.group("fw")  # "0" or "1"
        ln = m.group("ln")  # "0" or "1"

        color = COLOR_MAP.get((fw, ln))
        label = LABEL_MAP.get((fw, ln))
        if color is None or label is None:
            continue

        epochs, acc = read_curve(path)
        if epochs is None:
            continue

        curves.append((epochs, acc, label, color))

    if not curves:
        print("[ERROR] No usable curves found.")
        return

    # --------------------------------------------------
    # Plot（線4本あれば4本、欠けててもあるぶんだけ描く）
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    for epochs, acc, label, color in curves:
        plt.plot(epochs, acc, label=label, color=color, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.title("Layer normalization (FW vs RNN, with/without LN)")

    out_path = os.path.join(OUT_DIR, "ln_sweep_acc.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved → {out_path}")


if __name__ == "__main__":
    main()

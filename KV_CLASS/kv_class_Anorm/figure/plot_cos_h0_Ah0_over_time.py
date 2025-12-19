#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cos_h0_Ah0_over_time.py  (0〜1表示・負値は0に切り捨て)
------------------------------------------------------------
../results_A_kv/ 以下の
  - SloopVec_kv_<core>_S*_eta*_lam*_seed*.csv   （h と base を含む）
  - Amat_kv_<core>_S*_eta*_lam*_seed*.csv       （A 行列）
を対応付けて，

各時刻 t の「s=0 の h (= h^{(0)})」と A_t から
  cos(h^{(0)}, A_t h^{(0)})
を計算し，時系列プロットする（条件ごとに別々の図）。

★表示は 0〜1（負値は 0 に切り捨て）：
  cos_pos = clip(cos_raw, 0, 1)

出力:
  plots/sloop_h0Ah0/<model_id>/
    cos_h0_Ah0_over_time.png
    cos_h0_Ah0_over_time.csv

注意:
  - Sloop_kv_*.csv は h[...] 列が無いことがあるため，本スクリプトでは対象外
  - ベクトルが入っている SloopVec_kv_* のみ処理する
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# core 名（表示用）
# --------------------------------------------------
CORE_NAME = {
    "fw": "Ba-FW",
    "tanh": "SC-FW",
    "rnn": "RNN-LN",
}


# --------------------------------------------------
# cosine
# --------------------------------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def clip_01(x: float) -> float:
    """数値誤差も含めて [0,1] にクリップ（負値は0に切り捨て）"""
    return float(np.clip(x, 0.0, 1.0))


# --------------------------------------------------
# ファイル名パーサ（SloopVec）
#   SloopVec_kv_<core>_S... を想定
# --------------------------------------------------
def parse_sloopvec_filename(path: str):
    base = os.path.basename(path)
    pattern = r"SloopVec_kv_(fw|tanh|rnn)_S(\d+)_eta([0-9]+)_lam([0-9]+)_seed(\d+)\.csv$"
    m = re.match(pattern, base)
    if m is None:
        raise ValueError(f"Filename does not match SloopVec pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)
    return core, S, eta, lam, seed


# --------------------------------------------------
# ファイル名パーサ（Amat）
#   Amat_kv_<core>_S...
# --------------------------------------------------
def parse_amat_filename(path: str):
    base = os.path.basename(path)
    pattern = r"Amat_kv_(fw|tanh|rnn)_S(\d+)_eta([0-9]+)_lam([0-9]+)_seed(\d+)\.csv$"
    m = re.match(pattern, base)
    if m is None:
        raise ValueError(f"Filename does not match Amat pattern: {base}")

    core = m.group(1)
    S = int(m.group(2))
    eta = m.group(3)
    lam = m.group(4)
    seed = m.group(5)
    return core, S, eta, lam, seed


# --------------------------------------------------
# A 行列を読み込む（各行が t の A_t を flat で持つ想定）
#  - 't' 列や 'Unnamed: 0' 列があってもなるべく吸収
#  - 行ごとに「数値として読める成分だけ」取り出して正方化
# --------------------------------------------------
def load_A_list(csv_path: str):
    df = pd.read_csv(csv_path)

    # よくある不要列を除去（存在すれば）
    drop_cols = []
    if "t" in df.columns:
        drop_cols.append("t")
    for c in df.columns:
        if str(c).startswith("Unnamed:"):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))

    A_list = []
    for _, row in df.iterrows():
        vals = pd.to_numeric(row, errors="coerce").values.astype(float)
        vals = vals[~np.isnan(vals)]

        L = vals.shape[0]
        if L > 1 and (int(np.sqrt(L - 1)) ** 2 == (L - 1)):
            vals = vals[1:]
            L = vals.shape[0]

        d = int(np.sqrt(L))
        if d * d != L:
            raise ValueError(f"[Amat] row size {L} cannot form square in {csv_path}")

        A_list.append(vals.reshape(d, d))

    return A_list


# --------------------------------------------------
# SloopVec から s=0 の h を t ごとに取得
# --------------------------------------------------
def extract_h_s0_by_t(sloopvec_csv: str):
    df = pd.read_csv(sloopvec_csv)

    if "t" not in df.columns or "s" not in df.columns:
        raise ValueError(f"[SloopVec] 't' and 's' columns are required: {sloopvec_csv}")

    h_cols = [c for c in df.columns if c.startswith("h[")]
    if len(h_cols) == 0:
        raise ValueError(f"[SloopVec] No h[...] columns found: {sloopvec_csv}")

    df0 = df[df["s"] == 0].copy()
    if len(df0) == 0:
        raise ValueError(f"[SloopVec] No rows with s=0 found: {sloopvec_csv}")

    df0 = df0.sort_values(["t"]).groupby("t").head(1)

    t_list = df0["t"].astype(int).tolist()
    H0 = df0[h_cols].values.astype(float)
    return t_list, H0


# --------------------------------------------------
# 1 モデル分の処理
# --------------------------------------------------
def process_one_model(sloopvec_csv: str, amat_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    core, S, eta, lam, seed = parse_sloopvec_filename(sloopvec_csv)
    core_name = CORE_NAME.get(core, core)

    print(f"[PROCESS] core={core} S={S} eta={eta} lam={lam} seed={seed}")
    print(f"  sloopvec={sloopvec_csv}")
    print(f"  amat   ={amat_csv}")

    A_list = load_A_list(amat_csv)
    t_list, H0 = extract_h_s0_by_t(sloopvec_csv)

    out_rows = []
    ok_t = []
    cos_pos_list = []

    for idx, t in enumerate(t_list):
        if t < 0 or t >= len(A_list):
            continue

        h0 = H0[idx]
        A = A_list[t]
        Ah0 = A @ h0

        cos_raw = cosine(h0, Ah0)     # 本来のcos（念のため-1..1）
        cos_pos = clip_01(cos_raw)    # 表示用：負値は0

        ok_t.append(int(t))
        cos_pos_list.append(float(cos_pos))

        out_rows.append({
            "t": int(t),
            "cos_raw": float(cos_raw),
            "cos_pos": float(cos_pos),
        })

    if len(ok_t) == 0:
        print(f"[WARN] No valid (t,A_t) pairs. Skip: {out_dir}")
        return

    out_csv = os.path.join(out_dir, "cos_h0_Ah0_over_time.csv")
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)

    plt.figure(figsize=(7.0, 3.2))
    plt.plot(ok_t, cos_pos_list, marker="o")
    pad = 0.05  # 好みで 0.02〜0.05 くらい
    plt.ylim([-pad, 1.0 + pad])
    plt.grid(True)
    plt.xlabel("t (step)")
    plt.ylabel("max(cos(h^(0), A_t h^(0)), 0)")

    title = f"{core_name}, S={S}, η={int(eta)/1000.0:g}, λ={int(lam)/1000.0:g}, seed={seed}"
    plt.title(title, fontsize=10)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "cos_h0_Ah0_over_time.png")
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_png[:-4] + ".eps")
    plt.close()

    print(f"[DONE] Saved:")
    print(f"  - {out_png}")
    print(f"  - {out_csv}")


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="../results_A_kv",
                    help="SloopVec_kv_* と Amat_kv_* があるディレクトリ")
    ap.add_argument("--out_root", type=str, default="plots/sloop_h0Ah0",
                    help="出力ルートディレクトリ")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    sloopvec_list = sorted(glob.glob(os.path.join(args.dir, "SloopVec_kv_*.csv")))
    if len(sloopvec_list) == 0:
        print("[ERROR] No SloopVec_kv_*.csv found.")
        return

    amat_list = sorted(glob.glob(os.path.join(args.dir, "Amat_kv_*.csv")))
    if len(amat_list) == 0:
        print("[ERROR] No Amat_kv_*.csv found.")
        return

    print(f"[INFO] Found {len(sloopvec_list)} SloopVec files")
    print(f"[INFO] Found {len(amat_list)} Amat files")

    amat_map = {}
    for a in amat_list:
        try:
            core, S, eta, lam, seed = parse_amat_filename(a)
            amat_map[(core, S, eta, lam, seed)] = a
        except Exception:
            continue

    for sloopvec_csv in sloopvec_list:
        try:
            core, S, eta, lam, seed = parse_sloopvec_filename(sloopvec_csv)
        except Exception as e:
            print(f"[WARN] Skip (cannot parse): {sloopvec_csv} ({e})")
            continue

        amat_csv = amat_map.get((core, S, eta, lam, seed), None)
        if amat_csv is None:
            print(f"[WARN] Amat not found for: {sloopvec_csv}")
            continue

        model_id = f"{core}_S{S}_eta{eta}_lam{lam}_seed{seed}"
        out_dir = os.path.join(args.out_root, model_id)
        process_one_model(sloopvec_csv, amat_csv, out_dir)


if __name__ == "__main__":
    main()

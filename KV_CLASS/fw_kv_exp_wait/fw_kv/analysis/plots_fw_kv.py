# -*- coding: utf-8 -*-
# fw_kv/analysis/plots_fw_kv.py
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# 基本設定
# =========================================
CONFIG = dict(D_G=100, D_H=200)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(os.path.dirname(__file__), "fig")
os.makedirs(OUTDIR, exist_ok=True)
LW = 2.2  # 線の太さ

# =========================================
# CSV 読み込み関連
# =========================================
def list_csvs(search_paths):
    found = []
    for pat in search_paths:
        found.extend(glob.glob(pat))
    # dict.fromkeys で順序維持しつつ重複削除
    return sorted(list(dict.fromkeys(found)))

def try_parse_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def infer_beta_from_fname(path):
    base = os.path.basename(path)
    for pat in [r"_beta([0-9.]+)", r"beta([0-9.]+)"]:
        m = re.search(pat, base)
        if m:
            return try_parse_float(m.group(1))
    return np.nan

def robust_read_one_csv(path):
    try:
        df = pd.read_csv(path, dtype=str, skip_blank_lines=True)
    except Exception as e:
        print(f"[SKIP] read fail: {path} ({e})")
        return pd.DataFrame()

    if df.shape[1] > 0:
        # 先頭列にヘッダ行が何度も入っているケースを除外
        mask = ~(df.iloc[:, 0].isin({"epoch", "seed", "use_fw", "use_ln", "S", "T_bind"}))
        # 1行目は必ず残す（本来のヘッダ or 最初の行）
        mask.iloc[0] = True
        df = df[mask].copy()

    # 列名整形
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "val_acc": "valid_acc",
        "train_accu": "train_acc",
        "val_loss": "valid_loss",
        "t_bind": "T_bind",
        "dg": "d_g",
        "dh": "d_h",
        "lambda_decay": "lambda",
    }
    df.rename(columns=rename_map, inplace=True)

    # 必須列を埋める
    expected = [
        "epoch", "seed", "use_fw", "use_ln", "S", "T_bind",
        "d_g", "d_h", "lambda", "eta", "beta", "valid_acc", "delta_wait"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # beta が空ならファイル名から推定
    if df["beta"].isna().all():
        beta_guess = infer_beta_from_fname(path)
        if not np.isnan(beta_guess):
            df["beta"] = beta_guess

    # 数値化
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 最低限のキーがない行は落とす
    df = df.dropna(subset=["epoch", "S", "T_bind", "use_fw", "use_ln"])

    # int系
    for c in ["epoch", "seed", "use_fw", "use_ln", "S", "T_bind", "d_g", "d_h"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    df["__source__"] = os.path.basename(path)
    return df

def load_all_csvs(search_paths):
    paths = list_csvs(search_paths)
    if not paths:
        raise SystemExit("[ERROR] CSVが見つかりません。")

    frames = []
    for p in paths:
        one = robust_read_one_csv(p)
        if not one.empty:
            frames.append(one)

    if not frames:
        raise SystemExit("[ERROR] 有効なCSV行が存在しません。")

    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    after = len(df)
    print(f"[INFO] Dedup: {before} → {after} rows")

    return df

def final_rows(df):
    keys = [
        c for c in
        ["use_fw", "use_ln", "S", "T_bind", "lambda", "eta", "d_g", "d_h", "beta", "seed"]
        if c in df.columns
    ]
    if not keys:
        return df.copy()
    idx = df.groupby(keys, dropna=False)["epoch"].idxmax()
    return df.loc[idx].reset_index(drop=True)

# =========================================
# 理論ユーティリティ
# =========================================
def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]

def gain_S(S, rho):
    S = np.asarray(S, float)
    return (1.0 - np.power(rho, S + 1.0)) / (1.0 - rho)

def fit_rho_and_affine_for_S(S_vals, acc_vals, T_bind, d_g=64,
                             rho_grid=np.linspace(0.05, 0.98, 188)):
    S_vals = np.asarray(S_vals, float)
    acc_vals = np.asarray(acc_vals, float)
    scale = np.sqrt(T_bind / float(d_g))

    best = dict(mse=np.inf)
    for rho in rho_grid:
        p = gain_S(S_vals, rho) / scale
        a, b = linfit(p, acc_vals)
        mse = np.mean((acc_vals - (a * p + b)) ** 2)
        if mse < best["mse"]:
            best.update(mse=mse, rho=rho, a=a, b=b)
    return best

def fit_affine_for_capacity(T_vals, acc_vals, d_g=64):
    T_vals = np.asarray(T_vals, float)
    acc_vals = np.asarray(acc_vals, float)
    p = 1.0 / np.sqrt(T_vals / float(d_g))
    a, b = linfit(p, acc_vals)
    return dict(a=a, b=b)

def agg_mean_ci(df, keys, y="valid_acc"):
    g = df.groupby(keys, dropna=False)[y]
    mean = g.mean()
    std = g.std(ddof=1)
    n = g.count().clip(lower=1)
    ci = 1.96 * (std / np.sqrt(n))
    out = mean.reset_index().rename(columns={y: "mean"})
    out["ci"] = ci.values
    return out

def fig_wait_curve(df, tag, T_bind=8, S_fixed=1):
    need = {"epoch", "valid_acc", "delta_wait", "S", "T_bind", "use_fw", "use_ln", "beta"}
    if not need.issubset(df.columns):
        return

    # ここでは use_fw/use_ln は今まで通りの固定値でフィルタ
    df = df[
        (df["T_bind"] == T_bind)
        & (df["S"] == S_fixed)
        & (df["use_fw"] == 1)
        & (df["use_ln"] == 1)
    ]

    if len(df) == 0:
        print(f"[WARN] empty df for tag={tag}")
        return

    betas = sorted(pd.unique(df["beta"].dropna())) or [np.nan]

    for beta in betas:
        sub = df if np.isnan(beta) else df[np.isclose(df["beta"], beta)]
        if len(sub) == 0:
            continue

        plt.figure(figsize=(7, 5))
        waits = sorted(sub["delta_wait"].dropna().unique())
        if not waits:
            plt.close()
            continue

        for w in waits:
            s = sub[sub["delta_wait"] == w]
            grp = s.groupby("epoch")["valid_acc"]
            mean = grp.mean()
            std = grp.std(ddof=1)
            n = grp.count().clip(lower=1)
            ci = 1.96 * (std / np.sqrt(n))

            plt.plot(mean.index, mean, label=f"Δ_wait={int(w)}", lw=LW)
            plt.fill_between(mean.index, mean - ci, mean + ci, alpha=0.25)

        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"

        # ★ タイトルに tag を入れる
        plt.title(f"Wait Effect ({tag}, T_bind={T_bind}, S={S_fixed}, β={beta_label})")

        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.4)

        # ★ ファイル名にも tag を入れる
        out = os.path.join(
            OUTDIR,
            f"wait_curve_{tag}_T{T_bind}_S{S_fixed}_beta{beta_label}.png"
        )

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] {out}")


def fig_acc_vs_S_T_with_theory(df, T_bind, d_g=64):
    """
    Plot validation accuracy vs S with theory fit and 95% CI.
    Each Δ_wait is shown separately, optionally grouped by β.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, os

    required = {"T_bind", "use_fw", "use_ln", "S", "valid_acc", "delta_wait", "beta"}
    if not required.issubset(df.columns):
        print("[WARN] Missing columns:", required - set(df.columns))
        return

    f_all = df.query("T_bind == @T_bind and use_fw == 1 and use_ln == 1")
    betas = sorted(pd.unique(f_all["beta"].dropna())) or [np.nan]

    for beta in betas:
        sub_beta = f_all if np.isnan(beta) else f_all[np.isclose(f_all["beta"], beta)]
        if len(sub_beta) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        colors = plt.cm.tab10(np.linspace(0, 1, len(sub_beta["delta_wait"].unique())))
        for i, w in enumerate(sorted(sub_beta["delta_wait"].dropna().unique())):
            sub = sub_beta[sub_beta["delta_wait"] == w]
            fw_ci = agg_mean_ci(sub, ["S"]).sort_values("S")
            S_arr, mean_arr, ci_arr = fw_ci["S"], fw_ci["mean"], fw_ci["ci"]

            # 実測線 + CI帯
            ax.plot(S_arr, mean_arr, lw=LW, color=colors[i],
                    label=f"Δ_wait={int(w)} (empirical)")
            ax.fill_between(S_arr, mean_arr - ci_arr, mean_arr + ci_arr,
                            color=colors[i], alpha=0.25)

            # 理論フィット線
            fit = fit_rho_and_affine_for_S(S_arr, mean_arr, T_bind, d_g)
            S_grid = np.linspace(0, max(S_arr), 400)
            gain_grid = gain_S(S_grid, fit["rho"]) / np.sqrt(T_bind / float(d_g))
            acc_theory = fit["a"] * gain_grid + fit["b"]
            ax.plot(S_grid, acc_theory, "--", color=colors[i],
                    lw=LW - 0.5, label=f"Theory (ρ={fit['rho']:.2f})")

        # 軸・凡例・タイトル
        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        ax.set_xlabel("Inner-loop iterations $S$", fontsize=12)
        ax.set_ylabel("Validation accuracy", fontsize=12)
        ax.set_title(f"External memory model: T_bind={T_bind}, β={beta_label}", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=10, loc="lower right")

        # 保存
        out_name = f"acc_vs_S_T{T_bind}_beta{beta_label}.png"
        out_path = os.path.join(OUTDIR, out_name)
        fig.tight_layout()
        plt.savefig(out_path, format="png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved {out_path}")

def fig_capacity_curve_with_theory_CI(df, d_g=64, S_fixed=3):
    betas = sorted(pd.unique(df["beta"].dropna())) or [np.nan]
    for beta in betas:
        sub = df if np.isnan(beta) else df[np.isclose(df["beta"], beta)]
        fw = sub[
            (sub["use_fw"] == 1)
            & (sub["use_ln"] == 1)
            & (sub["S"] == S_fixed)
        ]
        if len(fw) == 0:
            continue

        fw_ci = agg_mean_ci(fw, ["T_bind"]).sort_values("T_bind")
        T = fw_ci["T_bind"].values
        mean = fw_ci["mean"].values
        ci = fw_ci["ci"].values

        plt.figure()
        plt.plot(T, mean, marker="o", lw=LW, label=f"FW+LN (S={S_fixed})")
        plt.fill_between(T, mean - ci, mean + ci, alpha=0.25)

        fit = fit_affine_for_capacity(T, mean, d_g)
        T_grid = np.linspace(min(T), max(T), 200)
        acc_theory = fit["a"] * (1 / np.sqrt(T_grid / float(d_g))) + fit["b"]
        plt.plot(T_grid, acc_theory, "--", lw=LW, label="Theory ~1/sqrt(T)")

        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        plt.xlabel("T_bind")
        plt.ylabel("Validation accuracy")
        plt.title(f"Capacity (S={S_fixed}, β={beta_label})")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.4)
        out = os.path.join(OUTDIR, f"capacity_Tbind_curve_CI_beta{beta_label}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=240)
        plt.close()
        print(f"[OK] {out}")

def fig_fw_ln_interaction(df, S_fixed=3, T_bind_fixed=8, seed_fixed=0):
    betas = sorted(pd.unique(df["beta"].dropna())) or [np.nan]
    groups = [
        ("RNN-only", (0, 0)),
        ("LN only", (0, 1)),
        ("FW only", (1, 0)),
        ("FW+LN", (1, 1)),
    ]

    for beta in betas:
        sub = df if np.isnan(beta) else df[np.isclose(df["beta"], beta)]
        sub = sub[
            (sub["S"] == S_fixed)
            & (sub["T_bind"] == T_bind_fixed)
            & (sub["seed"] == seed_fixed)
        ]
        if len(sub) == 0:
            continue

        plt.figure(figsize=(6, 4))
        for name, (fw, ln) in groups:
            vals = sub[(sub["use_fw"] == fw) & (sub["use_ln"] == ln)]
            if len(vals) == 0:
                continue
            vals_mean = (
                vals.groupby("epoch", as_index=False)["valid_acc"]
                .mean()
                .sort_values("epoch")
            )
            plt.plot(vals_mean["epoch"], vals_mean["valid_acc"], label=name, lw=LW)

        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(f"FW×LN (S={S_fixed}, T={T_bind_fixed}, β={beta_label})")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        out = os.path.join(
            OUTDIR,
            f"fw_ln_interaction_S{S_fixed}_T{T_bind_fixed}_beta{beta_label}.png",
        )
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[OK] {out}")

def fig_residual_vs_S(df, d_g=64, T_bind=8):
    betas = sorted(pd.unique(df["beta"].dropna())) or [np.nan]

    for beta in betas:
        sub = df if np.isnan(beta) else df[np.isclose(df["beta"], beta)]
        fw = sub[
            (sub["T_bind"] == T_bind)
            & (sub["use_fw"] == 1)
            & (sub["use_ln"] == 1)
        ]
        if len(fw) == 0:
            continue

        fw_ci = agg_mean_ci(fw, ["S"]).sort_values("S")
        S = fw_ci["S"].values
        mean = fw_ci["mean"].values

        fit = fit_rho_and_affine_for_S(S, mean, T_bind, d_g)
        p = gain_S(S, fit["rho"]) / np.sqrt(T_bind / float(d_g))
        res = mean - (fit["a"] * p + fit["b"])

        plt.figure()
        plt.axhline(0, color="black", lw=1.2)
        plt.plot(S, res, "o-", lw=LW)
        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        plt.xlabel("S")
        plt.ylabel("Residual")
        plt.title(f"Residual vs S (T={T_bind}, β={beta_label})")
        plt.grid(True, ls="--", alpha=0.4)
        out = os.path.join(
            OUTDIR, f"residual_vs_S_T{T_bind}_beta{beta_label}.png"
        )
        plt.tight_layout()
        plt.savefig(out, dpi=240)
        plt.close()
        print(f"[OK] {out}")

def fig_residual_vs_T(df, d_g=64, S_fixed=3):
    betas = sorted(pd.unique(df["beta"].dropna())) or [np.nan]

    for beta in betas:
        sub = df if np.isnan(beta) else df[np.isclose(df["beta"], beta)]
        fw = sub[
            (sub["use_fw"] == 1)
            & (sub["use_ln"] == 1)
            & (sub["S"] == S_fixed)
        ]
        if len(fw) == 0:
            continue

        fw_ci = agg_mean_ci(fw, ["T_bind"]).sort_values("T_bind")
        T = fw_ci["T_bind"].values
        mean = fw_ci["mean"].values

        fit = fit_affine_for_capacity(T, mean, d_g)
        res = mean - (fit["a"] * (1 / np.sqrt(T / float(d_g))) + fit["b"])

        plt.figure()
        plt.axhline(0, color="black", lw=1.2)
        plt.plot(T, res, "o-", lw=LW)
        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        plt.xlabel("T_bind")
        plt.ylabel("Residual")
        plt.title(f"Residual vs T_bind (S={S_fixed}, β={beta_label})")
        plt.grid(True, ls="--", alpha=0.4)
        out = os.path.join(
            OUTDIR, f"residual_vs_T_S{S_fixed}_beta{beta_label}.png"
        )
        plt.tight_layout()
        plt.savefig(out, dpi=240)
        plt.close()
        print(f"[OK] {out}")

def fig_dg_scaling_curve(final):
    betas = sorted(pd.unique(final["beta"].dropna())) or [np.nan]

    for beta in betas:
        sub = final if np.isnan(beta) else final[np.isclose(final["beta"], beta)]
        dg_values = sorted(int(x) for x in pd.unique(sub["d_g"].dropna()))
        if not dg_values:
            continue

        plt.figure()
        for dg in dg_values:
            s = sub[
                (sub["use_fw"] == 1)
                & (sub["use_ln"] == 1)
                & (sub["S"] == 3)
                & (sub["d_g"] == dg)
            ]
            if len(s) == 0:
                continue
            s = (
                s.groupby("T_bind", as_index=False)["valid_acc"]
                .mean()
                .sort_values("T_bind")
            )
            plt.plot(s["T_bind"], s["valid_acc"], "o-", label=f"d_g={dg}")

        beta_label = "all" if np.isnan(beta) else f"{beta:.2f}"
        plt.xscale("log")
        plt.xlabel("T_bind (log)")
        plt.ylabel("Validation Accuracy")
        plt.title(f"Capacity vs T_bind (d_g scaling, β={beta_label})")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        out = os.path.join(
            OUTDIR, f"capacity_vs_T_dg_scaling_beta{beta_label}.png"
        )
        plt.tight_layout()
        plt.savefig(out, dpi=240)
        plt.close()
        print(f"[OK] {out}")

# =========================================
# main
# =========================================
def main():
    exps = [
        ("results_wait_fw",  lambda df: fig_wait_curve(df, "fw",  T_bind=10, S_fixed=1)),
        ("results_wait_fw",  lambda df: fig_wait_curve(df, "fw",  T_bind=10, S_fixed=2)),
        ("results_wait_rnn", lambda df: fig_wait_curve(df, "rnn", T_bind=10, S_fixed=2)),
        ("results_fw_ln*", lambda df: fig_fw_ln_interaction(df, S_fixed=1, T_bind_fixed=10, seed_fixed=0)),
        ("results_fw_ln*", lambda df: fig_fw_ln_interaction(df, S_fixed=2, T_bind_fixed=10, seed_fixed=0)),
        ("results_fw_ln*", lambda df: fig_fw_ln_interaction(df, S_fixed=3, T_bind_fixed=10, seed_fixed=0)),
    ]

    for pattern, func in exps:
        search_paths = [os.path.join(ROOT, pattern, "*.csv")]
        found_files = glob.glob(search_paths[0])
        if not found_files:
            print(f"[SKIP] No CSVs found for {pattern}")
            continue

        print(f"[INFO] Processing {pattern} ...")
        try:
            df = load_all_csvs(search_paths)
        except SystemExit as e:
            print(f"[WARN] Skipped {pattern} due to load error: {e}")
            continue

        if df.empty:
            print(f"[SKIP] Empty data for {pattern}")
            continue

        df = df.sort_values(["__source__", "epoch"]).reset_index(drop=True)
        func(df)

    print(f"[DONE] All figures saved to: {OUTDIR}/")

if __name__ == "__main__":
    main()

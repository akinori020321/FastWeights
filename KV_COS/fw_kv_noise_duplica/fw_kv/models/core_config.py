# fw_kv/models/core_config.py

from dataclasses import dataclass

@dataclass
class CoreCfg:
    # =============================
    # 次元
    # =============================
    glimpse_dim: int = 64     # 入力 z_t 次元 d_g
    hidden_dim: int = 128     # 隠れ状態 h_t 次元 d_h

    # =============================
    # Fast Weights Hebbian 設定
    # =============================
    lambda_decay: float = 0.97       # A ← λA + ηΔA の λ
    eta: float = 0.3                 # A 更新係数 η
    epsilon: float = 1e-6            # 数値安定化項

    # =============================
    # 自己整合ループ（inner steps）
    # =============================
    inner_steps: int = 3             # S：Ah を読む内ループ

    # =============================
    # LayerNorm / FW の有無
    # =============================
    use_layernorm: bool = True
    use_A: bool = True               # Fast Weights を使うか

    # =============================
    # ★ 方向一致強度パラメータ（α）
    #  softplus(alpha_fw) + 1 = k_pos >= 1
    #  r^k_pos としてゲインに使う
    # =============================
    alpha_fw: float = 0.0            # 初期値：1.0 〜 3.0 が推奨

    # =============================
    # Reconstruction タスク用オプション
    # =============================
    use_output_norm: bool = False     # pred_vec を L2 normalize するか

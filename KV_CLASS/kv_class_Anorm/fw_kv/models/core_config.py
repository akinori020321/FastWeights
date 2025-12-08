# fw_kv/models/core_config.py

from dataclasses import dataclass

@dataclass
class CoreCfg:
    glimpse_dim: int = 64
    hidden_dim: int = 128
    lambda_decay: float = 0.97
    eta: float = 0.3
    epsilon: float = 1e-6
    inner_steps: int = 3
    use_layernorm: bool = True
    use_A: bool = True

    # ★ 方向一致制御 (α)：softplus(alpha_fw) + 1 → α_shape (>=1)
    alpha_fw: float = -2.0

    # ★ Ah のスケーリング係数（Ba-style Query 用）
    beta: float = 1.0
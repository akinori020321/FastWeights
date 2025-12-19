import os
import numpy as np
import matplotlib.pyplot as plt

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def k_from_alpha_fw(alpha_fw: float) -> float:
    if alpha_fw >= 0.0:
        return 1.0 + softplus(alpha_fw)
    else:
        return 1.0 / (1.0 + softplus(-alpha_fw))

def alpha_dyn(r, k):
    r = np.clip(r, 0.0, 1.0)
    return 1.0 - (1.0 - r)**k

alpha_fw_list = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
r = np.linspace(0.0, 1.0, 500)

plt.figure(figsize=(8, 4.8))
for a in alpha_fw_list:
    k = float(k_from_alpha_fw(a))
    plt.plot(r, alpha_dyn(r, k), linewidth=2, label=f"alpha_fw={a:.1f}, k={k:.2f}")

plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel(r"$R=\max\{0,\cos(\mathbf{h},A\mathbf{h})\}$")
plt.ylabel(r"$\alpha_{\mathrm{dyn}}$")
plt.title(r"Dynamic Update Strength $\alpha_{\mathrm{dyn}}$ vs. Directional Consistency $R$")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
os.makedirs("alpha_fig", exist_ok=True)
plt.savefig(os.path.join("alpha_fig", "alpha_dyn_vs_r.png"), dpi=200)
plt.savefig(os.path.join("alpha_fig", "alpha_dyn_vs_r.eps"))
plt.close()

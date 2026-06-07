"""
generate_data.py
----------------
Generates snapshot data for the bilinear Koopman
System:  x_{t+1} = 0.95*x_t - 0.2*x_t^3 + u_t
Domain:  X = [-1, 1],  U = [-1, 1]
Outputs
-------
data_train.npz  --  training snapshots  (x, u, y) with y = f(x, u)
data_test.npz   --  testing snapshots
scatter plot    --  (x, u) pairs for both splits
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── reproducibility ───────────────────────────────────────────────────────────
SEED_TRAIN = 0
SEED_VAL   = 1
N_TRAIN    = 300
N_TEST     = 500
SAVE_DIR   = "."

# ── dynamics ──────────────────────────────────────────────────────────────────
def f(x, u):
    return 0.95 * x - 0.2 * x**3 + u

# ── data generation ───────────────────────────────────────────────────────────
def generate_snapshots(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    y = f(x, u)
    return x, u, y

x_tr, u_tr, y_tr = generate_snapshots(N_TRAIN, SEED_TRAIN)
x_va, u_va, y_va = generate_snapshots(N_TEST,  SEED_VAL)

# ── save ──────────────────────────────────────────────────────────────────────
np.savez(os.path.join(SAVE_DIR, "data_train.npz"), x=x_tr, u=u_tr, y=y_tr)
np.savez(os.path.join(SAVE_DIR, "data_test.npz"),  x=x_va, u=u_va, y=y_va)
print(f"Saved  data_train.npz  ({N_TRAIN} samples)")
print(f"Saved  data_test.npz   ({N_TEST}  samples)")

# ── scatter plot: (x, u) pairs ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x_tr, u_tr, s=18, alpha=0.6, color="steelblue",
           label=f"Training", zorder=3)
ax.scatter(x_va, u_va, s=18, alpha=0.8, marker="^", color="tomato",
           label=f"Testing", zorder=4)
ax.axhline(-1, color="gray", lw=0.8, ls="--")
ax.axhline( 1, color="gray", lw=0.8, ls="--")
ax.axvline(-1, color="gray", lw=0.8, ls="--")
ax.axvline( 1, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$x$",  fontsize=13)
ax.set_ylabel(r"$u$",  fontsize=13)
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(-1.15, 1.3)
ax.set_ylim(-1.15, 1.3)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "case1_data.png"), dpi=150)
plt.close()
print("Saved  case1_data.png")
"""
main_bilinear_koopman.py
------------------------
Section III-C: Learn the bilinear Koopman operator M and validate
one-step state prediction for the system

    x_{t+1} = 0.95*x_t - 0.2*x_t^3 + u_t

Gram matrix (joint feature inner product):
    G_psi[i,j] = k(xi,xj) + v(ui,uj) + k(xi,xj)*v(ui,uj)
where
    k(xi,xj)  = (xi*xj) * wendland(xi, xj)   [linear-radial, state]
    v(ui,uj)  = (ui*uj) * wendland(ui, uj)    [linear-radial, input]

Closed-form solution to eq (3) (G_psi symmetric):
    Theta = (G_psi + N*beta*I)^{-1}

Prediction:
    x^+_hat = G_psi_query @ B,   B = (G_psi + N*beta*I)^{-1} @ Y_tr

Error bound (CDC Theorem 2, near-origin cone region):
    |error(x,u)| <= C * |(x,u)| = C * sqrt(x^2 + u^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from Helper_Wendland_Kernel import wendland_kernel, kernel_gram_matrix, pick_ell_for_sparsity

# ── 0. Load data ──────────────────────────────────────────────────────────────
train = np.load("data_train.npz")
test  = np.load("data_test.npz")
x_tr, u_tr, y_tr = train["x"], train["u"], train["y"]
x_te, u_te, y_te = test["x"],  test["u"],  test["y"]
N_tr = len(x_tr)
N_te = len(x_te)

X_tr = x_tr[:, None]
U_tr = u_tr[:, None]
Y_tr = y_tr[:, None]
X_te = x_te[:, None]
U_te = u_te[:, None]
y_true = y_te
d_x, d_u = 1, 1

# ── 1. Pick length-scales ─────────────────────────────────────────────────────
print("Picking length-scales ...")
ell_x = pick_ell_for_sparsity(X_tr, d_x=d_x, sparsity_target=(0.3, 0.5))
print(f"  ell_x = {ell_x:.4f}")
ell_u = pick_ell_for_sparsity(U_tr, d_x=d_u, sparsity_target=(0.3, 0.5))
print(f"  ell_u = {ell_u:.4f}")

# ── 2. Build linear-radial Gram matrices ──────────────────────────────────────
linear_dot = lambda a, b: float(a @ b)

W_xx_tr  = kernel_gram_matrix(X_tr, X_tr, linear_dot, wendland_kernel, d_x=d_x, ell=ell_x)
W_uu_tr  = kernel_gram_matrix(U_tr, U_tr, linear_dot, wendland_kernel, d_x=d_u, ell=ell_u)
G_psi_tr = W_xx_tr + W_uu_tr + W_xx_tr * W_uu_tr
print(f"\nG_psi_tr shape: {G_psi_tr.shape}")

# ── 3. Closed-form solution to eq (3) ─────────────────────────────────────────
beta = 1e-4
B    = np.linalg.solve(G_psi_tr + N_tr * beta * np.eye(N_tr), Y_tr)

# ── 4. Build query Gram matrix for test set ───────────────────────────────────
W_xx_te  = kernel_gram_matrix(X_te, X_tr, linear_dot, wendland_kernel, d_x=d_x, ell=ell_x)
W_uu_te  = kernel_gram_matrix(U_te, U_tr, linear_dot, wendland_kernel, d_x=d_u, ell=ell_u)
G_psi_te = W_xx_te + W_uu_te + W_xx_te * W_uu_te

# ── 5. One-step prediction ────────────────────────────────────────────────────
y_pred = (G_psi_te @ B).ravel()
error  = y_true - y_pred
rmse   = np.sqrt(np.mean(error**2))
print(f"\nTest RMSE: {rmse:.6f}")

 

 
fig1, axes = plt.subplots(1, 2, figsize=(11, 5))
# subplot (a): scatter of (x, u) training and testing pairs
ax = axes[0]
ax.scatter(x_tr, u_tr, s=18, alpha=0.6, color="steelblue",
           label="Training", zorder=3)
ax.scatter(x_te, u_te, s=18, alpha=0.8, marker="^", color="tomato",
           label="Testing", zorder=4)
ax.axhline(-1, color="gray", lw=0.8, ls="--")
ax.axhline( 1, color="gray", lw=0.8, ls="--")
ax.axvline(-1, color="gray", lw=0.8, ls="--")
ax.axvline( 1, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$x$", fontsize=13)
ax.set_ylabel(r"$u$", fontsize=13) 
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
 
# subplot (b): predicted vs true next state
ax = axes[1]
ax.scatter(y_true, y_pred, s=18, alpha=0.7, color="tomato", zorder=3)
lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
ax.set_xlabel(r"$x^+$",       fontsize=13)
ax.set_ylabel(r"$\hat{x}^+$", fontsize=13)  
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
 
fig1.tight_layout()
fig1.savefig("case1_figure2.png", dpi=150)
plt.close(fig1)
print("Saved  case1_figure2.png")
 
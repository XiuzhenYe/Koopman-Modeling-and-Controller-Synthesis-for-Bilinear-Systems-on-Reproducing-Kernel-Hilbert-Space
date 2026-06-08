"""
plot_C_vs_n.py
--------------
Two-subfigure plot:
  (a) 3D scatter of prediction errors inside cone bound +/- C*|(x,u)|
  (b) C(N) vs training size N with N^{-1/3} reference line

State prediction error bound (linear, not quadratic):
    |e_i| <= C * |(x_i, u_i)| = C * sqrt(x_i^2 + u_i^2)

C(N) = max_i |error_i| / |(x_i, u_i)|

Uses N = 50, 100, 150, 200, 250, 300.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from Helper_Wendland_Kernel import wendland_kernel, kernel_gram_matrix, pick_ell_for_sparsity

# ── 0. Load data ──────────────────────────────────────────────────────────────
train = np.load("data_train.npz")
test  = np.load("data_test.npz")

x_tr_all = train["x"]
u_tr_all = train["u"]
y_tr_all = train["y"]
x_te, u_te, y_te = test["x"], test["u"], test["y"]

# linear bound denominator: |(x,u)| = sqrt(x^2 + u^2)
xu_norm = np.sqrt(x_te**2 + u_te**2)
mask    = xu_norm > 1e-4
X_te    = x_te[:, None]
U_te    = u_te[:, None]

# ── 1. Sweep training sizes ───────────────────────────────────────────────────
n_values = np.linspace(50, x_tr_all.shape[0], 6, dtype=int).tolist()
beta     = 1e-4
d_x, d_u = 1, 1

C_values    = []
rmse_values = []
error_full  = None
C_full      = None

print(f"{'n':>6}  {'C(n)':>10}  {'RMSE':>10}")
print("-" * 32)

for n in n_values:
    idx  = np.random.default_rng(seed=0).choice(len(x_tr_all), size=n, replace=False)
    x_tr = x_tr_all[idx]
    u_tr = u_tr_all[idx]
    y_tr = y_tr_all[idx]

    X_tr = x_tr[:, None]
    U_tr = u_tr[:, None]
    Y_tr = y_tr[:, None]

    ell_x = pick_ell_for_sparsity(X_tr, d_x=d_x, sparsity_target=(0.3, 0.5))
    ell_u = pick_ell_for_sparsity(U_tr, d_x=d_u, sparsity_target=(0.3, 0.5))
    linear_dot = lambda a, b: float(a @ b)

    W_xx_tr  = kernel_gram_matrix(X_tr, X_tr, linear_dot, wendland_kernel, d_x=d_x, ell=ell_x)
    W_uu_tr  = kernel_gram_matrix(U_tr, U_tr, linear_dot, wendland_kernel, d_x=d_u, ell=ell_u)
    G_psi_tr = W_xx_tr + W_uu_tr + W_xx_tr * W_uu_tr

    B = np.linalg.solve(G_psi_tr + n * beta * np.eye(n), Y_tr)

    W_xx_te  = kernel_gram_matrix(X_te, X_tr, linear_dot, wendland_kernel, d_x=d_x, ell=ell_x)
    W_uu_te  = kernel_gram_matrix(U_te, U_tr, linear_dot, wendland_kernel, d_x=d_u, ell=ell_u)
    G_psi_te = W_xx_te + W_uu_te + W_xx_te * W_uu_te

    y_pred = (G_psi_te @ B).ravel()
    error  = y_te - y_pred
    rmse   = np.sqrt(np.mean(error**2))

    # linear bound: C(N) = max_i |error_i| / |(x_i, u_i)|
    C_n = np.max(np.abs(error[mask]) / xu_norm[mask])
    C_values.append(C_n)
    rmse_values.append(rmse)
    print(f"{n:>6}  {C_n:>10.4f}  {rmse:>10.6f}")

    if n == n_values[-1]:
        error_full = error
        C_full     = C_n

C_values = np.array(C_values)
n_arr    = np.array(n_values)

print(f"\nAll errors within cone (N={n_values[-1]}): "
      f"{np.all(np.abs(error_full[mask]) <= C_full * xu_norm[mask] + 1e-10)}")

# ── 2. N^{-1/3} reference line ───────────────────────────────────────────────
a_fit = np.median(C_values * n_arr ** (1/3))
C_ref = a_fit * n_arr ** (-1/3)

# ── 3. Cone surfaces: +/- C * |(x,u)| ────────────────────────────────────────
x_surf = np.linspace(-1.0, 1.0, 80)
u_surf = np.linspace(-1.0, 1.0, 80)
X_surf, U_surf = np.meshgrid(x_surf, u_surf)
norm_surf = np.sqrt(X_surf**2 + U_surf**2)
Z_upper   =  C_full * norm_surf    # linear cone
Z_lower   = -C_full * norm_surf
z_max     =  C_full * np.sqrt(2)   # cone height at corners x=1, u=1

# ── 4. Combined figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6))
gs  = GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.15)

# ---- subplot (a): 3D error cone at N=300 ------------------------------------
ax3d = fig.add_subplot(gs[0], projection="3d")

ax3d.plot_surface(X_surf, U_surf, Z_upper,
                  alpha=0.4, color="darkred",
                  linewidth=0, edgecolor="none")
ax3d.plot_surface(X_surf, U_surf, Z_lower,
                  alpha=0.4, color="darkblue",
                  linewidth=0, edgecolor="none")

norm_err    = plt.Normalize(vmin=-C_full*0.5, vmax=C_full*0.5)
face_colors = plt.cm.coolwarm(norm_err(error_full))
edge_colors = face_colors[:, :3] * 0.6

sc = ax3d.scatter(x_te, u_te, error_full,
                  c=error_full, cmap="coolwarm",
                  s=25, alpha=1.0, zorder=10,
                  edgecolors=edge_colors, linewidths=0.5,
                  vmin=-C_full*0.5, vmax=C_full*0.5)

ax3d.set_zlim(-z_max, z_max)
ax3d.set_xlabel(r"$x$",                fontsize=11, labelpad=6)
ax3d.set_ylabel(r"$u$",                fontsize=11, labelpad=6) 
ax3d.set_xticks([-1, 0, 1])
ax3d.set_yticks([-1, 0, 1])
ax3d.set_xticklabels(["-1", "0", "1"], fontsize=9)
ax3d.set_yticklabels(["-1", "0", "1"], fontsize=9)
ax3d.tick_params(axis="z", labelsize=9)
ax3d.view_init(elev=25, azim=-50)

cbar = fig.colorbar(sc, ax=ax3d, shrink=0.5, pad=0.08)
cbar.set_label(r"$x^+ - \hat{x}^+$", fontsize=10)

# ---- subplot (b): C(N) vs N -------------------------------------------------
ax = fig.add_subplot(gs[1])
ax.plot(n_arr, C_values, "o-", color="steelblue", lw=2, ms=8)
ax.plot(n_arr, C_ref,    "--", color="gray",    lw=1.8)

ax.set_xlabel(r"Training size $N$", fontsize=12)
ax.set_ylabel(r"$C$",            fontsize=12) 
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(n_arr)
ax.set_xticklabels([str(n) for n in n_arr], fontsize=9)

plt.savefig("case1_C_vs_n.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved  case1_C_vs_n.png")
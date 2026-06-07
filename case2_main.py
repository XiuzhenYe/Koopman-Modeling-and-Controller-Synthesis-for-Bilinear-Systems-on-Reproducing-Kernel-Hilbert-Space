"""
main_control.py
---------------
Section IV-C: Synthesize a stabilizing feedback controller via LP.

System:  x_{t+1} = 0.95*x_t - 0.2*x_t^3 + u_t
Domain:  X = [-1, 1],  U = [-1, 1]

Lyapunov function:  v(x) = x^2
Decay rate:         q(x) = 0.19*x^2   =>  v(x) - q(x) = 0.81*x^2

For each query state x, solve a small LP directly:
    find w_j >= 0 such that:
        sum_j w_j * (f(x, u_j(x)))^2 <= 0.81*x^2    (descent)
        sum_j w_j                     == 1            (normalization)
    then alpha(x) = sum_j w_j * u_j(x)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ── 0. Dynamics and parameters ────────────────────────────────────────────────
def f(x, u):
    return 0.95 * x - 0.2 * x**3 + u

N_j = 50        # number of input candidates per state
c   = 1.0       # scaling: u_j(x) = c * |x| * u_base[j]
u_base = np.linspace(-1.0, 1.0, N_j)   # fixed base input grid

# ── 1. Solve LP at a single state x ──────────────────────────────────────────
def solve_lp_at_x(x):
    """
    For query state x:
      1. Build input candidates: u_j(x) = c * |x| * u_base[j]
      2. Compute next states:    x^+_j  = f(x, u_j(x))
      3. Solve LP:
            find w_j >= 0
            s.t. sum_j w_j * (x^+_j)^2 <= 0.81*x^2
                 sum_j w_j             == 1
      4. Return alpha(x) = sum_j w_j * u_j(x)
    """
    # near origin: no control needed
    if abs(x) < 1e-6:
        return 0.0

    # input candidates at x
    u_j   = c * abs(x) * u_base                  # (N_j,)
    u_j   = np.clip(u_j, -1.0, 1.0)

    # next states and their Lyapunov values
    x_plus_j = f(x, u_j)                          # (N_j,)
    v_plus_j = x_plus_j**2                         # (N_j,)
    rhs_x    = 0.81 * x**2                         # scalar

    # feasibility check: is there at least one good input?
    if v_plus_j.min() > rhs_x + 1e-8:
        # no input achieves descent — return best available
        j_best = np.argmin(v_plus_j)
        return float(u_j[j_best])

    # solve LP
    w = cp.Variable(N_j, nonneg=True)
    constraints = [
        w @ v_plus_j <= rhs_x,    # Lyapunov descent
        cp.sum(w)    == 1.0,      # normalization
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        # fallback: use best single input
        j_best = np.argmin(v_plus_j)
        return float(u_j[j_best])

    w_val = w.value                 # (N_j,)
    return float(w_val @ u_j)      # alpha(x) = sum_j w_j * u_j(x)

# ── 2. Evaluate on fine grid ──────────────────────────────────────────────────
x_plot = np.linspace(-1.0, 1.0, 200)

print("Solving LP at each grid point ...")
alpha_plot = np.array([solve_lp_at_x(xi) for xi in x_plot])
x_plus_cl  = f(x_plot, alpha_plot)

# Lyapunov descent check
v_next  = x_plus_cl**2
v_curr  = x_plot**2
lyap_ok = np.all(v_next <= 0.81 * v_curr + 1e-4)
ratio   = np.sqrt(v_next[1:] / (v_curr[1:] + 1e-12))
print(f"Lyapunov descent satisfied : {lyap_ok}")
print(f"max |x^+|/|x|              : {ratio.max():.4f}  (should be <= 0.9)")

# ── 3. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# ---- (a) feedback policy alpha(x) vs x --------------------------------------
ax = axes[0]
ax.plot(x_plot, alpha_plot, color="steelblue", lw=2)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel(r"$x$",         fontsize=13)
ax.set_ylabel(r"$\alpha(x)$", fontsize=13) 
ax.grid(True, alpha=0.3)

# ---- (b) x^+ = f(x, alpha(x)) vs x with Lyapunov bounds --------------------
ax = axes[1]
ax.plot(x_plot, x_plus_cl,
        color="steelblue", lw=2)
ax.plot(x_plot, np.zeros_like(x_plot),
        color="k",         lw=1.2, ls="--")
ax.plot(x_plot,  0.9 * x_plot,
        color="tomato",    lw=1.2, ls="--")
ax.plot(x_plot, -0.9 * x_plot,
        color="tomato",    lw=1.2, ls="--")
ax.fill_between(x_plot,
                -0.9 * np.abs(x_plot),
                 0.9 * np.abs(x_plot),
                alpha=0.08, color="tomato")
ax.set_xlabel(r"$x$",   fontsize=13)
ax.set_ylabel(r"$x^+$", fontsize=13)  
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("case2_figure.png", dpi=150)
plt.close()
print("Saved case2_figure.png")
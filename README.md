# Koopman Modeling and Stabilization of Discrete-Time Nonlinear Control Systems: Bilinearity on a Reproducing Kernel Hilbert Space

Simulation code for the paper:

> **Koopman Modeling and Stabilization of Discrete-Time Nonlinear Control Systems: Bilinearity on a Reproducing Kernel Hilbert Space**
> Jarod Morris, Xiuzhen Ye, Wentao Tang
> *Allerton Conference on Communication, Control, and Computing*, 2026

The paper proves that a nonlinear control system, under mild smoothness conditions, admits a bilinear Koopman representation when the state and input are separately lifted into reproducing kernel Hilbert spaces (RKHSs) defined by linear–radial product kernels.  

---

## System

All simulations use nonlinear system with bilinearity.

---

## Case 1 — Bilinear Koopman Operator Learning (Section III-C)

**Step 1: Generate data**
```bash
python case1_data_generation.py
```
Produces `data_train.npz` (N=300 i.i.d. uniform training snapshots) and `data_test.npz` (N=500 i.i.d. uniform test snapshots) from $\mathbb{X} \times \mathbb{U}$.

**Step 2: Learn operator and plot**
```bash
python case1_main.py
```
Builds the three-term linear-radial Gram matrix:
$$G_{\mathring{\psi},ij} = \mathring{\kappa}(x_i, x_j) + \mathring{\varkappa}(u_i, u_j) + \mathring{\kappa}(x_i, x_j)\,\mathring{\varkappa}(u_i, u_j)$$
and solves the closed-form solution to eq. (3) of the paper:
$$\Theta = (G_{\mathring{\psi}} + N\beta I)^{-1}, \quad B = \Theta Y_{\mathrm{tr}}, \quad \hat{x}^+ = G_{\mathring{\psi},\mathrm{query}}\,B.$$

Produces:
- `case1_figure1.png` — (a) training/testing data scatter; (b) $\hat{x}^+$ vs $x^+$  
- `case1_figure2.png` — (a) $\hat{x}^+$ vs $x^+$; (b) 3D prediction errors inside paraboloid bound $\pm C|(x,u)|$

## Case 2 — Stabilizing Controller Synthesis (Section IV-C)

**Step 1: Generate control data**
```bash
python case2_data_generation.py
```
Samples $N_i = 200$ state points uniformly from $\mathbb{X}$ and constructs $N_j = 50$ scaled input candidates per state as $u_j(x_i) = c|x_i|\mathsf{u}_j$ with $c = 0.5$. Produces `data_control.npz`.

**Step 2: Synthesize controller and plot**
```bash
python case2_main.py
```
Sets the Lyapunov function $v(x) = x^2$ and decay rate $q(x) = 0.19x^2$. For each query state $x$, solves a small LP to find weights $w_j \geq 0$ satisfying the Lyapunov descent constraint:
$$\sum_j w_j\,v(f(x, u_j(x))) \leq 0.81\,x^2, \quad \sum_j w_j = 1,$$
and recovers the feedback policy $\alpha(x) = \sum_j w_j u_j(x)$.

Produces `case2_figure.png`:
- (a) Feedback policy $\alpha(x)$ vs $x$
- (b) Closed-loop $x^+ = f(x, \alpha(x))$ vs $x$, verified to lie within descent band $|x^+| \leq 0.9|x|$


## Dependencies

```
numpy
scipy
matplotlib
cvxpy          # for LP solver in Case 2
```
 
---

## Related Papers

This repository accompanies a series of related works on Koopman operator theory for nonlinear systems:

- **L4DC 2026** — Tang & Ye, *Koopman Operator for Stability Analysis: Theory with a Linear–Radial Product Reproducing Kernel* ([arXiv:2511.06063](https://arxiv.org/abs/2511.06063))
- **CCC 2026** — Tang & Ye, *Koopman-based Estimation of Lyapunov Functions: Theory on a Reproducing Kernel Hilbert Space*
- **CDC 2026** — Ye & Tang, *Dissipativity Analysis of Nonlinear Systems: A Linear–Radial Kernel-based Approach*

---

## Citation

```bibtex
@inproceedings{morris2026koopman,
  title     = {Koopman Modeling and Stabilization of Discrete-Time Nonlinear Control Systems:
               Bilinearity on a Reproducing Kernel Hilbert Space},
  author    = {Morris, Jarod and Ye, Xiuzhen and Tang, Wentao},
  booktitle = {Allerton Conference on Communication, Control, and Computing},
  year      = {2026}
}
```

---

## Acknowledgements

This work is supported by the National Science Foundation under CBET Award #2414369.

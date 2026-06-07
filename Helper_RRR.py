import numpy as np
from scipy.linalg import eigh


def solve_rrr(K_00, K_01, gamma, rank_r, eps=1e-8):
    """
    Solve (1/m^2) K_01 K_00 v = sigma^2 ( (1/m) K_00 + gamma I ) v
    Normalize v_i, i = 1, ..., r
    Obtain V = [v1, ..., vr]
    return Theta = (1/m^2) V V^T using the top-r eigenvectors.
    """
    m = K_00.shape[0]
    I = np.eye(m)

    # A = (1/m^2) K_01 K_00, B = (1/m) K_00 + gamma I
    A = K_01 @ K_00
    B = m * K_00 + (m ** 2) * gamma * I
 

    # symmetrize for numerical stability
    B = 0.5 * (B + B.T) + eps * I

    # step 1: solve Av = sigma B v, return sigma and v
    eigvals, eigvecs = eigh(A, B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # step 2: Normalization v_i^T (K_00/m + gamma I) (K_00/m) v_i = 1. Let v_i^T C v_i = 1
    C = (K_00/m + gamma * I) @ K_00/m
    for i in range(eigvecs.shape[1]):
        v = eigvecs[:,i]
        norm = np.sqrt(v.T @ C @ v)
        eigvecs[:, i] = v/norm

    # keep r eigvectors and has to be positive eigval
    keep = eigvals > 1e-12
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    rank_r = int(rank_r)
    r_eff = min(rank_r, eigvecs.shape[1])
    V = eigvecs[:, :r_eff]

    Theta = (V @ V.T) / (m**2)
    return eigvals[:r_eff], V, Theta 


def predict_rrr(K_va_tr, X1_train, Theta):
    """
    One-step prediction:
        x^+ = K_query_train @ Theta.T @ X1_train
    """
    return K_va_tr @ Theta.T @ X1_train
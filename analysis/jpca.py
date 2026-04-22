"""
jPCA: Joint Principal Component Analysis for rotational dynamics
Based on Churchland et al., 2012 (Neuron)

jPCA finds the plane of maximum rotational variance in high-dimensional
time-series data. It is commonly used to reveal rotational structure in
neural population activity.
"""

import numpy as np
from typing import Optional


def jpca(
    X: np.ndarray,
    num_jpcs: int = 2,
    num_pcs: int = 6,
    mean_subtract: bool = True,
) -> dict:
    """
    Perform jPCA on multi-condition time-series data.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n_conditions, n_times, n_features).
        Each condition is a trial/condition with time-varying observations.
    num_jpcs : int
        Number of jPC pairs to return. Each pair spans a 2D rotational plane.
        Must be even and <= num_pcs. Default: 2.
    num_pcs : int
        Number of PCs to project into before jPCA. Reduces dimensionality
        and noise. Default: 6.
    mean_subtract : bool
        If True, subtract the cross-condition mean trajectory before
        analysis. Recommended. Default: True.

    Returns
    -------
    dict with keys:
        'jpcs'          : np.ndarray (n_features, num_jpcs)
                          jPC axes in original feature space.
        'projections'   : np.ndarray (n_conditions, n_times, num_jpcs)
                          Data projected onto jPC axes.
        'eigenvalues'   : np.ndarray (num_jpcs,)
                          Imaginary parts of eigenvalues (rotation rates).
        'M_skew'        : np.ndarray (num_pcs, num_pcs)
                          Best-fit skew-symmetric matrix in PC space.
        'pcs'           : np.ndarray (n_features, num_pcs)
                          PCA axes used for preprocessing.
        'pc_projections': np.ndarray (n_conditions, n_times, num_pcs)
                          Data in PC space (before jPCA rotation).
        'explained_var' : float
                          Fraction of PC-space variance captured by jPCs.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be shape (conditions, times, features), got {X.shape}")

    n_cond, n_times, n_feat = X.shape

    if num_jpcs % 2 != 0:
        raise ValueError(f"num_jpcs must be even, got {num_jpcs}")
    if num_jpcs > num_pcs:
        raise ValueError(f"num_jpcs ({num_jpcs}) must be <= num_pcs ({num_pcs})")
    if num_pcs > min(n_feat, n_cond * n_times):
        raise ValueError(
            f"num_pcs ({num_pcs}) too large for data shape {X.shape}"
        )

    # ------------------------------------------------------------------ #
    # 1. Optional mean subtraction across conditions
    # ------------------------------------------------------------------ #
    if mean_subtract:
        mean_traj = X.mean(axis=0, keepdims=True)  # (1, T, F)
        X = X - mean_traj

    # ------------------------------------------------------------------ #
    # 2. PCA pre-processing
    # ------------------------------------------------------------------ #
    # Stack all conditions x times into rows for PCA
    X_flat = X.reshape(-1, n_feat)  # (C*T, F)

    # Center across all observations
    grand_mean = X_flat.mean(axis=0)
    X_centered = X_flat - grand_mean

    # SVD-based PCA (more numerically stable than eig on covariance)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pcs = Vt[:num_pcs].T  # (F, num_pcs)  — each column is a PC axis

    # Project into PC space
    X_pc_flat = X_centered @ pcs  # (C*T, num_pcs)
    X_pc = X_pc_flat.reshape(n_cond, n_times, num_pcs)  # (C, T, num_pcs)

    # ------------------------------------------------------------------ #
    # 3. Build the velocity dataset in PC space
    #    Churchland et al. fit  dX/dt ≈ M @ X  for a skew-symmetric M
    # ------------------------------------------------------------------ #
    # Use finite differences for velocity (exclude last time point)
    X_curr = X_pc[:, :-1, :]  # (C, T-1, num_pcs)
    X_next = X_pc[:, 1:, :]   # (C, T-1, num_pcs)
    dX = X_next - X_curr      # approximate dX/dt (units: per time step)

    # Reshape for regression
    X_curr_2d = X_curr.reshape(-1, num_pcs)  # (C*(T-1), num_pcs)
    dX_2d = dX.reshape(-1, num_pcs)          # (C*(T-1), num_pcs)

    # ------------------------------------------------------------------ #
    # 4. Fit best skew-symmetric M via closed-form solution
    #    Minimise ||dX - X @ M^T||_F  subject to M = -M^T
    #
    #    Unconstrained least-squares solution:  M_ls = (X^T X)^{-1} X^T dX
    #    Skew-symmetric projection:             M_skew = (M_ls - M_ls^T) / 2
    # ------------------------------------------------------------------ #
    XtX = X_curr_2d.T @ X_curr_2d               # (num_pcs, num_pcs)
    XtdX = X_curr_2d.T @ dX_2d                  # (num_pcs, num_pcs)

    # Regularised solve in case of near-singular XtX
    ridge = 1e-8 * np.trace(XtX) / num_pcs
    M_ls = np.linalg.solve(XtX + ridge * np.eye(num_pcs), XtdX)
    M_ls = M_ls.T  # shape (num_pcs, num_pcs), rows index output dims

    # Project onto skew-symmetric matrices
    M_skew = (M_ls - M_ls.T) / 2.0  # (num_pcs, num_pcs)

    # ------------------------------------------------------------------ #
    # 5. Eigen-decomposition of M_skew
    #    Eigenvalues come in conjugate pairs ±iω; eigenvectors are complex.
    #    Each pair defines a 2D rotation plane.
    # ------------------------------------------------------------------ #
    eigenvalues, eigenvectors = np.linalg.eig(M_skew)

    # Sort by magnitude of imaginary part (rotation rate), descending
    order = np.argsort(-np.abs(eigenvalues.imag))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ------------------------------------------------------------------ #
    # 6. Extract real jPC axes from complex eigenvectors
    #    For a pair (λ, λ*) with eigenvectors (v, v*):
    #      jPC_1 = Re(v) / ||Re(v)||
    #      jPC_2 = Im(v) / ||Im(v)||
    #    These two vectors span the plane of maximum rotation.
    # ------------------------------------------------------------------ #
    jpcs_pc = np.zeros((num_pcs, num_jpcs))  # axes in PC space

    for pair_idx in range(num_jpcs // 2):
        ev = eigenvectors[:, pair_idx * 2]   # one from the conjugate pair
        v_re = ev.real
        v_im = ev.imag

        norm_re = np.linalg.norm(v_re)
        norm_im = np.linalg.norm(v_im)

        if norm_re < 1e-12 or norm_im < 1e-12:
            # Degenerate case: fill with zeros
            jpcs_pc[:, pair_idx * 2] = 0.0
            jpcs_pc[:, pair_idx * 2 + 1] = 0.0
        else:
            jpcs_pc[:, pair_idx * 2] = v_re / norm_re
            jpcs_pc[:, pair_idx * 2 + 1] = v_im / norm_im

    # ------------------------------------------------------------------ #
    # 7. Map jPC axes back to original feature space
    # ------------------------------------------------------------------ #
    jpcs_orig = pcs @ jpcs_pc  # (n_features, num_jpcs)

    # ------------------------------------------------------------------ #
    # 8. Project data onto jPCs
    # ------------------------------------------------------------------ #
    # X_pc is (C, T, num_pcs); jpcs_pc is (num_pcs, num_jpcs)
    projections = X_pc @ jpcs_pc  # (C, T, num_jpcs)

    # ------------------------------------------------------------------ #
    # 9. Compute explained variance in PC space
    #    (variance of the jPC projections / total variance in PC space)
    # ------------------------------------------------------------------ #
    total_var = np.var(X_pc_flat, axis=0).sum()
    proj_flat = projections.reshape(-1, num_jpcs)
    jpc_var = np.var(proj_flat, axis=0).sum()
    explained_var = jpc_var / (total_var + 1e-12)

    # Rotation rates (imaginary parts of leading eigenvalues)
    rot_rates = np.array([
        eigenvalues[2 * k].imag for k in range(num_jpcs // 2)
        for _ in range(2)
    ])

    return {
        "jpcs": jpcs_orig,             # (n_features, num_jpcs)
        "projections": projections,    # (n_conditions, n_times, num_jpcs)
        "eigenvalues": rot_rates,      # (num_jpcs,)  imaginary parts
        "M_skew": M_skew,              # (num_pcs, num_pcs)
        "pcs": pcs,                    # (n_features, num_pcs)
        "pc_projections": X_pc,        # (n_conditions, n_times, num_pcs)
        "explained_var": float(explained_var),
    }


# --------------------------------------------------------------------------- #
# Convenience: project new data onto previously computed jPC axes
# --------------------------------------------------------------------------- #

def jpca_transform(
    X_new: np.ndarray,
    result: dict,
    mean_subtract: bool = True,
) -> np.ndarray:
    """
    Project new data onto jPC axes computed by a previous `jpca()` call.

    Parameters
    ----------
    X_new : np.ndarray
        Shape (n_conditions, n_times, n_features).
    result : dict
        Output dictionary from `jpca()`.
    mean_subtract : bool
        Subtract cross-condition mean of X_new before projecting.

    Returns
    -------
    projections : np.ndarray  (n_conditions, n_times, num_jpcs)
    """
    if mean_subtract:
        X_new = X_new - X_new.mean(axis=0, keepdims=True)

    n_cond, n_times, n_feat = X_new.shape
    X_flat = X_new.reshape(-1, n_feat)

    # Project into PC space (use grand mean = 0 since we already subtracted)
    X_pc = X_flat @ result["pcs"]
    X_pc = X_pc.reshape(n_cond, n_times, -1)

    # Project onto jPCs (in PC space)
    jpcs_pc = result["pcs"].T @ result["jpcs"]  # (num_pcs, num_jpcs)
    projections = X_pc @ jpcs_pc

    return projections


# --------------------------------------------------------------------------- #
# Quick demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate 20 conditions, 50 time steps, 100 neurons
    # with embedded rotational structure in a 2D subspace
    n_cond, n_times, n_feat = 20, 50, 100

    t = np.linspace(0, 2 * np.pi, n_times)
    angles = np.linspace(0, 2 * np.pi, n_cond, endpoint=False)

    # Two "rotational" dimensions
    rot_dim1 = rng.standard_normal(n_feat)
    rot_dim2 = rng.standard_normal(n_feat)
    rot_dim1 /= np.linalg.norm(rot_dim1)
    rot_dim2 /= np.linalg.norm(rot_dim2)
    rot_dim2 -= rot_dim2 @ rot_dim1 * rot_dim1  # orthogonalise
    rot_dim2 /= np.linalg.norm(rot_dim2)

    X = np.zeros((n_cond, n_times, n_feat))
    for c, angle in enumerate(angles):
        traj = (
            np.outer(np.cos(t + angle), rot_dim1)
            + np.outer(np.sin(t + angle), rot_dim2)
        )
        X[c] = traj + 0.3 * rng.standard_normal((n_times, n_feat))

    log

    print("jPCA demo")
    print("---------")
    print(f"Data shape            : {X.shape}")
    print(f"jPCs shape            : {result['jpcs'].shape}")
    print(f"Projections shape     : {result['projections'].shape}")
    print(f"Rotation rates (±iω)  : {result['eigenvalues'][:2]}")
    print(f"Explained variance    : {result['explained_var']:.1%}")

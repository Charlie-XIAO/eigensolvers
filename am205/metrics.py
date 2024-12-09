import numpy as np

machep = np.finfo(np.float64).eps


def orthogonality_loss(z, n):
    inner = np.abs(z.T @ z)
    np.fill_diagonal(inner, 0)
    return np.max(inner) / (n * machep)


def residual_norm(d, e, vals, z, n):
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    residuals = [
        np.linalg.norm(T @ z[:, i] - vals[i] * z[:, i]) for i in range(n)
    ]
    return np.max(residuals) / (np.linalg.norm(T) * n * machep)

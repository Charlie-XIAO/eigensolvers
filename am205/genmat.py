import numpy as np

from metrics import machep
from scipy.linalg.lapack import dsytrd


def _from_eigenvalues(eigenvalues, n):
    Q, _ = np.linalg.qr(np.random.random((n, n)))
    Lambda = np.diag(eigenvalues)
    A = Q @ Lambda @ Q.T
    _, d, e, _, info = dsytrd(A, lower=True)
    assert info == 0, f"{info=}"
    return d, e


def uniform_eps(n, rng):
    eigenvalues = machep * np.arange(1, n, dtype=np.float64)
    eigenvalues = np.append(eigenvalues, 1.0)
    return _from_eigenvalues(eigenvalues, n)


def uniform_sqeps(n, rng):
    eigenvalues = 1 + np.sqrt(machep) * np.arange(2, n, dtype=np.float64)
    eigenvalues = np.insert(eigenvalues, 0, machep)
    eigenvalues = np.append(eigenvalues, 2.0)
    return _from_eigenvalues(eigenvalues, n)


def uniform_equi(n, rng):
    tau = (1.0 - machep) / (n - 1)
    eigenvalues = machep + tau * np.arange(n, dtype=np.float64)
    return _from_eigenvalues(eigenvalues, n)


def geometric(n, rng):
    numers = n - np.arange(1, n + 1, dtype=np.float64)
    eigenvalues = np.exp(machep, numers / (n - 1))
    return _from_eigenvalues(eigenvalues, n)


def normal(n, rng):
    eigenvalues = rng.normal(0, 1, n)
    return _from_eigenvalues(eigenvalues, n)


def clustered_one(n, rng):
    eigenvalues = np.ones(n)
    eigenvalues[1:] += machep * rng.random(n - 1)
    eigenvalues[0] = machep
    return _from_eigenvalues(eigenvalues, n)


def clustered_pmone(n, rng):
    eigenvalues = np.ones(n)
    eigenvalues[1:] *= rng.choice([-1, 1], n - 1)
    eigenvalues[1:] += machep * rng.random(n - 1)
    eigenvalues[0] = machep
    return _from_eigenvalues(eigenvalues, n)


def clustered_machep(n, rng):
    eigenvalues = np.full(n, machep)
    eigenvalues[:-1] += machep * rng.random(n - 1)
    eigenvalues[-1] = 1.0
    return _from_eigenvalues(eigenvalues, n)


def clustered_pmmachep(n, rng):
    eigenvalues = np.full(n, machep)
    eigenvalues[:-1] *= rng.choice([-1, 1], n - 1)
    eigenvalues[:-1] += machep * rng.random(n - 1)
    eigenvalues[-1] = 1.0
    return _from_eigenvalues(eigenvalues, n)


def wilkinson(n, __unused):
    d = np.arange(1, n + 1, dtype=np.float64)
    e = np.full(n - 1, 1.0, dtype=np.float64)
    return d, e


random_matrix_generators = {
    "uniform_eps": uniform_eps,
    "uniform_sqeps": uniform_sqeps,
    "uniform_equi": uniform_equi,
    "geometric": geometric,
    "normal": normal,
    "clustered_one": clustered_one,
    "clustered_pmone": clustered_pmone,
    "clustered_machep": clustered_machep,
    "clustered_pmmachep": clustered_pmmachep,
    "wilkinson": wilkinson,
}

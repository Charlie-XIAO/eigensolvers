import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import dsytrd, get_lapack_funcs


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


def main(n, gen_func):
    print(f"{n=}, {gen_func=}")

    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()
    results_path = results_dir / f"random-{gen_func}.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    if f"{n=}" in results:
        print("⚠️ Skipped")
        return

    rng = np.random.default_rng(0)
    d, e = random_matrix_generators[gen_func](n, rng)

    result = {
        "compute_v=True": {},
        "compute_v=False": {},
        "orthogonality_loss": {},
        "residual_norm": {},
    }
    dstev, dstevd, dstevr, dstevx = get_lapack_funcs(
        ["stev", "stevd", "stevr", "stevx"], dtype=np.float64
    )

    for compute_v in [False, True]:
        start = time.perf_counter()
        vals, z, info = dstev(d, e, compute_v=compute_v)
        duration = time.perf_counter() - start
        assert info == 0, f"{info=}"
        print(f"✅ dstev  [{duration:.4f}s]")
        result[f"{compute_v=}"]["dstev"] = duration
        if compute_v:
            result["orthogonality_loss"]["dstev"] = orthogonality_loss(z, n)
            result["residual_norm"]["dstev"] = residual_norm(d, e, vals, z, n)

        start = time.perf_counter()
        vals, z, info = dstevd(d, e, compute_v=compute_v)
        duration = time.perf_counter() - start
        assert info == 0, f"{info=}"
        print(f"✅ dstevd [{duration:.4f}s]")
        result[f"{compute_v=}"]["dstevd"] = duration
        if compute_v:
            result["orthogonality_loss"]["dstevd"] = orthogonality_loss(z, n)
            result["residual_norm"]["dstevd"] = residual_norm(d, e, vals, z, n)

        start = time.perf_counter()
        m, vals, z, info = dstevr(d, e, 0, 0.0, 0.0, 0, 0, compute_v=compute_v)
        duration = time.perf_counter() - start
        assert info == 0, f"{info=}"
        print(f"✅ dstevr [{duration:.4f}s]")
        result[f"{compute_v=}"]["dstevr"] = duration
        if compute_v:
            result["orthogonality_loss"]["dstevr"] = orthogonality_loss(z, n)
            result["residual_norm"]["dstevr"] = residual_norm(d, e, vals, z, n)

        start = time.perf_counter()
        m, vals, z, info = dstevx(d, e, 0, 0.0, 0.0, 0, 0, compute_v=compute_v)
        duration = time.perf_counter() - start
        assert info == 0, f"{info=}"
        print(f"✅ dstevx [{duration:.4f}s]")
        result[f"{compute_v=}"]["dstevx"] = duration
        if compute_v:
            result["orthogonality_loss"]["dstevx"] = orthogonality_loss(z, n)
            result["residual_norm"]["dstevx"] = residual_norm(d, e, vals, z, n)

    results[f"{n=}"] = result
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    n = int(sys.argv[1])
    gen_func = sys.argv[2]
    assert gen_func in random_matrix_generators, f"{gen_func=}"
    main(n, gen_func)

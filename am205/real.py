import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import mmread
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


def main(i):
    name = f"bcsstk{i:02d}"
    print(f"{name=}")

    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()
    results_path = results_dir / f"real.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    if name in results:
        print("⚠️ Skipped")
        return

    T = mmread(data_dir / f"{name}.mtx")
    T = T.toarray()
    print(f"{T.shape=}, {T.dtype=}")
    n = T.shape[0]
    if n > 5000:
        print(f"⚠️ Matrix too large ({n=})")
        return

    _, d, e, _, info = dsytrd(T, lower=True)
    assert info == 0, f"{info=}"

    result = {
        "n": n,
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

    results[name] = result
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    i = int(sys.argv[1])
    main(i)

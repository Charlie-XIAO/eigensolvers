# AM205 Symmetric Tridiagonal Eigensolvers

This is my final project repository for Harvard AM205 (Fall 2024), titled *A Comparative Analysis of LAPACK Routines for Solving Eigenproblems on Real Symmetric Tridiagonal Matrices*. The report can be found [here](./am205/final-report.pdf). The project relies on `scipy` to implement wrappers of LAPACK routines (that are not already covered in `scipy.linalg.lapack`). Changes to the `scipy` codebase are mainly in `/scipy/linalg/flapack_other.pyf.src`. See [`6e8d0f8`](https://github.com/Charlie-XIAO/eigensolvers/commit/6e8d0f819fbac4a366103a21e86e3430d49d911f) for more details.

## Running Instructions

Follow the scipy documentation to install the system dependencies and set up the development environment. Then to obtain all the evaluation plots:

```bash
make devbuild
make prepare
make run-random
make run-real
```

All experiments are under the `/am205/` directory. See the root `Makefile` for brief explanations of these targets. The evaluation plots will be stored under the `/am205/plots/` directory.

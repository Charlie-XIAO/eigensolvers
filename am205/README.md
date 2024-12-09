This is my final project repository for Harvard AM205 (Fall 2024). The project relies on scipy to implement wrappers of LAPACK routines (that are not already covered in `scipy.linalg.lapack`).

Follow the scipy documentation to install the system dependencies and set up the development environment.  Then to obtain all the evaluation plots:

```bash
make devbuild  # Skip if scipy is already built
make prepare
make run-random
make run-real
```

The evaluation plots will be stored under `/am205/plots/`.

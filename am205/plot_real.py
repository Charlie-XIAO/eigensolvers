import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial import Polynomial

colors = sns.color_palette()
linestyles = ["-", "--", "-.", ":"]

routines = ["dstev", "dstevd", "dstevr", "dstevx"]


def main():
    results_dir = Path(__file__).parent / "results"
    plots_dir = Path(__file__).parent / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    # Collect results
    results_path = results_dir / f"real.json"
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # Orthogonality loss and residual norm plots
    for metric in ["orthogonality_loss", "residual_norm"]:
        data = {routine: {} for routine in routines}
        for name, result in results.items():
            for routine in routines:
                data[routine][name] = result[metric][routine]

        plt.figure(figsize=(12, 6))
        datasets = list(next(iter(data.values())).keys())
        x = np.arange(len(datasets))
        width = 0.8 / len(routines)
        for i, routine in enumerate(routines):
            targets = [data[routine][dataset] for dataset in datasets]
            plt.bar(x + i * width - 0.4, targets, width, alpha=0.5, color=colors[i], edgecolor="black", label=routine)

        plt.xticks(x, datasets, rotation=45)
        plt.yscale("log")
        if metric == "orthogonality_loss":
            plt.title(f"Orthogonality Loss")
        elif metric == "residual_norm":
            plt.title(f"Maximum Residual Norm")
        plt.grid(axis="y", linestyle="--", alpha=0.8)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / f"real-{metric}.png")
        plt.clf()
        print(f"Saved: {metric=}")
    return

    # Runtime plot for all routines
    for compute_v in [False, True]:
        ns, durations = [], []
        for result in results.values():
            if f"{compute_v=}" in result and len(result[f"{compute_v=}"]) > 0:
                ns.append(int(result["n"]))
                durations.append(result[f"{compute_v=}"])

        for routine, color in zip(routines, colors):
            routine_durations = [duration[routine] for duration in durations]
            x, y = np.log10(ns), np.log10(routine_durations)
            p = Polynomial.fit(x, y, deg=1)
            xs = np.linspace(x[0], x[-1], 100)

            plt.scatter(ns, routine_durations, color=color, alpha=0.5)
            plt.plot(10 ** xs, 10 ** p(xs), color=color, linestyle="--", label=f"{routine} ({p.convert().coef[1]:.2f})")

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Matrix size")
        plt.ylabel("Time taken (s)")
        title_text = "Real-world"
        if not compute_v:
            title_text += " (eigenvalues only)"
        plt.title(title_text)
        plt.tight_layout()
        plt.legend(loc="upper left")
        plt.savefig(plots_dir / f"real-{compute_v=}.png")
        print(f"Saved: {compute_v=}")
        plt.clf()


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial import Polynomial

colors = sns.color_palette()
linestyles = ["-", "--", "-.", ":"]
hatches = ["//", "\\\\", "++", "xx"]

routines = ["dstev", "dstevd", "dstevr", "dstevx"]

gen_funcs = [
    "uniform_eps",
    "uniform_sqeps",
    "uniform_equi",
    "geometric",
    "normal",
    "clustered_one",
    "clustered_pmone",
    "clustered_machep",
    "clustered_pmmachep",
    "wilkinson",
]

gen_funcs_styles = {
    "uniform_eps": {"color": colors[0], "linestyle": linestyles[0]},
    "uniform_sqeps": {"color": colors[0], "linestyle": linestyles[1]},
    "uniform_equi": {"color": colors[0], "linestyle": linestyles[2]},
    "geometric": {"color": colors[1], "linestyle": linestyles[0]},
    "normal": {"color": colors[2], "linestyle": linestyles[0]},
    "clustered_one": {"color": colors[3], "linestyle": linestyles[0]},
    "clustered_pmone": {"color": colors[3], "linestyle": linestyles[1]},
    "clustered_machep": {"color": colors[4], "linestyle": linestyles[0]},
    "clustered_pmmachep": {"color": colors[4], "linestyle": linestyles[1]},
    "wilkinson": {"color": colors[5], "linestyle": linestyles[0]},
}

def main():
    results_dir = Path(__file__).parent / "results"
    plots_dir = Path(__file__).parent / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    # Collect results
    all_results = {}
    for gen_func in gen_funcs:
        results_path = results_dir / f"random-{gen_func}.json"
        if not results_path.exists():
            continue
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        all_results[gen_func] = {
            int(n.split("=")[1]): result for n, result in results.items()
        }

    # Orthogonality loss and residual norm plots
    for metric in ["orthogonality_loss", "residual_norm"]:
        data = {
            routine: {gen_func: [] for gen_func in gen_funcs} for routine in routines
        }
        for gen_func, results in all_results.items():
            for result in results.values():
                for routine in routines:
                    data[routine][gen_func].append(result[metric][routine])

        targets = {
            routine: [np.max(data[routine][gen_func]) for gen_func in gen_funcs]
            for routine in routines
        }

        x = np.arange(len(routines))
        width = 0.8 / len(gen_funcs)
        for i, gen_func in enumerate(gen_funcs):
            targets_values = [targets[routine][i] for routine in routines]
            plt.bar(x + i * width - 0.4, targets_values, width, alpha=0.5, color=colors[i], edgecolor="black", label=gen_func)

        plt.xticks(x, routines)
        plt.yscale("log")
        if metric == "orthogonality_loss":
            plt.title(f"Orthogonality Loss")
        elif metric == "residual_norm":
            plt.title(f"Maximum Residual Norm")
        plt.grid(axis="y", linestyle="--", alpha=0.8)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / f"random-{metric}.png")
        plt.clf()
        print(f"Saved: {metric=}")

    # Runtime plot per routine for all generator functions
    for routine in routines:
        for compute_v in [False, True]:
            for gen_func, results in all_results.items():
                ns, durations = [], []
                for n, result in results.items():
                    ns.append(n)
                    durations.append(result[f"{compute_v=}"][routine])
                x, y = np.log10(ns), np.log10(durations)
                p = Polynomial.fit(x, y, deg=1)
                xs = np.linspace(x[0], x[-1], 100)

                plt.scatter(ns, durations, color=gen_funcs_styles[gen_func]["color"], alpha=0.5)
                plt.plot(10 ** xs, 10 ** p(xs), **gen_funcs_styles[gen_func], label=f"{gen_func} ({p.convert().coef[1]:.2f})")

            plt.xscale("log")
            plt.yscale("log")
            if compute_v:
                plt.ylim(1e-4, 1e1)
            else:
                plt.ylim(1e-4, 1e-1)
            plt.xlabel("Matrix size")
            plt.ylabel("Time taken (s)")
            title_text = routine
            if not compute_v:
                title_text += " (eigenvalues only)"
            plt.title(title_text)
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.savefig(plots_dir / f"random-{routine}-{compute_v=}.png")
            print(f"Saved: {routine=}, {compute_v=}")
            plt.clf()

    # Runtime plot per generator function for all routines
    for gen_func, results in all_results.items():
        for compute_v in [False, True]:
            ns, durations = [], []
            for n, result in results.items():
                if f"{compute_v=}" in result and len(result[f"{compute_v=}"]) > 0:
                    ns.append(n)
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
            title_text = gen_func
            if not compute_v:
                title_text += " (eigenvalues only)"
            plt.title(title_text)
            plt.tight_layout()
            plt.legend(loc="upper left")
            plt.savefig(plots_dir / f"random-{gen_func}-{compute_v=}.png")
            print(f"Saved: {gen_func=}, {compute_v=}")
            plt.clf()


if __name__ == "__main__":
    main()

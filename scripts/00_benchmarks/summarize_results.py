"""Make compact tables and scientific figures from completed comparisons."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/00_benchmarks"
FIGURES = ROOT / "figures/00_benchmarks"
VARIANTS = ["baseline", "rewrite_native", "rewrite_memmap"]
LABELS = ["Merged baseline", "Rewrite: DuckDB", "Rewrite: shared arrays"]
COLORS = ["#6c757d", "#167c80", "#d58422"]


def main():
    reports = {path.stem.removeprefix("final_"): json.loads(path.read_text())
               for path in sorted(DATA.glob("final_*.json"))}
    if not reports:
        raise ValueError("Run compare_execution.py or run_matrix.py first.")
    if any(report.get("sampler_version", 1) < 2 for report in reports.values()):
        raise ValueError("These results use the incomplete memory sampler. Rerun the comparisons.")
    metrics = {report["memory_metric"] for report in reports.values()}
    if len(metrics) != 1:
        raise ValueError("Summarize each platform separately; footprint and PSS are different measures.")
    memory_metric = metrics.pop()
    rows = []
    for case, report in reports.items():
        for result in report["results"]:
            times = result["phase_seconds"]
            fit = sum(times.get(phase, 0) for phase in ["fit_preparation", "shared_input_preparation",
                                                       "training", "fit_finalize"])
            phases = {name: value for name, value in result["phase_memory"].items() if value}
            rows.append(dict(case=case, variant=result["variant"], rows=report["fixture"]["rows"],
                ensembles=report["configuration"]["ensembles"], wall_seconds=result["wall_seconds"],
                partition_seconds=times["partitioning"], fit_seconds=fit,
                heldout_prediction_seconds=times["heldout_prediction"], evaluation_seconds=times["evaluation"],
                prediction_seconds=times["prediction"],
                peak_accounted_mb=result["process_tree_peak_accounted_mb"],
                prepared_pipeline_peak_mb=result.get("prepared_input_pipeline_peak_accounted_mb"),
                peak_rss_mb=result["process_tree_peak_rss_mb"],
                peak_phase=max(phases, key=lambda phase: phases[phase]["peak_accounted_mb"]),
                train_peak_mb=phases.get("training", {}).get("peak_accounted_mb"),
                prediction_peak_mb=phases.get("prediction", {}).get("peak_accounted_mb"),
                shared_file_mb=result["execution_info"]["shared_input_bytes"] / 1024**2
                    if result.get("execution_info") else 0,
                saved_models_mb=result["saved_models_bytes"] / 1024**2,
                max_prediction_difference=result.get("max_prediction_difference", 0),
                max_full_prediction_difference=result.get("max_full_prediction_difference", 0),
                same_nan_mask=result.get("same_nan_mask", True),
                same_full_nan_mask=result.get("same_full_nan_mask", True)))
    (DATA / "summary.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(f"{row['case']:25} {row['variant']:16} fit={row['fit_seconds']:6.2f}s "
              f"predict={row['prediction_seconds']:6.2f}s peak={row['peak_accounted_mb']:8.1f} MiB "
              f"({row['peak_phase']})")

    FIGURES.mkdir(parents=True, exist_ok=True)
    names = list(reports)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout="constrained")
    for position, (variant, label, color) in enumerate(zip(VARIANTS, LABELS, COLORS)):
        values = [next(row for row in rows if row["case"] == name and row["variant"] == variant) for name in names]
        for ax, field in zip(axes, ["wall_seconds", "peak_accounted_mb", "peak_rss_mb"]):
            ax.bar(x + (position - 1) * .25, [row[field] for row in values], width=.24,
                   label=label, color=color)
    axes[0].set_ylabel("Whole job wall time (seconds, including imports)")
    axes[1].set_ylabel(f"Whole job peak {memory_metric} (MiB)")
    axes[2].set_ylabel("Whole job peak summed process RSS (MiB)")
    for ax in axes:
        ax.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle("Training and prediction in the same job: measured local comparisons")
    fig.savefig(FIGURES / "whole_job_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.5 * len(names)), layout="constrained", squeeze=False)
    for index, name in enumerate(names):
        for position, (variant, label, color) in enumerate(zip(VARIANTS, LABELS, COLORS)):
            row = next(row for row in rows if row["case"] == name and row["variant"] == variant)
            axes[index, 0].bar(np.arange(3) + (position - 1) * .24,
                [row["fit_seconds"], row["heldout_prediction_seconds"], row["prediction_seconds"]],
                width=.23, color=color, label=label)
            axes[index, 1].bar(np.arange(2) + (position - 1) * .24,
                [row["train_peak_mb"], row["prediction_peak_mb"]], width=.23, color=color)
        axes[index, 0].set_xticks(range(3), ["Fit", "Held-out prediction", "Full prediction"])
        axes[index, 0].set_ylabel("Seconds")
        axes[index, 1].set_xticks(range(2), ["Training peak", "Prediction peak"])
        axes[index, 1].set_ylabel(f"{memory_metric} (MiB)")
        axes[index, 0].set_title(name.replace("_", " "))
        for ax in axes[index]:
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Phase trade-offs; phase peaks are not additive")
    fig.savefig(FIGURES / "phase_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()

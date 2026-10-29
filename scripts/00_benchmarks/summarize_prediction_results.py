"""Summarize repeated whole-pipeline comparisons and isolated aggregation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/00_benchmarks/streaming_prediction"
FIGURES = ROOT / "figures/00_benchmarks/streaming_prediction"
VARIANTS = ["baseline", "rewrite_native", "rewrite_memmap"]
LABELS = ["First rewrite", "Streaming + native input", "Streaming + shared training"]


def main():
    groups = {}
    for path in sorted((DATA/"repeated").glob("*_repeat*.json")):
        report = json.loads(path.read_text())
        if report["baseline"] != "53dc476" or report["sampler_version"] != 2:
            raise ValueError("Expected first-rewrite baseline and complete process sampler.")
        case = path.stem.split("_repeat")[0]
        for result in report["results"]:
            if result["variant"] != "baseline" and not result["equivalent_within_1e6"]:
                raise ValueError("Numerical equivalence failed.")
            times = result["phase_seconds"]
            memory = result["phase_memory"]
            row = dict(wall_seconds=result["wall_seconds"],
                fit_seconds=sum(times.get(k, 0) for k in ["fit_preparation",
                    "shared_input_preparation", "training", "fit_finalize"]),
                prediction_seconds=times["prediction"],
                heldout_prediction_seconds=times["heldout_prediction"],
                partition_seconds=times["partitioning"],
                peak_accounted_mb=result["process_tree_peak_accounted_mb"],
                peak_rss_mb=result["process_tree_peak_rss_mb"],
                train_peak_accounted_mb=memory["training"]["peak_accounted_mb"],
                prediction_peak_accounted_mb=memory["prediction"]["peak_accounted_mb"],
                saved_models_mb=result["saved_models_bytes"]/1024**2,
                largest_saved_model_mb=result["largest_saved_model_bytes"]/1024**2,
                shared_input_mb=(result.get("execution_info") or {}).get("shared_input_bytes", 0)/1024**2,
                max_prediction_difference=result.get("max_prediction_difference", 0),
                max_full_prediction_difference=result.get("max_full_prediction_difference", 0))
            groups.setdefault((case, result["variant"]), []).append(row)
    rows = []
    for (case, variant), values in groups.items():
        stats = {key: dict(median=float(np.median([v[key] for v in values])),
            minimum=float(min(v[key] for v in values)), maximum=float(max(v[key] for v in values)))
            for key in values[0]}
        rows.append(dict(case=case, variant=variant, n_repeats=len(values), statistics=stats))
        print(case, variant, {k: round(v["median"], 4) for k, v in stats.items()})
    (DATA/"repeated_summary.json").write_text(json.dumps(rows, indent=2))
    FIGURES.mkdir(parents=True, exist_ok=True)
    cases = list(dict.fromkeys(r["case"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for i, (variant, label) in enumerate(zip(VARIANTS, LABELS)):
        selected = [next(r for r in rows if r["case"] == case and r["variant"] == variant) for case in cases]
        for axis, field in zip(axes, ["wall_seconds", "prediction_seconds", "peak_accounted_mb"]):
            stats = [r["statistics"][field] for r in selected]
            median = np.array([s["median"] for s in stats])
            axis.bar(np.arange(len(cases))+(i-1)*.25, median, width=.24, label=label,
                yerr=[median-[s["minimum"] for s in stats], [s["maximum"] for s in stats]-median],
                capsize=3)
    for axis, title, ylabel in zip(axes,
        ["Whole pipeline", "Full prediction", "Whole-pipeline peak"],
        ["Seconds", "Seconds", "macOS footprint (MiB)"]):
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(np.arange(len(cases)), [c.replace("_", "\n") for c in cases], fontsize=8)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=7)
    fig.suptitle("Medians and ranges of three fresh-process repeats, two workers")
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES/f"repeated_pipeline.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

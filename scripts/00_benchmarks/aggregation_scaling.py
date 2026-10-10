"""Isolate mean/std aggregation memory from estimator and database costs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from compare_execution import process_memory


def worker(args):
    import numpy as np
    import pandas as pd
    from stemflow.lazyloading.prediction import PredictionMoments
    rng = np.random.default_rng(42)
    ids = pd.RangeIndex(args.rows)
    before = time.perf_counter()
    if args.mode == "matrix":
        values = np.empty((args.rows, args.ensembles))
        for i in range(args.ensembles):
            values[:, i] = rng.uniform(size=args.rows)
        frame = pd.DataFrame(values)
        mean, std = frame.mean(axis=1).to_numpy(), frame.std(axis=1).to_numpy()
        logical = values.nbytes
    else:
        reducer = PredictionMoments(ids, with_std=True)
        for _ in range(args.ensembles):
            reducer.update(pd.DataFrame({"pred": rng.uniform(size=args.rows)}, index=ids))
        mean, std = reducer.finish(1)
        logical = reducer.mean.nbytes + reducer.m2.nbytes + reducer.count.nbytes
    Path(args.result).write_text(json.dumps(dict(mode=args.mode, rows=args.rows,
        ensembles=args.ensembles, aggregation_seconds=time.perf_counter()-before,
        logical_storage_bytes=logical, mean_sum=float(mean.sum()), std_sum=float(std.sum()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500000)
    parser.add_argument("--ensembles", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/00_benchmarks/streaming_prediction/aggregation.json"))
    parser.add_argument("--mode", choices=["matrix", "streaming"])
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    results = []
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ["matrix", "streaming"]:
            result = str(Path(temporary)/f"{mode}.json")
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MPLCONFIGDIR": str(Path(temporary)/"matplotlib")}
            process = subprocess.Popen([sys.executable, __file__, "--mode", mode,
                "--rows", str(args.rows), "--ensembles", str(args.ensembles), "--result", result], env=env)
            peak_rss, peak_accounted = 0, 0
            while process.poll() is None:
                rss, accounted = process_memory(process.pid)
                peak_rss, peak_accounted = max(peak_rss, rss), max(peak_accounted, accounted)
                time.sleep(.02)
            if process.returncode:
                raise RuntimeError(f"Aggregation child exited with {process.returncode}.")
            record = json.loads(Path(result).read_text())
            record.update(peak_rss_bytes=peak_rss, peak_accounted_bytes=peak_accounted)
            results.append(record)
    report = dict(results=results, note="Synthetic aggregation-only workload; includes generation and alignment. "
        "Process peaks include imports; not a whole AdaSTEM benchmark.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Repeat isolated prediction comparisons against the first execution rewrite."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--baseline", default="53dc476")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT/"data/00_benchmarks/streaming_prediction/repeated")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = dict(acadian_100000_e8=["--source", "real", "--rows", "100000", "--features", "116"],
        forest_50000_e8_lazy=["--rows", "50000", "--features", "32",
            "--base-model", "random_forest", "--trees", "40", "--depth", "12"])
    for repeat in range(1, args.repeat+1):
        for name, options in cases.items():
            output = args.output_dir/f"{name}_repeat{repeat}.json"
            print(f"Running {name}, repeat {repeat}", flush=True)
            subprocess.run([sys.executable, str(ROOT/"scripts/00_benchmarks/compare_execution.py"),
                "--baseline", args.baseline, "--ensembles", "8", "--jobs", "2",
                "--output", str(output), *options], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

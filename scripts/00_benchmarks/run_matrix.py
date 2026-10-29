"""Run sequential end-to-end comparisons across data and ensemble sizes."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("acadian_30000_e4", ["--source", "real", "--rows", "30000", "--ensembles", "4", "--jobs", "2"]),
    ("acadian_100000_e4", ["--source", "real", "--rows", "100000", "--ensembles", "4", "--jobs", "4"]),
    ("acadian_100000_e8", ["--source", "real", "--rows", "100000", "--ensembles", "8", "--jobs", "4"]),
    ("acadian_100000_e8_j2", ["--source", "real", "--rows", "100000", "--ensembles", "8", "--jobs", "2"]),
    ("synthetic_200000_e4", ["--rows", "200000", "--features", "128", "--ensembles", "4", "--jobs", "4"]),
    ("synthetic_200000_e8", ["--rows", "200000", "--features", "128", "--ensembles", "8", "--jobs", "4"]),
    ("forest_50000_e8", ["--rows", "50000", "--features", "32", "--ensembles", "8", "--jobs", "4",
                         "--base-model", "random_forest", "--trees", "40", "--depth", "12", "--resident-models"]),
    ("forest_50000_e8_lazy", ["--rows", "50000", "--features", "32", "--ensembles", "8", "--jobs", "4",
                              "--base-model", "random_forest", "--trees", "40", "--depth", "12"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=[name for name, _ in CASES])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/00_benchmarks")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for repeat in range(args.repeat):
        for name, options in CASES:
            if args.cases and name not in args.cases:
                continue
            suffix = f"_repeat{repeat + 1}" if args.repeat > 1 else ""
            output = args.output_dir / f"final_{name}{suffix}.json"
            if args.skip_existing and output.exists():
                continue
            print(f"Running {name}{suffix}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name("compare_execution.py")),
                            *options, "--data-dir", str(args.data_dir), "--output", str(output)],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

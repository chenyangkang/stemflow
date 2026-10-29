"""Quick native-allocation check; run unchanged on macOS, Linux or Windows.

Example: python scripts/00_benchmarks/check_execution_monitor.py
Requires psutil. Retains one compact JSON report and removes all child processes.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stemflow.utils.execution_monitor import ExecutionMonitor, _ProcessMemoryReader


WORKER = """
import mmap, sys
size = int(sys.argv[1]) * 2**20
payload = mmap.mmap(-1, size)
for offset in range(0, size, 4096):
    payload[offset] = 1
print('ready', flush=True)
sys.stdin.readline()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, default=32)
    parser.add_argument("--output", type=Path,
        default=ROOT / "data/00_benchmarks/execution_monitor/validation.json")
    args = parser.parse_args()
    if args.memory_mib < 1:
        parser.error("--memory-mib must be positive")
    processes, costs = [], {}
    with ExecutionMonitor(sample_interval=.05, log_interval=None,
            output_path=args.output, detailed_memory=True, max_samples=100) as monitor:
        with monitor.phase("known_native_allocations"):
            try:
                for _ in range(2):
                    process = subprocess.Popen([sys.executable, "-c", WORKER, str(args.memory_mib)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    processes.append(process)
                    if process.stdout.readline().strip() != "ready":
                        raise RuntimeError("Allocation worker failed to start")
                expected = 2 * args.memory_mib * 2**20
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    latest = monitor.summary()["latest_sample"]
                    if (latest is not None and (latest["children_rss_bytes"] or 0) >= expected
                            and {p.pid for p in processes} <= set(latest["process_ids"])):
                        break
                    time.sleep(.02)
                else:
                    raise AssertionError("Sampler did not observe both native allocations")
                for detailed in (False, True):
                    reader, durations = _ProcessMemoryReader(detailed), []
                    for _ in range(20):
                        started = time.perf_counter()
                        reader.snapshot(monitor.pid)
                        durations.append(time.perf_counter() - started)
                    costs["detailed" if detailed else "rss_only"] = dict(
                        median_sample_ms=1000 * statistics.median(durations),
                        max_sample_ms=1000 * max(durations), processes=latest["process_count"])
            finally:
                for process in processes:
                    try:
                        process.communicate("exit\n", timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
        with monitor.phase("after_children_exit"):
            time.sleep(.1)
    report = monitor.summary()
    report["validation"] = dict(passed=True, child_allocation_bytes=expected,
                                sampling_cost=costs)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(platform=report["platform"], psutil=report["psutil_version"],
        passed=True, peak_children_mib=report["peak_children_rss_bytes"] / 2**20,
        sampling_cost=costs, report=str(args.output)), indent=2))


if __name__ == "__main__":
    main()

"""The benchmark sampler must include every child and grandchild process."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="Platform memory sampler")
def test_sampler_includes_all_descendants():
    path = Path(__file__).resolve().parents[1] / "scripts/00_benchmarks/compare_execution.py"
    spec = importlib.util.spec_from_file_location("benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    child_code = """
import os, subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; data=bytearray(8*1024**2); time.sleep(5)'])
data = bytearray(16*1024**2)
print(os.getpid(), grandchild.pid, flush=True)
try:
    time.sleep(5)
finally:
    grandchild.terminate()
    grandchild.wait()
"""
    children, expected = [], set()
    try:
        for _ in range(3):
            child = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
            children.append(child)
            expected.update(map(int, child.stdout.readline().split()))
        assert len(expected) == 6
        time.sleep(.1)
        rss, accounted, processes = benchmark.process_memory(os.getpid(), details=True)
        assert expected.issubset(processes)
        assert all(processes[pid][0] > 0 and processes[pid][1] > 0 for pid in expected)
        assert rss >= 3 * 24 * 1024**2
        assert accounted > 0
    finally:
        # Let each child reap its grandchild before finishing.
        for child in children:
            child.wait(timeout=10)

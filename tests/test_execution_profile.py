"""Opt-in profiling records attribution without changing returned results."""

import json
import sys
import tracemalloc

import numpy as np
import pytest

from stemflow.calibration import calibration_scores
from stemflow.utils.execution_profile import profile_execution


def test_disabled_profile_is_transparent(monkeypatch, tmp_path):
    monkeypatch.delenv("STEMFLOW_PROFILE_DIR", raising=False)
    assert profile_execution(lambda value: value + 1)(2) == 3
    assert not list(tmp_path.iterdir())


def test_function_line_and_allocation_records(monkeypatch, tmp_path):
    monkeypatch.setenv("STEMFLOW_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("STEMFLOW_PROFILE_LINES", "1")
    monkeypatch.setenv("STEMFLOW_PROFILE_ALLOCATIONS", "1")
    previous = sys.gettrace()
    score, _ = profile_execution(calibration_scores)(np.arange(100) % 2, np.full(100, .5))
    assert score["Brier"] == .25
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["error"] is None
    assert record["wall_seconds"] > 0
    assert record["functions"] and record["line_timings"]
    assert record["python_peak_traced_bytes"] > 0
    assert record["python_line_net_allocations"]
    assert sys.gettrace() is previous
    assert not tracemalloc.is_tracing()


def test_failed_call_records_error_and_restores_profiler(monkeypatch, tmp_path):
    monkeypatch.setenv("STEMFLOW_PROFILE_DIR", str(tmp_path))
    @profile_execution
    def failing():
        raise ValueError("original failure")
    with pytest.raises(ValueError, match="original failure"):
        failing()
    assert profile_execution(lambda: 42)() == 42
    records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert sorted(str(r["error"]) for r in records) == ["None", "ValueError"]


def test_profile_output_failure_preserves_call_result(monkeypatch, tmp_path):
    destination = tmp_path / "not_a_directory"
    destination.write_text("existing file")
    monkeypatch.setenv("STEMFLOW_PROFILE_DIR", str(destination))
    with pytest.warns(UserWarning, match="Could not save"):
        assert profile_execution(lambda: 42)() == 42

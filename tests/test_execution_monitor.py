"""Portable sampling, failure isolation and unchanged AdaSTEM predictions."""

import json
import os
import pickle
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

psutil = pytest.importorskip("psutil")

import stemflow.utils.execution_monitor as monitoring
from stemflow.utils.execution_monitor import ExecutionMonitor, _ProcessMemoryReader, current_monitor
from tests.test_execution_rewrite import make_model, sample, sources


def fake_sample(parent=10, children=20):
    return dict(tree_rss_bytes=parent + children, parent_rss_bytes=parent,
                children_rss_bytes=children, tree_uss_bytes=None, tree_pss_bytes=None,
                tree_physical_footprint_bytes=None, process_count=2,
                rss_processes=2, uss_processes=0, pss_processes=0,
                physical_footprint_processes=0, children_enumerated=True, rss_complete=True)


def test_inclusive_stages_and_simultaneous_peak(monkeypatch, tmp_path):
    state = dict(parent=10, children=100)
    monkeypatch.setattr(_ProcessMemoryReader, "snapshot", lambda *_: fake_sample(**state))
    output = tmp_path / "monitor.json"
    with ExecutionMonitor(sample_interval=60, log_interval=None, output_path=output) as monitor:
        with monitor.phase("fit"):
            with monitor.phase("training"):
                state.update(parent=100, children=10)
        running = monitor.summary()
        assert running["running"]
    report = monitor.summary()
    assert report["peak_tree_rss_bytes"] == 110  # Not the sum of individual peaks (200).
    assert report["peak_parent_rss_bytes"] == report["peak_children_rss_bytes"] == 100
    outer, inner = report["stages"]
    assert inner["parent_id"] == outer["id"]
    assert report["wall_seconds"] >= outer["wall_seconds"] >= inner["wall_seconds"] > 0
    assert inner["peak_tree_rss_bytes"] == outer["peak_tree_rss_bytes"] == 110
    assert report["recent_samples"] == []
    assert not report["running"] and report["error"] is None
    assert json.loads(output.read_text()) == report
    assert not list(tmp_path.glob("*.partial"))
    assert current_monitor() is None and not monitor._thread.is_alive()


def test_live_updates_and_bounded_trace(tmp_path, capsys):
    output = tmp_path / "monitor.json"
    with ExecutionMonitor(sample_interval=.01, log_interval=.02, max_samples=3,
                          output_path=output) as monitor:
        with monitor.phase("evaluation"):
            deadline = time.monotonic() + 5
            while monitor.summary()["samples"] < 6 and time.monotonic() < deadline:
                time.sleep(.01)
            live = json.loads(output.read_text())
            assert live["running"]
    report = monitor.summary()
    assert report["samples"] >= 6
    assert len(report["recent_samples"]) == 3
    assert all(s["tree_rss_bytes"] > 0 for s in report["recent_samples"])
    assert "[stemflow] running" in capsys.readouterr().out
    assert not json.loads(output.read_text())["running"]
    assert report["sampling_seconds"] > 0


@pytest.mark.parametrize("kwargs", [dict(sample_interval=0), dict(sample_interval=float("inf")),
    dict(sample_interval=True), dict(log_interval=-1), dict(max_samples=-1),
    dict(max_samples=True), dict(detailed_memory=1)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ExecutionMonitor(**kwargs)


def test_context_guards_and_thread_isolation():
    monitor = ExecutionMonitor(log_interval=None)
    with pytest.raises(RuntimeError, match="Enter the monitor"):
        monitor.summary()
    with monitor:
        with pytest.raises(RuntimeError, match="nested"):
            with ExecutionMonitor(log_interval=None):
                pass
        seen = []
        thread = threading.Thread(target=lambda: seen.append(current_monitor()))
        thread.start()
        thread.join()
        assert seen == [None]
        with pytest.raises(ValueError):
            with monitor.phase(""):
                pass
    with pytest.raises(RuntimeError, match="once"):
        with monitor:
            pass
    with pytest.raises(RuntimeError, match="inside"):
        with monitor.phase("too_late"):
            pass


def test_original_failure_and_failed_write_are_preserved(tmp_path):
    path = tmp_path / "directory_is_a_file"
    path.write_text("keep")
    with pytest.warns(RuntimeWarning, match="Could not save"):
        with pytest.raises(ValueError, match="model failed"):
            with ExecutionMonitor(log_interval=None, output_path=path / "monitor.json") as monitor:
                with monitor.phase("training"):
                    raise ValueError("model failed")
    report = monitor.summary()
    assert report["error"] == report["stages"][0]["error"] == "ValueError"
    assert report["output_error"] is not None
    assert not monitor._thread.is_alive() and current_monitor() is None
    assert path.read_text() == "keep"


def test_read_failure_is_not_zero_memory(monkeypatch):
    def failing(*_):
        raise OSError("cannot read process")
    monkeypatch.setattr(_ProcessMemoryReader, "snapshot", failing)
    with ExecutionMonitor(log_interval=None) as monitor:
        with monitor.phase("training"):
            result = 42
    report = monitor.summary()
    assert result == 42
    assert report["peak_tree_rss_bytes"] is None and report["latest_sample"] is None
    assert "cannot read process" in report["sampling_error"]


@pytest.mark.parametrize("api,fields", [
    ("old", {"uss": 4, "pss": 7}),  # Linux: USS and PSS.
    ("old", {"uss": 4}),  # macOS/Windows: no PSS.
    ("new", {"uss": 4, "pss": 7}),  # psutil 8 API.
    ("new", {}),  # A platform without extended memory accounting.
])
def test_optional_memory_capabilities(api, fields):
    process = SimpleNamespace(pid=123, children=lambda **_: [],
                              memory_info=lambda: SimpleNamespace(rss=10))
    setattr(process, "memory_full_info" if api == "old" else "memory_footprint",
            lambda: SimpleNamespace(**fields))
    reader = _ProcessMemoryReader(detailed_memory=True)
    reader.darwin = None
    reader.psutil = SimpleNamespace(Process=lambda _: process, Error=psutil.Error)
    report = reader.snapshot(123)
    assert report["tree_rss_bytes"] == 10 and report["rss_complete"]
    assert report["tree_uss_bytes"] == fields.get("uss")
    assert report["tree_pss_bytes"] == fields.get("pss")
    assert report["children_rss_bytes"] == 0


@pytest.mark.parametrize("failure", [psutil.AccessDenied, psutil.NoSuchProcess])
def test_disappearing_or_inaccessible_worker(failure):
    def failing():
        raise failure(456)
    child = SimpleNamespace(pid=456, memory_info=failing, memory_full_info=failing)
    root = SimpleNamespace(pid=123, children=lambda **_: [child],
                           memory_info=lambda: SimpleNamespace(rss=10), memory_full_info=failing)
    reader = _ProcessMemoryReader(detailed_memory=True)
    reader.darwin = None
    reader.psutil = SimpleNamespace(Process=lambda _: root, Error=psutil.Error)
    report = reader.snapshot(123)
    assert report["tree_rss_bytes"] == 10 and report["rss_processes"] == 1
    assert not report["rss_complete"] and report["children_rss_bytes"] is None
    assert report["tree_uss_bytes"] is report["tree_pss_bytes"] is None


def test_denied_enumeration_marks_unknown_children():
    def denied(**_):
        raise PermissionError("process listing denied")
    root = SimpleNamespace(pid=123, children=denied,
                           memory_info=lambda: SimpleNamespace(rss=10))
    reader = _ProcessMemoryReader()
    reader.darwin = None
    reader.psutil = SimpleNamespace(Process=lambda _: root, Error=psutil.Error)
    report = reader.snapshot(123)
    assert report["parent_rss_bytes"] == 10
    assert not report["children_enumerated"] and not report["rss_complete"]
    assert report["children_rss_bytes"] is None


def test_real_child_and_grandchild_native_memory():
    grandchild = """
import mmap, sys
size = 48 * 2**20
payload = mmap.mmap(-1, size)
for offset in range(0, size, 4096):
    payload[offset] = 1
print('ready', flush=True)
sys.stdin.readline()
"""
    child = """
import mmap, subprocess, sys
process = subprocess.Popen([sys.executable, '-c', sys.argv[1]],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
try:
    assert process.stdout.readline().strip() == 'ready'
    size = 32 * 2**20
    payload = mmap.mmap(-1, size)
    for offset in range(0, size, 4096):
        payload[offset] = 1
    print(process.pid, flush=True)
    sys.stdin.readline()
finally:
    process.communicate('exit\\n', timeout=10)
"""
    process = subprocess.Popen([sys.executable, "-c", child, grandchild],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        grandchild_pid = int(process.stdout.readline().strip())
        assert grandchild_pid > 0
        with ExecutionMonitor(log_interval=None, detailed_memory=True) as monitor:
            with monitor.phase("known_child_allocation"):
                report = monitor.summary()
                current = report["latest_sample"]
                assert {process.pid, grandchild_pid} <= set(current["process_ids"])
                assert current["children_rss_bytes"] >= 80 * 2**20
                assert current["tree_rss_bytes"] == current["parent_rss_bytes"] + current["children_rss_bytes"]
                assert current["process_count"] >= 3
        assert monitor.summary()["stages"][0]["peak_children_rss_bytes"] >= 80 * 2**20
    finally:
        process.communicate("exit\n", timeout=15)
    assert process.returncode == 0


@pytest.mark.parametrize("kind,backend,jobs,streaming", [
    ("pandas", "native", 1, True), ("duckdb", "native", 2, True),
    ("duckdb", "memmap", 2, False), ("parquet", "memmap", 2, True),
])
def test_adastem_monitor_preserves_results(sample, tmp_path, kind, backend, jobs, streaming):
    X, y = sample
    xs, ys = sources(X, y, kind, tmp_path)
    options = dict(n_jobs=jobs, ensemble_bootstrap=True, lazy_loading=True)
    plain = make_model(**options)
    plain.fit(xs, ys, temporal_window_prequery=True, training_data_backend=backend,
              persist_prediction_state=False)
    expected = plain.predict_proba(xs, return_std=True, temporal_window_prequery=True,
                                   streaming=streaming)
    observed = make_model(**options)
    with observed.monitor_execution(sample_interval=.05, log_interval=None) as monitor:
        observed.fit(xs, ys, temporal_window_prequery=True, training_data_backend=backend,
                     persist_prediction_state=False)
        fit_report = observed.execution_monitor_
        assert fit_report["running"]
        assert any(stage["name"] == "fit" for stage in fit_report["stages"])
        actual = observed.predict_proba(xs, return_std=True, temporal_window_prequery=True,
                                        streaming=streaming)
        assert len(observed.execution_monitor_["stages"]) > len(fit_report["stages"])
        serialized = pickle.dumps(observed)
    for a, b in zip(actual, expected):
        assert_allclose(a, b, atol=2e-15, equal_nan=True)
    assert_array_equal(plain.ensemble_df, observed.ensemble_df)
    report = monitor.summary()
    assert observed.execution_monitor_ == report
    assert not observed.execution_monitor_["running"]
    assert pickle.loads(pickle.dumps(observed)).execution_monitor_ == report
    stages = {stage["name"] for stage in report["stages"]}
    assert {"fit", "partitioning", "input_validation", "training_data_preparation",
            "training", "fit_cleanup", "prediction", "prediction_setup", "prediction_cleanup"} <= stages
    assert ("prediction_and_aggregation" if streaming else "prediction_aggregation") in stages
    assert report["peak_tree_rss_bytes"] > 0
    assert report["sampling_error"] is None
    if jobs > 1:
        assert report["peak_process_count"] >= 3
    assert b"stemflow-memory-monitor" not in serialized
    assert not hasattr(observed, "_shared_training_data")
    assert not os.path.exists(observed.joblib_tmp_dir)
    assert not monitor._thread.is_alive()


def test_report_survives_lazy_save_load(sample, tmp_path):
    X, y = sample
    model = make_model(lazy_loading=True, n_jobs=1)
    with model.monitor_execution(log_interval=None) as monitor:
        model.fit(X, y, temporal_window_prequery=True)
    report = monitor.summary()
    archive = tmp_path / "model.tar.gz"
    model.save(archive, remove_temporary_file=False, verbosity=0)
    loaded = type(model).load(archive, new_lazy_loading_path=str(tmp_path / "loaded"))
    assert loaded.execution_monitor_ == report
    assert_allclose(loaded.predict_proba(X), model.predict_proba(X), equal_nan=True)
    assert loaded.execution_monitor_ == report  # Unmonitored calls preserve the report.


def test_standalone_monitor_attaches_report_and_replaces_latest(sample):
    X, y = sample
    model = make_model(n_jobs=1)
    with ExecutionMonitor(log_interval=None) as fitting:
        model.fit(X, y, temporal_window_prequery=True)
    first = model.execution_monitor_
    assert first == fitting.summary()
    with model.monitor_execution(log_interval=None) as predicting:
        model.predict_proba(X)
    assert model.execution_monitor_ == predicting.summary()
    assert model.execution_monitor_ is not first
    assert not any(stage["name"] == "fit" for stage in model.execution_monitor_["stages"])
    model.execution_monitor_["stages"].clear()
    assert predicting.summary()["stages"]  # The saved report is an independent snapshot.


def test_failed_fit_retains_report(sample, monkeypatch):
    X, y = sample
    model = make_model(n_jobs=1)
    def failing(*args, **kwargs):
        raise ValueError("estimator failed")
    monkeypatch.setattr(model, "SAC_training", failing)
    with pytest.raises(ValueError, match="estimator failed"):
        with model.monitor_execution(log_interval=None):
            model.fit(X, y, temporal_window_prequery=True)
    report = model.execution_monitor_
    assert not report["running"] and report["error"] == "ValueError"
    assert any(stage["name"] == "training" and stage["error"] == "ValueError"
               for stage in report["stages"])


def test_disabled_hooks_do_not_require_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    @monitoring.monitor_step("unused")
    def example():
        with monitoring.execution_step("also_unused"):
            return 42
    assert example() == 42
    with pytest.raises(ImportError, match="monitoring"):
        with ExecutionMonitor():
            pass
    assert current_monitor() is None

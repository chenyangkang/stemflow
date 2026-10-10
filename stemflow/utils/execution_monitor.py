"""Opt-in wall timings and sampled host memory for a local process tree."""

import ctypes
import copy
import json
import math
import os
import platform
import threading
import time
import warnings
import weakref
from collections import deque
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path


_active_monitor = ContextVar("stemflow_execution_monitor", default=None)
_memory_fields = ("tree_rss_bytes", "parent_rss_bytes", "children_rss_bytes",
                  "tree_uss_bytes", "tree_pss_bytes", "tree_physical_footprint_bytes")


class _DarwinMemory:
    """Limited fallback when a sandbox blocks psutil's global process listing."""

    def __init__(self):
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.proc_listchildpids.restype = ctypes.c_int
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int

    def children(self, pid):
        size = 256
        while True:
            buffer = (ctypes.c_int * size)()
            ctypes.set_errno(0)
            count = self.lib.proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
            if count < 0 or (count == 0 and ctypes.get_errno()):
                raise OSError(ctypes.get_errno(), "Cannot enumerate child processes")
            # This API returns a PID count, rather than a byte count.
            if count < size:
                return list(buffer)[:count]
            size *= 2

    def memory(self, pid):
        # rusage_info_v2: a 16-byte UUID followed by eighteen uint64 fields.
        buffer = (ctypes.c_uint64 * 20)()
        if self.lib.proc_pid_rusage(pid, 2, ctypes.byref(buffer)) == 0:
            return int(buffer[8]), int(buffer[9])
        return None, None


class _ProcessMemoryReader:
    """Keep portable RSS separate from optional platform-specific accounting."""

    def __init__(self, detailed_memory=False):
        try:
            import psutil
        except ImportError as caught:
            raise ImportError('Memory monitoring requires psutil: pip install "stemflow[monitoring]"') from caught
        self.psutil = psutil
        self.detailed_memory = detailed_memory
        self.darwin = None
        if platform.system() == "Darwin":
            try:
                self.darwin = _DarwinMemory()
            except OSError:
                pass  # Portable RSS remains available without the fallback library.

    def snapshot(self, root_pid):
        psutil = self.psutil
        processes, tree_readable, backend = {}, True, "psutil"
        try:
            root = psutil.Process(root_pid)
            processes[root_pid] = root
            processes.update({p.pid: p for p in root.children(recursive=True)})
        except (psutil.Error, OSError):
            tree_readable, backend = False, "unavailable"
            if self.darwin is not None:
                tree_readable, backend = True, "darwin_libproc"
                pending, visited = [root_pid], set()
                while pending:
                    pid = pending.pop()
                    if pid <= 0 or pid in visited:
                        continue
                    visited.add(pid)
                    try:
                        processes[pid] = psutil.Process(pid)
                        pending.extend(self.darwin.children(pid))
                    except (psutil.Error, OSError):
                        processes.setdefault(pid, None)
                        tree_readable = False
        processes.setdefault(root_pid, None)
        records = {}
        for pid, process in processes.items():
            record = dict(rss=None, uss=None, pss=None, physical_footprint=None)
            if process is not None:
                try:
                    record["rss"] = process.memory_info().rss
                except (psutil.Error, OSError):
                    pass  # A process can exit or become inaccessible during a sample.
            if self.detailed_memory and process is not None:
                try:
                    # psutil 8 renamed this API; retain support for older releases.
                    method = getattr(process, "memory_footprint", None)
                    info = method() if method is not None else process.memory_full_info()
                    record["uss"], record["pss"] = getattr(info, "uss", None), getattr(info, "pss", None)
                except (psutil.Error, OSError, NotImplementedError):
                    pass
                method = getattr(process, "memory_extras", None)
                if method is not None and platform.system() == "Darwin":
                    try:
                        record["physical_footprint"] = getattr(method(), "phys_footprint", None)
                    except (psutil.Error, OSError, NotImplementedError):
                        pass
            if self.darwin is not None and (record["rss"] is None or self.detailed_memory):
                rss, footprint = self.darwin.memory(pid)
                if record["rss"] is None:
                    record["rss"] = rss
                if self.detailed_memory and record["physical_footprint"] is None:
                    record["physical_footprint"] = footprint
            records[pid] = record
        sample = dict(process_count=len(records), children_enumerated=tree_readable,
                      children_enumeration_backend=backend,
                      process_ids=sorted(records),
                      parent_rss_bytes=records[root_pid]["rss"])
        for name in ("rss", "uss", "pss", "physical_footprint"):
            values = [record[name] for record in records.values() if record[name] is not None]
            sample[f"tree_{name}_bytes"] = sum(values) if values else None
            sample[f"{name}_processes"] = len(values)
            sample[f"{name}_complete"] = tree_readable and len(values) == len(records)
        child_values = [record["rss"] for pid, record in records.items()
                        if pid != root_pid and record["rss"] is not None]
        sample["children_rss_bytes"] = (sum(child_values) if child_values
            else 0 if tree_readable and len(records) == 1 else None)
        sample["rss_complete"] = tree_readable and sample["rss_processes"] == len(records)
        return sample


class ExecutionMonitor:
    """Monitor fitting, prediction and custom stages in the calling process.

    Memory includes descendants on this node, including native estimator memory.
    RSS counts shared pages repeatedly. Optional USS, Linux PSS and macOS
    physical footprint are recorded separately when supported and accessible.
    These are not interchangeable or equal to a scheduler's memory charge. This is a
    sampler, not a memory limiter, allocation profiler or remote/GPU monitor.

    Args:
        sample_interval: Seconds between background memory samples (default 1).
        log_interval: Seconds between flushed stdout updates; None disables logs.
            Stage completions and workflow start/end are also logged when enabled.
        output_path: Optional compact JSON report. Updated alongside live logs
            and on exit, replacing one file rather than accumulating snapshots.
        detailed_memory: Collect USS/PSS and physical footprint when available.
            Default False keeps cheap RSS sampling; detailed reads can be slower.
        max_samples: Retain at most this many recent samples; default 0 stores
            summaries only. Stage records and peaks survive trace truncation.

    A context is single-use. Use stages from its calling thread. AdaSTEM models
    retain a plain report in execution_monitor_, updated after each outer model
    call and on context exit. The live sampler is never stored on the model.
    """

    def __init__(self, sample_interval=1.0, log_interval=30.0,
                 output_path=None, max_samples=0, detailed_memory=False):
        for name, value in [("sample_interval", sample_interval), ("log_interval", log_interval)]:
            if value is None and name == "log_interval":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number" + (" or None." if name == "log_interval" else "."))
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 0:
            raise ValueError("max_samples must be a nonnegative integer.")
        if not isinstance(detailed_memory, bool):
            raise ValueError("detailed_memory must be a bool.")
        self.detailed_memory = detailed_memory
        self.sample_interval = float(sample_interval)
        self.log_interval = log_interval
        self.output_path = Path(output_path) if output_path is not None else None
        self.max_samples = max_samples
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._trace = deque(maxlen=max_samples)
        self._stages, self._stack = [], []
        self._started = None
        self._ended = None
        self._error = None
        self._sampling_error = None
        self._output_error = None
        self._thread = None
        self._models = []
        self._model_call_depth = 0

    def _track_model(self, model):
        if not any(reference() is model for reference in self._models):
            self._models.append(weakref.ref(model))

    def _publish_reports(self):
        report = self.summary()
        for reference in self._models:
            model = reference()
            if model is not None:
                model.execution_monitor_ = copy.deepcopy(report)

    @staticmethod
    def _statistics():
        return dict(samples=0, partial_rss_samples=0, peak_process_count=0,
                    **{f"peak_{name}": None for name in _memory_fields})

    def __enter__(self):
        if self._started is not None:
            raise RuntimeError("An ExecutionMonitor context can only be entered once.")
        if current_monitor() is not None:
            raise RuntimeError("Use monitor.phase(...) for nested stages, rather than nested monitors.")
        self.pid, self._owner_thread = os.getpid(), threading.get_ident()
        self._reader = _ProcessMemoryReader(self.detailed_memory)
        self._sampling_seconds = 0.0
        self._sampling_attempts = 0
        self._max_sampling_seconds = 0.0
        self._max_sample_gap_seconds = 0.0
        self._statistics_total = self._statistics()
        self._started = time.perf_counter()
        self._token = _active_monitor.set(self)
        with self._lock:
            self._sample()
        self._thread = threading.Thread(target=self._run, name="stemflow-memory-monitor", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            _active_monitor.reset(self._token)
            raise
        self._log("started")
        return self

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._sample()
            self._ended = time.perf_counter()
            self._error = error_type.__name__ if error_type is not None else None
        _active_monitor.reset(self._token)
        self._log("finished" if error_type is None else f"failed ({self._error})")
        if self.log_interval is None:
            self._save_report(self.summary())
        self._publish_reports()
        return False

    def _save_report(self, report):
        if self.output_path is None or self._output_error is not None:
            return
        with self._lock:
            temporary = self.output_path.with_name(self.output_path.name + ".partial")
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                temporary.replace(self.output_path)
            except OSError as caught:
                self._output_error = f"{type(caught).__name__}: {caught}"
                self._warn(f"Could not save execution monitor report: {caught}")

    @staticmethod
    def _warn(message):
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        except Warning:
            pass  # A monitoring warning must not replace the model's exception.

    def _sample(self):
        started = time.perf_counter()
        self._sampling_attempts += 1
        try:
            sample = self._reader.snapshot(self.pid)
        except Exception as caught:
            self._sampling_error = f"{type(caught).__name__}: {caught}"
            return
        finally:
            duration = time.perf_counter() - started
            self._sampling_seconds += duration
            self._max_sampling_seconds = max(self._max_sampling_seconds, duration)
        sample["elapsed_seconds"] = time.perf_counter() - self._started
        if hasattr(self, "_latest"):
            self._max_sample_gap_seconds = max(self._max_sample_gap_seconds,
                sample["elapsed_seconds"] - self._latest["elapsed_seconds"])
        sample["stage"] = "/".join(stage["name"] for stage in self._stack) or "workflow"
        self._latest = sample
        self._trace.append(sample)
        for record in [self._statistics_total, *self._stack]:
            record["samples"] += 1
            record["partial_rss_samples"] += int(not sample["rss_complete"])
            record["peak_process_count"] = max(record["peak_process_count"], sample["process_count"])
            for name in _memory_fields:
                value, key = sample[name], f"peak_{name}"
                if value is not None and (record[key] is None or value > record[key]):
                    record[key] = value
                    if record is self._statistics_total and name == "tree_rss_bytes":
                        record["peak_rss_stage"] = sample["stage"]
                        record["peak_rss_elapsed_seconds"] = sample["elapsed_seconds"]

    def _run(self):
        next_log = time.perf_counter() + (self.log_interval or float("inf"))
        while not self._stop.wait(self.sample_interval):
            with self._lock:
                self._sample()
            now = time.perf_counter()
            if now >= next_log:
                self._log("running")
                next_log = now + self.log_interval

    def _log(self, message):
        if self.log_interval is None:
            return
        report = self.summary()
        self._save_report(report)
        sample = report["latest_sample"] or {}
        def mib(value):
            return "unavailable" if value is None else f"{value / 2**20:.1f} MiB"
        text = (f"[stemflow] {message}; elapsed={report['wall_seconds']:.2f}s; "
                f"stage={sample.get('stage', 'workflow')}; "
                f"RSS={mib(sample.get('tree_rss_bytes'))}; "
                f"RSS coverage={sample.get('rss_processes', 0)}/{sample.get('process_count', '?')}"
                f"{' (partial tree)' if not sample.get('rss_complete') else ''}; "
                f"peak RSS={mib(report['peak_tree_rss_bytes'])}; "
                f"parent={mib(sample.get('parent_rss_bytes'))}; "
                f"children={mib(sample.get('children_rss_bytes'))}")
        try:
            print(text, flush=True)
        except (OSError, ValueError):
            pass

    @contextmanager
    def phase(self, name):
        """Record an inclusive stage, including nested calls and worker memory."""
        if current_monitor() is not self or self._ended is not None:
            raise RuntimeError("Stages must run inside this monitor in its calling thread/process.")
        if not isinstance(name, str) or not name:
            raise ValueError("A stage name must be a nonempty string.")
        with self._lock:
            stage = dict(id=len(self._stages), name=name,
                         parent_id=self._stack[-1]["id"] if self._stack else None,
                         start_seconds=time.perf_counter() - self._started,
                         wall_seconds=None, error=None, **self._statistics())
            self._stages.append(stage)
            self._stack.append(stage)
            self._sample()
        try:
            yield self
        except BaseException as caught:
            stage["error"] = type(caught).__name__
            raise
        finally:
            with self._lock:
                self._sample()
                stage["wall_seconds"] = time.perf_counter() - self._started - stage["start_seconds"]
                self._stack.pop()
            self._log(f"completed {name} ({stage['wall_seconds']:.2f}s)" if stage["error"] is None
                      else f"failed {name} ({stage['error']})")

    def summary(self):
        """Return a JSON-compatible snapshot, also available during execution."""
        if self._started is None:
            raise RuntimeError("Enter the monitor context before requesting its summary.")
        with self._lock:
            now = self._ended or time.perf_counter()
            stages = [dict(stage) for stage in self._stages]
            for stage in stages:
                if stage["wall_seconds"] is None:
                    stage["wall_seconds"] = now - self._started - stage["start_seconds"]
            return dict(pid=self.pid, running=self._ended is None, error=self._error,
                        wall_seconds=now - self._started, sample_interval=self.sample_interval,
                        platform=platform.platform(), psutil_version=self._reader.psutil.__version__,
                        detailed_memory=self.detailed_memory,
                        sampling_attempts=self._sampling_attempts,
                        sampling_seconds=self._sampling_seconds,
                        max_sampling_seconds=self._max_sampling_seconds,
                        max_sample_gap_seconds=self._max_sample_gap_seconds,
                        sampling_error=self._sampling_error, output_error=self._output_error,
                        **self._statistics_total,
                        stages=stages, latest_sample=dict(self._latest) if hasattr(self, "_latest") else None,
                        recent_samples=[dict(sample) for sample in self._trace])


def current_monitor():
    """Workers must not inherit the parent's sampler or stage stack."""
    monitor = _active_monitor.get()
    if (monitor is not None and monitor.pid == os.getpid()
            and monitor._owner_thread == threading.get_ident()):
        return monitor
    return None


def execution_step(name):
    monitor = current_monitor()
    return monitor.phase(name) if monitor is not None else nullcontext()


def monitor_step(name):
    """Transparent method hook when no monitor is active."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            monitor = current_monitor()
            if monitor is None:
                return function(*args, **kwargs)
            if args and hasattr(args[0], "monitor_execution"):
                monitor._track_model(args[0])
            monitor._model_call_depth += 1
            try:
                with monitor.phase(name):
                    return function(*args, **kwargs)
            finally:
                monitor._model_call_depth -= 1
                if monitor._model_call_depth == 0:
                    monitor._publish_reports()
        return wrapped
    return decorate

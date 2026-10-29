"""Opt-in function timing and Python allocation attribution for execution audits."""

import cProfile
from functools import wraps
import itertools
import json
import os
from pathlib import Path
import pstats
import sys
import time
import tracemalloc
import warnings


_active = False
_sequence = itertools.count()


def profile_execution(function):
    """Enable with STEMFLOW_PROFILE_DIR; allocation snapshots are separately opt-in.

    cProfile attributes function calls to source lines. Allocation snapshots
    report retained Python allocations by line, not native estimator RAM or
    per-line peak RSS. Use a process-tree sampler for the whole-job peak.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _active
        destination = os.environ.get("STEMFLOW_PROFILE_DIR")
        if not destination or _active:
            return function(*args, **kwargs)
        _active = True
        allocation = os.environ.get("STEMFLOW_PROFILE_ALLOCATIONS") == "1"
        owns_trace = allocation and not tracemalloc.is_tracing()
        if owns_trace:
            tracemalloc.start(10)
        before = tracemalloc.take_snapshot() if allocation else None
        profiler, started = cProfile.Profile(), time.perf_counter()
        line_records, frames = {}, {}
        previous_trace = sys.gettrace()
        line_timing = os.environ.get("STEMFLOW_PROFILE_LINES") == "1"
        package = str(Path(__file__).resolve().parents[1]) + os.sep
        def trace(frame, event, arg):
            filename = frame.f_code.co_filename
            if not filename.startswith(package) or filename == __file__:
                return None
            now = time.perf_counter()
            last = frames.get(id(frame))
            if last is not None and event in {"line", "return"}:
                key = (filename, last[0])
                record = line_records.setdefault(key, [0, 0.])
                record[0], record[1] = record[0]+1, record[1]+now-last[1]
            if event == "return":
                frames.pop(id(frame), None)
            elif event in {"call", "line"}:
                frames[id(frame)] = (frame.f_lineno, now)
            return trace
        if line_timing:
            sys.settrace(trace)
        error = None
        profiler.enable()
        try:
            return function(*args, **kwargs)
        except BaseException as caught:
            error = type(caught).__name__
            raise
        finally:
            profiler.disable()
            if line_timing:
                sys.settrace(previous_trace)
            rows = []
            for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
                rows.append(dict(file=filename, line=line, function=name, calls=calls,
                                 own_seconds=own, cumulative_seconds=cumulative))
            report = dict(function=function.__qualname__, pid=os.getpid(), error=error,
                wall_seconds=time.perf_counter()-started,
                functions=sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:60])
            if line_timing:
                report["line_timings"] = sorted([dict(file=filename, line=line, visits=record[0],
                    elapsed_seconds_including_calls=record[1]) for (filename, line), record in line_records.items()],
                    key=lambda row: row["elapsed_seconds_including_calls"], reverse=True)[:80]
            if allocation:
                after = tracemalloc.take_snapshot()
                report["python_peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
                report["python_line_net_allocations"] = [dict(file=str(item.traceback[0]),
                    bytes=item.size_diff, count=item.count_diff)
                    for item in after.compare_to(before, "lineno")[:40]]
            if owns_trace:
                tracemalloc.stop()
            _active = False
            try:
                directory = Path(destination)
                directory.mkdir(parents=True, exist_ok=True)
                name = f"{function.__name__}_{os.getpid()}_{next(_sequence)}.json"
                (directory / name).write_text(json.dumps(report, indent=2))
            except OSError as caught:
                warnings.warn(f"Could not save execution profile: {caught}")
    return wrapped

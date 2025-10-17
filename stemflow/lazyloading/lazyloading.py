from __future__ import annotations
from pathlib import Path
import time
import tempfile
import shutil
from typing import Optional, Any, Callable
import threading
import joblib
from contextlib import contextmanager
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class LazyLoadingEstimator(BaseEstimator, MetaEstimatorMixin):
    """
    A thin wrapper around any scikit-learn estimator that supports
    dumping and loading from disk, with optional lazy auto-load
    and auto-dump behavior.

    Args:
        estimator (BaseEstimator | None): The sklearn estimator
            (e.g., RandomForestClassifier). Can be None if the model
            will be loaded from disk before first use.
        dump_dir (str | Path | None): Directory where model dumps are
            saved and loaded.
        filename (str | None): Base filename for dumps ('.pkl' auto-added).
            If None, a name is generated automatically.
        compress (Any): Compression option passed to joblib.dump.
            Default is 3.
        auto_load (bool): Whether to automatically load from disk when
            estimator is None. Default is True.
        auto_dump (bool): Whether to automatically dump after each call
            and clear estimator from memory. Default is False.
        keep_loaded (bool): Whether to keep estimator in memory after
            calls even if auto_dump is True. Default is False.

    Returns:
        LazyLoadingEstimator: The initialized wrapper instance.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator],
        dump_dir: Optional[Path | str] = None,
        filename: Optional[str] = None,
        compress: Any = 0,
        auto_load: bool = True,
        auto_dump: bool = False,
        keep_loaded: bool = False,
    ):
        self.estimator = estimator
        self.dump_dir = None if dump_dir is None else Path(dump_dir)
        self.filename = filename
        self.compress = compress
        self.auto_load = auto_load
        self.auto_dump = auto_dump
        self.keep_loaded = keep_loaded

        # intra-process safety (not cross-process)
        self._lock = threading.RLock()

    # ---------- Core sklearn API (delegate with lazy load / optional dump) ----------
    def fit(self, X, y=None, **fit_params):
        with self._loaded_estimator(write_intent=True) as est:
            est.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        with self._loaded_estimator() as est:
            return est.predict(X)

    def score(self, X, y=None):
        with self._loaded_estimator() as est:
            if hasattr(est, "score"):
                return est.score(X, y)
            raise AttributeError(f"{type(est).__name__} has no .score")

    def predict_proba(self, X):
        with self._loaded_estimator() as est:
            if hasattr(est, "predict_proba"):
                return est.predict_proba(X)
            raise AttributeError(f"{type(est).__name__} has no .predict_proba")

    def decision_function(self, X):
        with self._loaded_estimator() as est:
            if hasattr(est, "decision_function"):
                return est.decision_function(X)
            raise AttributeError(f"{type(est).__name__} has no .decision_function")

    # Make other attributes/methods transparently available (autoload if needed)
    def __getattr__(self, name):
        # Only called if attribute not found on self
        # Try autoloading and then delegate
        if name.startswith("__"):  # avoid dunder recursion
            raise AttributeError(name)
        with self._lock:
            if self.estimator is None and self.auto_load:
                self._load_inplace()
            if self.estimator is not None and hasattr(self.estimator, name):
                return getattr(self.estimator, name)
        # Fallback to default behavior
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    # ---------- Persistence helpers ----------
    def _resolve_path(self) -> Path:
        if self.dump_dir is None:
            raise ValueError("dump_dir is not set. Set wrapper.dump_dir or pass it in __init__.")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        fname = self.filename
        if not fname:
            ts = time.strftime("%Y%m%d-%H%M%S")
            base = type(self.estimator).__name__ if self.estimator is not None else "Estimator"
            fname = f"{base}-{ts}.pkl"
            self.filename = fname
        if not fname.endswith(".pkl"):
            fname += ".pkl"
        return self.dump_dir / fname

    def dump(self) -> Path:
        """
        Dump ONLY the inner estimator to disk (not the wrapper).
        Atomic write: write to temp file then replace.
        After dump, sets self.estimator=None (frees RAM).
        """
        with self._lock:
            if self.estimator is None:
                # Nothing to dump; ensure file exists?
                raise ValueError("No estimator in memory to dump (self.estimator is None).")
            path = self._resolve_path()
            tmp_dir = Path(tempfile.mkdtemp(dir=self.dump_dir))
            tmp_path = tmp_dir / (path.name + ".tmp")
            try:
                joblib.dump(self.estimator, tmp_path, compress=self.compress)
                # Atomic on same filesystem
                shutil.move(str(tmp_path), str(path))
                # Free memory
                self.estimator = None

            finally:
                # Best-effort cleanup
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    tmp_dir.rmdir()
                except Exception:
                    pass
            return path


    def load(self, path: Optional[Path | str] = None) -> "LazyLoadingEstimator":
        """
        Load the inner estimator from disk into this wrapper (in-place).
        """
        with self._lock:
            path = Path(path) if path is not None else self._resolve_path()
            self.estimator = joblib.load(path)
        return self

    @classmethod
    def load_from_dir(
        cls,
        dump_dir: Path | str,
        filename: Optional[str] = None,
        compress: Any = 0,
        **kwargs,
    ) -> "LazyLoadingEstimator":
        dump_dir = Path(dump_dir)
        if filename is None:
            pkls = sorted(dump_dir.glob("*.pkl"))
            if not pkls:
                raise FileNotFoundError(f"No .pkl files in {dump_dir}")
            filename = pkls[-1].name
        wrapper = cls(
            estimator=None,
            dump_dir=dump_dir,
            filename=filename,
            compress=compress,
            **kwargs,
        )
        return wrapper.load(dump_dir / filename)

    # ---------- sklearn compatibility niceties ----------
    def _more_tags(self):
        tags = {}
        with self._lock:
            if self.estimator is not None and hasattr(self.estimator, "_get_tags"):
                tags.update(self.estimator._get_tags())
        return tags

    # ---------- internals ----------
    def _load_inplace(self):
        # helper that assumes lock is held
        path = self._resolve_path()
        if not path.exists():
            raise FileNotFoundError(f"Expected to auto-load, but no model file at {path}")
        self.estimator = joblib.load(path)

    @contextmanager
    def _loaded_estimator(self, write_intent: bool = False):
        """
        Context manager that:
          1) auto-loads estimator if needed,
          2) yields it,
          3) auto-dumps after use if configured.
        """
        with self._lock:
            if self.estimator is None and self.auto_load:
                self._load_inplace()
            if self.estimator is None:
                raise ValueError(
                    "Estimator is not loaded. Set auto_load=True or call .load() first."
                )
            est = self.estimator

        try:
            yield est
        finally:
            # Auto-dump logic: dump after operations that likely changed state OR if user requests aggressive RAM saving
            # We dump after .fit (write_intent=True), or if auto_dump=True (always), unless keep_loaded=True.
            if (write_intent or self.auto_dump) and not self.keep_loaded:
                try:
                    self.dump()
                except Exception as e:
                    # Don't hide exceptions during user calls; re-raise to surface issues.
                    raise

    def __getstate__(self):
        """
        Exclude unpicklable runtime-only attributes from pickling.
        Locks (and similar OS handles) cannot be pickled.
        """
        state = self.__dict__.copy()
        # Drop the RLock so pickle/joblib won’t choke on it
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        """
        Rebuild runtime-only attributes after unpickling.
        """
        self.__dict__.update(state)
        # Recreate a fresh lock for this process
        self._lock = threading.RLock()
        
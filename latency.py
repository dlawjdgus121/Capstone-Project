"""
Lightweight CPU/wall-time profiler for latency spikes.

Usage examples:

    from latency import LatencyProfiler

    profiler = LatencyProfiler(include_modules={"vision", "vlm", "routes"})
    profiler.start()

    # Later, print the slowest functions:
    profiler.print_report(limit=30)

    # Or wrap a single function:
    from latency import profile_function

    heavy_processing = profile_function(heavy_processing)

The profiler records per-function call count, CPU time, wall time, and average
duration. It uses only Python's standard library, so it can be enabled on a
running development machine without installing extra packages.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType, ModuleType
from typing import Any, Callable, Iterable


CpuClock = Callable[[], float]
_ACTIVE_PROFILER: "LatencyProfiler | None" = None


def _cpu_clock() -> CpuClock:
    return getattr(time, "thread_time", time.process_time)


def _module_name(frame: FrameType) -> str:
    return frame.f_globals.get("__name__", "<unknown>")


def _function_key(frame: FrameType) -> str:
    module = _module_name(frame)
    code = frame.f_code
    return f"{module}.{code.co_name} ({os.path.basename(code.co_filename)}:{code.co_firstlineno})"


@dataclass
class FunctionStats:
    calls: int = 0
    wall_total_ms: float = 0.0
    cpu_total_ms: float = 0.0
    wall_self_ms: float = 0.0
    cpu_self_ms: float = 0.0
    wall_max_ms: float = 0.0
    cpu_max_ms: float = 0.0

    def add(
        self,
        wall_ms: float,
        cpu_ms: float,
        wall_child_ms: float,
        cpu_child_ms: float,
    ) -> None:
        self.calls += 1
        self.wall_total_ms += wall_ms
        self.cpu_total_ms += cpu_ms
        self.wall_self_ms += max(0.0, wall_ms - wall_child_ms)
        self.cpu_self_ms += max(0.0, cpu_ms - cpu_child_ms)
        self.wall_max_ms = max(self.wall_max_ms, wall_ms)
        self.cpu_max_ms = max(self.cpu_max_ms, cpu_ms)

    @property
    def avg_wall_ms(self) -> float:
        return self.wall_total_ms / self.calls if self.calls else 0.0

    @property
    def avg_cpu_ms(self) -> float:
        return self.cpu_total_ms / self.calls if self.calls else 0.0


@dataclass
class _StackEntry:
    key: str
    wall_start: float
    cpu_start: float
    wall_child_ms: float = 0.0
    cpu_child_ms: float = 0.0


class LatencyProfiler:
    """Collect function/module CPU usage with minimal setup.

    Args:
        include_modules: Optional module prefixes to record, e.g. {"vision"}.
            When omitted, every non-stdlib module is recorded.
        exclude_modules: Module prefixes to skip.
        min_cpu_ms: Functions below this total CPU time are hidden in reports.
        print_at_exit: Print a summary automatically when the process exits.
    """

    def __init__(
        self,
        include_modules: Iterable[str] | None = None,
        exclude_modules: Iterable[str] | None = None,
        min_cpu_ms: float = 0.0,
        print_at_exit: bool = True,
        report_interval_s: float = 0.0,
        reset_each_report: bool = True,
    ) -> None:
        self.include_modules = tuple(include_modules or ())
        self.exclude_modules = tuple(exclude_modules or ("latency",))
        self.min_cpu_ms = float(min_cpu_ms)
        self.print_at_exit = print_at_exit
        self.report_interval_s = max(0.0, float(report_interval_s))
        self.reset_each_report = bool(reset_each_report)
        self.stats: dict[str, FunctionStats] = defaultdict(FunctionStats)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._cpu_clock = _cpu_clock()
        self._enabled = False
        self._started_at_wall = 0.0
        self._started_at_cpu = 0.0
        self._reporter_thread: threading.Thread | None = None
        self._reporter_stop = threading.Event()

    def start(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._started_at_wall = time.perf_counter()
        self._started_at_cpu = self._cpu_clock()
        threading.setprofile(self._profile)
        sys.setprofile(self._profile)
        if self.print_at_exit:
            atexit.register(self.print_report)
        if self.report_interval_s > 0:
            self._start_reporter()

    def stop(self) -> None:
        if not self._enabled:
            return
        self._reporter_stop.set()
        sys.setprofile(None)
        threading.setprofile(None)
        self._enabled = False

    def reset(self) -> None:
        with self._lock:
            self.stats.clear()
        self._started_at_wall = time.perf_counter()
        self._started_at_cpu = self._cpu_clock()

    def record_scope(self, name: str, wall_ms: float, cpu_ms: float) -> None:
        key = f"scope.{name}"
        with self._lock:
            self.stats[key].add(
                wall_ms=wall_ms,
                cpu_ms=cpu_ms,
                wall_child_ms=0.0,
                cpu_child_ms=0.0,
            )

    def snapshot(self) -> list[tuple[str, FunctionStats]]:
        with self._lock:
            rows = [
                (key, FunctionStats(**vars(stat)))
                for key, stat in self.stats.items()
                if stat.cpu_total_ms >= self.min_cpu_ms
            ]
        rows.sort(key=lambda item: item[1].cpu_self_ms, reverse=True)
        return rows

    def module_snapshot(self) -> list[tuple[str, FunctionStats]]:
        merged: dict[str, FunctionStats] = defaultdict(FunctionStats)
        for key, stat in self.snapshot():
            module = key.split(" ", 1)[0].rsplit(".", 1)[0]
            merged[module].calls += stat.calls
            merged[module].wall_total_ms += stat.wall_total_ms
            merged[module].cpu_total_ms += stat.cpu_total_ms
            merged[module].wall_self_ms += stat.wall_self_ms
            merged[module].cpu_self_ms += stat.cpu_self_ms
            merged[module].wall_max_ms = max(merged[module].wall_max_ms, stat.wall_max_ms)
            merged[module].cpu_max_ms = max(merged[module].cpu_max_ms, stat.cpu_max_ms)
        rows = sorted(merged.items(), key=lambda item: item[1].cpu_self_ms, reverse=True)
        return rows

    def scope_snapshot(self) -> list[tuple[str, FunctionStats]]:
        rows = [(key, stat) for key, stat in self.snapshot() if key.startswith("scope.")]
        rows.sort(key=lambda item: item[1].cpu_self_ms, reverse=True)
        return rows

    def print_report(self, limit: int = 25) -> None:
        elapsed_wall = max(0.001, time.perf_counter() - self._started_at_wall)
        elapsed_cpu = max(0.0, self._cpu_clock() - self._started_at_cpu)
        print(
            "\n[LATENCY] "
            f"elapsed={elapsed_wall:.2f}s cpu={elapsed_cpu:.2f}s "
            f"functions={len(self.stats)}"
        )
        self._print_rows("MODULE CPU", self.module_snapshot(), limit=min(limit, 15))
        self._print_rows("SCOPE CPU", self.scope_snapshot(), limit=limit)
        self._print_rows("FUNCTION CPU", self.snapshot(), limit=limit)

    def _start_reporter(self) -> None:
        if self._reporter_thread and self._reporter_thread.is_alive():
            return
        self._reporter_stop.clear()
        self._reporter_thread = threading.Thread(
            target=self._report_loop,
            name="latency-profiler-reporter",
            daemon=True,
        )
        self._reporter_thread.start()

    def _report_loop(self) -> None:
        while not self._reporter_stop.wait(self.report_interval_s):
            self.print_report()
            if self.reset_each_report:
                self.reset()

    def _print_rows(
        self,
        title: str,
        rows: list[tuple[str, FunctionStats]],
        limit: int,
    ) -> None:
        print(f"[LATENCY] {title}")
        print(
            "  cpu_self  cpu_total  wall_self wall_total calls  avg_cpu  max_cpu  name"
        )
        for key, stat in rows[:limit]:
            print(
                f"  {stat.cpu_self_ms:8.2f} "
                f"{stat.cpu_total_ms:9.2f} "
                f"{stat.wall_self_ms:9.2f} "
                f"{stat.wall_total_ms:10.2f} "
                f"{stat.calls:5d} "
                f"{stat.avg_cpu_ms:8.2f} "
                f"{stat.cpu_max_ms:8.2f}  "
                f"{key}"
            )

    def _should_record(self, frame: FrameType) -> bool:
        module = _module_name(frame)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in self.exclude_modules):
            return False
        if self.include_modules:
            return any(module == prefix or module.startswith(f"{prefix}.") for prefix in self.include_modules)
        filename = frame.f_code.co_filename
        return "site-packages" not in filename and "Lib" not in filename

    def _stack(self) -> list[_StackEntry]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if not self._enabled:
            return
        if event == "call":
            if not self._should_record(frame):
                return
            self._stack().append(
                _StackEntry(
                    key=_function_key(frame),
                    wall_start=time.perf_counter(),
                    cpu_start=self._cpu_clock(),
                )
            )
            return
        if event not in ("return", "c_return", "exception", "c_exception"):
            return

        stack = self._stack()
        if not stack:
            return

        key = _function_key(frame)
        entry_index = None
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].key == key:
                entry_index = index
                break
        if entry_index is None:
            return

        entry = stack.pop(entry_index)
        wall_ms = (time.perf_counter() - entry.wall_start) * 1000.0
        cpu_ms = (self._cpu_clock() - entry.cpu_start) * 1000.0

        with self._lock:
            self.stats[entry.key].add(
                wall_ms=wall_ms,
                cpu_ms=cpu_ms,
                wall_child_ms=entry.wall_child_ms,
                cpu_child_ms=entry.cpu_child_ms,
            )

        if stack:
            stack[-1].wall_child_ms += wall_ms
            stack[-1].cpu_child_ms += cpu_ms


def profile_function(func: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
    """Decorate one sync or async function and print per-call CPU/wall time."""

    label = name or f"{func.__module__}.{func.__qualname__}"
    cpu_clock = _cpu_clock()

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            wall_start = time.perf_counter()
            cpu_start = cpu_clock()
            try:
                return await func(*args, **kwargs)
            finally:
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                cpu_ms = (cpu_clock() - cpu_start) * 1000.0
                print(f"[LATENCY] {label} cpu={cpu_ms:.2f}ms wall={wall_ms:.2f}ms")

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wall_start = time.perf_counter()
        cpu_start = cpu_clock()
        try:
            return func(*args, **kwargs)
        finally:
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            cpu_ms = (cpu_clock() - cpu_start) * 1000.0
            print(f"[LATENCY] {label} cpu={cpu_ms:.2f}ms wall={wall_ms:.2f}ms")

    return wrapper


@contextmanager
def latency_scope(name: str):
    """Measure one code block and add it to the active profiler report."""

    if _ACTIVE_PROFILER is None:
        yield
        return

    cpu_clock = _cpu_clock()
    wall_start = time.perf_counter()
    cpu_start = cpu_clock()
    try:
        yield
    finally:
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cpu_ms = (cpu_clock() - cpu_start) * 1000.0
        if _ACTIVE_PROFILER is not None:
            _ACTIVE_PROFILER.record_scope(name, wall_ms, cpu_ms)


def profile_module_functions(
    module: ModuleType,
    *,
    include_private: bool = False,
    names: Iterable[str] | None = None,
) -> None:
    """Wrap functions defined in a module with per-call latency logging.

    This is useful for a quick local run:

        import vision
        from latency import profile_module_functions
        profile_module_functions(vision)
    """

    selected = set(names or ())
    for attr_name, value in list(vars(module).items()):
        if selected and attr_name not in selected:
            continue
        if not include_private and attr_name.startswith("_"):
            continue
        if not inspect.isfunction(value):
            continue
        if value.__module__ != module.__name__:
            continue
        setattr(module, attr_name, profile_function(value))


def start_from_env() -> LatencyProfiler | None:
    """Start profiler when LATENCY_PROFILE=1 is set.

    Optional env vars:
        LATENCY_MODULES=vision,vlm,routes
        LATENCY_MIN_CPU_MS=1
        LATENCY_REPORT_INTERVAL_S=5
        LATENCY_RESET_EACH_REPORT=1
    """

    if os.getenv("LATENCY_PROFILE", "0") != "1":
        return None
    modules = [
        item.strip()
        for item in os.getenv("LATENCY_MODULES", "vision,vlm,routes").split(",")
        if item.strip()
    ]
    try:
        min_cpu_ms = float(os.getenv("LATENCY_MIN_CPU_MS", "0"))
    except ValueError:
        min_cpu_ms = 0.0
    try:
        report_interval_s = float(os.getenv("LATENCY_REPORT_INTERVAL_S", "0"))
    except ValueError:
        report_interval_s = 0.0
    reset_each_report = os.getenv("LATENCY_RESET_EACH_REPORT", "1") != "0"
    global _ACTIVE_PROFILER

    profiler = LatencyProfiler(
        include_modules=modules,
        min_cpu_ms=min_cpu_ms,
        report_interval_s=report_interval_s,
        reset_each_report=reset_each_report,
    )
    _ACTIVE_PROFILER = profiler
    profiler.start()
    print(
        f"[LATENCY] profiler started modules={modules} "
        f"report_interval_s={report_interval_s}"
    )
    return profiler

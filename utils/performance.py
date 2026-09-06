"""Aggregate wall/stream timings without synchronizing every GPU operation."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import wraps
import json
from pathlib import Path
from time import perf_counter

import torch


def validate_performance_config(config: dict) -> dict:
    values = config.setdefault("performance", {})
    if not isinstance(values, dict):
        raise ValueError("performance must be a mapping.")
    for key in ("enabled", "cuda_events"):
        values.setdefault(key, True)
        if not isinstance(values[key], bool):
            raise ValueError(f"performance.{key} must be a boolean.")
    values.setdefault("evaluation_backend", "shared")
    if values["evaluation_backend"] not in {"shared", "clients"}:
        raise ValueError("performance.evaluation_backend must be shared or clients.")
    return values


class StageTimings:
    def __init__(self, device, *, enabled=True, cuda_events=True):
        self.device = torch.device(device)
        self.enabled = enabled
        self.cuda_events = enabled and cuda_events and self.device.type == "cuda"
        self.stages = {}
        self._pending = []

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start_event = end_event = None
        if self.cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            record = self.stages.setdefault(name, {
                "calls": 0, "wall_seconds": 0.0,
                "cuda_stream_seconds": 0.0 if self.cuda_events else None,
            })
            record["calls"] += 1
            record["wall_seconds"] += elapsed
            if end_event is not None:
                end_event.record(torch.cuda.current_stream(self.device))
                self._pending.append((name, start_event, end_event))
                if len(self._pending) >= 4096:
                    # Bound event storage during large candidate-pool audits.
                    self.flush()
                    if len(self._pending) >= 4096:
                        self.flush(wait=True)

    def flush(self, *, wait=False):
        if wait and self._pending:
            torch.cuda.synchronize(self.device)
        pending = []
        for name, start, end in self._pending:
            if wait or end.query():
                self.stages[name]["cuda_stream_seconds"] += start.elapsed_time(end) / 1000
            else:
                pending.append((name, start, end))
        self._pending = pending

    def save(self, path, *, status):
        if not self.enabled:
            return
        self.flush(wait=True)
        payload = {
            "status": status,
            "device": str(self.device),
            "scope": "run covers ServerBase.train; excludes model/data initialization; outputs.write also includes initial server metadata",
            "wall_clock": "perf_counter; host elapsed time, CUDA may be asynchronous",
            "cuda_clock": (
                "CUDA events on the current device stream; includes stream idle time"
                if self.cuda_events else "disabled"
            ),
            "aggregation": "inclusive nested stages; do not sum parent and child times",
            "stages": dict(sorted(self.stages.items())),
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def measure_stage(owner, name):
    timings = getattr(owner, "timings", None)
    return timings.measure(name) if timings is not None else nullcontext()


def timed_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            with measure_stage(self, name):
                return function(self, *args, **kwargs)
        return wrapped
    return decorate


def timed_torch_save(owner, *args, **kwargs):
    with measure_stage(owner, "outputs.write"):
        return torch.save(*args, **kwargs)

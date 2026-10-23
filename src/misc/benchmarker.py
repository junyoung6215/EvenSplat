import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.peak_memory = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        # Kernels launch asynchronously; synchronize so the interval covers the GPU work.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time()
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    @contextmanager
    def memory(self, tag: str):
        """Record the peak memory allocated inside the block, above what was already held."""
        if not torch.cuda.is_available():
            yield
            return
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            self.peak_memory[tag].append(torch.cuda.max_memory_allocated() - base)

    def dump_stage_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.peak_memory), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

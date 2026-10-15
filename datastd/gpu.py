"""
GPU detection and worker allocation.

Detects NVIDIA GPUs via nvidia-smi and assigns one persistent audio.cpp server
to each GPU. The server's default model capacity is one active request.
"""

import json
import subprocess
import os
from dataclasses import dataclass


@dataclass
class GPU:
    index: int
    name: str
    vram_mb: int


def detect_gpus() -> list[GPU]:
    """Detect NVIDIA GPUs using nvidia-smi. Returns empty list on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(GPU(index=int(parts[0]), name=parts[1], vram_mb=int(parts[2])))
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def estimate_workers_per_gpu(vram_mb: int, model_key: str) -> int:
    """Return the supported per-GPU request capacity for one server/model."""
    return 1


def build_allocation(model_key: str) -> dict:
    """
    Build a worker allocation plan.
    Returns {total_workers, gpus: [{index, name, vram_mb, workers, env: {CUDA_VISIBLE_DEVICES}}]}
    """
    gpus = detect_gpus()
    if not gpus:
        # CPU fallback: single worker
        return {
            "total_workers": 1,
            "gpus": [{"index": -1, "name": "CPU", "vram_mb": 0, "workers": 1, "backend": "cpu"}],
        }

    plan = []
    total = 0
    for gpu in gpus:
        w = estimate_workers_per_gpu(gpu.vram_mb, model_key)
        plan.append({
            "index": gpu.index,
            "name": gpu.name,
            "vram_mb": gpu.vram_mb,
            "workers": w,
            "backend": "cuda",
        })
        total += w

    return {"total_workers": total, "gpus": plan}

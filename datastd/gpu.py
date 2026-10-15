"""
GPU detection and worker allocation.

Detects NVIDIA GPUs via nvidia-smi, decides how many parallel TTS workers
each GPU can handle based on VRAM, and returns an allocation plan.
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
    """
    Estimate how many parallel audiocpp_cli processes a GPU can run.
    Conservative: ~2-4 GB per TTS worker depending on model size.
    """
    # ponytail: rough heuristic, good enough for a dataset tool
    model_vram = {
        "maya1": 3000,              # ~3 GB
        "omnivoice": 2500,          # ~2.5 GB (0.6B model)
        "qwen3_tts_voicedesign": 4000,  # ~4 GB (1.7B model)
    }
    per_worker = model_vram.get(model_key, 3000)
    # Leave ~1 GB headroom for the system
    available = max(0, vram_mb - 1024)
    workers = max(1, available // per_worker)
    return workers


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
            "backend": "best",
        })
        total += w

    return {"total_workers": total, "gpus": plan}

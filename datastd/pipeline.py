"""
Core pipeline — orchestrates text generation → TTS synthesis across multiple GPUs.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .gpu import build_allocation
from .models import MODELS, build_instruct, build_text_gen_prompt, build_cli_args
from .textgen import generate_batch

log = logging.getLogger(__name__)

# Average seconds of audio per TTS utterance (3-6 sentences ≈ 15-25s)
AVG_UTTERANCE_SECS = 20
SECS_PER_HOUR = 3600


def estimate_utterance_count(hours: float) -> int:
    """How many text utterances we need to fill `hours` of audio."""
    return max(1, int((hours * SECS_PER_HOUR) / AVG_UTTERANCE_SECS))


class Job:
    """Tracks a dataset generation job."""

    def __init__(self, job_id: str, model_key: str, speaker_type: str, hours: float, output_dir: str):
        self.job_id = job_id
        self.model_key = model_key
        self.speaker_type = speaker_type
        self.hours = hours
        self.output_dir = Path(output_dir)
        self.status = "pending"
        self.progress = 0  # 0-100
        self.total_utterances = 0
        self.completed_utterances = 0
        self.error = None
        self.allocation = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "model_key": self.model_key,
            "speaker_type": self.speaker_type,
            "hours": self.hours,
            "status": self.status,
            "progress": self.progress,
            "total_utterances": self.total_utterances,
            "completed_utterances": self.completed_utterances,
            "error": self.error,
            "allocation": self.allocation,
        }


# Global job store — ponytail: in-memory is fine for a single-session tool
_jobs: dict[str, Job] = {}


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    return [j.to_dict() for j in _jobs.values()]


def _find_cli() -> str:
    """Find the audiocpp_cli binary. Checks ./bin/ then PATH."""
    candidates = [
        Path("bin/audiocpp_cli"),
        Path("bin/audiocpp_cli.exe"),
        Path("audiocpp_cli"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # Fall back to PATH
    return "audiocpp_cli"


def _find_model(model_key: str) -> str:
    """Find the downloaded GGUF model file. Checks ./models/ dir."""
    filename = MODELS[model_key]["gguf_filename"]
    path = Path("models") / filename
    if path.exists():
        return str(path.resolve())
    raise FileNotFoundError(f"Model file not found: {path}. Download it first via setup.py.")


async def _synthesize_one(
    cli: str, model_path: str, model_key: str, speaker_type: str,
    text: str, output_path: str, gpu_index: int, backend: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Run a single audiocpp_cli TTS call."""
    cmd = build_cli_args(
        cli=cli,
        model_key=model_key,
        model_path=model_path,
        backend=backend,
        text=text,
        speaker_type=speaker_type,
        output_path=output_path,
    )

    env = os.environ.copy()
    if backend == "cuda" and gpu_index >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Ensure binary's directory is in PATH so dynamic libraries are located
    bin_dir = str(Path(cli).parent.resolve())
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    async with semaphore:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.warning(f"TTS failed for {output_path}: {stderr.decode()[:500]}")
                return False
            return True
        except Exception as e:
            log.warning(f"TTS error for {output_path}: {e}")
            return False


async def run_job(job: Job):
    """Execute a dataset generation job."""
    _jobs[job.job_id] = job
    job.status = "running"

    try:
        # 1. Figure out how many utterances we need
        job.total_utterances = estimate_utterance_count(job.hours)
        log.info(f"Job {job.job_id}: generating {job.total_utterances} utterances for {job.hours}h")

        # 2. Build GPU allocation
        job.allocation = build_allocation(job.model_key)
        total_workers = job.allocation["total_workers"]
        log.info(f"Job {job.job_id}: {total_workers} parallel workers across {len(job.allocation['gpus'])} device(s)")

        # 3. Generate text data
        job.status = "generating_text"
        prompt = build_text_gen_prompt(job.model_key, job.speaker_type)

        def on_text_progress(cur, total):
            job.progress = int((cur / total) * 20)

        texts = await generate_batch(prompt, job.total_utterances, on_progress=on_text_progress)
        if not texts:
            job.status = "error"
            job.error = "Failed to generate text data. Ensure Ollama/LLM is running at LLM_BASE_URL."
            return

        job.total_utterances = len(texts)  # adjust to actual count

        # 4. Save text manifest
        job.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, text in enumerate(texts):
            wav_name = f"utt_{i:06d}.wav"
            manifest.append({"id": i, "text": text, "audio": wav_name})

        manifest_path = job.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        # Also save a simple text-only file for reference
        texts_path = job.output_dir / "texts.txt"
        texts_path.write_text("\n---\n".join(texts))

        # 5. Synthesize audio with parallel workers
        job.status = "synthesizing"
        cli = _find_cli()
        model_path = _find_model(job.model_key)

        # Build per-GPU semaphores and task assignment
        gpu_configs = job.allocation["gpus"]
        gpu_semaphores = {}
        for gc in gpu_configs:
            gpu_semaphores[gc["index"]] = asyncio.Semaphore(gc["workers"])

        tasks = []
        for i, entry in enumerate(manifest):
            gc = gpu_configs[i % len(gpu_configs)]
            out_path = str(job.output_dir / entry["audio"])
            sem = gpu_semaphores[gc["index"]]

            tasks.append(_synthesize_one(
                cli, model_path, job.model_key, job.speaker_type,
                entry["text"], out_path, gc["index"], gc["backend"], sem,
            ))

        # Run all tasks, updating progress (20% to 100%)
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            if result:
                job.completed_utterances += 1
            job.progress = 20 + int(((i + 1) / len(tasks)) * 80)

        # 6. Save metadata
        meta = {
            "model": job.model_key,
            "speaker_type": job.speaker_type,
            "target_hours": job.hours,
            "total_utterances": job.total_utterances,
            "completed": job.completed_utterances,
            "allocation": job.allocation,
        }
        (job.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        job.status = "done"
        job.progress = 100
        log.info(f"Job {job.job_id}: done. {job.completed_utterances}/{job.total_utterances} succeeded.")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        log.exception(f"Job {job.job_id} failed")

"""
Text data generator — uses native llama.cpp server (llama-server) running a fast,
compact GGUF instruction-tuned model (e.g. Qwen2.5-1.5B-Instruct-GGUF).
No Ollama or external daemon required. Automatically launches and manages llama-server.
Also supports any custom external OpenAI-compatible endpoint via LLM_BASE_URL.
"""

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

LLM_PORT = int(os.environ.get("LLM_PORT", 8080))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", f"http://127.0.0.1:{LLM_PORT}/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "local")

# Subprocess handle for native llama-server
_llama_process: subprocess.Popen | None = None


def find_llama_server() -> str | None:
    """Find prebuilt llama-server binary in ./bin (including subdirs) or PATH."""
    candidates = []
    bin_dir = Path("bin")
    if bin_dir.exists():
        # Check direct candidates
        for name in ("llama-server", "llama-server.exe"):
            p = bin_dir / name
            if p.exists() and p.is_file():
                candidates.append(p)
        # Check subdirectories (e.g. bin/llama-b11658/llama-server)
        for p in bin_dir.rglob("llama-server*"):
            if p.is_file() and p.name in ("llama-server", "llama-server.exe"):
                if p not in candidates:
                    candidates.append(p)

    candidates.extend([
        Path("llama-server"),
        Path("llama-server.exe"),
    ])
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c.resolve())
    which_bin = shutil.which("llama-server")
    if which_bin:
        return which_bin
    return None


def find_llm_model() -> str | None:
    """Find the GGUF model file in ./models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return None

    # Priority 1: user-specified model
    target = models_dir / LLM_MODEL
    if target.exists():
        return str(target.resolve())

    # Priority 2: any Qwen instruct or LLM GGUF
    preferred = [
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "llama-3.2-1b-instruct-q4_k_m.gguf",
    ]
    for name in preferred:
        p = models_dir / name
        if p.exists():
            return str(p.resolve())

    # Priority 3: any .gguf with instruct/qwen/llama in name
    for f in models_dir.glob("*.gguf"):
        name_lower = f.name.lower()
        if "instruct" in name_lower or "qwen" in name_lower or "chat" in name_lower:
            return str(f.resolve())

    return None


def is_llm_server_running() -> bool:
    """Check if an LLM server is already responding at LLM_BASE_URL."""
    try:
        health_url = LLM_BASE_URL.replace("/v1", "") + "/health"
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(health_url)
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{LLM_BASE_URL}/models")
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    return False


def start_llama_server() -> bool:
    """Start local llama-server in background if binary and model exist."""
    global _llama_process

    if is_llm_server_running():
        log.info(f"LLM server already active at {LLM_BASE_URL}")
        return True

    server_bin = find_llama_server()
    if not server_bin:
        log.warning("llama-server binary not found in ./bin or PATH. Run setup.py to download it.")
        return False

    model_file = find_llm_model()
    if not model_file:
        log.warning(f"GGUF LLM model not found in ./models. Run setup.py to download {LLM_MODEL}.")
        return False

    server_path = Path(server_bin)
    if os.name != "nt":
        try:
            os.chmod(server_path, 0o755)
        except Exception:
            pass
        # Clean up any orphaned llama-server processes holding the port
        try:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
            time.sleep(0.3)
        except Exception:
            pass

    # Build LD_LIBRARY_PATH and PATH with server's directory and all bin subdirectories
    env = os.environ.copy()
    lib_dirs = [str(server_path.parent.resolve()), str(Path("bin").resolve())]
    if Path("bin").exists():
        for p in Path("bin").iterdir():
            if p.is_dir():
                lib_dirs.append(str(p.resolve()))

    ld_str = os.pathsep.join(dict.fromkeys(lib_dirs))
    env["LD_LIBRARY_PATH"] = ld_str + (os.pathsep + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")
    env["PATH"] = ld_str + os.pathsep + env.get("PATH", "")

    threads = min(os.cpu_count() or 4, 4)
    cmd = [
        server_bin,
        "-m", model_file,
        "--port", str(LLM_PORT),
        "--host", "127.0.0.1",
        "-c", "2048",
        "-ngl", "99",  # Offload layers to GPU if available; falls back to CPU cleanly
        "--threads", str(threads),
    ]

    log.info(f"Starting native llama-server: {' '.join(cmd)}")
    log_file_path = Path("bin/llama_server.log")
    log_file_path.parent.mkdir(exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")

    try:
        _llama_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        log.error(f"Failed to launch llama-server process: {e}")
        log_file.close()
        return False

    # Wait for server readiness, checking process liveness on every poll
    health_url = f"http://127.0.0.1:{LLM_PORT}/health"
    for _ in range(40):
        time.sleep(0.5)

        if _llama_process.poll() is not None:
            log_file.flush()
            err_output = ""
            if log_file_path.exists():
                err_output = log_file_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            log.error(
                f"llama-server terminated unexpectedly with exit code {_llama_process.returncode}:\n{err_output}"
            )
            return False

        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(health_url)
                if r.status_code == 200:
                    log.info(f"Native llama-server online at {LLM_BASE_URL}")
                    return True
        except Exception:
            continue

    log.warning("llama-server launched but timed out waiting for /health check.")
    return False


def stop_llama_server():
    """Terminate llama-server subprocess on exit."""
    global _llama_process
    if _llama_process is not None:
        try:
            log.info("Stopping native llama-server...")
            _llama_process.terminate()
            _llama_process.wait(timeout=3)
        except Exception:
            _llama_process.kill()
        finally:
            _llama_process = None


atexit.register(stop_llama_server)


async def generate_text(prompt: str) -> str:
    """
    Send prompt to llama-server (or external OpenAI-compatible endpoint) and return text.
    """
    if not is_llm_server_running():
        started = start_llama_server()
        if not started and not is_llm_server_running():
            raise RuntimeError(
                f"No LLM server running at {LLM_BASE_URL} and could not auto-start llama-server. "
                "Ensure bin/llama-server and models/*.gguf exist (run `python setup.py`), "
                "or point LLM_BASE_URL to an external endpoint."
            )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a specialized TTS text dataset generator. "
                            "Follow the user instructions precisely. Output ONLY the natural spoken text "
                            "with requested tags. Do NOT output conversational chatter, quotes, or markdown formatting."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 512,
                "stop": ["\n\n\n", "<|im_end|>", "<|endoftext|>"],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()

        # Clean reasoning tokens or think tags
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip surrounding quotes if model wrapped full response in quotes
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        return text


async def generate_batch(prompt: str, count: int, on_progress=None) -> list[str]:
    """Generate `count` text samples sequentially."""
    results = []
    if not is_llm_server_running():
        start_llama_server()

    for i in range(count):
        try:
            text = await generate_text(prompt)
            if text:
                results.append(text)
            log.info(f"Generated text {i+1}/{count}")
        except Exception as e:
            log.warning(f"Text generation {i+1} failed: {e}")
            if not is_llm_server_running():
                log.error("LLM server is unavailable. Aborting batch generation early.")
                break
        if on_progress:
            on_progress(i + 1, count)
    return results

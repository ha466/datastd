"""
kaggle.py — Run DataSTD on Kaggle Notebook with Free GPU & Public Gateway.

Kaggle provides free NVIDIA GPUs (T4, 2x T4, or P100). This script sets up DataSTD,
native audio.cpp inference, native llama.cpp LLM server (Qwen2.5-1.5B GGUF),
and establishes a free public Cloudflare Gateway so you can control it from your browser or API.

================================================================================
HOW TO RUN IN A KAGGLE NOTEBOOK
================================================================================

Option A (Recommended — 2-line quickstart):
    !git clone https://github.com/ha466/datastd.git /kaggle/working/datastd
    %cd /kaggle/working/datastd
    !pip install uv && python kaggle.py

Option B (Paste this entire script into a Kaggle notebook cell):
    !pip install uv
    # Paste kaggle.py and run the cell. It will auto-clone the repo if missing!

================================================================================
GATEWAYS TO ACCESS THE SERVER
================================================================================
1. Cloudflare Tunnel (Default, Automatic, Zero Signup):
   - A public URL like https://xxxxxx.trycloudflare.com will be printed below.
   - Open this URL in any browser to access the interactive web interface.

2. Ngrok Gateway (Optional):
   - If you prefer Ngrok, set your auth token in notebook before running:
     import os; os.environ["NGROK_AUTHTOKEN"] = "your_token_here"

================================================================================
REST API ROUTES & GATEWAY USAGE
================================================================================
You can control dataset generation remotely via the Gateway URL:

1. List Supported Models & Tags:
   curl https://<gateway-url>/api/models

2. Check Detected GPUs & Allocation:
   curl https://<gateway-url>/api/gpus

3. Preview Worker Allocation:
   curl -X POST https://<gateway-url>/api/allocation \\
     -H "Content-Type: application/json" \\
     -d '{"model_key": "maya1", "speaker_type": "young female narrator", "hours": 1.0}'

4. Start Dataset Generation Job:
   curl -X POST https://<gateway-url>/api/jobs/start \\
     -H "Content-Type: application/json" \\
     -d '{"model_key": "maya1", "speaker_type": "young female narrator", "hours": 1.0}'
   # Returns: {"job_id": "abc12345", "status": "started"}

5. Poll Job Status & Progress:
   curl https://<gateway-url>/api/jobs/<job_id>

6. List All Jobs:
   curl https://<gateway-url>/api/jobs
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/ha466/datastd.git"


def run(cmd: str, **kw):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


def setup_uv():
    """Ensure uv package manager is available."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[SETUP] Installing uv package manager...")
        run("curl -LsSf https://astral.sh/uv/install.sh | sh")
        os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")


def ensure_repo():
    """Ensure project directory exists and is active."""
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() and (cwd / "datastd").exists():
        return cwd

    kaggle_working = Path("/kaggle/working/datastd")
    if (kaggle_working / "pyproject.toml").exists():
        os.chdir(kaggle_working)
        return kaggle_working

    # Auto-clone repository if not present
    print(f"[SETUP] Cloning DataSTD repository from {REPO_URL}...")
    kaggle_working.parent.mkdir(parents=True, exist_ok=True)
    run(f"git clone {REPO_URL} {kaggle_working}")
    os.chdir(kaggle_working)
    return kaggle_working


def start_optional_ngrok(port: int) -> str | None:
    """Optional Ngrok gateway if NGROK_AUTHTOKEN is provided."""
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        return None
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        print(f"[GATEWAY] Ngrok Gateway Online: {url}")
        return url
    except Exception as e:
        print(f"[WARN] Ngrok setup failed: {e}")
        return None


def main():
    print("=" * 70)
    print("🎙️ DataSTD — Kaggle Cloud GPU Dataset Generator")
    print("=" * 70)

    # 1. Setup uv
    setup_uv()

    # 2. Locate or clone project
    project_dir = ensure_repo()
    print(f"[SETUP] Active project directory: {project_dir}")

    # 3. Download native binaries (audio.cpp + llama-server) and default models
    print("\n[SETUP] Downloading audio.cpp, llama.cpp (llama-server), and models...")
    run("uv run python setup.py --platform linux --default")

    # 4. Configure Server & Gateway
    port = int(os.environ.get("PORT", 7860))
    os.environ["PORT"] = str(port)

    # Check optional Ngrok gateway
    ngrok_url = start_optional_ngrok(port)

    print("\n" + "=" * 70)
    print("🚀 Launching DataSTD Server & Cloudflare Tunnel Gateway...")
    print("  Look for the public gateway URL in the output below (e.g. trycloudflare.com)")
    print("  Open it in your browser to access the Web UI and start generation!")
    print("=" * 70 + "\n")

    # 5. Start datastd server (FastAPI + pycloudflared tunnel + auto llama-server)
    run("uv run datastd")


if __name__ == "__main__":
    main()

"""
FastAPI server — serves the HTML UI and provides API endpoints for the pipeline.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .gpu import build_allocation, detect_gpus
from .models import MODELS
from .pipeline import Job, get_job, list_jobs, run_job
from .textgen import is_llm_server_running, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="DataSTD — Synthetic Voice Dataset Generator")

# Serve static UI files
UI_DIR = Path(__file__).parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# --- API Models ---

class StartJobRequest(BaseModel):
    model_key: str
    speaker_type: str
    hours: float
    output_dir: str = "output"


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DataSTD</h1><p>UI not found. Place files in ./ui/</p>")


@app.get("/api/models")
async def api_models():
    """List available models with their config, description, and supported tags."""
    return {
        key: {
            "display_name": m["display_name"],
            "description": m.get("description", ""),
            "task": m.get("task", "tts"),
            "languages": m["languages"],
            "inline_tags": m["inline_tags"],
            "instruct_template": m["instruct_template"],
        }
        for key, m in MODELS.items()
    }


@app.get("/api/gpus")
async def api_gpus():
    """Detect GPUs and return info."""
    gpus = detect_gpus()
    return {
        "gpus": [{"index": g.index, "name": g.name, "vram_mb": g.vram_mb} for g in gpus],
        "count": len(gpus),
    }


@app.post("/api/allocation")
async def api_allocation(req: StartJobRequest):
    """Preview worker allocation without starting a job."""
    return build_allocation(req.model_key)


@app.post("/api/jobs/start")
async def api_start_job(req: StartJobRequest, bg: BackgroundTasks):
    """Start a new dataset generation job."""
    job_id = uuid.uuid4().hex[:8]
    job = Job(
        job_id=job_id,
        model_key=req.model_key,
        speaker_type=req.speaker_type,
        hours=req.hours,
        output_dir=os.path.join(req.output_dir, job_id),
    )
    bg.add_task(run_job, job)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/jobs")
async def api_list_jobs():
    """List all jobs."""
    return list_jobs()


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    """Get job status."""
    job = get_job(job_id)
    if not job:
        return {"error": "Job not found"}
    return job.to_dict()


PUBLIC_TUNNEL_URL: str | None = None


@app.get("/api/info")
async def api_info():
    """Server status, LLM status, and public URL info."""
    return {
        "status": "ok",
        "tunnel_url": PUBLIC_TUNNEL_URL,
        "llm_online": is_llm_server_running(),
        "llm_model": LLM_MODEL,
    }


def main():
    """Entry point for `datastd` command or `uv run datastd`."""
    global PUBLIC_TUNNEL_URL
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    log.info(f"Starting DataSTD server on port {port}")

    # Try to start cloudflared tunnel
    try:
        from pycloudflared import try_cloudflare
        tunnel = try_cloudflare(port=port)
        PUBLIC_TUNNEL_URL = getattr(tunnel, "tunnel_url", str(tunnel))
        print("\n" + "=" * 60)
        print(f"  DataSTD Public Tunnel URL: {PUBLIC_TUNNEL_URL}")
        print("=" * 60 + "\n")
        log.info(f"Cloudflare tunnel online: {PUBLIC_TUNNEL_URL}")
    except Exception as e:
        log.warning(f"Cloudflare tunnel failed (optional): {e}")

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

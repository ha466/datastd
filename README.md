# DataSTD — Synthetic Voice Dataset Generator

Generate hours of high-quality synthetic voice datasets using native [audio.cpp](https://github.com/0xShug0/audio.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp) engines with GGUF models.

**Zero heavy Python ML dependencies. Zero external daemons. No Ollama needed.**

DataSTD is a self-contained, high-throughput dataset generation studio that pairs a fast, compact instruction-tuned LLM (`Qwen2.5-1.5B-Instruct-GGUF`) for realistic conversational script generation with native C++ multi-GPU TTS synthesis.

---

## Key Highlights

- **Native C++ Performance**: Inference runs directly via pre-compiled `audio.cpp` and `llama-server` binaries with CUDA and CPU acceleration.
- **Self-Contained LLM Text Generation**: Ships with native `llama.cpp` integration that automatically launches and manages an instruction-tuned 1.5B GGUF model (`Qwen2.5-1.5B-Instruct-GGUF`).
- **Rich Expression Tags**: Full support for Maya1's 20 emotion/paralinguistic tags and OmniVoice's non-verbal cues.
- **Kaggle & Cloud GPU Ready**: Single-command startup with automatic repository cloning and Cloudflare / Ngrok gateway exposure.
- **Multi-GPU Parallel Workers**: Automatically detects NVIDIA GPUs via `nvidia-smi` and allocates parallel workers per device based on available VRAM.
- **Full REST API & Modern Web UI**: Inspect models, preview GPU worker allocations, trigger generation jobs, and track progress in real time.

---

## Supported TTS Models & Tag Visibility

| Model Family | Key | Task | Languages | Tags Support | Highlights |
|---|---|---|---|---|---|
| **Maya1** | `maya1` | TTS / Voice Design | English (`en`) | **20 Inline Emotion Tags** | Expressive storytelling, natural-language persona conditioning, SNAC neural codec |
| **OmniVoice** | `omnivoice` | Zero-Shot TTS | 600+ Languages | **Non-Verbal Expression Tags** | Multilingual synthesis, attribute-based voice design, non-verbal markers |
| **Qwen3-TTS VoiceDesign** | `qwen3_tts_voicedesign` | Voice Design | 10 Languages (`en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`) | Natural Prompt Conditioning | Deep voice control (age, gender, timbre, accent, emotion) from text descriptions |

### Inline Expression & Non-Verbal Tags Reference

#### 1. Maya1 Tags (20 Supported Tokens)
Maya1 supports localized inline performance markers placed directly within conversational text:
```
<angry>       <appalled>       <chuckle>      <cry>
<curious>     <disappointed>   <excited>      <exhale>
<gasp>        <giggle>         <gulp>         <laugh>
<laugh_harder><mischievous>    <sarcastic>    <scream>
<sigh>        <sing>           <snort>        <whisper>
```
*Example input:*
> "Wait, did you actually hear that? <gasp> <whisper>It sounded like someone whispering behind the door.</whisper> <laugh>Unbelievable!</laugh>"

#### 2. OmniVoice Non-Verbal Tags
OmniVoice accepts paralinguistic and discourse reaction tokens:
```
[laughter]              [sigh]                  [confirmation-en]
[question-en]           [question-ah]           [question-oh]
[surprise-ah]           [surprise-oh]           [surprise-wa]
[dissatisfaction-hnn]
```
*Example input:*
> "I never expected them to agree [laughter], but here we are [sigh]!"

---

## Kaggle Cloud Notebook Deployment

Kaggle provides free NVIDIA GPUs (T4, 2x T4, or P100). DataSTD includes a native Kaggle orchestrator in `kaggle.py`.

### How to Run on Kaggle

#### Method 1: Git Clone Quickstart (Recommended)
In a fresh Kaggle notebook cell:
```bash
!git clone https://github.com/ha466/datastd.git /kaggle/working/datastd
%cd /kaggle/working/datastd
!pip install uv && python kaggle.py
```

#### Method 2: Single-Cell Execution
If you paste `kaggle.py` directly into a notebook cell, the script will automatically detect the environment, clone `https://github.com/ha466/datastd.git` to `/kaggle/working/datastd`, download required native binaries, and start the server with a public Cloudflare tunnel.

### Gateways & Remote Access
When running on Kaggle, DataSTD automatically establishes a secure public gateway:
1. **Cloudflare Tunnel (Default, Free, Zero-Config)**:
   - Prints a public gateway URL: `https://xxxxxx.trycloudflare.com`
   - Open in any web browser to access the complete GUI, or query the REST API directly.
2. **Ngrok Gateway (Optional)**:
   - To use Ngrok instead, set your token before launching:
     ```python
     import os
     os.environ["NGROK_AUTHTOKEN"] = "your_ngrok_token_here"
     ```

---

## Remote REST API Reference

All features of DataSTD can be controlled programmatically over HTTP or the public gateway URL:

### 1. List Available Models & Supported Tags
```http
GET /api/models
```
**Response:**
```json
{
  "maya1": {
    "display_name": "Maya1 — Expressive Voice Design (English)",
    "description": "Maya Research 3B expressive TTS with natural-language voice design and 20 inline emotion/expression tags.",
    "task": "tts",
    "languages": ["en"],
    "inline_tags": ["<angry>", "<laugh>", "<whisper>", "..."],
    "instruct_template": "{speaker_type}, expressive storytelling style, natural pacing, clear articulation."
  }
}
```

### 2. Detect Hardware & GPUs
```http
GET /api/gpus
```
**Response:**
```json
{
  "gpus": [
    {"index": 0, "name": "Tesla T4", "vram_mb": 15360}
  ],
  "count": 1
}
```

### 3. Preview Worker Allocation
```http
POST /api/allocation
Content-Type: application/json

{
  "model_key": "maya1",
  "speaker_type": "young female narrator",
  "hours": 1.0
}
```
**Response:**
```json
{
  "total_workers": 4,
  "gpus": [
    {"index": 0, "name": "Tesla T4", "vram_mb": 15360, "workers": 4, "backend": "cuda"}
  ]
}
```

### 4. Start Dataset Generation Job
```http
POST /api/jobs/start
Content-Type: application/json

{
  "model_key": "maya1",
  "speaker_type": "young female narrator, warm conversational delivery",
  "hours": 1.0,
  "output_dir": "output"
}
```
**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "started"
}
```

### 5. Check Job Status & Progress
```http
GET /api/jobs/{job_id}
```
**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "model_key": "maya1",
  "speaker_type": "young female narrator, warm conversational delivery",
  "hours": 1.0,
  "status": "synthesizing",
  "progress": 65,
  "total_utterances": 180,
  "completed_utterances": 117,
  "error": null
}
```

### 6. Server Health & Gateway Info
```http
GET /api/info
```
**Response:**
```json
{
  "status": "ok",
  "tunnel_url": "https://sample-tunnel.trycloudflare.com",
  "llm_online": true,
  "llm_model": "qwen2.5-1.5b-instruct-q4_k_m.gguf"
}
```

---

## Local Quick Start

### Windows
```bat
REM 1. One-click launch
start_windows.bat
```
*(Automatically downloads `uv`, native `audiocpp_cli.exe`, `llama-server.exe`, and default models on first run).*

### Linux / Colab
```bash
chmod +x start_linux.sh
./start_linux.sh
```

### Manual CLI Setup
```bash
# 1. Install dependencies
uv sync

# 2. Download native binaries & default models
uv run python setup.py --default

# 3. Launch the server
uv run datastd
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Web server port |
| `LLM_PORT` | `8080` | Native llama-server port |
| `LLM_MODEL` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Local GGUF LLM filename in `models/` |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible LLM endpoint (can point to any remote LLM) |
| `LLM_API_KEY` | `local` | LLM authorization key (if using external endpoint) |
| `NGROK_AUTHTOKEN` | *(empty)* | Optional Ngrok auth token for public gateway |

---

## Output Structure

Each completed generation job outputs:
```
output/<job_id>/
├── manifest.json       # Array of {"id": 0, "text": "...", "audio": "utt_000000.wav"}
├── texts.txt           # Raw text lines with inline emotion tags
├── metadata.json       # Job parameters, GPU allocation, stats, timestamps
├── utt_000000.wav      # Synthesized 24 kHz WAV audio files
├── utt_000001.wav
└── ...
```

---

## Verification Self-Check

DataSTD includes a lightweight runnable verification check:
```bash
python test_check.py
```

---

## License & Credits

- Core framework powered by [audio.cpp](https://github.com/0xShug0/audio.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp).
- Models (`Maya1`, `OmniVoice`, `Qwen3-TTS`) retain the open licenses from their respective creators.

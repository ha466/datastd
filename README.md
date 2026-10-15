# DataSTD — Synthetic Voice Dataset Generator

High-throughput synthetic voice dataset generator powered by native [audio.cpp](https://github.com/0xShug0/audio.cpp) (TTS) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (text generation) engines with GGUF models.

**Zero Python ML dependencies. Zero external daemons.**

---

## Quick Start

### Kaggle / Colab (Free GPU)
```bash
!git clone https://github.com/ha466/datastd.git /kaggle/working/datastd
%cd /kaggle/working/datastd
!pip install uv && python kaggle.py
```
*Auto-detects GPU, downloads native binaries, launches local `llama-server` (`Qwen2.5-1.5B`), and outputs a free public Cloudflare Gateway URL (`https://*.trycloudflare.com`).*

### Windows
```bat
start_windows.bat
```

### Linux
```bash
chmod +x start_linux.sh && ./start_linux.sh
```

---

## Supported Models & Tags

| Model | Task | Languages | Tags Support | Description |
|---|---|---|---|---|
| **Maya1** (`maya1`) | TTS / Voice Design | English (`en`) | **20 Inline Emotion Tags** | 3B expressive TTS with natural-language persona conditioning |
| **OmniVoice** (`omnivoice`) | Zero-Shot TTS | 600+ Languages | **Non-Verbal Reaction Tags** | Massive multilingual synthesis with paralinguistic tokens |
| **Qwen3-TTS** (`qwen3_tts_voicedesign`) | Voice Design | 10 Languages | Persona Prompt Conditioning | 1.7B VoiceDesign from natural-language descriptions |

### Tag Reference

- **Maya1**: `<angry>`, `<appalled>`, `<chuckle>`, `<cry>`, `<curious>`, `<disappointed>`, `<excited>`, `<exhale>`, `<gasp>`, `<giggle>`, `<gulp>`, `<laugh>`, `<laugh_harder>`, `<mischievous>`, `<sarcastic>`, `<scream>`, `<sigh>`, `<sing>`, `<snort>`, `<whisper>`.
- **OmniVoice**: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[dissatisfaction-hnn]`.

---

## Hugging Face Dataset Upload

Push generated datasets (WAVs, `manifest.json`, and metadata) directly to Hugging Face Hub:

1. **Web UI**: Enter your repo ID (e.g. `username/my-voice-dataset`) and token in the form, or click **"🤗 Upload to Hugging Face"** on any completed job card.
2. **Auto-upload via API / Job Start**: Include `"hf_repo_id": "username/my-dataset"` in your start request.
3. **Manual API Upload**:
   ```bash
   curl -X POST http://localhost:7860/api/jobs/<job_id>/upload_hf \
     -H "Content-Type: application/json" \
     -d '{"repo_id": "username/voice-dataset", "token": "hf_..."}'
   ```
   *(Or set `export HF_TOKEN="hf_..."` to omit the token parameter).*

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List models, supported tags, and instruct templates |
| `GET` | `/api/gpus` | Detected NVIDIA GPUs and VRAM |
| `POST` | `/api/allocation` | Preview GPU worker distribution |
| `POST` | `/api/jobs/start` | Submit generation job (`model_key`, `speaker_type`, `hours`, optional `hf_repo_id`) |
| `GET` | `/api/jobs` | List all jobs |
| `GET` | `/api/jobs/{id}` | Job progress, utterance count, and status |
| `POST` | `/api/jobs/{id}/upload_hf` | Upload completed job dataset to Hugging Face Hub |
| `GET` | `/api/info` | Server status, gateway URL, and LLM engine health |

### Example: Start Job & Upload
```bash
curl -X POST http://localhost:7860/api/jobs/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "maya1",
    "speaker_type": "young female narrator, warm and engaging",
    "hours": 0.5,
    "hf_repo_id": "username/synthetic-voice-dataset"
  }'
```

---

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server port |
| `LLM_PORT` | `8080` | Native llama-server port |
| `LLM_MODEL` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Local GGUF LLM filename in `models/` |
| `LLM_PARALLEL` | `2` | llama-server decoding slots for concurrent samples |
| `LLM_BATCH_SIZE` | `4` | Independent prompts included in each native llama.cpp batch request |
| `LLM_CONTEXT_SIZE` | `4096` | Total llama-server context size shared across slots |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible endpoint (local or remote) |
| `TTS_TEXT_BATCH_MAX` | `64` | Maximum generated-text batch before valid-audio duration is measured again |
| `TTS_MAX_RETRIES` | `3` | Synthesis attempts per utterance before it is counted as failed |
| `TTS_MAX_NO_PROGRESS_BATCHES` | `3` | Consecutive batches without valid audio before generation stops |
| `TTS_REQUEST_TIMEOUT_SECONDS` | `600` | Per-utterance audio.cpp request timeout |
| `TTS_SERVER_STARTUP_TIMEOUT_SECONDS` | `120` | Maximum wait for each audio.cpp server to become ready |
| `TTS_SERVER_THREADS` | `2` | CPU threads assigned to each audio.cpp server |
| `HF_TOKEN` | *(empty)* | Hugging Face write token for dataset uploads |
| `NGROK_AUTHTOKEN` | *(empty)* | Optional Ngrok auth token for public gateway |

---

## Output Structure

```
output/<job_id>/
├── manifest.json       # Successful WAVs only, with measured duration_seconds
├── texts.txt           # Generated lines with inline emotion tags
├── metadata.json       # Job config, GPU allocation, stats
└── utt_000000.wav      # Validated synthesized WAV audio files
```

---

## Verification

```bash
python test_check.py
```

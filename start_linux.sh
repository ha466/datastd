#!/bin/bash
# DataSTD — Linux/Colab Startup Script
# Powered by audio.cpp (TTS) and llama.cpp (LLM textgen) — zero external daemons needed!
set -e

echo ""
echo "========================================"
echo "  DataSTD - Synthetic Voice Dataset"
echo "========================================"
echo ""

# Check uv
if ! command -v uv &> /dev/null; then
    echo "[SETUP] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Setup: download native binaries (audio.cpp + llama-server) and default models
if [ ! -f "bin/audiocpp_cli" ] || [ ! -f "bin/llama-server" ]; then
    echo "[SETUP] Downloading native audio.cpp and llama.cpp binaries and models..."
    uv run python setup.py --platform linux --default
fi

if [ ! -d "models" ] || [ -z "$(ls -A models/ 2>/dev/null)" ]; then
    echo "[SETUP] Downloading default models..."
    uv run python setup.py --models-only --default
fi

# Start the server
echo "[START] Launching DataSTD server..."
uv run datastd

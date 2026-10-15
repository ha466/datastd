"""
setup.py — Downloads audio.cpp & llama.cpp binaries and GGUF models.
Usage:
    uv run python setup.py                       # auto-detects platform, downloads binaries + default models
    uv run python setup.py --platform linux      # explicit platform
    uv run python setup.py --default             # default model (maya1 + qwen2.5-1.5b)
    uv run python setup.py --all-models          # download all supported models
    uv run python setup.py --binary-only         # binaries only
    uv run python setup.py --models-only         # models only
"""

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

AUDIO_BINARIES = {
    "linux": {
        "url": "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.1/audio-v0.9.1-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
        "filename": "audio-v0.9.1-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
    },
    "windows": {
        "url": "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.1/audio-v0.9.1-bin-windows-x64-cuda12.4.zip",
        "filename": "audio-v0.9.1-bin-windows-x64-cuda12.4.zip",
    },
}

AUDIO_CUDA_RUNTIME = {
    "windows": {
        "url": "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.1/audio-v0.9.1-cudart-windows-x64-cuda12.4.zip",
        "filename": "audio-v0.9.1-cudart-windows-x64-cuda12.4.zip",
    },
}

AUDIO_CPU_BINARIES = {
    "linux": {
        "url": "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.1/audio-v0.9.1-bin-ubuntu-x64-cpu.tar.gz",
        "filename": "audio-v0.9.1-bin-ubuntu-x64-cpu.tar.gz",
    },
    "windows": {
        "url": "https://github.com/0xShug0/audio.cpp/releases/download/v0.9.1/audio-v0.9.1-bin-windows-x64-cpu.zip",
        "filename": "audio-v0.9.1-bin-windows-x64-cpu.zip",
    },
}

LLAMA_BINARIES = {
    "linux": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11658/llama-b11658-bin-ubuntu-cuda-12.8-x64.tar.gz",
        "filename": "llama-b11658-bin-ubuntu-cuda-12.8-x64.tar.gz",
    },
    "windows": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11658/llama-b11658-bin-win-cuda-12.4-x64.zip",
        "filename": "llama-b11658-bin-win-cuda-12.4-x64.zip",
    },
}

LLAMA_CPU_BINARIES = {
    "linux": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11658/llama-b11658-bin-ubuntu-x64.tar.gz",
        "filename": "llama-b11658-bin-ubuntu-x64.tar.gz",
    },
    "windows": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b11658/llama-b11658-bin-win-cpu-x64.zip",
        "filename": "llama-b11658-bin-win-cpu-x64.zip",
    },
}

TTS_MODELS = {
    "maya1-orig.gguf": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Maya1-GGUF/maya1-orig.gguf",
    "omnivoice-f16.gguf": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/OmniVoice-GGUF/omnivoice-f16.gguf",
    "qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf",
}

LLM_MODELS = {
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

DEFAULT_MODELS = {
    "maya1-orig.gguf": TTS_MODELS["maya1-orig.gguf"],
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": LLM_MODELS["qwen2.5-1.5b-instruct-q4_k_m.gguf"],
}


def download(url: str, dest: str):
    """Download with progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")
    urlretrieve(
        url,
        dest,
        reporthook=lambda c, bs, ts: print(
            f"\r  {c * bs / (1024 * 1024):.1f} MB" if ts <= 0 else f"\r  {c * bs / (1024 * 1024):.1f} MB / {ts / (1024 * 1024):.1f} MB",
            end="",
            flush=True,
        ),
    )
    print()


def _extract_archive(archive_path: Path, dest_dir: Path):
    """Extract tar.gz or zip archive."""
    if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(dest_dir, filter="tar")
            else:
                tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)


def setup_binaries(platform: str, use_cpu: bool = False):
    """Download and extract both audio.cpp and llama.cpp binaries (GPU accelerated by default)."""
    bin_dir = Path("bin")
    bin_dir.mkdir(exist_ok=True)

    cli_name = "audiocpp_cli.exe" if platform == "windows" else "audiocpp_cli"
    server_name = "audiocpp_server.exe" if platform == "windows" else "audiocpp_server"
    llama_name = "llama-server.exe" if platform == "windows" else "llama-server"

    # Pre-clean collisions if a directory exists with the same name as binary
    for name in (cli_name, server_name, llama_name):
        target = bin_dir / name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)

    # 1. audio.cpp (GPU build by default)
    audio_info = AUDIO_CPU_BINARIES[platform] if use_cpu else AUDIO_BINARIES[platform]
    audio_archive = bin_dir / audio_info["filename"]
    gpu_label = "CPU-only" if use_cpu else "GPU accelerated"
    if not audio_archive.exists():
        print(f"\n[1/4] Downloading audio.cpp binary ({platform} {gpu_label})...")
        download(audio_info["url"], str(audio_archive))
    else:
        print(f"\n[1/4] audio.cpp binary archive already exists ({gpu_label}): {audio_archive}")

    print(f"  Extracting audio.cpp to {bin_dir}/...")
    _extract_archive(audio_archive, bin_dir)

    # Windows CUDA builds need the matching runtime DLL archive beside the binaries.
    if not use_cpu and platform in AUDIO_CUDA_RUNTIME:
        runtime_info = AUDIO_CUDA_RUNTIME[platform]
        runtime_archive = bin_dir / runtime_info["filename"]
        if not runtime_archive.exists():
            print(f"\n[1b/4] Downloading audio.cpp CUDA runtime ({platform})...")
            download(runtime_info["url"], str(runtime_archive))
        else:
            print(f"\n[1b/4] audio.cpp CUDA runtime archive already exists: {runtime_archive}")
        print(f"  Extracting audio.cpp CUDA runtime to {bin_dir}/...")
        _extract_archive(runtime_archive, bin_dir)

    # 2. llama.cpp (llama-server) — CUDA GPU by default for high-speed LLM text generation
    llama_info = LLAMA_CPU_BINARIES[platform] if use_cpu else LLAMA_BINARIES[platform]
    llama_archive = bin_dir / llama_info["filename"]
    mode_str = "CPU-only" if use_cpu else "CUDA GPU"
    if not llama_archive.exists():
        print(f"\n[2/4] Downloading llama.cpp binary ({platform} {mode_str})...")
        download(llama_info["url"], str(llama_archive))
    else:
        print(f"\n[2/4] llama.cpp binary archive already exists ({mode_str}): {llama_archive}")

    print(f"  Extracting llama.cpp to {bin_dir}/...")
    _extract_archive(llama_archive, bin_dir)

    # Flatten binaries and libraries to bin_dir and set executable permissions
    for root, dirs, files in os.walk(bin_dir):
        for f in files:
            if f in (cli_name, server_name, llama_name) or f.endswith(".dll") or ".so" in f:
                src = Path(root) / f
                dst = bin_dir / f
                if src.resolve() != dst.resolve():
                    try:
                        if dst.is_dir() and not dst.is_symlink():
                            shutil.rmtree(dst)
                        elif dst.is_symlink() or dst.exists():
                            dst.unlink()
                        shutil.copy2(src, dst, follow_symlinks=True)
                    except Exception:
                        pass
                if platform != "windows":
                    try:
                        os.chmod(dst, 0o755)
                    except Exception:
                        pass

    # Verify and fallback search for binaries
    target_cli = bin_dir / cli_name
    if not target_cli.is_file():
        for p in bin_dir.rglob(cli_name):
            if p.is_file():
                if target_cli.is_dir() and not target_cli.is_symlink():
                    shutil.rmtree(target_cli)
                elif target_cli.is_symlink() or target_cli.exists():
                    target_cli.unlink()
                shutil.copy2(p, target_cli, follow_symlinks=True)
                break

    target_server = bin_dir / server_name
    if not target_server.is_file():
        for p in bin_dir.rglob(server_name):
            if p.is_file():
                if target_server.is_dir() and not target_server.is_symlink():
                    shutil.rmtree(target_server)
                elif target_server.is_symlink() or target_server.exists():
                    target_server.unlink()
                shutil.copy2(p, target_server, follow_symlinks=True)
                break

    target_llama = bin_dir / llama_name
    if not target_llama.is_file():
        for p in bin_dir.rglob(llama_name):
            if p.is_file():
                if target_llama.is_dir() and not target_llama.is_symlink():
                    shutil.rmtree(target_llama)
                elif target_llama.is_symlink() or target_llama.exists():
                    target_llama.unlink()
                shutil.copy2(p, target_llama, follow_symlinks=True)
                break

    if platform != "windows":
        for target in (target_cli, target_server, target_llama):
            if target.is_file():
                try:
                    os.chmod(target, 0o755)
                except Exception:
                    pass

    if not target_server.is_file():
        raise RuntimeError(
            f"audiocpp_server was not found in the extracted audio.cpp archive: {audio_archive}"
        )

    print(f"  [OK] audio.cpp CLI ready:    {target_cli}")
    print(f"  [OK] audio.cpp server ready: {target_server}")
    print(f"  [OK] llama-server ready:  {target_llama}")


def setup_models(targets: dict[str, str]):
    """Download GGUF model files."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print(f"\n[3/4] Downloading {len(targets)} GGUF model(s)...")
    for filename, url in targets.items():
        dest = models_dir / filename
        if dest.exists():
            print(f"  [OK] Already exists: {dest}")
            continue
        download(url, str(dest))
        print(f"  [OK] Downloaded: {dest}")


def main():
    parser = argparse.ArgumentParser(description="DataSTD Setup — download native binaries and GGUF models")
    parser.add_argument("--platform", choices=["windows", "linux"], default=None,
                        help="Platform for binaries (auto-detected if omitted)")
    parser.add_argument("--default", action="store_true",
                        help="Download default models (Maya1 TTS + Qwen2.5-1.5B-Instruct LLM)")
    parser.add_argument("--all-models", action="store_true",
                        help="Download all supported TTS models + LLM models")
    parser.add_argument("--binary-only", action="store_true", help="Download binaries only")
    parser.add_argument("--models-only", action="store_true", help="Download models only")
    parser.add_argument("--llm-only", action="store_true", help="Download LLM model only")
    parser.add_argument("--cpu", action="store_true", help="Download CPU-only llama-server instead of CUDA GPU")
    parser.add_argument("--model", action="append", help="Download specific model(s) by filename")
    args = parser.parse_args()

    platform = args.platform
    if platform is None:
        platform = "windows" if sys.platform == "win32" else "linux"
        print(f"Auto-detected platform: {platform}")

    if not args.models_only and not args.llm_only:
        setup_binaries(platform, use_cpu=args.cpu)

    if not args.binary_only:
        if args.llm_only:
            setup_models(LLM_MODELS)
        elif args.model:
            all_dict = {**TTS_MODELS, **LLM_MODELS}
            targets = {k: all_dict[k] for k in args.model if k in all_dict}
            setup_models(targets)
        elif args.all_models:
            setup_models({**TTS_MODELS, **LLM_MODELS})
        else:
            # Default minimal setup: maya1 + qwen2.5-1.5b-instruct
            setup_models(DEFAULT_MODELS)

    print("\n✓ Setup complete!")
    print("  Run DataSTD with: `uv run python -m datastd.server`")


if __name__ == "__main__":
    main()

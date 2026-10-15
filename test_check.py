"""
Runnable self-check script for DataSTD (Ponytail rule: smallest assert-based check).
Run with: python test_check.py
"""

from datastd.models import MODELS, build_instruct, build_text_gen_prompt, build_cli_args
from datastd.gpu import estimate_workers_per_gpu, build_allocation
from datastd.pipeline import estimate_utterance_count, Job


def test_models():
    # 1. Verify all 3 models configured
    assert "maya1" in MODELS, "Missing maya1"
    assert "omnivoice" in MODELS, "Missing omnivoice"
    assert "qwen3_tts_voicedesign" in MODELS, "Missing qwen3_tts_voicedesign"

    # Verify Maya1 complete 20 tags
    assert len(MODELS["maya1"]["inline_tags"]) == 20, f"Expected 20 Maya1 tags, got {len(MODELS['maya1']['inline_tags'])}"
    assert "<whisper>" in MODELS["maya1"]["inline_tags"]
    assert "<laugh>" in MODELS["maya1"]["inline_tags"]
    assert "<curious>" in MODELS["maya1"]["inline_tags"]

    # Verify OmniVoice nonverbal tags
    assert len(MODELS["omnivoice"]["inline_tags"]) > 0, "OmniVoice should have nonverbal tags"
    assert "[laughter]" in MODELS["omnivoice"]["inline_tags"]
    assert "[sigh]" in MODELS["omnivoice"]["inline_tags"]

    # 2. Instruct formatting
    speaker = "young female narrator"
    maya1_instruct = build_instruct("maya1", speaker)
    assert speaker in maya1_instruct, f"Speaker missing in instruct: {maya1_instruct}"

    # 3. Text gen prompt formatting & tags
    prompt_maya1 = build_text_gen_prompt("maya1", speaker)
    assert "<laugh>" in prompt_maya1, "Maya1 inline tag missing in LLM prompt"
    assert speaker in prompt_maya1

    prompt_omni = build_text_gen_prompt("omnivoice", speaker)
    assert "[laughter]" in prompt_omni, "OmniVoice nonverbal tag missing in LLM prompt"
    assert speaker in prompt_omni

    # 4. CLI arguments construction
    cmd_maya1 = build_cli_args(
        cli="audiocpp_cli",
        model_key="maya1",
        model_path="models/maya1-orig.gguf",
        backend="cuda",
        text="Hello world <laugh> haha </laugh>",
        speaker_type=speaker,
        output_path="out.wav",
    )
    assert "--task" in cmd_maya1 and cmd_maya1[cmd_maya1.index("--task") + 1] == "tts"
    assert "--family" in cmd_maya1 and cmd_maya1[cmd_maya1.index("--family") + 1] == "maya1"
    assert any(arg.startswith("instruct=") for arg in cmd_maya1), "Maya1 must use instruct in request-option"

    cmd_qwen = build_cli_args(
        cli="audiocpp_cli",
        model_key="qwen3_tts_voicedesign",
        model_path="models/qwen3-tts.gguf",
        backend="best",
        text="Testing Qwen",
        speaker_type=speaker,
        output_path="out2.wav",
        device_index=1,
    )
    assert "--task" in cmd_qwen and cmd_qwen[cmd_qwen.index("--task") + 1] == "vdes"
    assert "--instruct" in cmd_qwen, "Qwen VoiceDesign must use --instruct"
    assert "--device" in cmd_qwen and cmd_qwen[cmd_qwen.index("--device") + 1] == "1"
    assert "--backend" in cmd_qwen and cmd_qwen[cmd_qwen.index("--backend") + 1] == "best"
    print("[OK] Model templates and CLI argument generation verified.")


def test_gpu_allocation():
    # Test worker estimation math
    # 8GB GPU (~8192 MB)
    w_maya = estimate_workers_per_gpu(8192, "maya1")
    assert w_maya >= 2, f"Expected at least 2 workers on 8GB GPU, got {w_maya}"

    # Fallback plan when no nvidia-smi
    plan = build_allocation("maya1")
    assert "total_workers" in plan
    assert plan["total_workers"] >= 1
    assert len(plan["gpus"]) >= 1
    print("[OK] GPU worker allocation logic verified.")


def test_pipeline():
    # 1 hour = 3600s / 20s ≈ 180 utterances
    count_1h = estimate_utterance_count(1.0)
    assert count_1h == 180, f"Expected 180 utterances for 1h, got {count_1h}"

    job = Job(
        job_id="test1234",
        model_key="maya1",
        speaker_type="young female",
        hours=0.5,
        output_dir="output/test1234",
        hf_repo_id="testuser/test-dataset",
        hf_private=True,
    )
    d = job.to_dict()
    assert d["job_id"] == "test1234"
    assert d["status"] == "pending"
    assert d["progress"] == 0
    assert d["hf_repo_id"] == "testuser/test-dataset"
    assert "hf_url" in d
    print("[OK] Pipeline estimation and job tracking (with HF Hub support) verified.")


def test_binary_discovery():
    import tempfile
    import os
    from pathlib import Path
    from datastd.textgen import find_llama_server
    from datastd.pipeline import _find_cli

    # Test nested bin discovery
    test_bin = Path("bin/test_llama_dir")
    test_bin.mkdir(parents=True, exist_ok=True)
    dummy_server = test_bin / ("llama-server.exe" if os.name == "nt" else "llama-server")
    dummy_server.write_text("dummy")

    try:
        found = find_llama_server()
        assert found is not None, "Failed to locate nested llama-server in bin subfolder"
        assert "llama-server" in found

        # Verify _find_cli executes safely
        cli_result = _find_cli()
        assert isinstance(cli_result, str), "_find_cli should return string binary path or name"
    finally:
        dummy_server.unlink(missing_ok=True)
        test_bin.rmdir()

    print("[OK] Binary discovery (including subdirectories and _find_cli) verified.")


if __name__ == "__main__":
    test_models()
    test_gpu_allocation()
    test_pipeline()
    test_binary_discovery()
    print("\nALL CHECKS PASSED!")

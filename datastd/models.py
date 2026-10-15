"""
Pre-built chat templates & inline tags for each supported TTS model family.

Each model has:
  - family: the --family flag for audiocpp_cli
  - gguf_url: HuggingFace direct download URL
  - gguf_filename: local filename after download
  - instruct_template: a Python format-string the user fills in with speaker_type
  - inline_tags: list of available inline emotion/paralinguistic tags
  - sample_text_prompt: prompt sent to a small LLM to generate tagged text data
"""

MODELS = {
    "maya1": {
        "family": "maya1",
        "task": "tts",
        "instruct_style": "request_option",
        "display_name": "Maya1 — Expressive Voice Design (English)",
        "description": "Maya Research 3B expressive TTS with natural-language voice design and 20 inline emotion/expression tags.",
        "gguf_url": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Maya1-GGUF/maya1-orig.gguf",
        "gguf_filename": "maya1-orig.gguf",
        "languages": ["en"],
        "inline_tags": [
            "<angry>",
            "<appalled>",
            "<chuckle>",
            "<cry>",
            "<curious>",
            "<disappointed>",
            "<excited>",
            "<exhale>",
            "<gasp>",
            "<giggle>",
            "<gulp>",
            "<laugh>",
            "<laugh_harder>",
            "<mischievous>",
            "<sarcastic>",
            "<scream>",
            "<sigh>",
            "<sing>",
            "<snort>",
            "<whisper>",
        ],
        # {speaker_type} is replaced at runtime with user input like "young female narrator"
        "instruct_template": "{speaker_type}, expressive storytelling style, natural pacing, clear articulation.",
        "sample_text_prompt": (
            "You are a TTS text dataset generator. Generate a single spoken paragraph (3-5 sentences) "
            "for a voice described as: {speaker_type}.\n"
            "The text must be natural, engaging spoken conversational English.\n"
            "Naturally embed 1 to 3 emotion/expression tags from this supported set: {tags}.\n"
            "Tags should wrap or precede emotional delivery (e.g. '<gasp> Wait, did you actually hear that? "
            "<whisper>It sounded like someone whispering behind the door.</whisper> <laugh>Unbelievable!</laugh>').\n"
            "Output ONLY the tagged spoken text. Do not include quotes, markdown formatting, or explanations."
        ),
    },
    "omnivoice": {
        "family": "omnivoice",
        "task": "tts",
        "instruct_style": "instruct",
        "display_name": "OmniVoice — Multilingual Zero-Shot TTS (600+ Languages)",
        "description": "k2-fsa massively multilingual zero-shot TTS supporting attribute-based voice design and non-verbal expression tags.",
        "gguf_url": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/OmniVoice-GGUF/omnivoice-f16.gguf",
        "gguf_filename": "omnivoice-f16.gguf",
        "languages": ["en", "zh", "es", "fr", "de", "ja", "ko", "ar", "hi", "pt", "ru", "it"],
        "inline_tags": [
            "[laughter]",
            "[sigh]",
            "[confirmation-en]",
            "[question-en]",
            "[question-ah]",
            "[question-oh]",
            "[surprise-ah]",
            "[surprise-oh]",
            "[surprise-wa]",
            "[dissatisfaction-hnn]",
        ],
        "instruct_template": "{speaker_type}, natural pace, clear articulation, clean studio recording.",
        "sample_text_prompt": (
            "You are a TTS text dataset generator. Generate a single spoken paragraph (3-5 sentences) "
            "for a voice described as: {speaker_type}.\n"
            "The text must be natural, conversational, and suitable for multilingual speech synthesis.\n"
            "You can optionally include 1 or 2 natural non-verbal expression tags from this supported set: {tags}.\n"
            "Example: 'I couldn't believe they actually pulled it off [laughter]! It was such a relief [sigh].'\n"
            "Output ONLY the spoken text. Do not include quotes, markdown formatting, or explanations."
        ),
    },
    "qwen3_tts_voicedesign": {
        "family": "qwen3_tts",
        "task": "vdes",
        "instruct_style": "instruct",
        "display_name": "Qwen3-TTS 1.7B — VoiceDesign (10 Languages)",
        "description": "Alibaba Qwen3-TTS 1.7B VoiceDesign generating custom voices from rich natural-language persona descriptions.",
        "gguf_url": "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf",
        "gguf_filename": "qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf",
        "languages": ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        "inline_tags": [],  # VoiceDesign uses natural language persona description in instruct
        "instruct_template": "{speaker_type}, clear articulation, warm authentic pitch, expressive delivery.",
        "sample_text_prompt": (
            "You are a TTS text dataset generator. Generate a single spoken paragraph (3-5 sentences) "
            "for a voice described as: {speaker_type}.\n"
            "The text must be natural and conversational with rhythmic variety, perfect for speech synthesis.\n"
            "Output ONLY the spoken text. Do not include quotes, markdown formatting, or explanations."
        ),
    },
}


def build_instruct(model_key: str, speaker_type: str) -> str:
    """Build the instruct string for audiocpp_cli."""
    return MODELS[model_key]["instruct_template"].format(speaker_type=speaker_type)


def build_text_gen_prompt(model_key: str, speaker_type: str) -> str:
    """Build the prompt to send to a small LLM for generating tagged text."""
    m = MODELS[model_key]
    tags_str = ", ".join(m["inline_tags"]) if m["inline_tags"] else "none"
    return m["sample_text_prompt"].format(speaker_type=speaker_type, tags=tags_str)


def build_cli_args(
    cli: str,
    model_key: str,
    model_path: str,
    backend: str,
    text: str,
    speaker_type: str,
    output_path: str,
    device_index: int = -1,
) -> list[str]:
    """Build the exact command-line arguments for audiocpp_cli."""
    m = MODELS[model_key]
    instruct = build_instruct(model_key, speaker_type)
    task = m.get("task", "tts")
    family = m["family"]

    cmd = [
        cli,
        "--task", task,
        "--family", family,
        "--model", model_path,
        "--backend", backend,
        "--text", text,
    ]

    if device_index >= 0:
        cmd.extend(["--device", str(device_index)])

    if m.get("instruct_style") == "request_option":
        cmd.extend(["--request-option", f"instruct={instruct}"])
    else:
        cmd.extend(["--instruct", instruct])

    cmd.extend(["--out", output_path])
    return cmd

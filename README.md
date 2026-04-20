# HSE Emotion Assistant

Multimodal emotion assistant with visual emotion analysis, Qwen realtime voice interaction, and local RAG knowledge base.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![LLM](https://img.shields.io/badge/API-DashScope%20%2F%20Qwen-green)

[中文文档](README_ZH.md) | [English Guide](README_EN.md)

## Highlights

- Visual emotion analysis with local camera input
- Realtime voice assistant based on Qwen Omni / ASR / TTS models
- Local knowledge-base retrieval with embedding and rerank support
- Desktop GUI for text chat, voice mode, and environment setup

## Quick Start

```powershell
conda create -n hse_emotion python=3.10 -y
conda activate hse_emotion
pip install -r requirements_stable.txt
pip install -r requirements_llm_ui_v0.1.3.txt
copy .env.example .env
python -m hsemotion_ui
```

Before running the app, edit `.env` and set:

```env
DASHSCOPE_API_KEY=your_key_here
```

## Required Models

This project is currently configured around these DashScope / Qwen model IDs:

- `qwen3.5-plus`
- `qwen3-omni-flash-realtime`
- `qwen3-asr-flash-realtime`
- `qwen3-tts-flash-realtime`
- `text-embedding-v4`
- `qwen3-rerank`

## Important Notes

- No API key is included in this repository
- You must provide your own DashScope / Qwen API key
- Runtime artifacts such as `.env`, logs, vector databases, and model caches are intentionally excluded

## Documentation

- For full Chinese deployment instructions, see [README_ZH.md](README_ZH.md)
- For full English deployment instructions, see [README_EN.md](README_EN.md)
- For environment setup with a small installer UI, run `tools\run_env_installer.bat`

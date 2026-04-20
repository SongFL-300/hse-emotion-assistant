# HSE Emotion Assistant Open-Source Edition

## 1. Overview

This project is a multimodal emotional support assistant for Chinese-language interaction scenarios. It combines:

- local visual emotion analysis
- structured emotion context injection
- professional knowledge-base retrieval (RAG)
- text chat
- realtime voice conversation

This open-source edition does not include any default API key. You must use your own DashScope / Qwen API.

## 2. Environment

- OS: Windows recommended
- Python: `3.10.x` recommended
- Conda environment recommended
- Camera and microphone are required for visual / voice modes

Recommended setup:

```powershell
conda create -n hse_emotion python=3.10 -y
conda activate hse_emotion
```

## 3. Fresh Deployment

### 3.1 Enter the project directory

```powershell
cd HSE-Emotion-Assistant-OpenSource
```

### 3.2 Install dependencies

```powershell
pip install -r requirements_stable.txt
pip install -r requirements_llm_ui_v0.1.3.txt
```

Optional visual backends:

```powershell
pip install -r requirements_optional.clean.txt
```

### 3.3 Configure your API

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Then edit `.env` and fill in at least:

```env
DASHSCOPE_API_KEY=your_key_here
```

If you do not configure your API key, the app will fail at startup by design.

## 4. Required Qwen / DashScope Models

The current project configuration uses these model IDs:

- text chat: `qwen3.5-plus`
- realtime voice assistant: `qwen3-omni-flash-realtime`
- realtime ASR: `qwen3-asr-flash-realtime`
- realtime TTS: `qwen3-tts-flash-realtime`
- embedding: `text-embedding-v4`
- reranker: `qwen3-rerank`

Your DashScope account must have access to the models you plan to use.

## 5. Where to Get the API Key

You need an API key from Alibaba Cloud Bailian / DashScope.

Typical flow:

1. Open the Alibaba Cloud Bailian / DashScope console
2. Sign in with your own account
3. Create or manage an API key in the API / application section
4. Paste that key into `.env` as `DASHSCOPE_API_KEY`

If you use an international endpoint, also set:

```env
DASHSCOPE_BASE_HTTP_API_URL=https://dashscope-intl.aliyuncs.com/api/v1
```

## 6. Run the Project

```powershell
conda activate hse_emotion
cd HSE-Emotion-Assistant-OpenSource
python -m hsemotion_ui
```

Or use the setup helper:

```powershell
tools\run_env_installer.bat
```

The setup helper can:

- install / repair dependencies
- optionally install DeepFace / LibreFace
- write your own API config to `.env`
- run an environment check

## 7. Common Issues

### 7.1 `Missing DASHSCOPE_API_KEY`

Your API is not configured. Check whether `.env` exists and whether `DASHSCOPE_API_KEY` is empty.

### 7.2 DeepFace / LibreFace not installed

These are optional backends. The project can still run without them, but some visual features will be limited.

### 7.3 LibreFace is slow on first run

It may download weights the first time. That is expected.

## 8. What Was Removed in the Open-Source Release

- default API key
- runtime `.env`
- logs
- local RAG database
- downloaded model caches / weights

## 9. Repository Layout

- `hsemotion_ui`: desktop UI entry
- `hsemotion_llm`: chat, speech, retrieval, config loading
- `face_mesh_detector`: face mesh and alignment
- `emotion_analyzer`: compatibility emotion backends
- `Emotion-recognition`: Mini-Xception model assets
- `tools`: environment setup and installer scripts

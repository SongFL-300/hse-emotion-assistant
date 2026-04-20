# HSE Emotion Assistant 开源版

## 1. 项目简介

这是一个面向中文情感陪伴场景的多模态助手项目。它把以下能力组合到一个桌面应用中：

- 本地视觉情绪分析
- 结构化情绪上下文注入
- 专业知识库 RAG
- 文本对话
- 纯语音实时对话

本开源版不包含任何默认 API Key。你必须接入你自己的 DashScope / Qwen API 才能运行。

## 2. 运行环境

- 操作系统：Windows 优先
- Python：建议 `3.10.x`
- 建议使用 Conda 独立环境
- 摄像头、麦克风在语音/视觉模式下需要可用

推荐：

```powershell
conda create -n hse_emotion python=3.10 -y
conda activate hse_emotion
```

## 3. 从零部署

### 3.1 克隆或进入项目目录

```powershell
cd HSE-Emotion-Assistant-OpenSource
```

### 3.2 安装核心依赖

```powershell
pip install -r requirements_stable.txt
pip install -r requirements_llm_ui_v0.1.3.txt
```

如果你还想启用可选视觉后端：

```powershell
pip install -r requirements_optional.clean.txt
```

### 3.3 配置 API

复制 `.env.example` 为 `.env`：

```powershell
copy .env.example .env
```

然后编辑 `.env`，至少填写：

```env
DASHSCOPE_API_KEY=你的Key
```

如果不配置 API，项目会在启动时直接报错并阻止运行。这是开源版的预期行为。

## 4. 需要的千问 / DashScope 模型

本项目当前默认会用到这些模型：

- 文本对话：`qwen3.5-plus`
- 纯语音主模型：`qwen3-omni-flash-realtime`
- 实时语音转写：`qwen3-asr-flash-realtime`
- 实时语音合成：`qwen3-tts-flash-realtime`
- 向量 embedding：`text-embedding-v4`
- RAG rerank：`qwen3-rerank`

你至少需要确保你的 DashScope 账户对这些模型具备调用权限。

## 5. API 去哪里获取

你需要的是阿里云百炼 / DashScope 的 API Key。

一般路径：

1. 打开阿里云百炼 / DashScope 控制台
2. 登录你自己的账号
3. 在 API Key 管理或应用调用页面创建 Key
4. 把生成的 Key 填入 `.env` 的 `DASHSCOPE_API_KEY`

如果你使用国际站或非默认地域，可以额外配置：

```env
DASHSCOPE_BASE_HTTP_API_URL=https://dashscope-intl.aliyuncs.com/api/v1
```

## 6. 如何运行

```powershell
conda activate hse_emotion
cd HSE-Emotion-Assistant-OpenSource
python -m hsemotion_ui
```

如果你更习惯用安装器：

```powershell
tools\run_env_installer.bat
```

安装器支持：

- 安装 / 修复依赖
- 可选安装 DeepFace / LibreFace
- 写入你自己的 API 到 `.env`
- 环境检测

## 7. 常见问题

### 7.1 启动时报 `Missing DASHSCOPE_API_KEY`

说明你还没有配置自己的 API。检查 `.env` 是否存在，以及 `DASHSCOPE_API_KEY` 是否为空。

### 7.2 DeepFace / LibreFace 未安装

它们属于可选后端，不安装也能运行项目，但部分视觉能力会受影响。

### 7.3 第一次运行 LibreFace 很慢

首次运行时可能需要下载权重，这是正常现象。

## 8. 开源版说明

开源版做了这些处理：

- 删除默认 API
- 不包含 `.env`
- 不包含日志、RAG 数据库、缓存权重
- 保留项目代码、依赖、安装器与文档

## 9. 仓库结构

- `hsemotion_ui`：桌面前端入口
- `hsemotion_llm`：聊天、语音、RAG、配置加载
- `face_mesh_detector`：人脸关键点与对齐
- `emotion_analyzer`：兼容情绪后端
- `Emotion-recognition`：Mini-Xception 模型资源
- `tools`：环境安装与配置脚本

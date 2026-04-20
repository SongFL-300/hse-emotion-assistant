from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .dotenv import load_dotenv


@dataclass(frozen=True)
class DashScopeConfig:
    api_key: str
    model: str = "qwen3.5-plus"
    base_http_api_url: str | None = None
    incremental_output: bool = True
    enable_thinking: bool = True
    thinking_budget: int = 1999


@dataclass(frozen=True)
class EmotionConfig:
    enabled: bool = True
    camera_index: int = 0
    sample_fps: float = 60.0
    engine: str = "libreface"  # libreface | tf_mini_xception | deepface
    mediapipe_offline_mode: bool = True
    window_count: int = 4
    blink_enabled: bool = True
    au_enabled: bool = True
    context_compaction_enabled: bool = True
    rmes_enabled: bool = True
    rmes_mode: str = "experiment"
    rmes_acceptance_gate: bool = True
    greeting_text: str = "你好 (^_^)，我是智能表情变化以及微表情等数据结合的心理聊天助手！"
    greeting_voice_text: str = "你好，我是你的情绪聊天助手。"


@dataclass(frozen=True)
class RagConfig:
    enabled: bool = False
    db_path: Path = Path(".rag") / "rag_store.sqlite3"
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024
    rerank_model: str = "qwen3-rerank"
    top_k: int = 6
    chunk_chars: int = 600
    chunk_overlap: int = 80


@dataclass(frozen=True)
class RealtimeAsrConfig:
    enabled: bool = False
    model: str = "gummy-realtime-v1"
    sample_rate: int = 16000


@dataclass(frozen=True)
class RealtimeTtsConfig:
    enabled: bool = False
    base_ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime"
    voice: str = "Cherry"
    sample_rate: int = 24000
    language_type: str = "Auto"


@dataclass(frozen=True)
class OmniRealtimeConfig:
    enabled: bool = True
    model: str = "qwen3-omni-flash-realtime"
    base_ws_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    voice: str = "Cherry"
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    enable_video: bool = True
    video_fps: float = 1.0
    local_barge_in_enabled: bool = True
    local_barge_in_rms: float = 900.0
    local_barge_in_cooldown_s: float = 0.8
    instructions: str = (
        "你是一个实时全模态情绪陪伴助手。充分利用语音语气、用户说话节奏和摄像头画面中的表情线索，"
        "自然、直接地回应。优先短句、口语化、低延迟，不要机械复述检测结果。"
    )
    voice_instructions: str = (
        "当前是纯语音对话。请尽量短答：默认 1 到 2 句，优先一句话说完。"
        "除非用户明确追问或要求展开，否则不要长解释、不要分点、不要铺垫。"
        "如果只是接话、安抚、确认、追问，尽量控制在 8 到 24 个中文字符的自然口语范围。"
    )
    transcription_model: str = "qwen3-asr-flash-realtime"


@dataclass(frozen=True)
class AppConfig:
    dashscope: DashScopeConfig
    emotion: EmotionConfig = EmotionConfig()
    rag: RagConfig = RagConfig()
    asr: RealtimeAsrConfig = RealtimeAsrConfig()
    tts: RealtimeTtsConfig = RealtimeTtsConfig()
    omni: OmniRealtimeConfig = OmniRealtimeConfig()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() not in {"0", "false", "False", "no", "NO", "off", "OFF", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip())


def load_config_from_env() -> AppConfig:
    dotenv_path = load_dotenv(override=False)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        if dotenv_path is None:
            raise ValueError("Missing DASHSCOPE_API_KEY（未找到 .env；请在系统环境变量或项目根目录 .env 中设置）")
        raise ValueError(
            f"Missing DASHSCOPE_API_KEY（已加载 .env: {dotenv_path}，但未读取到有效值；请检查是否为空、拼写是否正确）"
        )

    model = os.getenv("QWEN_MODEL", "qwen3.5-plus")
    base_http_api_url = os.getenv("DASHSCOPE_BASE_HTTP_API_URL") or None

    return AppConfig(
        dashscope=DashScopeConfig(
            api_key=api_key,
            model=model,
            base_http_api_url=base_http_api_url,
            enable_thinking=_env_bool("QWEN_ENABLE_THINKING", True),
            thinking_budget=min(1999, max(0, _env_int("QWEN_THINKING_BUDGET", 1999))),
        ),
        emotion=EmotionConfig(
            enabled=_env_bool("HSEMOTION_EMOTION_ENABLED", True),
            camera_index=_env_int("HSEMOTION_CAMERA_INDEX", 0),
            sample_fps=_env_float("HSEMOTION_EMOTION_FPS", 60.0),
            engine=_env_str("HSEMOTION_EMOTION_ENGINE", "libreface"),
            mediapipe_offline_mode=_env_bool("HSEMOTION_MEDIAPIPE_OFFLINE", True),
            window_count=_env_int("HSEMOTION_WINDOW_COUNT", 4),
            blink_enabled=_env_bool("HSEMOTION_BLINK_ENABLED", True),
            au_enabled=_env_bool("HSEMOTION_AU_ENABLED", True),
            context_compaction_enabled=_env_bool("HSEMOTION_CONTEXT_COMPACTION_ENABLED", True),
            rmes_enabled=_env_bool("HSEMOTION_RMES_ENABLED", True),
            rmes_mode=_env_str("HSEMOTION_RMES_MODE", "experiment"),
            rmes_acceptance_gate=_env_bool("HSEMOTION_RMES_ACCEPTANCE_GATE", True),
            greeting_text=_env_str("HSEMOTION_GREETING_TEXT", EmotionConfig.greeting_text),
            greeting_voice_text=_env_str("HSEMOTION_GREETING_VOICE_TEXT", EmotionConfig.greeting_voice_text),
        ),
        rag=RagConfig(
            enabled=_env_bool("HSEMOTION_RAG_ENABLED", False),
            db_path=_env_path("HSEMOTION_RAG_DB_PATH", RagConfig.db_path),
            embedding_model=_env_str("HSEMOTION_EMBEDDING_MODEL", RagConfig.embedding_model),
            embedding_dimension=_env_int("HSEMOTION_EMBEDDING_DIM", RagConfig.embedding_dimension),
            rerank_model=_env_str("HSEMOTION_RERANK_MODEL", RagConfig.rerank_model),
            top_k=_env_int("HSEMOTION_RAG_TOP_K", RagConfig.top_k),
            chunk_chars=_env_int("HSEMOTION_RAG_CHUNK_CHARS", RagConfig.chunk_chars),
            chunk_overlap=_env_int("HSEMOTION_RAG_CHUNK_OVERLAP", RagConfig.chunk_overlap),
        ),
        asr=RealtimeAsrConfig(
            enabled=_env_bool("HSEMOTION_ASR_ENABLED", False),
            model=_env_str("HSEMOTION_ASR_MODEL", RealtimeAsrConfig.model),
            sample_rate=_env_int("HSEMOTION_ASR_SAMPLE_RATE", RealtimeAsrConfig.sample_rate),
        ),
        tts=RealtimeTtsConfig(
            enabled=_env_bool("HSEMOTION_TTS_ENABLED", False),
            base_ws_url=_env_str("HSEMOTION_TTS_WS_URL", RealtimeTtsConfig.base_ws_url),
            voice=_env_str("HSEMOTION_TTS_VOICE", RealtimeTtsConfig.voice),
            sample_rate=_env_int("HSEMOTION_TTS_SAMPLE_RATE", RealtimeTtsConfig.sample_rate),
            language_type=_env_str("HSEMOTION_TTS_LANGUAGE", RealtimeTtsConfig.language_type),
        ),
        omni=OmniRealtimeConfig(
            enabled=_env_bool("HSEMOTION_OMNI_ENABLED", True),
            model=_env_str("HSEMOTION_OMNI_MODEL", OmniRealtimeConfig.model),
            base_ws_url=_env_str("HSEMOTION_OMNI_WS_URL", OmniRealtimeConfig.base_ws_url),
            voice=_env_str("HSEMOTION_OMNI_VOICE", OmniRealtimeConfig.voice),
            input_sample_rate=_env_int("HSEMOTION_OMNI_INPUT_SAMPLE_RATE", OmniRealtimeConfig.input_sample_rate),
            output_sample_rate=_env_int("HSEMOTION_OMNI_OUTPUT_SAMPLE_RATE", OmniRealtimeConfig.output_sample_rate),
            enable_video=_env_bool("HSEMOTION_OMNI_ENABLE_VIDEO", OmniRealtimeConfig.enable_video),
            video_fps=_env_float("HSEMOTION_OMNI_VIDEO_FPS", OmniRealtimeConfig.video_fps),
            local_barge_in_enabled=_env_bool(
                "HSEMOTION_OMNI_LOCAL_BARGE_IN_ENABLED", OmniRealtimeConfig.local_barge_in_enabled
            ),
            local_barge_in_rms=_env_float("HSEMOTION_OMNI_LOCAL_BARGE_IN_RMS", OmniRealtimeConfig.local_barge_in_rms),
            local_barge_in_cooldown_s=_env_float(
                "HSEMOTION_OMNI_LOCAL_BARGE_IN_COOLDOWN_S", OmniRealtimeConfig.local_barge_in_cooldown_s
            ),
            instructions=_env_str("HSEMOTION_OMNI_INSTRUCTIONS", OmniRealtimeConfig.instructions),
            voice_instructions=_env_str("HSEMOTION_OMNI_VOICE_INSTRUCTIONS", OmniRealtimeConfig.voice_instructions),
            transcription_model=_env_str("HSEMOTION_OMNI_TRANSCRIPTION_MODEL", OmniRealtimeConfig.transcription_model),
        ),
    )

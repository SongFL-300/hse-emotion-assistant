from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import customtkinter as ctk  # type: ignore
except Exception:  # noqa: BLE001
    ctk = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]

from hsemotion_llm.chat_orchestrator import ChatHooks, EmotionAwareChat
from hsemotion_llm.config import AppConfig, load_config_from_env
from hsemotion_llm.emotion_queue import EmotionQueue
from hsemotion_llm.emotion.visual_tracker import VisualEmotionTracker
from hsemotion_llm.rag.indexer import RagIndexer
from hsemotion_llm.rag.retriever import RagRetriever
from hsemotion_llm.rag.store import RagStore
from hsemotion_llm.session_log import ChatSessionLogger
from hsemotion_llm.speech.omni_realtime import (
    OmniAssistantTranscriptEvent,
    OmniRealtimeSession,
    OmniUserTranscriptEvent,
)
from hsemotion_llm.speech.asr_realtime import RealtimeAsr, AsrEvent
from hsemotion_llm.speech.tts_realtime import RealtimeTts, TtsTextChunker


APP_BG = "#f3f5f7"
PANEL_BG = "#fbfcfd"
PANEL_ALT_BG = "#eef2f5"
SURFACE_BG = "#ffffff"
VOICE_BG = "#102126"
TEXT_MUTED = "#61707d"
TEXT_PRIMARY = "#16202a"
TEXT_INVERSE = "#f4f8fb"
ACCENT = "#1f8a70"
ACCENT_SOFT = "#dff2ec"
ACCENT_DEEP = "#166c58"
BORDER = "#d6dde4"
DANGER = "#b54747"
USER_BUBBLE = "#dff2ec"
ASSISTANT_BUBBLE = "#ffffff"
SYSTEM_BUBBLE = "#eef2f5"


@dataclass(frozen=True)
class _UiEvent:
    type: str
    payload: object | None = None


def _safe_import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return None
    return sd


def _format_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s//60:02d}:{s%60:02d}"


def _tone_hint(emotion: str | None) -> str:
    e = (emotion or "neutral").lower().strip()
    if e in {"sad", "fear"}:
        return "更温和、更慢一点，先安抚再追问"
    if e == "angry":
        return "更稳、更短句，先接住情绪再讨论事实"
    if e == "happy":
        return "更轻松、更互动，适当跟随节奏"
    if e == "surprise":
        return "先确认发生了什么，再给选择题式建议"
    return "自然、不过度分析"


def _build_suggestions(emotion: str | None) -> list[str]:
    e = (emotion or "neutral").lower().strip()
    if e == "sad":
        return [
            "我有点难受，想找个人聊聊。",
            "今天心情不太好，你能陪我说说话吗？",
            "我想整理下最近的压力，你能帮我一起梳理吗？",
        ]
    if e == "angry":
        return [
            "我有点烦/生气，先让我把事情说完可以吗？",
            "我想冷静一下，你能帮我把问题拆小一点吗？",
            "我现在情绪有点上来，你能先听我讲讲发生了什么吗？",
        ]
    if e == "fear":
        return [
            "我有点紧张/担心，想确认一下我这样想是否合理。",
            "我对接下来要做的事没把握，你能帮我制定一个小计划吗？",
            "我有点慌，你能先陪我做个快速的情绪稳定吗？",
        ]
    if e == "happy":
        return [
            "我今天挺开心的，想分享一件小事。",
            "我们来聊点轻松的：给我 3 个有趣话题？",
            "我想把好心情延续下去，你能推荐个小挑战吗？",
        ]
    if e == "surprise":
        return [
            "我刚刚有点震惊，你能帮我判断这件事正常吗？",
            "我遇到一个意外情况，想听听你的看法。",
            "我有点没反应过来，你能先帮我复盘一下吗？",
        ]
    return [
        "我们从一个简单问题开始：你今天过得怎么样？",
        "如果你不知道聊什么：说说你现在最在意的一件事？",
        "想不出话题也没关系：你更想被倾听还是一起解决问题？",
    ]


_PROFESSIONAL_KB_PDFS = [
    Path(r"C:\Users\Administrator\Downloads\9787117311380-chi.pdf"),
    Path(r"C:\Users\Administrator\Downloads\9789240003910-chi.pdf"),
    Path(r"C:\Users\Administrator\Downloads\mhgap-chi(OCR).pdf"),
    Path(r"C:\Users\Administrator\Downloads\WHO-MSD-MER-16.2-chi.pdf"),
    Path(r"C:\Users\Administrator\Downloads\WHO-MSD-MER-16.4-chi.pdf"),
]


class EmotionChatTkApp:
    def __init__(self, config: AppConfig):
        self._config = config

        if ctk is not None:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("green")
            self._root = ctk.CTk()
        else:
            self._root = tk.Tk()
        self._root.title("HSE Emotion Assistant")
        self._root.geometry("1180x820")
        self._root.minsize(1080, 760)
        try:
            self._root.configure(bg=APP_BG)
        except Exception:
            pass
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._style = ttk.Style(self._root)
        self._configure_theme()

        self._ui_queue: queue.Queue[_UiEvent] = queue.Queue()
        self._llm_thread: threading.Thread | None = None
        self._closing = False

        # Emotion queue (tracks notable changes around turns)
        self._emotion_queue_lock = threading.Lock()
        self._emotion_queue = EmotionQueue()
        self._emoqueue_last_idle_ts = 0.0

        # Pure voice chat mode
        self._voice_only_mode = False
        self._voice_asr_buffer = ""
        self._voice_asr_last_update_ts = 0.0
        self._voice_asr_last_sent = ""
        self._voice_asr_paused_for_reply = False
        self._voice_reply_text = ""
        self._last_omni_error_text = ""
        self._last_omni_error_ts = 0.0
        self._omni_reply_window_open = False
        self._pending_enter_voice_only = False
        self._knowledge_status_var = tk.StringVar(master=self._root, value="专业知识库：未触发")

        # Toggles
        self._rag_enabled = tk.BooleanVar(master=self._root, value=bool(config.rag.enabled))
        self._rag_mode = tk.StringVar(master=self._root, value="auto")
        self._tts_enabled = tk.BooleanVar(master=self._root, value=bool(config.tts.enabled))
        # “会打断的”动态调节：默认开启，但只在大变化时触发
        self._realtime_adjust = tk.BooleanVar(master=self._root, value=True)
        self._adjust_threshold = tk.DoubleVar(master=self._root, value=0.45)
        self._backend_var = tk.StringVar(master=self._root, value=str(config.emotion.engine))

        # Local vision emotion tracker
        self._emotion_tracker = VisualEmotionTracker(config.emotion)
        self._emotion_tracker.prepare_async()

        # RAG
        self._rag_store = RagStore(config.rag.db_path)
        self._rag_indexer = RagIndexer(dashscope=config.dashscope, rag=config.rag, store=self._rag_store)
        self._rag_retriever = RagRetriever(dashscope=config.dashscope, rag=config.rag, store=self._rag_store)

        hooks = ChatHooks(
            get_emotion_summary=self._get_emotion_summary_for_prompt,
            get_emotion_trace=self._get_emotion_trace,
            get_emotion_status=self._emotion_tracker.get_status,
            get_rag_snippets=self._build_rag_snippets_for_query,
            get_rag_snippets_with_meta=self._build_rag_snippets_with_meta,
        )
        self._chat = EmotionAwareChat(config, hooks=hooks)

        # ASR
        self._asr = RealtimeAsr(dashscope=config.dashscope, asr=config.asr)
        self._asr_running = False
        self._omni_session = OmniRealtimeSession(
            dashscope=config.dashscope,
            omni=config.omni,
            get_frame_bgr=self._emotion_tracker.get_latest_raw_frame_bgr,
            get_visual_signal=self._emotion_tracker.get_structured_signal,
            on_user_transcript=self._on_omni_user_transcript,
            on_assistant_transcript=self._on_omni_assistant_transcript,
            on_state=self._on_omni_state,
            on_error=self._on_omni_error,
        )

        # Logging / history
        self._logger: ChatSessionLogger | None = None
        try:
            log_dir = Path(os.getenv("HSEMOTION_LOG_DIR", ".hsemotion_logs"))
            self._logger = ChatSessionLogger(config=config, base_dir=log_dir)
        except Exception:
            self._logger = None

        # TTS progress state
        self._tts_lock = threading.Lock()
        self._tts_play_s = 0.0
        self._tts_text_chars = 0
        self._tts_sampling_stop = threading.Event()
        self._tts_sampler_thread: threading.Thread | None = None
        self._tts_active = False

        # Current assistant streaming bubble
        self._assistant_label: tk.Label | None = None
        self._assistant_text = ""

        self._build_ui()
        self._insert_greeting()
        self._root.after(30, self._drain_ui_queue)
        self._root.after(120, self._check_voice_autosend)
        self._root.after(250, self._update_emotion_panel)
        self._root.after(16, self._update_video_preview)

    def run(self) -> None:
        self._root.mainloop()

    def _configure_theme(self) -> None:
        try:
            if "clam" in self._style.theme_names():
                self._style.theme_use("clam")
        except Exception:
            pass
        self._style.configure("App.TFrame", background=APP_BG)
        self._style.configure("Card.TFrame", background=PANEL_BG, relief="flat")
        self._style.configure("Panel.TLabelframe", background=PANEL_BG, borderwidth=1, relief="solid")
        self._style.configure("Panel.TLabelframe.Label", background=PANEL_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        self._style.configure("App.TLabel", background=APP_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 10))
        self._style.configure("Muted.TLabel", background=APP_BG, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        self._style.configure("CardTitle.TLabel", background=PANEL_BG, foreground=TEXT_PRIMARY, font=("Segoe UI Semibold", 11))
        self._style.configure("CardCaption.TLabel", background=PANEL_BG, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        self._style.configure("Accent.Horizontal.TProgressbar", troughcolor=PANEL_ALT_BG, background=ACCENT, bordercolor=PANEL_ALT_BG, lightcolor=ACCENT, darkcolor=ACCENT)
        self._style.configure("App.Vertical.TScrollbar", background=SURFACE_BG, troughcolor=APP_BG, bordercolor=APP_BG, arrowcolor=TEXT_MUTED)
        self._style.map("App.Vertical.TScrollbar", background=[("active", SURFACE_BG)])
        self._style.configure(
            "App.TCombobox",
            fieldbackground=SURFACE_BG,
            background=SURFACE_BG,
            foreground=TEXT_PRIMARY,
            arrowcolor=TEXT_PRIMARY,
            bordercolor=BORDER,
            lightcolor=SURFACE_BG,
            darkcolor=SURFACE_BG,
            insertcolor=TEXT_PRIMARY,
        )

    def _make_frame(
        self,
        parent: tk.Widget,
        *,
        bg: str,
        border: bool = False,
        corner_radius: int = 18,
        pack_kwargs: dict | None = None,
    ):
        if ctk is not None:
            frame = ctk.CTkFrame(
                parent,
                fg_color=bg,
                corner_radius=corner_radius,
                border_width=1 if border else 0,
                border_color=BORDER,
            )
        else:
            frame = tk.Frame(parent, bg=bg, highlightthickness=1 if border else 0, highlightbackground=BORDER)
        if pack_kwargs:
            frame.pack(**pack_kwargs)
        return frame

    def _make_label(
        self,
        parent: tk.Widget,
        *,
        text: str | None = None,
        textvariable=None,
        bg: str,
        fg: str,
        font: tuple[str, int] | tuple[str, int, str],
        **kwargs,
    ):
        if ctk is not None:
            label = ctk.CTkLabel(parent, text=text or "", textvariable=textvariable, fg_color="transparent", text_color=fg, font=font, **kwargs)
        else:
            label = tk.Label(parent, text=text, textvariable=textvariable, bg=bg, fg=fg, font=font, **kwargs)
        return label

    def _make_checkbutton(self, parent: tk.Widget, *, text: str, variable, bg: str):
        if ctk is not None:
            return ctk.CTkCheckBox(
                parent,
                text=text,
                variable=variable,
                fg_color=ACCENT,
                hover_color=ACCENT_DEEP,
                border_color=BORDER,
                text_color=TEXT_PRIMARY,
                checkmark_color=TEXT_INVERSE,
                font=("Segoe UI", 10),
            )
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg=bg,
            fg=TEXT_PRIMARY,
            activebackground=bg,
            activeforeground=TEXT_PRIMARY,
            selectcolor=SURFACE_BG,
        )

    def _make_button(
        self,
        parent: tk.Widget,
        *,
        text: str,
        command,
        primary: bool = False,
        width: int | None = None,
    ) -> tk.Button:
        if ctk is not None:
            return ctk.CTkButton(
                parent,
                text=text,
                command=command,
                width=width * 10 if width else 0,
                height=38,
                corner_radius=12,
                fg_color=ACCENT if primary else SURFACE_BG,
                hover_color=ACCENT_DEEP if primary else "#243244",
                border_width=0 if primary else 1,
                border_color=ACCENT if primary else BORDER,
                text_color=TEXT_INVERSE if primary else TEXT_PRIMARY,
                font=("Segoe UI Semibold", 10),
            )
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=ACCENT if primary else SURFACE_BG,
            fg=TEXT_INVERSE if primary else TEXT_PRIMARY,
            activebackground=ACCENT_DEEP if primary else PANEL_ALT_BG,
            activeforeground=TEXT_INVERSE if primary else TEXT_PRIMARY,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=("Segoe UI Semibold", 10),
            highlightthickness=1,
            highlightbackground=ACCENT if primary else BORDER,
            highlightcolor=ACCENT,
            cursor="hand2",
        )

    def _make_section_label(self, parent: tk.Widget, text: str, *, bg: str = PANEL_BG) -> tk.Label:
        return self._make_label(parent, text=text, bg=bg, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 11), anchor="w")

    # UI
    def _build_ui(self) -> None:
        top = self._make_frame(self._root, bg=APP_BG, pack_kwargs={"side": tk.TOP, "fill": tk.X, "padx": 16, "pady": (16, 10)})

        title_box = self._make_frame(top, bg=APP_BG, pack_kwargs={"side": tk.LEFT, "fill": tk.X, "expand": True})
        self._make_label(
            title_box,
            text="HSE Emotion Assistant",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 18),
            anchor="w",
        ).pack(anchor="w")
        self._make_label(
            title_box,
            text="面向情感陪伴场景的多模态助手，整合视觉情绪、知识库与语音对话。",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        self._emotion_var = tk.StringVar(value="本地识别模型：初始化中…")
        self._emotion_err_var = tk.StringVar(value="")
        status_card = self._make_frame(top, bg=PANEL_BG, border=True, pack_kwargs={"side": tk.RIGHT})
        self._make_label(status_card, text="运行状态", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=(12, 0))
        self._make_label(status_card, textvariable=self._emotion_var, bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 10), anchor="w", justify=tk.LEFT).pack(anchor="w", padx=16, pady=(2, 0))
        self._make_label(status_card, textvariable=self._emotion_err_var, bg=PANEL_BG, fg=DANGER, font=("Segoe UI", 9), anchor="w", justify=tk.LEFT).pack(anchor="w", padx=16, pady=(2, 12))

        body = self._make_frame(self._root, bg=APP_BG, pack_kwargs={"side": tk.TOP, "fill": tk.BOTH, "expand": True, "padx": 16, "pady": (0, 16)})

        self._left = self._make_frame(body, bg=APP_BG, pack_kwargs={"side": tk.LEFT, "fill": tk.BOTH, "expand": True})

        right_shell = self._make_frame(body, bg=APP_BG)
        right_shell.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        right_shell.pack_propagate(False)
        right_shell.configure(width=340)

        self._right_canvas = tk.Canvas(right_shell, width=340, highlightthickness=0, bd=0, bg=APP_BG)
        self._right_scroll = ttk.Scrollbar(right_shell, orient="vertical", style="App.Vertical.TScrollbar", command=self._right_canvas.yview)
        self._right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._right_canvas.configure(yscrollcommand=self._right_scroll.set)

        right = tk.Frame(self._right_canvas, width=320, bg=APP_BG, highlightthickness=0, bd=0)
        self._right_window = self._right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind(
            "<Configure>",
            lambda _e: self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all")),
        )
        self._right_canvas.bind(
            "<Configure>",
            lambda e: self._right_canvas.itemconfigure(self._right_window, width=max(260, e.width - 2)),
        )
        self._right_panel_shell = right_shell

        # Video preview
        video_box = self._make_frame(self._left, bg=PANEL_BG, border=True)
        video_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 12))
        try:
            video_box.configure(height=290)
        except Exception:
            pass
        video_box.pack_propagate(False)
        self._make_label(
            video_box,
            text="实时视频与表情检测",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI Semibold", 11),
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X, padx=14, pady=(12, 12))
        self._video_label = tk.Label(
            video_box,
            text="视频预览：情绪监测未启动",
            bg="#0f1519",
            fg="#d7e3eb",
            anchor="center",
        )
        self._video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self._video_photo = None

        # Stack (chat / voice)
        self._stack = self._make_frame(self._left, bg=APP_BG, pack_kwargs={"side": tk.TOP, "fill": tk.BOTH, "expand": True})

        self._chat_frame = tk.Frame(self._stack, bg=PANEL_BG, highlightthickness=1, highlightbackground=BORDER)
        self._voice_frame = tk.Frame(self._stack, bg=VOICE_BG, highlightthickness=1, highlightbackground="#1b353c")
        self._chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_chat_view(parent=self._chat_frame)
        self._build_voice_view(parent=self._voice_frame)

        # Right controls
        control_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(control_card, "主控制台").pack(anchor="w")
        self._make_label(control_card, text="情绪检测、知识库、语音与模式切换。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        self._monitor_btn = self._make_button(control_card, text="提前启动情绪监测", command=self._toggle_monitoring, primary=True)
        self._monitor_btn.pack(fill=tk.X, pady=(6, 0))
        self._monitor_btn.configure(state=tk.DISABLED)

        backend_box = self._make_frame(control_card, bg=PANEL_BG)
        backend_box.pack(fill=tk.X, pady=(8, 0))
        self._make_label(backend_box, text="视觉后端", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        backend_select = ttk.Combobox(
            backend_box,
            textvariable=self._backend_var,
            values=("libreface", "tf_mini_xception", "deepface"),
            state="readonly",
            style="App.TCombobox",
            width=17,
        )
        backend_select.pack(side=tk.RIGHT)
        backend_select.bind("<<ComboboxSelected>>", self._on_backend_change)

        self._make_checkbutton(control_card, text="启用专业知识库", variable=self._rag_enabled, bg=PANEL_BG).pack(anchor="w", pady=(8, 0))
        rag_mode_box = self._make_frame(control_card, bg=PANEL_BG)
        rag_mode_box.pack(fill=tk.X, pady=(4, 0))
        self._make_label(rag_mode_box, text="检索模式", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        ttk.Combobox(
            rag_mode_box,
            textvariable=self._rag_mode,
            values=("auto", "force"),
            state="readonly",
            style="App.TCombobox",
            width=10,
        ).pack(side=tk.RIGHT)
        self._make_checkbutton(control_card, text="启用 TTS（语音模式遮挡文字）", variable=self._tts_enabled, bg=PANEL_BG).pack(anchor="w")
        self._make_checkbutton(control_card, text="情绪大变化时动态调节（可能打断）", variable=self._realtime_adjust, bg=PANEL_BG).pack(anchor="w", pady=(8, 0))

        th_box = self._make_frame(control_card, bg=PANEL_BG)
        th_box.pack(fill=tk.X, pady=(4, 0))
        self._make_label(th_box, text="触发阈值", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Scale(
            th_box,
            from_=0.10,
            to=0.90,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._adjust_threshold,
            length=180,
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            troughcolor=PANEL_ALT_BG,
            highlightthickness=0,
        ).pack(side=tk.RIGHT)

        kb_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(kb_card, "知识库").pack(anchor="w")
        self._make_label(kb_card, text="导入专业文档并维护本地检索索引。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))
        self._make_button(kb_card, text="导入专业知识库", command=self._index_professional_corpus, primary=True).pack(fill=tk.X, pady=(0, 6))
        self._make_button(kb_card, text="上传 / 索引知识库…", command=self._pick_and_index).pack(fill=tk.X, pady=(0, 6))
        self._make_button(kb_card, text="清空知识库", command=self._clear_kb).pack(fill=tk.X)

        voice_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(voice_card, "语音模式").pack(anchor="w")
        self._make_label(voice_card, text="麦克风转写与纯语音全模态对话。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))
        self._mic_btn = self._make_button(voice_card, text="开始麦克风 ASR", command=self._toggle_asr)
        self._mic_btn.pack(fill=tk.X, pady=(0, 6))
        self._voice_only_btn = self._make_button(voice_card, text="进入纯语音对话", command=self._toggle_voice_only, primary=True)
        self._voice_only_btn.pack(fill=tk.X)

        suggest_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(suggest_card, "建议开场").pack(anchor="w")
        self._make_label(suggest_card, text="根据当前情绪状态生成更自然的起手句。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))
        self._suggest_btns: list[tk.Button] = []
        for _ in range(3):
            b = self._make_button(suggest_card, text="", command=lambda: None)
            b.configure(wraplength=280, justify=tk.LEFT, anchor="w", font=("Segoe UI", 10), pady=8)
            b.pack(fill=tk.X, pady=4)
            self._suggest_btns.append(b)
        self._refresh_suggestions(None)

        self._make_button(suggest_card, text="AI 帮我开个头", command=self._ai_starter, primary=True).pack(fill=tk.X, pady=(12, 0))

        ops_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(ops_card, "日志与历史").pack(anchor="w")
        self._make_label(ops_card, text="查看会话日志并恢复历史上下文。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))
        self._make_button(ops_card, text="打开日志目录", command=self._open_log_dir).pack(fill=tk.X, pady=(0, 6))
        self._make_button(ops_card, text="载入历史…", command=self._load_history).pack(fill=tk.X)

        insight_card = self._make_frame(right, bg=PANEL_BG, border=True, pack_kwargs={"fill": tk.X, "pady": (0, 12)})
        self._make_section_label(insight_card, "情绪洞察").pack(anchor="w")
        self._make_label(insight_card, text="结构化窗口、滚动摘要与 RMES 实验信息。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        self._make_label(insight_card, text="当前窗口", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(0, 4))
        self._window_text = tk.Text(insight_card, height=11, wrap="word", font=("Consolas", 9), bg=PANEL_ALT_BG, fg=TEXT_PRIMARY, relief="flat")
        self._window_text.pack(fill=tk.X)
        self._window_text.configure(state=tk.DISABLED)

        self._make_label(insight_card, text="最近 4 轮摘要", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(10, 4))
        self._rolling_text = tk.Text(insight_card, height=9, wrap="word", font=("Consolas", 9), bg=PANEL_ALT_BG, fg=TEXT_PRIMARY, relief="flat")
        self._rolling_text.pack(fill=tk.X)
        self._rolling_text.configure(state=tk.DISABLED)

        self._make_label(insight_card, text="RMES 调试", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 10)).pack(anchor="w", pady=(10, 4))
        self._rmes_text = tk.Text(insight_card, height=7, wrap="word", font=("Consolas", 9), bg=PANEL_ALT_BG, fg=TEXT_PRIMARY, relief="flat")
        self._rmes_text.pack(fill=tk.X)
        self._rmes_text.configure(state=tk.DISABLED)

        # Footer info
        self._kb_var = tk.StringVar(value=f"知识库 chunks: {self._rag_store.count()}")
        self._make_label(right, textvariable=self._kb_var, bg=APP_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

        self._bind_right_panel_scroll_recursive(self._right_panel_shell)

        # Gate chat until local vision models are ready (disable Send only)
        self._send_btn.configure(state=tk.DISABLED)

    def _get_emotion_trace(self) -> str | None:
        with self._emotion_queue_lock:
            return self._emotion_queue.to_prompt()

    def _get_emotion_summary_for_prompt(self) -> str | None:
        summary = self._emotion_tracker.get_summary()
        if summary:
            return summary
        if not self._config.emotion.enabled:
            return None
        status = self._emotion_tracker.get_status()
        if not status.model_ready:
            return "本地情绪模型未就绪"
        if not status.monitoring:
            return "情绪监测未启动"
        return "暂无稳定读数（请对准镜头）"

    def _record_emotion_cue(self, *, stage: str, role: str, text: str | None) -> None:
        try:
            status = self._emotion_tracker.get_status()
            snapshot = status.timeline.latest()
            summary = status.timeline.summary()
        except Exception:  # noqa: BLE001
            snapshot = None
            summary = None
        with self._emotion_queue_lock:
            self._emotion_queue.record(stage=stage, role=role, text=text, snapshot=snapshot, summary=summary)

    def _build_rag_snippets_for_query(self, query: str) -> str | None:
        if not self._rag_enabled.get():
            return None
        mode = (self._rag_mode.get() or "auto").strip().lower()
        if mode == "force":
            return self._rag_retriever.build_snippets(query)
        if self._rag_retriever.should_retrieve(query):
            return self._rag_retriever.build_snippets(query)
        return None

    def _build_rag_snippets_with_meta(self, query: str) -> tuple[str | None, int, list[str]]:
        if not self._rag_enabled.get():
            return None, 0, []
        mode = (self._rag_mode.get() or "auto").strip().lower()
        if mode == "force":
            snippets, hits = self._rag_retriever.build_snippets_with_hits(query)
            return snippets, len(hits), self._extract_hit_sources(hits)
        if self._rag_retriever.should_retrieve(query):
            snippets, hits = self._rag_retriever.build_snippets_with_hits(query)
            return snippets, len(hits), self._extract_hit_sources(hits)
        return None, 0, []

    def _extract_hit_sources(self, hits) -> list[str]:
        source_names: list[str] = []
        for hit in hits[:3]:
            try:
                src = hit.chunk.meta.get("source_name") or hit.chunk.source
            except Exception:
                src = None
            if src and str(src) not in source_names:
                source_names.append(str(src))
        return source_names

    def _update_text_rag_status(self, *, query: str) -> None:
        meta = dict(self._chat.last_reply_meta or {})
        hit_count = int(meta.get("rag_hit_count") or 0)
        injected = bool(meta.get("rag_injected"))
        triggered = bool(meta.get("rag_retrieval_triggered"))
        sources = [str(x) for x in (meta.get("rag_sources") or []) if str(x).strip()]
        query_preview = (query or "").strip().replace("\n", " ")
        if len(query_preview) > 24:
            query_preview = query_preview[:24] + "..."

        if not self._rag_enabled.get():
            status_text = "专业知识库：当前关闭"
        elif injected and hit_count > 0:
            src_preview = "、".join(sources) if sources else "来源未解析"
            status_text = f"专业知识库：文本模式已为“{query_preview}”检索并注入；top_k={hit_count}；来源={src_preview}"
        elif triggered:
            status_text = f"专业知识库：文本模式已触发检索，但未形成可注入片段（query=“{query_preview}”）"
        else:
            status_text = "专业知识库：文本模式本轮未触发或未命中检索"

        self._knowledge_status_var.set(status_text)
        if self._logger:
            self._logger.event(
                "rag_injection",
                source="text-chat",
                query=query.strip(),
                hit_count=hit_count,
                injected=injected,
                triggered=triggered,
                sources=sources,
            )

    def _build_chat_view(self, *, parent: tk.Widget) -> None:
        parent.configure(bg=PANEL_BG)

        header = tk.Frame(parent, bg=PANEL_BG, padx=18, pady=14)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(header, text="文字对话", bg=PANEL_BG, fg=TEXT_PRIMARY, font=("Segoe UI Semibold", 14)).pack(anchor="w")
        tk.Label(header, text="自然聊天、情绪辅助调节与知识库注入统一在一个工作区。", bg=PANEL_BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

        msg_box = tk.Frame(parent, bg=PANEL_BG)
        msg_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14)

        self._chat_canvas = tk.Canvas(msg_box, bg=PANEL_ALT_BG, highlightthickness=0)
        self._chat_scroll = ttk.Scrollbar(msg_box, orient="vertical", style="App.Vertical.TScrollbar", command=self._chat_canvas.yview)
        self._chat_canvas.configure(yscrollcommand=self._chat_scroll.set)

        self._chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._chat_inner = tk.Frame(self._chat_canvas, bg=PANEL_ALT_BG)
        self._chat_window = self._chat_canvas.create_window((0, 0), window=self._chat_inner, anchor="nw")

        self._bubble_labels: list[tk.Label] = []
        self._bubble_wraplength = 560

        def _on_inner_config(_e) -> None:
            self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))

        def _on_canvas_config(e) -> None:
            self._chat_canvas.itemconfigure(self._chat_window, width=e.width)
            self._bubble_wraplength = max(240, int(e.width * 0.72))
            for lbl in self._bubble_labels:
                try:
                    lbl.configure(wraplength=self._bubble_wraplength)
                except Exception:
                    pass

        self._chat_inner.bind("<Configure>", _on_inner_config)
        self._chat_canvas.bind("<Configure>", _on_canvas_config)

        def _on_mousewheel(ev) -> None:
            try:
                self._chat_canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")
            except Exception:
                pass

        self._chat_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        input_frame = tk.Frame(parent, bg=PANEL_BG, padx=14, pady=14)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X)

        input_shell = tk.Frame(input_frame, bg=SURFACE_BG, highlightthickness=1, highlightbackground=BORDER)
        input_shell.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._input = tk.Text(
            input_shell,
            height=4,
            font=("Segoe UI", 11),
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY,
            relief="flat",
            insertbackground=TEXT_PRIMARY,
            padx=12,
            pady=10,
        )
        self._input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._input.bind("<Control-Return>", lambda _e: self._send())

        btns = tk.Frame(input_frame, bg=PANEL_BG)
        btns.pack(side=tk.RIGHT, padx=(8, 0))

        self._send_btn = self._make_button(btns, text="发送 (Ctrl+Enter)", command=self._send, width=16, primary=True)
        self._send_btn.pack(side=tk.TOP, pady=(0, 6))
        self._make_button(btns, text="新建对话", command=self._new_chat, width=16).pack(side=tk.TOP, pady=(0, 6))
        self._make_button(btns, text="清空对话", command=self._clear_chat, width=16).pack(side=tk.TOP)

    def _build_voice_view(self, *, parent: tk.Widget) -> None:
        parent.configure(bg=VOICE_BG)

        title = tk.Label(
            parent,
            text="语音会话",
            bg=VOICE_BG,
            fg=TEXT_INVERSE,
            font=("Segoe UI Semibold", 16),
        )
        title.pack(anchor="w", padx=16, pady=(16, 6))
        tk.Label(
            parent,
            text="低延迟语音交互，持续展示识别与知识库注入状态。",
            bg=VOICE_BG,
            fg="#b7c7cf",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(0, 14))

        self._voice_hint_var = tk.StringVar(value="播放中…（文字已隐藏）")
        tk.Label(parent, textvariable=self._voice_hint_var, bg=VOICE_BG, fg="#d9e6ec", font=("Segoe UI", 10)).pack(
            anchor="w", padx=16, pady=(0, 14)
        )

        tk.Label(parent, text="实时识别（将自动发送）", bg=VOICE_BG, fg="#d9e6ec", font=("Segoe UI Semibold", 10)).pack(
            anchor="w", padx=16, pady=(0, 6)
        )
        self._voice_asr_var = tk.StringVar(value="")
        tk.Label(
            parent,
            textvariable=self._voice_asr_var,
            bg="#173038",
            fg=TEXT_INVERSE,
            justify=tk.LEFT,
            wraplength=860,
            padx=14,
            pady=12,
            font=("Segoe UI", 11),
        ).pack(fill=tk.X, padx=16, pady=(0, 14))

        self._voice_bar = ttk.Progressbar(parent, orient="horizontal", mode="determinate", maximum=10.0, style="Accent.Horizontal.TProgressbar")
        self._voice_bar.pack(fill=tk.X, padx=16)

        self._voice_time_var = tk.StringVar(value="00:00 / --:--")
        tk.Label(parent, textvariable=self._voice_time_var, bg=VOICE_BG, fg="#d9e6ec", font=("Segoe UI", 10)).pack(
            anchor="w", padx=16, pady=(10, 0)
        )

        self._voice_emotion_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._voice_emotion_var, bg=VOICE_BG, fg="#8fd3c1", font=("Segoe UI", 10)).pack(
            anchor="w", padx=16, pady=(10, 0)
        )
        tk.Label(
            parent,
            textvariable=self._knowledge_status_var,
            bg=VOICE_BG,
            fg="#a8e0c0",
            justify=tk.LEFT,
            wraplength=860,
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(8, 0))

        btns = tk.Frame(parent, bg=VOICE_BG)
        btns.pack(fill=tk.X, padx=16, pady=(18, 0))
        self._make_button(btns, text="返回文字界面", command=self._on_voice_back, width=16).pack(side=tk.LEFT)

    def _show_voice_mode(self) -> None:
        self._chat_frame.pack_forget()
        self._voice_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _show_chat_mode(self) -> None:
        self._voice_frame.pack_forget()
        self._chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_voice_back(self) -> None:
        if self._voice_only_mode:
            self._toggle_voice_only()
        else:
            self._show_chat_mode()

    def _enter_voice_only_runtime(self) -> None:
        self._voice_only_mode = True
        self._pending_enter_voice_only = False
        if self._config.emotion.enabled and not self._emotion_tracker.is_monitoring():
            self._emotion_tracker.start()
            self._record_emotion_cue(stage="monitor_start", role="system", text="monitor_start")

        self._show_voice_mode()
        self._voice_hint_var.set("聆听中…（纯语音对话）")
        self._voice_asr_var.set("")
        self._voice_asr_buffer = ""
        self._voice_asr_last_update_ts = 0.0
        self._voice_asr_last_sent = ""
        self._voice_asr_paused_for_reply = False
        self._voice_reply_text = ""
        try:
            self._voice_only_btn.configure(text="退出纯语音对话")
        except Exception:
            pass

        if self._config.omni.enabled:
            self._omni_session.start()
        elif not self._asr_running:
            self._start_asr()

    def _toggle_voice_only(self) -> None:
        if self._closing:
            return

        if not self._voice_only_mode:
            if not self._emotion_tracker.is_model_ready():
                self._ui_queue.put(_UiEvent(type="toast", payload="本地识别模型仍在初始化，请稍候…"))
                return
            try:
                if self._llm_thread and self._llm_thread.is_alive():
                    self._pending_enter_voice_only = True
                    self._show_voice_mode()
                    self._voice_hint_var.set("当前文字回复尚未结束，结束后自动切入纯语音…")
                    self._voice_asr_var.set("")
                    self._voice_reply_text = ""
                    self._knowledge_status_var.set("专业知识库：等待进入纯语音后启用")
                    try:
                        self._voice_only_btn.configure(text="退出纯语音对话")
                    except Exception:
                        pass
                else:
                    self._enter_voice_only_runtime()
            except Exception as exc:  # noqa: BLE001
                self._pending_enter_voice_only = False
                self._voice_only_mode = False
                self._show_chat_mode()
                self._ui_queue.put(_UiEvent(type="toast", payload=f"纯语音模式启动失败：{exc}"))
            return

        # disable
        self._pending_enter_voice_only = False
        self._voice_only_mode = False
        self._omni_reply_window_open = False
        try:
            self._voice_only_btn.configure(text="进入纯语音对话")
        except Exception:
            pass
        self._voice_asr_buffer = ""
        self._voice_asr_last_update_ts = 0.0
        self._voice_asr_paused_for_reply = False
        self._voice_reply_text = ""
        self._voice_hint_var.set("已退出纯语音对话，正在清理会话…")
        self._knowledge_status_var.set("专业知识库：未触发")
        self._show_chat_mode()
        self._send_btn.configure(state=tk.NORMAL if self._emotion_tracker.is_model_ready() else tk.DISABLED)
        threading.Thread(target=self._shutdown_voice_runtime, name="VoiceOnlyShutdown", daemon=True).start()

    def _shutdown_voice_runtime(self) -> None:
        try:
            if self._config.omni.enabled:
                self._omni_session.stop()
        except Exception as exc:  # noqa: BLE001
            self._ui_queue.put(_UiEvent(type="toast", payload=f"退出纯语音时清理 Omni 会话失败：{exc}"))
        try:
            self._stop_asr()
        except Exception as exc:  # noqa: BLE001
            self._ui_queue.put(_UiEvent(type="toast", payload=f"退出纯语音时停止 ASR 失败：{exc}"))
        self._ui_queue.put(_UiEvent(type="voice_shutdown_done"))

    def _maybe_send_voice_asr(self, *, force: bool) -> None:
        if not self._voice_only_mode:
            return
        if self._closing:
            return
        if self._llm_thread and self._llm_thread.is_alive():
            return
        text = (self._voice_asr_buffer or "").strip()
        if not text:
            return
        if text == (self._voice_asr_last_sent or "").strip():
            return
        # if not force, keep it conservative (avoid premature send)
        if not force:
            now = time.time()
            if now - float(self._voice_asr_last_update_ts) < 1.4:
                return

        self._voice_asr_last_sent = text
        self._voice_asr_buffer = ""
        try:
            self._voice_asr_var.set("")
        except Exception:
            pass
        self._send_text(text, source="voice")

    def _check_voice_autosend(self) -> None:
        try:
            if self._voice_only_mode and (self._voice_asr_buffer or "").strip():
                self._maybe_send_voice_asr(force=False)
        finally:
            if not self._closing:
                self._root.after(120, self._check_voice_autosend)

    def _add_bubble(self, *, role: str, text: str, style: str = "normal") -> tk.Label:
        bg = PANEL_ALT_BG
        outer = tk.Frame(self._chat_inner, bg=bg)
        outer.pack(fill=tk.X, pady=6, padx=8)

        if style == "system":
            bubble_bg = SYSTEM_BUBBLE
            fg = TEXT_MUTED
            side = tk.LEFT
            pad = (12, 72)
        else:
            if role == "user":
                bubble_bg = USER_BUBBLE
                fg = TEXT_PRIMARY
                side = tk.RIGHT
                pad = (72, 12)
            else:
                bubble_bg = ASSISTANT_BUBBLE
                fg = TEXT_PRIMARY
                side = tk.LEFT
                pad = (12, 72)

        bubble = tk.Frame(
            outer,
            bg=bubble_bg,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        bubble.pack(side=side, padx=pad)

        label = tk.Label(
            bubble,
            text=text,
            bg=bubble_bg,
            fg=fg,
            justify=tk.LEFT,
            wraplength=self._bubble_wraplength,
            font=("Segoe UI", 11),
            padx=12,
            pady=10,
        )
        label.pack()
        self._bubble_labels.append(label)

        self._chat_canvas.update_idletasks()
        self._chat_canvas.yview_moveto(1.0)
        return label

    def _add_toast(self, text: str) -> None:
        self._add_bubble(role="system", text=text, style="system")

    def _new_chat(self) -> None:
        if self._llm_thread and self._llm_thread.is_alive():
            self._add_toast("当前仍在生成回复，请稍后再新建对话。")
            return
        self._pending_enter_voice_only = False
        if self._voice_only_mode:
            try:
                self._toggle_voice_only()
            except Exception:
                pass
        self._clear_chat()
        self._knowledge_status_var.set("专业知识库：未触发")
        try:
            self._input.delete("1.0", tk.END)
        except Exception:
            pass
        self._add_toast("已新建对话。")

    def _clear_chat(self) -> None:
        self._chat.reset()
        for w in list(self._chat_inner.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        self._bubble_labels.clear()
        self._assistant_label = None
        self._assistant_text = ""
        self._insert_greeting()

    def _toggle_monitoring(self) -> None:
        if self._closing:
            return
        status = self._emotion_tracker.get_status()
        if not status.model_ready:
            self._ui_queue.put(_UiEvent(type="toast", payload="本地识别模型仍在初始化，请稍候…"))
            return
        if status.monitoring:
            self._emotion_tracker.stop()
            self._record_emotion_cue(stage="monitor_stop", role="system", text="monitor_stop")
        else:
            self._emotion_tracker.start()
            self._record_emotion_cue(stage="monitor_start", role="system", text="monitor_start")

    def _send(self) -> None:
        user_text = self._input.get("1.0", tk.END).strip()
        if not user_text:
            return
        self._input.delete("1.0", tk.END)
        self._send_text(user_text, source="text")

    def _send_text(self, user_text: str, *, source: str) -> None:
        if self._closing:
            return
        if self._llm_thread and self._llm_thread.is_alive():
            return
        if not self._emotion_tracker.is_model_ready():
            self._ui_queue.put(_UiEvent(type="toast", payload="本地识别模型仍在初始化，请稍候…"))
            return

        # 默认：情绪监测与对话同步启动；也可通过右侧按钮提前启动
        if self._config.emotion.enabled and not self._emotion_tracker.is_monitoring():
            self._emotion_tracker.start()

        # Voice-only: pause ASR during assistant reply to avoid echo loops.
        if self._voice_only_mode and self._asr_running:
            self._voice_asr_paused_for_reply = True
            self._stop_asr()

        if self._voice_only_mode:
            try:
                self._voice_hint_var.set("我在想…")
            except Exception:
                pass

        stage = "user_send_voice" if source == "voice" else "user_send"
        self._last_user_query_for_rag = user_text
        self._record_emotion_cue(stage=stage, role="user", text=user_text)
        self._add_bubble(role="user", text=user_text)
        if self._logger:
            status = self._emotion_tracker.get_status()
            latest = status.timeline.latest()
            meta = {
                "source": str(source),
                "voice_only_mode": bool(self._voice_only_mode),
                "emotion_summary": self._emotion_tracker.get_summary(),
                "latest_emotion": getattr(latest, "emotion", None),
                "tone_hint": _tone_hint(getattr(latest, "emotion", None) if latest else None),
                "realtime_adjust_enabled": bool(self._realtime_adjust.get()),
                "adjust_threshold": float(self._adjust_threshold.get()),
            }
            self._logger.log_message(role="user", text=user_text, meta=meta)

        self._send_btn.configure(state=tk.DISABLED)
        self._ui_queue.put(_UiEvent(type="assistant_start"))
        self._llm_thread = threading.Thread(target=self._llm_worker, args=(user_text,), daemon=True)
        self._llm_thread.start()

    def _start_tts_sampler(self) -> None:
        if not self._logger:
            return
        if self._tts_sampler_thread and self._tts_sampler_thread.is_alive():
            return
        self._tts_sampling_stop.clear()

        def _loop():
            last_logged = -1.0
            while not self._tts_sampling_stop.is_set() and not self._closing:
                with self._tts_lock:
                    play_s = float(self._tts_play_s)
                if play_s - last_logged >= 0.5:
                    last_logged = play_s
                    self._logger.log_emotion_sample(play_s=play_s, emotion_summary=self._emotion_tracker.get_summary())
                time.sleep(0.15)

        self._tts_sampler_thread = threading.Thread(target=_loop, daemon=True)
        self._tts_sampler_thread.start()

    def _stop_tts_sampler(self) -> None:
        self._tts_sampling_stop.set()

    def _llm_worker(self, user_text: str) -> None:
        tts_enabled = bool(self._tts_enabled.get())
        realtime_adjust = bool(self._realtime_adjust.get())
        adjust_threshold = float(self._adjust_threshold.get())

        tts: RealtimeTts | None = None
        tts_chunker: TtsTextChunker | None = None
        audio_stream = None

        assistant_parts: list[str] = []
        emotion_start = self._emotion_tracker.get_summary()

        if tts_enabled:
            sd = _safe_import_sounddevice()
            if sd is None:
                self._ui_queue.put(_UiEvent(type="toast", payload="未安装 sounddevice，无法启用 TTS。"))
                tts_enabled = False
            else:
                try:
                    with self._tts_lock:
                        self._tts_play_s = 0.0
                        self._tts_text_chars = 0
                        self._tts_active = True

                    audio_stream = sd.RawOutputStream(
                        samplerate=self._config.tts.sample_rate,
                        channels=1,
                        dtype="int16",
                    )
                    audio_stream.start()

                    played_frames = 0
                    last_push = 0.0

                    def _audio_cb(b: bytes) -> None:
                        nonlocal played_frames, last_push
                        try:
                            audio_stream.write(b)
                        except Exception:
                            return
                        played_frames += max(0, len(b) // 2)
                        now = time.time()
                        if now - last_push < 0.15:
                            return
                        last_push = now
                        play_s = float(played_frames) / float(self._config.tts.sample_rate)
                        with self._tts_lock:
                            self._tts_play_s = play_s
                        self._ui_queue.put(_UiEvent(type="tts_progress", payload=play_s))

                    tts = RealtimeTts(
                        dashscope=self._config.dashscope,
                        tts=self._config.tts,
                        audio_callback=_audio_cb,
                    )
                    tts.start()
                    tts_chunker = TtsTextChunker(min_chars=8, max_chars=70) if self._voice_only_mode else TtsTextChunker()
                    self._ui_queue.put(_UiEvent(type="tts_started"))
                except Exception as exc:  # noqa: BLE001
                    self._ui_queue.put(_UiEvent(type="toast", payload=f"TTS 初始化失败：{exc}"))
                    tts_enabled = False

        try:
            for delta in self._chat.stream_reply(
                user_text,
                realtime_adjust=realtime_adjust,
                adjust_threshold=adjust_threshold,
            ):
                if delta:
                    assistant_parts.append(delta)
                    self._ui_queue.put(_UiEvent(type="assistant_delta", payload=delta))

                if tts_enabled and tts and tts_chunker and delta:
                    for frag in tts_chunker.push(delta):
                        with self._tts_lock:
                            self._tts_text_chars += len(frag)
                        tts.append_text(frag)

            if tts_enabled and tts and tts_chunker:
                for frag in tts_chunker.flush():
                    with self._tts_lock:
                        self._tts_text_chars += len(frag)
                    tts.append_text(frag)
                tts.finish()
                tts.wait_done(timeout_s=12.0)
        except Exception as exc:  # noqa: BLE001
            self._ui_queue.put(_UiEvent(type="assistant_error", payload=str(exc)))
            if self._logger:
                self._logger.event("assistant_error", error=str(exc))
        finally:
            assistant_text = "".join(assistant_parts).strip()
            emotion_end = self._emotion_tracker.get_summary()

            if assistant_text:
                self._record_emotion_cue(stage="assistant_done", role="assistant", text=assistant_text)

            if self._logger and assistant_text:
                status = self._emotion_tracker.get_status()
                latest = status.timeline.latest()
                meta = {
                    "emotion_start": emotion_start,
                    "emotion_end": emotion_end,
                    "latest_emotion": getattr(latest, "emotion", None),
                    "tone_hint": _tone_hint(getattr(latest, "emotion", None) if latest else None),
                    "voice_only_mode": bool(self._voice_only_mode),
                    "tts_enabled": bool(tts_enabled),
                    "realtime_adjust_ui_enabled": bool(realtime_adjust),
                    "adjust_threshold": float(adjust_threshold),
                    "llm_meta": dict(self._chat.last_reply_meta),
                    "emotion_trace": self._get_emotion_trace() or "",
                }
                self._logger.log_message(role="assistant", text=assistant_text, meta=meta)

            if tts:
                try:
                    tts.stop()
                except Exception:
                    pass
            if audio_stream is not None:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception:
                    pass
            with self._tts_lock:
                self._tts_active = False
            self._ui_queue.put(_UiEvent(type="assistant_end"))

    # RAG
    def _pick_and_index(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择要加入知识库的文件（txt/md/pdf）",
            filetypes=[("Knowledge Files", "*.txt *.md *.pdf"), ("All", "*.*")],
        )
        if not paths:
            return

        def _worker():
            try:
                res = self._rag_indexer.index_paths([Path(p) for p in paths])
                self._ui_queue.put(_UiEvent(type="toast", payload=f"索引完成：新增 chunks={res.indexed_chunks}"))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(_UiEvent(type="toast", payload=f"索引失败：{exc}"))
            finally:
                self._ui_queue.put(_UiEvent(type="kb_refresh"))

        threading.Thread(target=_worker, daemon=True).start()

    def _index_professional_corpus(self) -> None:
        paths = [p for p in _PROFESSIONAL_KB_PDFS if p.exists()]
        if not paths:
            self._add_toast("未找到预设专业知识库 PDF。")
            return

        def _worker():
            try:
                res = self._rag_indexer.index_paths(paths)
                self._ui_queue.put(_UiEvent(type="toast", payload=f"专业知识库导入完成：新增 chunks={res.indexed_chunks}"))
            except Exception as exc:  # noqa: BLE001
                self._ui_queue.put(_UiEvent(type="toast", payload=f"专业知识库导入失败：{exc}"))
            finally:
                self._ui_queue.put(_UiEvent(type="kb_refresh"))

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_kb(self) -> None:
        if not messagebox.askyesno("确认", "确定清空本地知识库吗？"):
            return
        self._rag_store.clear()
        self._kb_var.set(f"知识库 chunks: {self._rag_store.count()}")

    # ASR
    def _start_asr(self) -> bool:
        if self._asr_running:
            return True

        def _on_event(ev: AsrEvent) -> None:
            self._ui_queue.put(_UiEvent(type="asr", payload=ev))

        try:
            self._asr.start(_on_event)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("ASR 启动失败", str(exc))
            return False

        self._asr_running = True
        try:
            self._mic_btn.configure(text="停止麦克风 ASR")
        except Exception:
            pass
        return True

    def _stop_asr(self) -> None:
        if not self._asr_running:
            return
        try:
            self._asr.stop()
        except Exception:
            pass
        self._asr_running = False
        try:
            self._mic_btn.configure(text="开始麦克风 ASR")
        except Exception:
            pass

    def _toggle_asr(self) -> None:
        if self._asr_running:
            self._stop_asr()
        else:
            self._start_asr()

    def _on_omni_user_transcript(self, event: OmniUserTranscriptEvent) -> None:
        self._ui_queue.put(_UiEvent(type="omni_user_transcript", payload=event))

    def _on_omni_assistant_transcript(self, event: OmniAssistantTranscriptEvent) -> None:
        self._ui_queue.put(_UiEvent(type="omni_assistant_transcript", payload=event))

    def _on_omni_state(self, text: str) -> None:
        self._ui_queue.put(_UiEvent(type="omni_state", payload=text))

    def _on_omni_error(self, text: str) -> None:
        self._ui_queue.put(_UiEvent(type="omni_error", payload=text))

    def _ai_starter(self) -> None:
        text = "你先跟我打个招呼吧，我们轻松聊聊。给我 2 个你觉得适合开场的话题也行。"
        self._input.delete("1.0", tk.END)
        self._input.insert(tk.END, text)

    def _insert_greeting(self) -> None:
        greeting = (self._config.emotion.greeting_text or "").strip()
        if not greeting:
            return
        self._chat.reset()
        self._emotion_tracker.begin_dialogue_window(mode="startup", reason="greeting")
        self._add_bubble(role="assistant", text=greeting)
        self._chat.session.add_assistant(greeting)
        if self._logger:
            self._logger.log_message(
                role="assistant",
                text=greeting,
                meta={"source": "system-greeting", "voice_text": self._config.emotion.greeting_voice_text},
            )

    def _on_backend_change(self, _event=None) -> None:
        backend = (self._backend_var.get() or "").strip()
        try:
            self._emotion_tracker.set_backend(backend)
            self._add_toast(f"已切换视觉后端：{backend}")
        except Exception as exc:  # noqa: BLE001
            self._add_toast(f"切换视觉后端失败：{exc}")

    def _set_readonly_text(self, widget: tk.Text, text: str) -> None:
        try:
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert("1.0", text)
            widget.configure(state=tk.DISABLED)
        except Exception:
            pass

    # Logs
    def _open_log_dir(self) -> None:
        if not self._logger:
            messagebox.showinfo("日志", "当前未启用日志（logger 初始化失败）。")
            return
        p = self._logger.paths.jsonl.parent
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("日志目录", str(p))

    def _load_history(self) -> None:
        base = Path(os.getenv("HSEMOTION_LOG_DIR", ".hsemotion_logs"))
        path = filedialog.askopenfilename(
            title="选择历史记录（jsonl）",
            initialdir=str(base) if base.exists() else None,
            filetypes=[("JSONL", "*.jsonl"), ("All", "*.*")],
        )
        if not path:
            return

        try:
            for w in list(self._chat_inner.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass
            self._bubble_labels.clear()
            self._assistant_label = None
            self._assistant_text = ""
            self._chat.reset()

            import json

            with Path(path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("type") != "message":
                        continue
                    payload = rec.get("payload") or {}
                    role = payload.get("role")
                    text = payload.get("text") or ""
                    if role == "user":
                        self._add_bubble(role="user", text=text)
                        self._chat.session.add_user(text)
                    elif role == "assistant":
                        self._add_bubble(role="assistant", text=text)
                        self._chat.session.add_assistant(text)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("载入失败", str(exc))

    # Periodic updates
    def _drain_ui_queue(self) -> None:
        try:
            while True:
                ev = self._ui_queue.get_nowait()
                if ev.type == "assistant_start":
                    self._emotion_tracker.begin_dialogue_window(mode="text", reason="assistant_reply_start")
                    self._assistant_text = ""
                    self._assistant_label = self._add_bubble(role="assistant", text="思考中…", style="muted")
                elif ev.type == "assistant_delta":
                    if self._assistant_label is None:
                        self._assistant_label = self._add_bubble(role="assistant", text="思考中…", style="muted")
                    if not self._assistant_text:
                        try:
                            self._assistant_label.configure(fg="#111", text="")
                        except Exception:
                            pass
                    self._assistant_text += str(ev.payload or "")
                    try:
                        self._assistant_label.configure(text=self._assistant_text)
                    except Exception:
                        pass
                    self._chat_canvas.update_idletasks()
                    self._chat_canvas.yview_moveto(1.0)
                elif ev.type == "assistant_error":
                    self._add_toast(f"错误：{ev.payload}")
                elif ev.type == "assistant_end":
                    try:
                        if not self._voice_only_mode and self._chat.last_reply_meta:
                            self._update_text_rag_status(query=getattr(self, "_last_user_query_for_rag", ""))
                    except Exception:
                        pass
                    if self._assistant_label is not None and not self._assistant_text.strip():
                        try:
                            self._assistant_label.configure(text="（本轮未收到可显示的正文）", fg="#666")
                        except Exception:
                            pass
                    self._assistant_label = None
                    self._assistant_text = ""
                    self._stop_tts_sampler()
                    if self._pending_enter_voice_only and not self._voice_only_mode:
                        try:
                            self._enter_voice_only_runtime()
                        except Exception as exc:
                            self._pending_enter_voice_only = False
                            self._show_chat_mode()
                            self._add_toast(f"纯语音模式启动失败：{exc}")
                        continue
                    if self._voice_only_mode:
                        self._show_voice_mode()
                        try:
                            self._voice_hint_var.set("聆听中…（纯语音对话）")
                        except Exception:
                            pass
                        if self._voice_asr_paused_for_reply:
                            self._voice_asr_paused_for_reply = False
                            self._start_asr()
                    else:
                        if not (self._tts_enabled.get() and self._tts_active):
                            self._show_chat_mode()
                        if self._emotion_tracker.is_model_ready():
                            self._send_btn.configure(state=tk.NORMAL)
                elif ev.type == "toast":
                    self._add_toast(str(ev.payload))
                elif ev.type == "kb_refresh":
                    self._kb_var.set(f"知识库 chunks: {self._rag_store.count()}")
                elif ev.type == "voice_shutdown_done":
                    if not self._voice_only_mode:
                        try:
                            self._voice_hint_var.set("已退出纯语音对话")
                        except Exception:
                            pass
                elif ev.type == "asr":
                    asr_ev: AsrEvent = ev.payload  # type: ignore[assignment]
                    if self._voice_only_mode:
                        self._voice_asr_buffer = str(asr_ev.text or "")
                        self._voice_asr_last_update_ts = time.time()
                        try:
                            self._voice_asr_var.set(self._voice_asr_buffer)
                        except Exception:
                            pass
                        if asr_ev.is_final:
                            self._maybe_send_voice_asr(force=True)
                    else:
                        self._input.delete("1.0", tk.END)
                        self._input.insert(tk.END, asr_ev.text)
                elif ev.type == "omni_user_transcript":
                    transcript: OmniUserTranscriptEvent = ev.payload  # type: ignore[assignment]
                    self._voice_asr_buffer = transcript.text
                    self._voice_asr_var.set(transcript.text)
                    if transcript.is_final:
                        self._omni_reply_window_open = False
                        if self._logger and transcript.text.strip():
                            self._logger.log_message(
                                role="user",
                                text=transcript.text.strip(),
                                meta={"source": "omni-realtime", "voice_only_mode": True, "emotion_summary": self._emotion_tracker.get_summary()},
                            )
                        try:
                            snippets, hit_count, source_names = self._build_rag_snippets_with_meta(transcript.text)
                        except Exception:
                            snippets, hit_count, source_names = None, 0, []
                        try:
                            self._omni_session.update_external_knowledge(snippets)
                        except Exception:
                            pass
                        if snippets:
                            query_preview = transcript.text.strip().replace("\n", " ")[:24]
                            src_preview = "、".join(source_names) if source_names else "来源未解析"
                            status_text = (
                                f"专业知识库：已为“{query_preview}”检索并注入当前 Omni 会话；"
                                f"top_k={hit_count}；来源={src_preview}"
                            )
                        elif self._rag_enabled.get():
                            status_text = "专业知识库：本轮纯语音未命中或未触发检索"
                        else:
                            status_text = "专业知识库：当前关闭"
                        self._knowledge_status_var.set(status_text)
                        if self._logger:
                            self._logger.event(
                                "rag_injection",
                                source="omni-realtime",
                                query=transcript.text.strip(),
                                hit_count=int(hit_count),
                                injected=bool(snippets),
                            )
                elif ev.type == "omni_assistant_transcript":
                    transcript: OmniAssistantTranscriptEvent = ev.payload  # type: ignore[assignment]
                    if transcript.text.strip() and not self._omni_reply_window_open:
                        self._omni_reply_window_open = True
                        self._emotion_tracker.begin_dialogue_window(mode="voice", reason="omni_reply_start")
                    self._voice_reply_text = transcript.text
                    if transcript.is_final and self._logger and transcript.text.strip():
                        self._logger.log_message(
                            role="assistant",
                            text=transcript.text.strip(),
                            meta={
                                "source": "omni-realtime",
                                "voice_only_mode": True,
                                "emotion_summary": self._emotion_tracker.get_summary(),
                            },
                        )
                    if transcript.is_final:
                        self._omni_reply_window_open = False
                elif ev.type == "omni_state":
                    self._voice_hint_var.set(str(ev.payload))
                elif ev.type == "omni_error":
                    text = str(ev.payload or "").strip()
                    now = time.time()
                    if text and (
                        text != self._last_omni_error_text or now - float(self._last_omni_error_ts) >= 3.0
                    ):
                        self._last_omni_error_text = text
                        self._last_omni_error_ts = now
                        self._add_toast(f"Omni error: {text}")
                    self._voice_hint_var.set("纯语音全模态会话异常，请退出后重试")
                elif ev.type == "tts_started":
                    self._show_voice_mode()
                    if self._voice_only_mode:
                        try:
                            self._voice_hint_var.set("我在说…（纯语音对话）")
                        except Exception:
                            pass
                    self._start_tts_sampler()
                elif ev.type == "tts_progress":
                    play_s = float(ev.payload or 0.0)
                    with self._tts_lock:
                        chars = int(self._tts_text_chars)
                    chars_per_sec = 6.0
                    est_total = max(1.0, float(chars) / chars_per_sec)
                    self._voice_bar.configure(maximum=est_total)
                    self._voice_bar["value"] = min(est_total, play_s)
                    self._voice_time_var.set(f"{_format_mmss(play_s)} / ~{_format_mmss(est_total)}")
                    self._voice_emotion_var.set(self._emotion_tracker.get_summary() or "")
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                self._root.after(30, self._drain_ui_queue)

    def _update_video_preview(self) -> None:
        if self._closing:
            return

        if Image is None or ImageTk is None:
            self._video_label.configure(text="未安装 pillow，无法显示视频预览", image="")
            self._video_photo = None
            self._root.after(800, self._update_video_preview)
            return

        frame = self._emotion_tracker.get_latest_frame_bgr()
        if frame is None:
            status = self._emotion_tracker.get_status()
            if not status.model_ready:
                text = "视频预览：本地识别模型初始化失败" if status.last_error else "视频预览：本地识别模型初始化中…"
            elif not status.monitoring:
                text = "视频预览：情绪监测未启动"
            else:
                text = "视频预览：正在打开摄像头…"
            self._video_label.configure(text=text, image="")
            self._video_photo = None
            self._root.after(120, self._update_video_preview)
            return

        try:
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)

            target_w = int(self._video_label.winfo_width() or 0)
            if target_w < 80:
                target_w = 720

            w, h = img.size
            scale = min(1.0, target_w / float(max(1, w)))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h))

            photo = ImageTk.PhotoImage(img)
            self._video_photo = photo
            self._video_label.configure(image=photo, text="")
            self._video_label.image = photo
        except Exception as exc:  # noqa: BLE001
            self._video_label.configure(text=f"视频预览错误：{exc}", image="")
            self._video_photo = None
        finally:
            self._root.after(16, self._update_video_preview)

    def _on_right_panel_mousewheel(self, event) -> None:
        try:
            delta = -1 if event.delta > 0 else 1
            self._right_canvas.yview_scroll(delta, "units")
            return "break"
        except Exception:
            return

    def _on_right_panel_mousewheel_linux(self, event) -> None:
        try:
            delta = -1 if getattr(event, "num", None) == 4 else 1
            self._right_canvas.yview_scroll(delta, "units")
            return "break"
        except Exception:
            return

    def _bind_right_panel_scroll_recursive(self, widget: tk.Widget) -> None:
        try:
            widget.bind("<MouseWheel>", self._on_right_panel_mousewheel, add="+")
            widget.bind("<Button-4>", self._on_right_panel_mousewheel_linux, add="+")
            widget.bind("<Button-5>", self._on_right_panel_mousewheel_linux, add="+")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._bind_right_panel_scroll_recursive(child)

    def _update_emotion_panel(self) -> None:
        if self._closing:
            return
        status = self._emotion_tracker.get_status()
        latest = status.timeline.latest()

        if not status.model_ready:
            self._emotion_var.set(
                f"本地识别模型({self._emotion_tracker.get_backend_display_name()})：初始化失败" if status.last_error else f"本地识别模型({self._emotion_tracker.get_backend_display_name()})：初始化中…"
            )
            self._emotion_err_var.set(status.last_error or "")
            self._send_btn.configure(state=tk.DISABLED)
            self._monitor_btn.configure(state=tk.DISABLED)
            self._refresh_suggestions(None)
            self._set_readonly_text(self._window_text, "当前窗口：模型未就绪。")
            self._set_readonly_text(self._rolling_text, "最近窗口摘要：模型未就绪。")
            self._set_readonly_text(self._rmes_text, "RMES 调试：模型未就绪。")
            self._root.after(250, self._update_emotion_panel)
            return

        if not (self._llm_thread and self._llm_thread.is_alive()):
            self._send_btn.configure(state=tk.DISABLED if self._voice_only_mode else tk.NORMAL)

        if not self._config.emotion.enabled:
            self._monitor_btn.configure(state=tk.DISABLED, text="情绪监测已禁用")
            self._emotion_var.set("情绪信号：已禁用")
        else:
            self._monitor_btn.configure(state=tk.NORMAL)
            self._monitor_btn.configure(text="停止情绪监测" if status.monitoring else "提前启动情绪监测")
            if status.monitoring:
                summary = status.timeline.summary() or "暂无数据（请对准镜头）"
                self._emotion_var.set(f"情绪信号（{self._emotion_tracker.get_backend_display_name()}）：" + summary)
            else:
                self._emotion_var.set(f"情绪信号（{self._emotion_tracker.get_backend_display_name()}）：未启动（发送后将自动同步启动）")

        self._emotion_err_var.set(status.last_error or "")
        self._refresh_suggestions(latest.emotion if latest else None)
        self._set_readonly_text(self._window_text, self._emotion_tracker.get_current_window_text())
        self._set_readonly_text(self._rolling_text, self._emotion_tracker.get_recent_windows_text())
        self._set_readonly_text(self._rmes_text, self._emotion_tracker.get_rmes_debug_text())
        for notice in self._emotion_tracker.get_and_clear_notices():
            self._add_toast(f"提示：{notice}")

        # Pre-chat / idle emotion tracking: keep a lightweight baseline before the user speaks.
        if (
            self._config.emotion.enabled
            and status.monitoring
            and not (self._llm_thread and self._llm_thread.is_alive())
        ):
            now = time.time()
            if now - float(self._emoqueue_last_idle_ts) >= 2.0:
                self._emoqueue_last_idle_ts = now
                self._record_emotion_cue(stage="idle", role="system", text="idle")

        self._root.after(250, self._update_emotion_panel)

    def _refresh_suggestions(self, emotion: str | None) -> None:
        suggestions = _build_suggestions(emotion)
        for btn, text in zip(self._suggest_btns, suggestions, strict=False):
            btn.configure(text=text, command=lambda t=text: self._set_input(t))

    def _set_input(self, text: str) -> None:
        self._input.delete("1.0", tk.END)
        self._input.insert(tk.END, text)

    def _on_close(self) -> None:
        self._closing = True
        self._stop_tts_sampler()
        try:
            self._asr.stop()
        except Exception:
            pass
        try:
            self._omni_session.stop()
        except Exception:
            pass
        try:
            self._emotion_tracker.shutdown()
        except Exception:
            pass
        try:
            self._rag_store.close()
        except Exception:
            pass
        self._root.destroy()


def main() -> None:
    try:
        config = load_config_from_env()
    except Exception as exc:  # noqa: BLE001
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "配置错误",
            f"{exc}\n\n请先设置环境变量 DASHSCOPE_API_KEY，并可选设置 QWEN_MODEL=qwen-plus-latest。",
        )
        root.destroy()
        return

    app = EmotionChatTkApp(config)
    app.run()

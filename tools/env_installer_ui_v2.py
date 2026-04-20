from __future__ import annotations

import importlib.metadata as md
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_STABLE = PROJECT_ROOT / "requirements_stable.clean.txt"
REQ_LLM = PROJECT_ROOT / "requirements_llm_ui_v0.1.3.clean.txt"
REQ_OPTIONAL = PROJECT_ROOT / "requirements_optional.clean.txt"
REQ_CONSTRAINTS = PROJECT_ROOT / "requirements_constraints.txt"
REQ_BUNDLE = PROJECT_ROOT / "requirements_runtime_bundle.txt"
ENV_FILE = PROJECT_ROOT / ".env"

UNINSTALL_CONFLICTS = [
    "tensorflow",
    "tensorflow-intel",
    "mediapipe",
    "protobuf",
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "keras",
    "h5py",
]


@dataclass(frozen=True)
class Requirement:
    name: str
    spec: str | None


def _parse_requirements(path: Path) -> list[Requirement]:
    if not path.exists():
        return []
    out: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*([<>=!~].+)?$", line)
        if not match:
            continue
        out.append(Requirement(match.group(1), match.group(2).strip() if match.group(2) else None))
    return out


def _safe_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:
        return None


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[^\d]+", text) if x.isdigit())


def _version_ok(installed: str | None, spec: str | None) -> bool:
    if not spec:
        return installed is not None
    if installed is None:
        return False
    if spec.startswith("=="):
        return installed == spec[2:].strip()
    if spec.startswith(">="):
        return _version_tuple(installed) >= _version_tuple(spec[2:].strip())
    return True


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _write_env(path: Path, updates: dict[str, str]) -> None:
    current = _load_env(path)
    current.update(updates)
    lines = [f"{k}={v}" for k, v in current.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask(value: str | None) -> str:
    if not value:
        return "未设置"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]}"


class InstallerUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("HSE Emotion Assistant Setup")
        self.root.geometry("1040x760")

        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False

        self.var_clean = tk.BooleanVar(value=True)
        self.var_install_stable = tk.BooleanVar(value=True)
        self.var_install_llm = tk.BooleanVar(value=True)
        self.var_install_optional = tk.BooleanVar(value=False)
        self.var_pip_check = tk.BooleanVar(value=True)
        self.var_write_api = tk.BooleanVar(value=False)
        self.var_rag = tk.BooleanVar(value=False)

        env = _load_env(ENV_FILE)
        self.api_key_var = tk.StringVar(value=env.get("DASHSCOPE_API_KEY", ""))
        self.base_url_var = tk.StringVar(value=env.get("DASHSCOPE_BASE_HTTP_API_URL", ""))
        self.qwen_model_var = tk.StringVar(value=env.get("QWEN_MODEL", "qwen3.5-plus"))
        self.omni_model_var = tk.StringVar(value=env.get("HSEMOTION_OMNI_MODEL", "qwen3-omni-flash-realtime"))
        self.engine_var = tk.StringVar(value=env.get("HSEMOTION_EMOTION_ENGINE", "libreface"))

        self.status_var = tk.StringVar(value="就绪")
        self.summary_var = tk.StringVar(value="")
        self.api_source_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.after(80, self._drain)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=14, pady=12)
        ttk.Label(top, text="HSE Emotion Assistant Setup", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(top, text="Install dependencies, configure your own DashScope API key, and validate the runtime in one place.", foreground="#555").pack(anchor="w")

        body = tk.Frame(self.root)
        body.pack(fill=tk.X, padx=14, pady=(0, 8))

        install = tk.LabelFrame(body, text="安装选项")
        install.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Checkbutton(install, text="修复核心依赖", variable=self.var_install_stable).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Checkbutton(install, text="安装 LLM / 语音 / UI 依赖", variable=self.var_install_llm).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(install, text="安装可选视觉后端（DeepFace / LibreFace）", variable=self.var_install_optional).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(install, text="安装前清理冲突包", variable=self.var_clean).pack(anchor="w", padx=12, pady=2)
        ttk.Checkbutton(install, text="安装后执行 pip check", variable=self.var_pip_check).pack(anchor="w", padx=12, pady=(2, 10))

        api = tk.LabelFrame(body, text="API 配置（必填后才能调用模型）")
        api.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Checkbutton(api, text="把下面的配置写入项目 .env", variable=self.var_write_api).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 8))
        ttk.Label(api, text="DashScope API Key").grid(row=1, column=0, sticky="w", padx=12, pady=4)
        ttk.Entry(api, textvariable=self.api_key_var, show="*", width=42).grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Label(api, text="Base HTTP URL").grid(row=2, column=0, sticky="w", padx=12, pady=4)
        ttk.Entry(api, textvariable=self.base_url_var, width=42).grid(row=2, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Label(api, text="文本模型").grid(row=3, column=0, sticky="w", padx=12, pady=4)
        ttk.Entry(api, textvariable=self.qwen_model_var, width=42).grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Label(api, text="语音助手模型").grid(row=4, column=0, sticky="w", padx=12, pady=4)
        ttk.Entry(api, textvariable=self.omni_model_var, width=42).grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Label(api, text="默认视觉后端").grid(row=5, column=0, sticky="w", padx=12, pady=4)
        ttk.Combobox(api, textvariable=self.engine_var, values=("libreface", "tf_mini_xception", "deepface"), state="readonly").grid(row=5, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Checkbutton(api, text="默认启用知识库", variable=self.var_rag).grid(row=6, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 10))
        api.grid_columnconfigure(1, weight=1)

        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=14, pady=(0, 8))
        ttk.Button(buttons, text="环境检测", command=self.start_detect).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="一键安装 / 修复 / 配置", command=self.start_install).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="仅保存 API", command=self.save_api_only).pack(side=tk.LEFT)
        ttk.Button(buttons, text="打开项目目录", command=self.open_project_dir).pack(side=tk.RIGHT)

        status = tk.Frame(self.root)
        status.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Label(status, text="状态：").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.api_source_var, foreground="#555").pack(side=tk.RIGHT)
        ttk.Label(status, textvariable=self.summary_var, foreground="#444").pack(side=tk.RIGHT, padx=(0, 18))

        ttk.Progressbar(self.root, variable=self.progress_var, maximum=100).pack(fill=tk.X, padx=14, pady=(0, 10))

        log_frame = tk.LabelFrame(self.root, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 10))
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scroll.set)

    def open_project_dir(self) -> None:
        try:
            os.startfile(PROJECT_ROOT)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _queue_log(self, text: str, level: str = "INFO") -> None:
        self.queue.put(("log", f"{level}|{text}"))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    level, text = payload.split("|", 1)
                    self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] [{level}] {text}\n")
                    self.log.see(tk.END)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "summary":
                    self.summary_var.set(payload)
                elif kind == "api":
                    self.api_source_var.set(payload)
                elif kind == "progress":
                    self.progress_var.set(float(payload))
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._drain)

    def _selected_requirements(self) -> list[Requirement]:
        reqs: list[Requirement] = []
        if self.var_install_stable.get():
            reqs.extend(_parse_requirements(REQ_STABLE))
        if self.var_install_llm.get():
            reqs.extend(_parse_requirements(REQ_LLM))
        if self.var_install_optional.get():
            reqs.extend(_parse_requirements(REQ_OPTIONAL))
        unique: list[Requirement] = []
        seen: set[str] = set()
        for req in reqs:
            key = req.name.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(req)
        return unique

    def _write_bundle(self, reqs: list[Requirement]) -> None:
        lines = [f"{r.name}{r.spec or ''}" for r in reqs]
        REQ_BUNDLE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _current_api_source(self) -> str:
        env = _load_env(ENV_FILE)
        file_key = env.get("DASHSCOPE_API_KEY")
        sys_key = os.getenv("DASHSCOPE_API_KEY")
        if sys_key:
            return f"当前 API：系统环境变量（{_mask(sys_key)}）"
        if file_key:
            return f"当前 API：项目 .env（{_mask(file_key)}）"
        return "当前 API：未配置"

    def _save_api(self) -> None:
        updates = {
            "DASHSCOPE_API_KEY": self.api_key_var.get().strip(),
            "QWEN_MODEL": self.qwen_model_var.get().strip() or "qwen3.5-plus",
            "HSEMOTION_OMNI_MODEL": self.omni_model_var.get().strip() or "qwen3-omni-flash-realtime",
            "HSEMOTION_EMOTION_ENGINE": self.engine_var.get().strip() or "libreface",
            "HSEMOTION_RAG_ENABLED": "1" if self.var_rag.get() else "0",
        }
        base_url = self.base_url_var.get().strip()
        if base_url:
            updates["DASHSCOPE_BASE_HTTP_API_URL"] = base_url
        _write_env(ENV_FILE, updates)

    def save_api_only(self) -> None:
        if not self.var_write_api.get():
            messagebox.showinfo("提示", "请先勾选写入 .env。")
            return
        if not self.api_key_var.get().strip():
            messagebox.showerror("缺少 API Key", "请先输入你自己的 DASHSCOPE_API_KEY。")
            return
        self._save_api()
        self.api_source_var.set(self._current_api_source())
        messagebox.showinfo("完成", f"已更新 {ENV_FILE}")

    def start_detect(self) -> None:
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._detect_worker, daemon=True).start()

    def start_install(self) -> None:
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _detect_worker(self) -> None:
        try:
            reqs = self._selected_requirements()
            self.queue.put(("status", "正在检测环境…"))
            self.queue.put(("progress", "5"))
            self._queue_log(f"Python 版本: {sys.version.split()[0]}")
            self._queue_log(f"Python 路径: {sys.executable}")
            if os.getenv("CONDA_PREFIX"):
                self._queue_log(f"当前 Conda 环境: {os.getenv('CONDA_PREFIX')}")
            ok_count = 0
            for idx, req in enumerate(reqs, start=1):
                installed = _safe_version(req.name)
                ok = _version_ok(installed, req.spec)
                self._queue_log(f"{req.name}{req.spec or ''} => {installed or '缺失'} [{'OK' if ok else 'WARN'}]", "INFO" if ok else "WARN")
                ok_count += int(ok)
                self.queue.put(("progress", str(5 + int(70 * idx / max(1, len(reqs))))))
            self._queue_log(self._current_api_source())
            self.queue.put(("api", self._current_api_source()))
            self.queue.put(("summary", f"依赖通过 {ok_count}/{len(reqs)}"))
            self.queue.put(("status", "检测完成"))
            self.queue.put(("progress", "100"))
        finally:
            self.busy = False

    def _install_worker(self) -> None:
        try:
            reqs = self._selected_requirements()
            self._write_bundle(reqs)
            self.queue.put(("status", "开始安装 / 修复 / 配置…"))
            self.queue.put(("progress", "3"))
            if self.var_write_api.get():
                if not self.api_key_var.get().strip():
                    raise RuntimeError("未填写 DASHSCOPE_API_KEY。开源版不提供默认 API，请输入你自己的 API Key。")
                self._queue_log("写入 .env 中的 API 与模型配置…")
                self._save_api()
                self.queue.put(("api", self._current_api_source()))
            if self.var_clean.get():
                self._queue_log("清理冲突包…")
                self._run_command([sys.executable, "-m", "pip", "uninstall", "-y", *UNINSTALL_CONFLICTS], allow_fail=True)
                self.queue.put(("progress", "15"))
            if reqs:
                self._queue_log("单次安装/修复全部已选依赖…")
                self._run_command([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--upgrade-strategy",
                    "only-if-needed",
                    "--constraint",
                    str(REQ_CONSTRAINTS),
                    "-r",
                    str(REQ_BUNDLE),
                ])
                self.queue.put(("progress", "82"))
            if self.var_pip_check.get():
                self._queue_log("执行 pip check…")
                self._run_command([sys.executable, "-m", "pip", "check"], allow_fail=True)
                self.queue.put(("progress", "90"))
            self._queue_log("安装完成，开始复检。")
            self._detect_worker()
        except Exception as exc:
            self._queue_log(str(exc), "ERROR")
            self.queue.put(("status", "安装失败"))
            self.busy = False

    def _run_command(self, cmd: list[str], *, allow_fail: bool = False) -> None:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                self._queue_log(line)
        code = proc.wait()
        if code != 0 and not allow_fail:
            raise RuntimeError(f"命令执行失败，退出码={code}: {' '.join(cmd)}")
        if code != 0:
            self._queue_log(f"命令退出码={code}", "WARN")
        else:
            self._queue_log("命令执行完成", "OK")

    def run(self) -> None:
        self.api_source_var.set(self._current_api_source())
        self.root.mainloop()


def main() -> None:
    InstallerUI().run()


if __name__ == "__main__":
    main()

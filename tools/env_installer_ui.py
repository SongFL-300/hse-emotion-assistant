from __future__ import annotations

import importlib
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
REQ_STABLE = PROJECT_ROOT / "requirements_stable.txt"
REQ_LLM = PROJECT_ROOT / "requirements_llm_ui_v0.1.3.txt"
REQ_STABLE_CLEAN = PROJECT_ROOT / "requirements_stable.clean.txt"
REQ_LLM_CLEAN = PROJECT_ROOT / "requirements_llm_ui_v0.1.3.clean.txt"
REQ_OPTIONAL_CLEAN = PROJECT_ROOT / "requirements_optional.clean.txt"
REQ_CONSTRAINTS = PROJECT_ROOT / "requirements_constraints.txt"
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


OPTIONAL_PACKAGES = [
    Requirement(name="deepface", spec=None),
    Requirement(name="libreface", spec=None),
]


def _strip_inline_comment(line: str) -> str:
    # Remove inline comments while preserving URLs or hashes inside quotes.
    if "#" not in line:
        return line
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line


def _ensure_clean_requirements(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    lines = []
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line).strip()
        if line:
            lines.append(line)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _build_constraints(paths: list[Path], dst: Path) -> None:
    # Keep only pinned versions (==) from the main requirements as constraints.
    pins: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = _strip_inline_comment(line).strip()
            if "==" in line:
                pins.append(line)
    dst.write_text("\n".join(pins) + ("\n" if pins else ""), encoding="utf-8")


def _parse_requirements(path: Path) -> list[Requirement]:
    if not path.exists():
        return []
    out: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line).strip()
        if not line:
            continue
        # very small parser: "pkg==x", "pkg>=x", "pkg"
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([<>=!~].+)?$", line)
        if not m:
            continue
        name = m.group(1).strip()
        spec = m.group(2).strip() if m.group(2) else None
        out.append(Requirement(name=name, spec=spec))
    return out


def _safe_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:
        return None


def _version_tuple(v: str) -> tuple:
    parts = re.split(r"[^\d]+", v)
    nums = [int(p) for p in parts if p.isdigit()]
    return tuple(nums)


def _version_satisfy(installed: str | None, spec: str | None) -> bool:
    if not spec:
        return installed is not None
    if installed is None:
        return False
    spec = spec.strip()
    if spec.startswith("=="):
        return installed == spec[2:].strip()
    if spec.startswith(">="):
        return _version_tuple(installed) >= _version_tuple(spec[2:].strip())
    if spec.startswith("<="):
        return _version_tuple(installed) <= _version_tuple(spec[2:].strip())
    if spec.startswith(">"):
        return _version_tuple(installed) > _version_tuple(spec[1:].strip())
    if spec.startswith("<"):
        return _version_tuple(installed) < _version_tuple(spec[1:].strip())
    return True


class InstallerUI:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.title("HSE Emotion Chat 环境检测与安装")
        self._root.geometry("980x700")

        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._busy = False

        self._var_clean = tk.BooleanVar(value=True)
        self._var_install_llm = tk.BooleanVar(value=True)
        self._var_install_stable = tk.BooleanVar(value=True)
        self._var_install_optional = tk.BooleanVar(value=False)

        self._status_var = tk.StringVar(value="就绪")
        self._env_summary_var = tk.StringVar(value="")
        self._progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._root.after(80, self._drain_queue)

    def run(self) -> None:
        self._root.mainloop()

    def _build_ui(self) -> None:
        top = tk.Frame(self._root)
        top.pack(fill=tk.X, padx=12, pady=10)

        ttk.Label(top, text="HSE Emotion Chat 环境检测与安装", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        ttk.Button(top, text="打开项目目录", command=self._open_project_dir).pack(side=tk.RIGHT)

        mid = tk.Frame(self._root)
        mid.pack(fill=tk.X, padx=12, pady=(0, 8))

        opt = tk.LabelFrame(mid, text="安装选项")
        opt.pack(fill=tk.X, padx=2, pady=4)

        tk.Checkbutton(opt, text="清理冲突包（更稳，但更慢）", variable=self._var_clean).pack(anchor="w", padx=10, pady=4)
        tk.Checkbutton(opt, text="安装视觉稳定依赖 requirements_stable.txt", variable=self._var_install_stable).pack(anchor="w", padx=10)
        tk.Checkbutton(opt, text="安装 LLM/UI 依赖 requirements_llm_ui_v0.1.3.txt", variable=self._var_install_llm).pack(anchor="w", padx=10, pady=(0, 6))
        tk.Checkbutton(opt, text="安装可选视觉后端（DeepFace / LibreFace）", variable=self._var_install_optional).pack(anchor="w", padx=10, pady=(0, 6))

        btns = tk.Frame(mid)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="环境检测", command=self._start_detect).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="一键安装/修复", command=self._start_install).pack(side=tk.LEFT)
        ttk.Button(btns, text="导出检测报告", command=self._export_report).pack(side=tk.RIGHT)

        status = tk.Frame(self._root)
        status.pack(fill=tk.X, padx=12, pady=(0, 6))
        ttk.Label(status, text="状态：").pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._env_summary_var, foreground="#444").pack(side=tk.RIGHT)

        bar = ttk.Progressbar(self._root, variable=self._progress_var, maximum=100)
        bar.pack(fill=tk.X, padx=12, pady=(0, 10))

        log_frame = tk.LabelFrame(self._root, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._log = tk.Text(log_frame, height=24, wrap="word", font=("Consolas", 10))
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self._log.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.configure(yscrollcommand=scroll.set)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _log_line(self, text: str, level: str = "INFO") -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.insert(tk.END, f"[{ts}] [{level}] {text}\n")
        self._log.see(tk.END)

    def _queue_log(self, text: str, level: str = "INFO") -> None:
        self._queue.put(("log", f"{level}|{text}"))

    def _queue_status(self, text: str) -> None:
        self._queue.put(("status", text))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    level, msg = payload.split("|", 1)
                    self._log_line(msg, level=level)
                elif kind == "status":
                    self._set_status(payload)
                elif kind == "progress":
                    self._progress_var.set(float(payload))
                elif kind == "summary":
                    self._env_summary_var.set(payload)
        except queue.Empty:
            pass
        finally:
            self._root.after(80, self._drain_queue)

    def _open_project_dir(self) -> None:
        try:
            os.startfile(PROJECT_ROOT)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showerror("打开失败", f"无法打开目录：{PROJECT_ROOT}")

    def _export_report(self) -> None:
        report = self._build_report()
        out = PROJECT_ROOT / "env_check_report.txt"
        out.write_text(report, encoding="utf-8")
        messagebox.showinfo("完成", f"已导出报告：{out}")

    def _build_report(self) -> str:
        self._prepare_clean_requirements()
        lines = []
        lines.append("HSE Emotion Chat 环境检测报告")
        lines.append("=" * 48)
        lines.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"Python 可执行: {sys.executable}")
        lines.append(f"CONDA_PREFIX: {os.getenv('CONDA_PREFIX', '')}")
        lines.append(f"VENV: {os.getenv('VIRTUAL_ENV', '')}")
        lines.append("")
        lines.append("依赖检测:")
        for req in self._collect_requirements():
            installed = _safe_version(req.name)
            ok = _version_satisfy(installed, req.spec)
            lines.append(f"- {req.name}{req.spec or ''} => {installed or '缺失'} [{'OK' if ok else 'FAIL'}]")
        lines.append("")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        lines.append(f"DASHSCOPE_API_KEY: {'已设置' if api_key else '未设置'}")
        lines.append(f".env 文件: {'存在' if ENV_FILE.exists() else '不存在'} ({ENV_FILE})")
        return "\n".join(lines)

    def _prepare_clean_requirements(self) -> None:
        _ensure_clean_requirements(REQ_STABLE, REQ_STABLE_CLEAN)
        _ensure_clean_requirements(REQ_LLM, REQ_LLM_CLEAN)
        if not REQ_OPTIONAL_CLEAN.exists():
            lines = [f"{r.name}{r.spec or ''}" for r in OPTIONAL_PACKAGES]
            REQ_OPTIONAL_CLEAN.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        _build_constraints([REQ_STABLE_CLEAN, REQ_LLM_CLEAN], REQ_CONSTRAINTS)

    def _collect_requirements(self) -> list[Requirement]:
        self._prepare_clean_requirements()
        reqs: list[Requirement] = []
        if self._var_install_stable.get():
            reqs.extend(_parse_requirements(REQ_STABLE_CLEAN if REQ_STABLE_CLEAN.exists() else REQ_STABLE))
        if self._var_install_llm.get():
            reqs.extend(_parse_requirements(REQ_LLM_CLEAN if REQ_LLM_CLEAN.exists() else REQ_LLM))
        if self._var_install_optional.get():
            reqs.extend(_parse_requirements(REQ_OPTIONAL_CLEAN))
        # 去重（按 name）
        seen: set[str] = set()
        out: list[Requirement] = []
        for r in reqs:
            if r.name.lower() in seen:
                continue
            seen.add(r.name.lower())
            out.append(r)
        return out

    def _start_detect(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._worker = threading.Thread(target=self._detect_worker, daemon=True)
        self._worker.start()

    def _start_install(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._worker = threading.Thread(target=self._install_worker, daemon=True)
        self._worker.start()

    def _detect_worker(self) -> None:
        self._prepare_clean_requirements()
        self._queue_status("正在检测环境…")
        self._queue.put(("progress", "5"))
        self._queue_log("开始检测 Python 与依赖环境。")

        python_version = sys.version.split()[0]
        self._queue_log(f"Python 版本: {python_version}")
        self._queue_log(f"Python 路径: {sys.executable}")

        conda = os.getenv("CONDA_PREFIX")
        venv = os.getenv("VIRTUAL_ENV")
        if conda:
            self._queue_log(f"当前 Conda 环境: {conda}")
        if venv:
            self._queue_log(f"当前 venv 环境: {venv}")

        reqs = self._collect_requirements()
        ok_count = 0
        for i, req in enumerate(reqs, start=1):
            installed = _safe_version(req.name)
            ok = _version_satisfy(installed, req.spec)
            level = "OK" if ok else "WARN"
            self._queue_log(
                f"{req.name}{req.spec or ''} => {installed or '缺失'} [{level}]",
                level="INFO" if ok else "WARN",
            )
            ok_count += 1 if ok else 0
            self._queue.put(("progress", str(5 + int(70 * i / max(1, len(reqs))))))

        api_key = os.getenv("DASHSCOPE_API_KEY")
        self._queue_log(f"DASHSCOPE_API_KEY: {'已设置' if api_key else '未设置'}")
        self._queue_log(f".env 文件: {'存在' if ENV_FILE.exists() else '不存在'} ({ENV_FILE})")

        summary = f"依赖通过 {ok_count}/{len(reqs)}"
        self._queue.put(("summary", summary))
        self._queue_status("检测完成")
        self._queue.put(("progress", "100"))
        self._busy = False

    def _install_worker(self) -> None:
        self._queue_status("开始安装/修复…")
        self._queue.put(("progress", "3"))
        self._queue_log("准备安装依赖。")

        if self._var_clean.get():
            self._queue_log("清理冲突包中…")
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *UNINSTALL_CONFLICTS]
            self._run_command(cmd)

        steps = []
        if self._var_install_stable.get():
            steps.append(
                (
                    "安装稳定依赖",
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--upgrade-strategy",
                        "only-if-needed",
                        "-r",
                        str(REQ_STABLE_CLEAN if REQ_STABLE_CLEAN.exists() else REQ_STABLE),
                    ],
                )
            )
        if self._var_install_llm.get():
            steps.append(
                (
                    "安装 LLM/UI 依赖",
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--upgrade-strategy",
                        "only-if-needed",
                        "-r",
                        str(REQ_LLM_CLEAN if REQ_LLM_CLEAN.exists() else REQ_LLM),
                    ],
                )
            )

        for idx, (title, cmd) in enumerate(steps, start=1):
            self._queue_log(f"{title}…")
            self._run_command(cmd)
            self._queue.put(("progress", str(10 + int(80 * idx / max(1, len(steps))))))

        if self._var_install_optional.get():
            self._queue_log("安装可选视觉后端…")
            self._run_command(
                [
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
                    str(REQ_OPTIONAL_CLEAN),
                ]
            )

        self._queue_log("安装完成，开始重新检测…")
        self._detect_worker()

    def _run_command(self, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception as exc:
            self._queue_log(f"命令启动失败: {exc}", level="ERROR")
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            self._queue_log(line.rstrip())
        code = proc.wait()
        if code != 0:
            self._queue_log(f"命令执行失败，退出码={code}", level="ERROR")
        else:
            self._queue_log("命令执行完成", level="OK")


def main() -> None:
    app = InstallerUI()
    app.run()


if __name__ == "__main__":
    main()

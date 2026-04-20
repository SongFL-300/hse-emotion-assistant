from __future__ import annotations

from pathlib import Path
import os


def find_dotenv(start_dir: Path | None = None, filename: str = ".env") -> Path | None:
    start = (start_dir or Path.cwd()).resolve()
    for base in (start, *start.parents):
        candidate = base / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_dotenv(
    dotenv_path: Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """
    轻量 .env 加载器（不额外依赖 python-dotenv）。

    - 默认从当前目录向上查找 `.env`
    - `override=False` 时，已有环境变量优先（更安全）
    """
    path: Path | None = None
    if dotenv_path is not None:
        path = Path(dotenv_path)
    else:
        # 可选：显式指定 .env 路径，避免“跑错工作目录”
        explicit = os.getenv("HSEMOTION_DOTENV_PATH")
        if explicit and explicit.strip():
            path = Path(explicit.strip())
        else:
            # 优先按本模块所在目录向上找（最符合“项目根目录 .env”的预期）
            path = find_dotenv(start_dir=Path(__file__).resolve().parent)
            # 兜底：再按当前工作目录向上找（常规 CLI / 终端运行）
            if not path:
                path = find_dotenv(start_dir=Path.cwd())
    if not path:
        return None

    # utf-8-sig 可自动吞掉 BOM（Windows 里很常见）
    content = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    for raw in content:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if not key:
            continue

        # 去掉引号（不做转义解析，保持简单）
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            # 非引号值：允许行尾注释（以空格#开头）
            for token in (" #", "\t#"):
                if token in value:
                    value = value.split(token, 1)[0].rstrip()
                    break

        if not override and key in os.environ and os.environ.get(key, ""):
            continue
        os.environ[key] = value

    return path

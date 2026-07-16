from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from utils.log_paths import get_service_log_dir


@dataclass(frozen=True)
class ReflectionConfig:
    runtime_dir: Path
    db_path: Path
    prompt_dir: Path
    export_root: Path
    reflection_base_url: str


def load_config() -> ReflectionConfig:
    runtime_dir = Path(get_service_log_dir()) / "thinking"
    port = (os.getenv("PROXY_PORT") or "4000").strip()
    return ReflectionConfig(
        runtime_dir=runtime_dir,
        db_path=runtime_dir / "thinking.db",
        prompt_dir=Path(os.getenv("REFLECTION_PROMPT_DIR", "src/thinking_reflection/prompt")),
        export_root=Path(os.getenv("REFLECTION_EXPORT_ROOT", "logs_session_eval/reflection")),
        reflection_base_url=f"http://127.0.0.1:{port}",
    )

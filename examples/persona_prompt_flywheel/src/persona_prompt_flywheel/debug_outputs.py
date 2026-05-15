from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from persona_prompt_flywheel.io_utils import ensure_dir, write_json, write_jsonl


def write_agent_debug_bundle(
    *,
    run_dir: str | Path,
    agent_name: str,
    input_snapshot: Any,
    output_files: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write per-agent debug inputs, file copies, and a manifest."""

    agent_dir = ensure_dir(Path(run_dir) / "output" / agent_name)
    _write_snapshot(agent_dir, input_snapshot)

    copied_files: list[str] = []
    missing_files: list[str] = []
    for file_path in output_files:
        source = Path(run_dir) / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if source.exists() and source.is_file():
            destination = agent_dir / source.name
            shutil.copy2(source, destination)
            copied_files.append(destination.name)
        else:
            missing_files.append(str(file_path))

    manifest = {
        "agent_name": agent_name,
        "debug_directory": str(agent_dir),
        "copied_output_files": copied_files,
        "missing_output_files": missing_files,
        "metadata": _json_safe(metadata or {}),
    }
    write_json(agent_dir / "output_manifest.json", manifest)
    return agent_dir


def _write_snapshot(agent_dir: Path, snapshot: Any) -> None:
    safe_snapshot = _json_safe(snapshot)
    if isinstance(safe_snapshot, list):
        write_jsonl(agent_dir / "input_snapshot.jsonl", safe_snapshot)
    else:
        write_json(agent_dir / "input_snapshot.json", safe_snapshot)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


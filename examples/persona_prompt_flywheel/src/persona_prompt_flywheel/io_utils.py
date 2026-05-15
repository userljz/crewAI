from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return dict(data)


def write_yaml(path: str | Path, data: JsonDict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def read_jsonl(path: str | Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return records
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: list[JsonDict]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: str | Path, rows: list[JsonDict], fieldnames: list[str] | None = None) -> None:
    if not fieldnames:
        fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


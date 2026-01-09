"""Shared path helpers for turn-based artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SessionPaths:
    root: Path
    session_id: str

    @property
    def session_root(self) -> Path:
        return self.root / "sessions" / self.session_id

    @property
    def images_dir(self) -> Path:
        return self.session_root / "images"

    @property
    def turns_dir(self) -> Path:
        return self.session_root / "turns"

    def turn_dir(self, turn_idx: int) -> Path:
        return self.turns_dir / f"turn_{turn_idx:04d}"

    def ground_dir(self, turn_idx: int, ground_id: int) -> Path:
        return self.turn_dir(turn_idx) / "ground" / f"ground_{ground_id:04d}"

    def attn_dir(self, turn_idx: int, kind: str, attn_id: int) -> Path:
        if kind not in {"i2t", "t2i"}:
            raise ValueError("kind must be 'i2t' or 't2i'")
        return self.turn_dir(turn_idx) / "attn" / kind / f"attn_{attn_id:04d}"

    def image_path(self, image_id: str) -> Path:
        return self.images_dir / image_id


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

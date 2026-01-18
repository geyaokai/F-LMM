"""Prompt override helpers for backend/worker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from mmengine.config import Config


def _read_prompt_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(path).expanduser()
    if not payload_path.is_absolute():
        payload_path = Path.cwd() / payload_path
    text = payload_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {"system": text}
    return {"system": text}


def apply_prompt_overrides(cfg: Config, args, prompt_file: Optional[str]) -> None:
    overrides = _read_prompt_file(prompt_file)
    if not overrides:
        return
    if "extra_prompt" in overrides:
        args.extra_prompt = overrides.get("extra_prompt") or ""
    if "roi_extra_prompt" in overrides:
        args.roi_extra_prompt = overrides.get("roi_extra_prompt") or ""
    if "phrase_extract_prompt" in overrides:
        args.phrase_extract_prompt = overrides.get("phrase_extract_prompt") or ""
    if "phrase_rerank_prompt" in overrides:
        args.phrase_rerank_prompt = overrides.get("phrase_rerank_prompt") or ""
    system_text = overrides.get("system_prompt") or overrides.get("system")
    if system_text:
        if not hasattr(cfg, "prompt_template") or cfg.prompt_template is None:
            cfg.prompt_template = {}
        cfg.prompt_template["SYSTEM"] = str(system_text)
    template_override = overrides.get("prompt_template")
    if isinstance(template_override, dict):
        if not hasattr(cfg, "prompt_template") or cfg.prompt_template is None:
            cfg.prompt_template = {}
        cfg.prompt_template.update(template_override)

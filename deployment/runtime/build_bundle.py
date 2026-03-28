#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INCLUDE_PATHS = [
    "LICENSE",
    "README.md",
    "configs",
    "deepseek_vl",
    "deployment/runtime",
    "flmm",
    "hpt",
    "llava",
    "mgm",
    "requirements",
    "scripts/demo",
    "segment_anything",
    "src/mmdet",
    "xtuner-cfg",
]

DEFAULT_EXCLUDE_PATHS = [
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "build",
    "dist",
    "results",
    "results_iso_debug",
    "tensorboard",
    "wandb",
    "work_dirs",
    "scripts/demo/showcase_bundle",
    "scripts/demo/showcase_bundle.v1.tar.gz",
    "scripts/demo/web/frontend",
    "tests",
]

OPTIONAL_MODEL_ASSET_PATHS = [
    "checkpoints",
    "data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a versioned runtime bundle without the deprecated frontend."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runtime",
        help="Directory where the versioned tar.gz bundle will be written.",
    )
    parser.add_argument(
        "--bundle-prefix",
        default="flmm-runtime",
        help="Bundle filename prefix.",
    )
    parser.add_argument(
        "--include-model-assets",
        action="store_true",
        help="Also include ignored model assets such as checkpoints/ and data/.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the temporary staging directory after the tarball is created.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def git_metadata() -> dict:
    status_output = git_output("status", "--short")
    status_lines = [line for line in status_output.splitlines() if line.strip()]
    return {
        "short_sha": git_output("rev-parse", "--short", "HEAD") or "nogit",
        "full_sha": git_output("rev-parse", "HEAD") or "",
        "dirty": bool(status_lines),
        "status_short": status_lines,
    }


def bundle_name(prefix: str, short_sha: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}-{stamp}-{short_sha}"


def normalize_rel(path: str | Path) -> str:
    return Path(path).as_posix().strip("/")


def should_exclude(rel_path: str, exclude_paths: Iterable[str]) -> bool:
    normalized = normalize_rel(rel_path)
    for excluded in exclude_paths:
        excluded_norm = normalize_rel(excluded)
        if not excluded_norm:
            continue
        if normalized == excluded_norm or normalized.startswith(f"{excluded_norm}/"):
            return True
    return False


def build_ignore(root: Path, exclude_paths: List[str]):
    root = root.resolve()

    def _ignore(current_dir: str, names: List[str]) -> List[str]:
        ignored: List[str] = []
        current_path = Path(current_dir).resolve()
        try:
            current_rel = current_path.relative_to(root).as_posix()
        except ValueError:
            current_rel = ""
        for name in names:
            rel = "/".join(part for part in [current_rel, name] if part)
            if should_exclude(rel, exclude_paths):
                ignored.append(name)
        return ignored

    return _ignore


def copy_entry(src_rel: str, stage_root: Path, exclude_paths: List[str]) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        raise FileNotFoundError(f"Required path does not exist: {src_rel}")
    dst = stage_root / src_rel
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=build_ignore(REPO_ROOT, exclude_paths),
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_runtime_metadata(
    stage_root: Path,
    *,
    name: str,
    git_info: dict,
    include_paths: List[str],
    exclude_paths: List[str],
    include_model_assets: bool,
) -> None:
    version_path = stage_root / "VERSION"
    version_path.write_text(f"{name}\n", encoding="utf-8")

    manifest = {
        "schema_version": "runtime_bundle_v1",
        "bundle_name": name,
        "created_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "git": git_info,
        "build_options": {
            "include_model_assets": include_model_assets,
            "included_paths": include_paths,
            "excluded_paths": exclude_paths,
        },
        "runtime": {
            "backend_entry": "scripts.demo.web.backend.main:app",
            "worker_entry": "python -m scripts.demo.web.backend.task_queue.worker --sleep 0.5",
            "env_example": "deployment/runtime/.env.example",
            "smoke_test": "deployment/runtime/smoke_test.py",
            "start_backend_script": "deployment/runtime/bin/start_backend.sh",
            "start_worker_script": "deployment/runtime/bin/start_worker.sh",
        },
        "notes": [
            "Historical in-repo frontend prototype path is excluded when present.",
            "Model checkpoints and data are excluded by default; add --include-model-assets when needed.",
        ],
    }
    (stage_root / "RUNTIME_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def make_tarball(stage_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(stage_root, arcname=stage_root.name)


def main() -> None:
    args = parse_args()
    git_info = git_metadata()
    name = bundle_name(args.bundle_prefix, git_info["short_sha"])
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_paths = list(DEFAULT_INCLUDE_PATHS)
    exclude_paths = list(DEFAULT_EXCLUDE_PATHS)
    if args.include_model_assets:
        include_paths.extend(OPTIONAL_MODEL_ASSET_PATHS)

    staging_parent = Path(
        tempfile.mkdtemp(prefix=f"{name}-", dir=str(output_dir))
    ).resolve()
    stage_root = staging_parent / name
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        for rel_path in include_paths:
            if should_exclude(rel_path, exclude_paths):
                continue
            copy_entry(rel_path, stage_root, exclude_paths)
        write_runtime_metadata(
            stage_root,
            name=name,
            git_info=git_info,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            include_model_assets=args.include_model_assets,
        )
        tarball_path = output_dir / f"{name}.tar.gz"
        make_tarball(stage_root, tarball_path)
        print(f"[runtime-bundle] created: {tarball_path}")
        print(f"[runtime-bundle] staging: {stage_root}")
        print(f"[runtime-bundle] files: {count_files(stage_root)}")
    finally:
        if not args.keep_staging:
            shutil.rmtree(staging_parent, ignore_errors=True)


if __name__ == "__main__":
    main()

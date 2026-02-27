#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Model artifact resolution for local files or ModelScope downloads."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path


GLOBAL_MODELSCOPE_REPO_ENV = "DEEPDOC_MODELSCOPE_REPO"
GLOBAL_MODELSCOPE_REVISION_ENV = "DEEPDOC_MODELSCOPE_REVISION"


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or os.getenv("DEEPDOC_MODEL_PROVIDER", "auto")).strip().lower()
    aliases = {
        "ms": "modelscope",
        "remote": "modelscope",
        "filesystem": "local",
        "user": "local",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "local", "modelscope"}:
        raise ValueError("Unsupported model provider '{}'. Use one of: auto, local, modelscope.".format(normalized))
    return normalized


def _model_home_path(model_home: str | None) -> Path:
    configured = model_home or os.getenv("DEEPDOC_MODEL_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home().joinpath(".cache", "deepdoc")


@dataclass(frozen=True)
class BundleSpec:
    name: str
    subdir: str
    required_files: tuple[str, ...]
    local_dir_env: str
    repo_env: str
    repo_default: str
    revision_env: str
    revision_default: str = "master"


BUNDLES: dict[str, BundleSpec] = {
    "vision": BundleSpec(
        name="vision",
        subdir="vision",
        required_files=(
            "det.onnx",
            "layout.onnx",
            "layout.laws.onnx",
            "layout.manual.onnx",
            "layout.paper.onnx",
            "ocr.res",
            "rec.onnx",
            "tsr.onnx",
        ),
        local_dir_env="DEEPDOC_VISION_MODEL_DIR",
        repo_env="DEEPDOC_MODELSCOPE_VISION_REPO",
        repo_default="Xorbits/deepdoc",
        revision_env="DEEPDOC_MODELSCOPE_VISION_REVISION",
    ),
    "xgb": BundleSpec(
        name="xgb",
        subdir="xgb",
        required_files=("updown_concat_xgb.model",),
        local_dir_env="DEEPDOC_XGB_MODEL_DIR",
        repo_env="DEEPDOC_MODELSCOPE_XGB_REPO",
        repo_default="Xorbits/deepdoc",
        revision_env="DEEPDOC_MODELSCOPE_XGB_REVISION",
    ),
    "tokenizer": BundleSpec(
        name="tokenizer",
        subdir="tokenizer",
        required_files=("huqie.txt",),
        local_dir_env="DEEPDOC_TOKENIZER_MODEL_DIR",
        repo_env="DEEPDOC_MODELSCOPE_TOKENIZER_REPO",
        repo_default="Xorbits/deepdoc",
        revision_env="DEEPDOC_MODELSCOPE_TOKENIZER_REVISION",
    ),
}


def _resolve_modelscope_repo_id(spec: BundleSpec) -> tuple[str, bool]:
    """Resolve ModelScope repo id, supporting a shared combined repo.

    Precedence:
      1) per-bundle env (e.g. DEEPDOC_MODELSCOPE_VISION_REPO)
      2) shared env (DEEPDOC_MODELSCOPE_REPO)
      3) per-bundle default (spec.repo_default)

    Returns (repo_id, use_shared_download_dir).
    """
    explicit = os.getenv(spec.repo_env)
    if explicit and explicit.strip():
        repo_id = explicit.strip()
        shared_repo = os.getenv(GLOBAL_MODELSCOPE_REPO_ENV)
        use_shared_dir = bool(shared_repo and shared_repo.strip() and shared_repo.strip() == repo_id)
        return repo_id, use_shared_dir

    shared_repo = os.getenv(GLOBAL_MODELSCOPE_REPO_ENV)
    if shared_repo and shared_repo.strip():
        return shared_repo.strip(), True

    repo_id = spec.repo_default.strip()
    # If all bundle defaults point at the same repo, treat it as a combined repo and share the download directory.
    default_repos = {bundle.repo_default.strip() for bundle in BUNDLES.values()}
    use_shared_dir = len(default_repos) == 1 and repo_id in default_repos
    return repo_id, use_shared_dir


def _resolve_modelscope_revision(spec: BundleSpec) -> str:
    """Resolve ModelScope revision with an optional shared default."""
    explicit = os.getenv(spec.revision_env)
    if explicit and explicit.strip():
        return explicit.strip()

    shared = os.getenv(GLOBAL_MODELSCOPE_REVISION_ENV)
    if shared and shared.strip():
        return shared.strip()

    return spec.revision_default


def _slugify_repo_path(value: str) -> str:
    return value.strip().replace("/", "__").replace(":", "__").replace("\\", "__")


def _modelscope_shared_download_dir(model_home: str | None, repo_id: str, revision: str) -> Path:
    """Stable directory for a combined repo snapshot to avoid collisions."""
    base = _model_home_path(model_home)
    return base.joinpath("modelscope", _slugify_repo_path(repo_id), _slugify_repo_path(revision))


def _validate_bundle_dir(spec: BundleSpec, base_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in spec.required_files if not base_dir.joinpath(name).exists()]
    return not missing, missing


def _discover_bundle_dir(spec: BundleSpec, roots: list[Path]) -> Path | None:
    for root in roots:
        exists, _ = _validate_bundle_dir(spec, root)
        if exists:
            return root

    for root in roots:
        if not root.exists():
            continue
        for required in spec.required_files:
            for hit in root.rglob(required):
                candidate = hit.parent
                exists, _ = _validate_bundle_dir(spec, candidate)
                if exists:
                    return candidate
    return None


def _import_modelscope_snapshot_download():
    try:
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

        return snapshot_download
    except Exception:
        try:
            from modelscope import snapshot_download  # type: ignore

            return snapshot_download
        except Exception as exc:  # pragma: no cover - import behavior depends on runtime env
            raise RuntimeError("ModelScope provider requires the 'modelscope' package. Install it or switch DEEPDOC_MODEL_PROVIDER=local.") from exc


def _download_modelscope_repo(*, repo_id: str, revision: str, target_dir: Path, offline: bool) -> Path:
    snapshot_download = _import_modelscope_snapshot_download()

    if not repo_id:
        raise RuntimeError(f"ModelScope repo id is empty. Set {GLOBAL_MODELSCOPE_REPO_ENV} or a bundle-specific env like DEEPDOC_MODELSCOPE_VISION_REPO.")

    target_dir.mkdir(parents=True, exist_ok=True)

    signature = inspect.signature(snapshot_download)
    params = signature.parameters
    kwargs: dict[str, object] = {}

    if "model_id" in params:
        kwargs["model_id"] = repo_id
    elif "repo_id" in params:
        kwargs["repo_id"] = repo_id

    if "revision" in params:
        kwargs["revision"] = revision

    if "cache_dir" in params:
        kwargs["cache_dir"] = str(target_dir.parent)

    if "local_dir" in params:
        kwargs["local_dir"] = str(target_dir)

    if "local_dir_use_symlinks" in params:
        kwargs["local_dir_use_symlinks"] = False

    if "local_files_only" in params:
        kwargs["local_files_only"] = offline

    if "model_id" in kwargs or "repo_id" in kwargs:
        snapshot_root = snapshot_download(**kwargs)
    else:
        snapshot_root = snapshot_download(repo_id, **kwargs)

    resolved_snapshot = Path(snapshot_root).expanduser().resolve()
    logging.info("Downloaded ModelScope repo %s@%s to %s", repo_id, revision, resolved_snapshot)
    return resolved_snapshot


def resolve_bundle_dir(
    bundle: str,
    *,
    model_home: str | None = None,
    provider: str | None = None,
    offline: bool | None = None,
) -> str:
    """Resolve a model bundle directory from local files or ModelScope."""

    if bundle not in BUNDLES:
        raise ValueError(f"Unknown model bundle '{bundle}'. Expected one of: {', '.join(BUNDLES)}")

    spec = BUNDLES[bundle]
    provider_name = _normalize_provider(provider)
    offline_mode = offline if offline is not None else _parse_bool(os.getenv("DEEPDOC_OFFLINE"), default=False)

    explicit_local = os.getenv(spec.local_dir_env)
    if explicit_local:
        local_bundle_dir = Path(explicit_local).expanduser().resolve()
    else:
        local_bundle_dir = _model_home_path(model_home).joinpath(spec.subdir)

    roots_to_scan = [local_bundle_dir]

    # Prefer reusing already-downloaded ModelScope artifacts from our stable
    # `model_home/modelscope/<repo>/<revision>/...` location when using shared repos.
    shared_download_dir: Path | None = None
    shared_repo_id: str | None = None
    shared_revision: str | None = None
    use_shared_repo_dir = False
    if provider_name in {"auto", "modelscope"} and not offline_mode:
        shared_repo_id, use_shared_repo_dir = _resolve_modelscope_repo_id(spec)
        if use_shared_repo_dir:
            shared_revision = _resolve_modelscope_revision(spec)
            shared_download_dir = _modelscope_shared_download_dir(model_home, shared_repo_id, shared_revision)
            roots_to_scan.append(shared_download_dir)

    discovered = _discover_bundle_dir(spec, roots_to_scan)
    if discovered:
        return str(discovered)

    if provider_name == "local":
        _, missing = _validate_bundle_dir(spec, local_bundle_dir)
        raise FileNotFoundError(
            "Missing required files for local '{}' bundle under {}: {}. Set {} or DEEPDOC_MODEL_HOME to a directory containing these files.".format(
                spec.name, local_bundle_dir, ", ".join(missing), spec.local_dir_env
            )
        )

    if provider_name == "local" or offline_mode:
        raise FileNotFoundError("Bundle '{}' was not found locally at {} and remote download is disabled. Disable DEEPDOC_OFFLINE or provide local model files.".format(spec.name, local_bundle_dir))

    repo_id = shared_repo_id
    revision = shared_revision
    if not repo_id or revision is None or not use_shared_repo_dir:
        repo_id, use_shared_repo_dir = _resolve_modelscope_repo_id(spec)
        revision = _resolve_modelscope_revision(spec)
        download_dir = _modelscope_shared_download_dir(model_home, repo_id, revision) if use_shared_repo_dir else local_bundle_dir
    else:
        # We already computed the shared repo download dir above.
        download_dir = shared_download_dir or _modelscope_shared_download_dir(model_home, repo_id, revision)

    snapshot_root = _download_modelscope_repo(
        repo_id=repo_id,
        revision=revision,
        target_dir=download_dir,
        offline=offline_mode,
    )
    discovered = _discover_bundle_dir(spec, [local_bundle_dir, download_dir, snapshot_root])
    if discovered:
        return str(discovered)

    raise FileNotFoundError(
        "Downloaded ModelScope repo '{}@{}' for bundle '{}' but could not locate the required files. "
        "Expected the following files to be colocated under a single directory in the repo (e.g. '{}/'): {}. "
        "Configured via {} / {} and {} / {}. "
        "Searched under: {}, {}.".format(
            repo_id,
            revision,
            spec.name,
            spec.subdir,
            ", ".join(spec.required_files),
            spec.repo_env,
            GLOBAL_MODELSCOPE_REPO_ENV,
            spec.revision_env,
            GLOBAL_MODELSCOPE_REVISION_ENV,
            download_dir,
            snapshot_root,
        )
    )


def validate_bundle_dir(bundle: str, directory: str | Path) -> tuple[bool, list[str]]:
    """Validate a local model bundle directory and report missing required files."""

    if bundle not in BUNDLES:
        raise ValueError(f"Unknown model bundle '{bundle}'. Expected one of: {', '.join(BUNDLES)}")

    spec = BUNDLES[bundle]
    candidate = Path(directory).expanduser().resolve()
    return _validate_bundle_dir(spec, candidate)


def resolve_vision_model_dir(
    *,
    model_home: str | None = None,
    provider: str | None = None,
    offline: bool | None = None,
) -> str:
    return resolve_bundle_dir("vision", model_home=model_home, provider=provider, offline=offline)


def resolve_xgb_model_dir(
    *,
    model_home: str | None = None,
    provider: str | None = None,
    offline: bool | None = None,
) -> str:
    return resolve_bundle_dir("xgb", model_home=model_home, provider=provider, offline=offline)


def resolve_tokenizer_dict_prefix(
    *,
    model_home: str | None = None,
    provider: str | None = None,
    offline: bool | None = None,
) -> str:
    bundle_dir = Path(resolve_bundle_dir("tokenizer", model_home=model_home, provider=provider, offline=offline))
    return str(bundle_dir.joinpath("huqie"))

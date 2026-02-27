import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_RESOURCE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "punkt",
        (
            "tokenizers/punkt",
            "tokenizers/punkt.zip",
            "tokenizers/punkt_tab",
            "tokenizers/punkt_tab.zip",
        ),
    ),
    ("wordnet", ("corpora/wordnet", "corpora/wordnet.zip")),
    (
        "averaged_perceptron_tagger",
        (
            "taggers/averaged_perceptron_tagger",
            "taggers/averaged_perceptron_tagger.zip",
            "taggers/averaged_perceptron_tagger_eng",
            "taggers/averaged_perceptron_tagger_eng.zip",
        ),
    ),
)

_lock = threading.Lock()
_ensured_keys: set[tuple[str, bool]] = set()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_nltk_data_dir(data_dir: str | None) -> Path | None:
    # Resolution precedence:
    # 1) explicit arg
    # 2) DEEPDOC_NLTK_DATA_DIR
    # 3) NLTK_DATA (common NLTK env var)
    # 4) fall back to a stable Deepdoc cache location so parsers can pick it up automatically
    configured = data_dir or os.getenv("DEEPDOC_NLTK_DATA_DIR") or os.getenv("NLTK_DATA")
    if configured and configured.strip():
        return Path(configured).expanduser().resolve()

    model_home = os.getenv("DEEPDOC_MODEL_HOME")
    if model_home and model_home.strip():
        base = Path(model_home).expanduser().resolve()
    else:
        base = Path.home().joinpath(".cache", "deepdoc")
    return base.joinpath("nltk_data")


def _ensure_search_path(nltk_module, data_path: Path | None):
    if not data_path:
        return
    data_path.mkdir(parents=True, exist_ok=True)
    text_path = str(data_path)
    if text_path not in nltk_module.data.path:
        nltk_module.data.path.insert(0, text_path)
    os.environ["NLTK_DATA"] = text_path


def _resource_exists(nltk_module, candidates: tuple[str, ...]) -> bool:
    for resource_path in candidates:
        try:
            nltk_module.data.find(resource_path)
            return True
        except LookupError:
            continue
        except Exception as exc:
            logger.warning("NLTK resource check failed for %s: %s", resource_path, exc)
            continue
    return False


def ensure_nltk_data(
    *,
    data_dir: str | None = None,
    offline: bool | None = None,
) -> None:
    """Ensure required NLTK resources are available for tokenizer usage."""

    import nltk

    resolved_dir = _resolve_nltk_data_dir(data_dir)
    offline_mode = offline if offline is not None else _parse_bool(os.getenv("DEEPDOC_OFFLINE"), default=False)
    auto_download_mode = not offline_mode

    _ensure_search_path(nltk, resolved_dir)
    cache_key = (str(resolved_dir) if resolved_dir else "", offline_mode)

    with _lock:
        if cache_key in _ensured_keys:
            return

        missing_packages: list[str] = []
        for package, candidates in _RESOURCE_SPECS:
            if not _resource_exists(nltk, candidates):
                missing_packages.append(package)

        if missing_packages and auto_download_mode:
            download_dir = str(resolved_dir) if resolved_dir else None
            for package in list(missing_packages):
                try:
                    success = nltk.download(package, quiet=True, download_dir=download_dir)
                except Exception as exc:
                    logger.warning("Failed to download NLTK package %s: %s", package, exc)
                    success = False
                if success and _resource_exists(nltk, dict(_RESOURCE_SPECS)[package]):
                    missing_packages.remove(package)

        if missing_packages:
            searched_paths = ", ".join(nltk.data.path)
            raise RuntimeError(
                "Missing required NLTK packages: {}. Searched paths: {}. "
                "Set DEEPDOC_NLTK_DATA_DIR to a local NLTK data path, or disable offline mode by setting "
                "DEEPDOC_OFFLINE=0."
                .format(
                    ", ".join(missing_packages),
                    searched_paths,
                )
            )

        _ensured_keys.add(cache_key)

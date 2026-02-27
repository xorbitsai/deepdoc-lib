from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from .common.model_store import (
    resolve_vision_model_dir,
    resolve_xgb_model_dir,
    validate_bundle_dir,
)

ProviderType = Literal["local", "modelscope", "auto"]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_provider(provider: str) -> ProviderType:
    normalized = provider.strip().lower()
    aliases = {
        "ms": "modelscope",
        "remote": "modelscope",
        "filesystem": "local",
        "user": "local",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"local", "modelscope", "auto"}:
        raise ValueError(
            "Unsupported model provider '{}'. Use one of: local, modelscope, auto.".format(provider)
        )
    return normalized  # type: ignore[return-value]


def _require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(message)


def _default_packaged_dict_path() -> str:
    packaged_dict = resources.files("deepdoc").joinpath("dict", "huqie.txt")
    dict_path = Path(str(packaged_dict))
    _require_file(
        dict_path,
        "Packaged tokenizer dictionary not found: {}".format(dict_path),
    )
    return str(dict_path)


@dataclass(frozen=True)
class TokenizerConfig:
    dict_path: str | None = None
    offline: bool = False
    nltk_data_dir: str | None = None

    def resolve_dict_path(self) -> str:
        if self.dict_path:
            dictionary = Path(self.dict_path).expanduser().resolve()
            if dictionary.is_dir():
                dictionary = dictionary.joinpath("huqie.txt")
            if dictionary.suffix != ".txt":
                raise ValueError(
                    "TokenizerConfig.dict_path must point to a '.txt' dictionary file, got: {}".format(dictionary)
                )
            _require_file(
                dictionary,
                "Tokenizer dictionary not found: {}. Provide a valid TokenizerConfig.dict_path."
                .format(dictionary),
            )
            return str(dictionary)

        return _default_packaged_dict_path()

    def resolve_dict_prefix(self) -> str:
        return str(Path(self.resolve_dict_path()).with_suffix(""))

    @classmethod
    def from_env(cls) -> "TokenizerConfig":
        tokenizer_dir = os.getenv("DEEPDOC_TOKENIZER_MODEL_DIR")
        dict_path = str(Path(tokenizer_dir).expanduser().resolve().joinpath("huqie.txt")) if tokenizer_dir else None

        return cls(
            dict_path=dict_path,
            offline=_parse_bool(os.getenv("DEEPDOC_OFFLINE"), default=False),
            nltk_data_dir=os.getenv("DEEPDOC_NLTK_DATA_DIR"),
        )


@dataclass(frozen=True)
class PdfModelConfig:
    vision_model_dir: str | None = None
    xgb_model_dir: str | None = None
    ascend_model_dir: str | None = None
    model_home: str | None = None
    model_provider: ProviderType = "auto"

    def normalized_provider(self) -> ProviderType:
        return _normalize_provider(self.model_provider)

    def _resolve_bundle_dir(self, bundle: str, explicit_dir: str | None) -> str:
        if explicit_dir:
            candidate = Path(explicit_dir).expanduser().resolve()
            exists, missing = validate_bundle_dir(bundle, candidate)
            if not exists:
                raise FileNotFoundError(
                    "Missing required files for '{}' bundle in {}: {}"
                    .format(bundle, candidate, ", ".join(missing))
                )
            return str(candidate)

        model_provider = self.normalized_provider()
        model_offline = model_provider == "local"

        if bundle == "vision":
            return resolve_vision_model_dir(
                model_home=self.model_home,
                provider=model_provider,
                offline=model_offline,
            )

        if bundle == "xgb":
            return resolve_xgb_model_dir(
                model_home=self.model_home,
                provider=model_provider,
                offline=model_offline,
            )

        raise ValueError(f"Unsupported PDF model bundle '{bundle}'")

    def resolve_vision_model_dir(self) -> str:
        return self._resolve_bundle_dir("vision", self.vision_model_dir)

    def resolve_xgb_model_dir(self) -> str:
        return self._resolve_bundle_dir("xgb", self.xgb_model_dir)

    def resolve_ascend_model_dir(self) -> str | None:
        if not self.ascend_model_dir:
            return None
        candidate = Path(self.ascend_model_dir).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Ascend model directory does not exist: {candidate}")
        return str(candidate)

    @classmethod
    def from_env(cls) -> "PdfModelConfig":
        return cls(
            vision_model_dir=os.getenv("DEEPDOC_VISION_MODEL_DIR"),
            xgb_model_dir=os.getenv("DEEPDOC_XGB_MODEL_DIR"),
            ascend_model_dir=os.getenv("DEEPDOC_ASCEND_MODEL_DIR"),
            model_provider=_normalize_provider(os.getenv("DEEPDOC_MODEL_PROVIDER", "auto")),
        )


@dataclass(frozen=True)
class ParserRuntimeConfig:
    tokenizer: TokenizerConfig
    pdf_models: PdfModelConfig

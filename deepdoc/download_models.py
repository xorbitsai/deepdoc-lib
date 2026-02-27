from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from deepdoc.common import model_store

    parser = argparse.ArgumentParser(
        prog="deepdoc-download-models",
        description="Download/cache all Deepdoc model bundles (and optional NLTK data) for offline use.",
    )

    # Default behavior: no args downloads everything using the remote provider into the default cache dirs
    # (~/.cache/deepdoc unless DEEPDOC_MODEL_HOME is set).
    parser.add_argument(
        "--provider",
        default="modelscope",
        choices=("auto", "local", "modelscope"),
        help="Model provider to use (default: %(default)s).",
    )
    parser.add_argument(
        "--model-home",
        default=None,
        help="Model cache root (default: $DEEPDOC_MODEL_HOME or ~/.cache/deepdoc).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable remote downloads (also disables NLTK auto-download).",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=tuple(model_store.BUNDLES.keys()),
        help="Bundle(s) to download. Repeatable. Default: all bundles.",
    )
    parser.add_argument(
        "--no-nltk",
        action="store_true",
        help="Skip downloading required NLTK resources used by the tokenizer.",
    )
    parser.add_argument(
        "--nltk-data-dir",
        default=None,
        help="Where to store NLTK data (default: $DEEPDOC_NLTK_DATA_DIR, $NLTK_DATA, or ~/.cache/deepdoc/nltk_data).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (repeatable).",
    )

    args = parser.parse_args(argv)
    if not args.bundle:
        args.bundle = list(model_store.BUNDLES.keys())
    return args


def download_all(*, provider: str = "modelscope", model_home: str | None = None, offline: bool = False) -> dict[str, str]:
    """Download/cache all bundles into the configured cache directories."""
    from deepdoc.common import model_store

    resolved: dict[str, str] = {}
    for bundle in model_store.BUNDLES:
        resolved[bundle] = model_store.resolve_bundle_dir(
            bundle,
            model_home=model_home,
            provider=provider,
            offline=offline,
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    model_home: str | None
    if args.model_home:
        model_home = str(Path(args.model_home).expanduser().resolve())
    else:
        model_home = None

    from deepdoc.common import model_store

    failures: list[str] = []
    resolved: dict[str, str] = {}

    for bundle in args.bundle:
        try:
            resolved_dir = model_store.resolve_bundle_dir(
                bundle,
                model_home=model_home,
                provider=args.provider,
                offline=args.offline,
            )
            resolved[bundle] = resolved_dir
        except Exception as exc:
            failures.append(f"{bundle}: {exc}")

    if not args.no_nltk:
        try:
            from deepdoc.depend.nltk_manager import ensure_nltk_data

            ensure_nltk_data(
                data_dir=args.nltk_data_dir,
                offline=args.offline,
            )
        except Exception as exc:
            failures.append(f"nltk: {exc}")

    for bundle_name in sorted(resolved):
        print(f"{bundle_name}\t{resolved[bundle_name]}")

    if failures:
        for item in failures:
            print(f"ERROR\t{item}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


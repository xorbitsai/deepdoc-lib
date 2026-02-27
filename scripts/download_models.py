#!/usr/bin/env python3
"""Backwards-compatible wrapper for `deepdoc-download-models`."""

from __future__ import annotations

from deepdoc.download_models import main


if __name__ == "__main__":
    raise SystemExit(main())

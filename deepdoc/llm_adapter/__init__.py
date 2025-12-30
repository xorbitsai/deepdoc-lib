"""
LLM Adapter Layer for DeepDoc

This module provides a thin adapter layer that handles LLM-related dependencies,
allowing DeepDoc to work in both FenixAOS environments and standalone configurations.
"""

from .adapter import LLMAdapter, LLMType
from .vision import vision_llm_chunk

__all__ = [
    "LLMAdapter",
    "LLMType",
    "vision_llm_chunk",
]

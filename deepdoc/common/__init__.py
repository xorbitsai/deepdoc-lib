"""
Common utilities for DeepDoc independent library.

This module provides common utilities that were previously imported from RAGFlow.
These are simplified versions suitable for an independent library.
"""

from .file_utils import get_project_base_directory, traversal_files
from .token_utils import num_tokens_from_string, total_token_count_from_response, truncate
from .misc_utils import pip_install_torch
from .connection_utils import timeout
from .config_utils import get_base_config, get_config_value

__all__ = [
    # file_utils
    "get_project_base_directory",
    "traversal_files",

    # token_utils
    "num_tokens_from_string",
    "total_token_count_from_response",
    "truncate",

    # misc_utils
    "pip_install_torch",

    # connection_utils
    "timeout",

    # config_utils
    "get_base_config",
    "get_config_value",
]

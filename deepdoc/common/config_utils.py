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

"""
Basic configuration utilities for DeepDoc independent library.
This is a simplified version compared to RAGFlow's complex configuration system.
"""

import os
from typing import Any, Dict, Optional


def get_base_config(service_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get base configuration for a service.
    This is a simplified version that uses environment variables.

    Args:
        service_name: Name of the service
        default_config: Default configuration to return

    Returns:
        Configuration dictionary
    """
    if default_config is None:
        default_config = {}

    # For tcadp_parser, provide basic Tencent Cloud configuration
    if service_name == "tcadp" or "tencent" in service_name.lower():
        return {
            "secret_id": os.getenv("TENCENT_CLOUD_SECRET_ID", ""),
            "secret_key": os.getenv("TENCENT_CLOUD_SECRET_KEY", ""),
            "region": os.getenv("TENCENT_CLOUD_REGION", "ap-guangzhou"),
        }

    return default_config


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get configuration value from environment variables.

    Args:
        key: Configuration key
        default: Default value if not found

    Returns:
        Configuration value
    """
    return os.getenv(key, default)

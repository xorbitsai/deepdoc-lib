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
Simplified settings for DeepDoc independent library.
Only includes configurations needed by DeepDoc components.
"""

import logging

# GPU device count detection
PARALLEL_DEVICES: int = 0

def check_and_install_torch():
    """
    Check for PyTorch and detect GPU devices.
    Simplified version for independent library.
    """
    global PARALLEL_DEVICES
    try:
        import torch.cuda
        PARALLEL_DEVICES = torch.cuda.device_count()
        logging.info(f"Found {PARALLEL_DEVICES} GPUs")
    except Exception as e:
        logging.info("Can't import package 'torch' or access GPU: %s", str(e))
        PARALLEL_DEVICES = 0

# Initialize on import
check_and_install_torch()

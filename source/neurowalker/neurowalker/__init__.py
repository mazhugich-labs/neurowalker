# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

import os

# Conveniences to other module directories via relative paths
ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ASSETS_DATA_DIR = os.path.join(ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

# Register custom modules
from .controllers import *
from .robots import *

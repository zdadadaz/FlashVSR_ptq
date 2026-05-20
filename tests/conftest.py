"""Test configuration for pytest."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock ComfyUI modules before any imports
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_filename_list = MagicMock(return_value=[])
sys.modules['folder_paths'].models_dir = "/tmp/models"
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.utils'].ProgressBar = MagicMock()

# Ensure project root is in path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

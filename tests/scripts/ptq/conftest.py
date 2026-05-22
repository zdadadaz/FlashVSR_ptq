"""Test configuration for pytest - mocks ComfyUI modules for ptq tests."""

import sys
from unittest.mock import MagicMock

# Mock ComfyUI modules before any imports from the project
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_filename_list = MagicMock(return_value=[])
sys.modules['folder_paths'].models_dir = "/tmp/models"
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.utils'].ProgressBar = MagicMock()
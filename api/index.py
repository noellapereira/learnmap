from __future__ import annotations

import os
import sys

# Ensure backend directory is in the Python module search path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app  # noqa: E402


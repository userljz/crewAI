from __future__ import annotations

import sys
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXAMPLE_ROOT / "src"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


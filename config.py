"""
NOTE: This module has been moved to `legacy/config.py`.
This stub remains for backward compatibility and import forwarding.
"""
import sys
from pathlib import Path

LEGACY_DIR = str(Path(__file__).resolve().parent / "legacy")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from legacy.config import *  # noqa: F401, F403

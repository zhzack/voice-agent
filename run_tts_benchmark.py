"""
NOTE: This script has been moved to `legacy/run_tts_benchmark.py`.
This stub remains for backward compatibility, import forwarding, and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_DIR = str(Path(__file__).resolve().parent / "legacy")
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from legacy.run_tts_benchmark import *  # noqa: F401, F403

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "run_tts_benchmark.py"

if __name__ == "__main__":
    print(f"[Notice] 'run_tts_benchmark.py' has moved to 'legacy/run_tts_benchmark.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

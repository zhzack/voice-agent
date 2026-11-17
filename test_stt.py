"""
NOTE: This script has been moved to `legacy/test_stt.py`.
This stub remains for backward compatibility and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "test_stt.py"

if __name__ == "__main__":
    print(f"[Notice] 'test_stt.py' has moved to 'legacy/test_stt.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

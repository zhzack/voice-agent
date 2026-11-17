"""
NOTE: This script has been moved to `legacy/check_mic.py`.
This stub remains for backward compatibility and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "check_mic.py"

if __name__ == "__main__":
    print(f"[Notice] 'check_mic.py' has moved to 'legacy/check_mic.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

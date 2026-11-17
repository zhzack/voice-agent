"""
NOTE: This script has been moved to `legacy/realtime_call.py`.
This stub remains for backward compatibility and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "realtime_call.py"

if __name__ == "__main__":
    print(f"[Notice] 'realtime_call.py' has moved to 'legacy/realtime_call.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

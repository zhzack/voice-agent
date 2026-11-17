"""
NOTE: This script has been moved to `legacy/main_server.py`.
This stub remains for backward compatibility and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "main_server.py"

if __name__ == "__main__":
    print(f"[Notice] 'main_server.py' has moved to 'legacy/main_server.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

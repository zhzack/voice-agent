"""
NOTE: This script has been moved to `legacy/prosody_studio.py`.
This stub remains for backward compatibility and CLI forwarding.
"""
import sys
import subprocess
from pathlib import Path

LEGACY_TARGET = Path(__file__).resolve().parent / "legacy" / "prosody_studio.py"

if __name__ == "__main__":
    print(f"[Notice] 'prosody_studio.py' has moved to 'legacy/prosody_studio.py'. Forwarding execution...\n")
    sys.exit(subprocess.call([sys.executable, str(LEGACY_TARGET)] + sys.argv[1:]))

"""Probe that Streamlit serves the application, then stop the test process."""

import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py"), "--server.headless", "true", "--server.port", "8502", "--server.fileWatcherType", "none"]
process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"Streamlit exited early ({process.returncode}): {output}")
        try:
            with urlopen("http://localhost:8502", timeout=1) as response:
                print("Dashboard HTTP status:", response.status)
                print("Streamlit startup: OK")
                break
        except Exception:
            continue
    else:
        raise RuntimeError("Timed out waiting for Streamlit startup")
finally:
    process.terminate()
    process.wait(timeout=10)

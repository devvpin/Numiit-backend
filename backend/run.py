#!/usr/bin/env python3
"""NumiIT - cross-platform project runner.

Usage:
    python run.py setup          # one-time: create venv + install deps + flutter pub get
    python run.py backend        # start FastAPI dev server on 0.0.0.0:8000
    python run.py frontend       # flutter run (auto-picks connected device)
    python run.py frontend chrome  # flutter run -d chrome
    python run.py frontend android # flutter run on Android with correct API URL
    python run.py build-web      # flutter build web
"""

import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = REPO_ROOT / "backend"
FRONTEND_ROOT = REPO_ROOT / "frontend"

IS_WINDOWS = platform.system() == "Windows"
VENV_DIR = BACKEND_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python")


def find_flutter() -> str:
    """Return the path to the flutter executable."""
    # Check if flutter is on PATH
    cmd = "flutter.bat" if IS_WINDOWS else "flutter"
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return cmd
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check the old local SDK path used in this project
    local_sdk = Path(r"d:\college 3rd year\coin\flutter_sdk\bin\flutter.bat")
    if local_sdk.exists():
        return str(local_sdk)

    # Common install locations on Windows
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "flutter" / "bin" / "flutter.bat",
        Path(os.environ.get("USERPROFILE", "")) / "flutter" / "bin" / "flutter.bat",
        Path("C:/flutter/bin/flutter.bat"),
        Path("D:/flutter/bin/flutter.bat"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    print("ERROR: Flutter not found. Install from https://docs.flutter.dev/get-started/install")
    sys.exit(1)


def get_local_ip() -> str:
    """Get the laptop's WiFi / LAN IP so phones on the same network can reach the backend."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# -- Commands ----------------------------------------------------------

def cmd_setup():
    """One-time setup: create venv, install backend deps, flutter pub get."""
    print("=" * 60)
    print("  NumiIT - Setup")
    print("=" * 60)

    # Backend venv
    if not VENV_PYTHON.exists():
        print("\n-> Creating Python virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    print("\n-> Upgrading pip...")
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("\n-> Installing backend dependencies...")
    subprocess.run([str(VENV_PYTHON), "-m", "pip", "install", "-e", str(BACKEND_ROOT)], check=True)

    # Frontend
    flutter = find_flutter()
    print("\n-> Running flutter pub get...")
    subprocess.run([flutter, "pub", "get"], cwd=str(FRONTEND_ROOT), check=True)

    print("\n[OK] Setup complete!")
    print(f"  Backend venv:  {VENV_DIR}")
    print(f"  Flutter:       {flutter}")


def cmd_backend():
    """Start the FastAPI backend on 0.0.0.0:8000 so phones can reach it."""
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_ROOT)

    local_ip = get_local_ip()
    print("=" * 60)
    print("  NumiIT Backend - FastAPI")
    print("=" * 60)
    print(f"  Local:   http://127.0.0.1:8000")
    print(f"  Network: http://{local_ip}:8000")
    print(f"  Health:  http://{local_ip}:8000/api/v1/health")
    print()
    print("  For Android phone on same WiFi, use:")
    print(f"    python run.py frontend android --ip {local_ip}")
    print("=" * 60)

    subprocess.run(
        [python, "-m", "uvicorn", "app.main:app",
         "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_ROOT),
        env=env,
    )


def _load_env_file(env_path: Path) -> dict[str, str]:
    """Read a .env file into a dict, ignoring comments and blank lines."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def cmd_frontend(args: list[str]):
    """Run the Flutter frontend with the correct API URL for the target device."""
    flutter = find_flutter()
    local_ip = get_local_ip()

    target = args[0] if args else ""

    # Check for --ip flag for physical device
    custom_ip = None
    if "--ip" in args:
        idx = args.index("--ip")
        if idx + 1 < len(args):
            custom_ip = args[idx + 1]

    # Load API URL from root .env
    root_env = _load_env_file(REPO_ROOT / ".env")
    env_api_url = root_env.get("NUMIIT_API_BASE_URL", "")

    cmd = [flutter, "run"]

    if target == "chrome":
        cmd += ["-d", "chrome"]
        # For web/chrome: inject production URL from .env if available
        if env_api_url:
            cmd += [f"--dart-define=NUMIIT_API_BASE_URL={env_api_url}"]
            print(f"  Chrome API URL -> {env_api_url}")
    elif target == "android":
        # For physical Android device, use laptop's network IP
        ip = custom_ip or local_ip
        api_url = f"http://{ip}:8000/api/v1"
        cmd += [f"--dart-define=NUMIIT_API_BASE_URL={api_url}"]
        print(f"  Android API URL -> {api_url}")
    elif target == "ios":
        ip = custom_ip or local_ip
        api_url = f"http://{ip}:8000/api/v1"
        cmd += [f"--dart-define=NUMIIT_API_BASE_URL={api_url}"]
        print(f"  iOS API URL -> {api_url}")
    elif target == "windows":
        cmd += ["-d", "windows"]
        if env_api_url:
            cmd += [f"--dart-define=NUMIIT_API_BASE_URL={env_api_url}"]
    elif target:
        cmd += ["-d", target]
        if env_api_url:
            cmd += [f"--dart-define=NUMIIT_API_BASE_URL={env_api_url}"]

    print(f"\n  Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=str(FRONTEND_ROOT))


def cmd_build_web():
    """Build the Flutter web frontend."""
    flutter = find_flutter()
    print("-> Building Flutter web...")
    subprocess.run([flutter, "pub", "get"], cwd=str(FRONTEND_ROOT), check=True)
    subprocess.run([flutter, "build", "web"], cwd=str(FRONTEND_ROOT), check=True)
    print(f"\n[OK] Web build ready at: {FRONTEND_ROOT / 'build' / 'web'}")
    print("  Start the backend and open http://127.0.0.1:8000")


# -- Entry point -------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands: setup, backend, frontend, build-web")
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "setup":
        cmd_setup()
    elif command == "backend":
        cmd_backend()
    elif command == "frontend":
        cmd_frontend(sys.argv[2:])
    elif command in ("build-web", "buildweb", "build_web"):
        cmd_build_web()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: setup, backend, frontend, build-web")
        sys.exit(1)


if __name__ == "__main__":
    main()

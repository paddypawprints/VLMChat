#!/usr/bin/env bash
set -euo pipefail

# scripts/setup-pi-camera-venv.sh
# Assist setting up a Python virtualenv that can access the Raspberry Pi camera
# (using libcamera / picamera2). The script is conservative: it checks for
# required binaries and suggests installing system packages with sudo.
#
# Usage:
#   ./scripts/setup-pi-camera-venv.sh [--no-install]
#
# Options:
#   --no-install  : do not attempt to install system packages (just create venv)

NO_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --no-install) NO_INSTALL=1 ;;
    *) ;;
  esac
done

echo "This script helps prepare a venv that can access the Pi camera (picamera2/libcamera)."

# Helper: command exists
cmd_exists() { command -v "$1" >/dev/null 2>&1; }

if ! cmd_exists python3; then
  echo "python3 not found on PATH. Please install Python 3 and retry." >&2
  exit 1
fi

# Detect apt (Debian/Ubuntu) — required for system package installs
if ! cmd_exists apt && [ "$NO_INSTALL" -eq 0 ]; then
  echo "apt not found; skipping system package installation. You will need to ensure libcamera and picamera2 are installed by other means." >&2
  NO_INSTALL=1
fi

if [ "$NO_INSTALL" -eq 0 ]; then
  echo "Checking for libcamera and picamera2 system packages..."
  MISSING_PKGS=()
  # These are common package names; actual names vary by distro.
  # We'll check for libcamera-hello binary as a proxy for libcamera.
  if ! cmd_exists libcamera-hello; then
    MISSING_PKGS+=(libcamera-apps libcamera-dev)
  fi
  # Check for python3-picamera2
  if ! python3 -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('picamera2') else 1)" 2>/dev/null; then
    MISSING_PKGS+=(python3-picamera2)
  fi

  if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    echo "The following system packages may be missing: ${MISSING_PKGS[*]}"
    echo "I can attempt to install them via apt (requires sudo)."
    read -p "Install packages now? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      sudo apt update
      # install a reasonable set of packages for libcamera/picamera2 usage
      sudo apt install -y libcamera-apps libcamera-dev libtiff5-dev libjpeg-dev \
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        python3-libcamera python3-picamera2 || {
        echo "apt install failed; please install the required packages manually." >&2
        exit 1
      }
    else
      echo "Skipping system package install. You must ensure libcamera and picamera2 are installed." >&2
    fi
  else
    echo "libcamera and picamera2 look available." 
  fi
fi

# Ensure user is in the 'video' group so camera device nodes are accessible
if groups "$USER" | grep -qw video; then
  echo "User $USER is already in the 'video' group."
else
  echo "Adding user $USER to 'video' group to allow camera access (requires sudo)."
  echo "You will need to log out and log back in for the group addition to take effect."
  read -p "Add $USER to video group now? [y/N] " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    sudo usermod -aG video "$USER"
    echo "Added $USER to video group. Please log out and back in (or reboot) before continuing."
  else
    echo "Skipping adding user to 'video' group. If you see permission errors, add the user to the video group and retry." >&2
  fi
fi

# Create venv with system-site-packages so python can see system-installed picamera2
VENV_DIR=${VENV_DIR:-.venv_camera}
PYTHON=${PYTHON:-python3}

if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv directory $VENV_DIR already exists. Skipping creation."
else
  echo "Creating virtualenv at $VENV_DIR (system-site-packages enabled)."
  $PYTHON -m venv --system-site-packages "$VENV_DIR"
fi

echo "Activate the venv with: source $VENV_DIR/bin/activate"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# If picamera2 isn't importable from system packages inside venv, offer to pip install it
if python -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('picamera2') else 1)" 2>/dev/null; then
  echo "picamera2 importable in venv (via system site-packages)."
else
  echo "picamera2 is not importable inside the venv. You have two options:"
  echo "  1) Install picamera2 system package and use --system-site-packages venv (recommended), or"
  echo "  2) Try installing picamera2 into the venv with pip (may fail if system libs missing)."
  read -p "Try to pip install picamera2 into the venv now? [y/N] " ans
  if [[ "$ans" =~ ^[Yy]$ ]]; then
    pip install picamera2 || {
      echo "pip install picamera2 failed. You likely need system libraries (libcamera) installed via apt. Try rerunning with system packages installed." >&2
      deactivate || true
      exit 1
    }
  else
    echo "Skipping pip install of picamera2. Ensure that picamera2 is available to the venv via system packages or reinstall with pip later." 
  fi
fi

# Quick runtime checks
echo "Running quick camera checks..."
if cmd_exists libcamera-hello; then
  echo "Running libcamera-hello (1s) to ensure libcamera works (will open preview window briefly):"
  libcamera-hello --qt-preview --timeout 1000 || echo "libcamera-hello returned non-zero (may be expected in headless environments)"
else
  echo "libcamera-hello not found; skip libcamera test."
fi

python - <<'PY'
try:
    from picamera2 import Picamera2
    print('picamera2 import OK')
    p = Picamera2()
    print('Created Picamera2 instance:', type(p))
except Exception as e:
    print('picamera2 import/create failed:', e)
    raise SystemExit(1)
PY

echo "Setup complete. If you added yourself to the video group, please log out and log back in (or reboot) before trying to access the camera from your venv."

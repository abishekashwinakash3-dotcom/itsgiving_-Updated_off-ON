#!/usr/bin/env bash
#
# setup.sh — one command to get the meme cam ready. macOS and Linux.
#
#   ./setup.sh
#
# Builds a virtualenv, installs the pinned dependencies, then runs doctor.py
# and tells you exactly what (if anything) is still wrong.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
step() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }
warn() { printf "\033[33m    %s\033[0m\n" "$1"; }
die()  { printf "\033[31m\nSetup stopped: %s\033[0m\n" "$1" >&2; exit 1; }

bold "it's giving — setup"

# ------------------------------------------------------------------ 1. python
# mediapipe 0.10.21 has no wheel for 3.13+, so prefer 3.12 and walk down.
step "Looking for a usable Python"
PY=""
for cand in python3.12 python3.11 python3.10 python3.9 python3 python; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)' 2>/dev/null; then
        PY="$cand"; break
    fi
done

if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        warn "found $(python3 -V 2>&1), which mediapipe 0.10.21 has no wheel for."
    fi
    echo "    Install Python 3.12, then run this again:"
    case "$(uname -s)" in
        Darwin) echo "      brew install python@3.12" ;;
        *)      echo "      sudo apt install python3.12 python3.12-venv" ;;
    esac
    die "no Python between 3.9 and 3.12 found"
fi
echo "    using $PY ($("$PY" -V 2>&1))"

# ------------------------------------------------------------------- 2. venv
step "Creating the virtualenv"
if [ -d venv ]; then
    echo "    venv/ already exists, reusing it"
else
    "$PY" -m venv venv || die "could not create venv (on Debian/Ubuntu: sudo apt install python3-venv)"
    echo "    created venv/"
fi

# shellcheck disable=SC1091
source venv/bin/activate || die "could not activate venv/"

# --------------------------------------------------------------- 3. installs
step "Installing dependencies (a minute or two the first time)"
python -m pip install --quiet --upgrade pip || warn "could not upgrade pip; continuing"
if ! python -m pip install --quiet -r requirements.txt; then
    die "pip install failed — scroll up for the reason. Do not unpin requirements.txt to
    get past it; those pins are what keep mediapipe from crashing at runtime."
fi
echo "    done"

# ------------------------------------------------------ 4. virtual-cam backend
step "Checking the virtual-camera backend"
case "$(uname -s)" in
    Darwin)
        if [ -d "/Applications/OBS.app" ]; then
            echo "    OBS Studio is installed"
        else
            warn "OBS Studio not found. The virtual camera needs it on macOS:"
            warn "  download https://obsproject.com, open OBS once, then quit it."
        fi
        ;;
    Linux)
        if lsmod 2>/dev/null | grep -q v4l2loopback; then
            echo "    v4l2loopback is loaded"
        else
            warn "v4l2loopback is not loaded. The virtual camera needs it:"
            warn "  sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback"
        fi
        ;;
esac

# ------------------------------------------------------------------ 5. doctor
step "Running the full check"
python doctor.py
DOCTOR=$?

# ------------------------------------------------------------- 6. calibration
if [ $DOCTOR -eq 0 ] && [ -t 0 ]; then
    NEEDS_CALIB=$(python - <<'EOF'
import hashlib, os, sys
p = "calibration.json"
SHIPPED = "9daa39e0d253d36c02faa44ba3bee8c130291aad239ed0f965ff967cdf9fa166"
if not os.path.exists(p):
    print("yes")
elif hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest() == SHIPPED:
    print("yes")
else:
    print("no")
EOF
)
    if [ "$NEEDS_CALIB" = "yes" ]; then
        step "Calibration"
        echo "    The calibration.json in this repo is somebody else's face. Yours takes"
        echo "    seven seconds: sit still with a neutral expression when the window opens."
        printf "    Calibrate now? [Y/n] "
        read -r ans
        case "${ans:-y}" in
            [Nn]*) warn "skipped — run 'python its_giving_v2.py --calibrate' before you rely on it" ;;
            *)     python its_giving_v2.py --calibrate ;;
        esac
    fi
fi

# -------------------------------------------------------------------- 7. done
echo
if [ $DOCTOR -ne 0 ]; then
    bold "Setup finished, but the check above found blocking problems."
    echo "Fix those, then re-run:  source venv/bin/activate && python doctor.py"
    exit 1
fi

bold "Ready."
cat <<'EOF'

Every new terminal, first:
    source venv/bin/activate

Then:
    python its_giving_v2.py --hotkeys      # starts OFF - your plain webcam

In Meet: settings > Video > Camera > the virtual camera named above,
and turn Meet's own effects off (... > Apply visual effects > None).

Type into the terminal running the script:
    manual        arm it, memes only when you ask
    heart         show that meme for a couple of seconds
    auto 60       auto for 60 seconds, then it disarms ITSELF
    <blank Enter> panic - straight back to a plain webcam

Leave it running in 'off' for serious calls. Quitting removes the camera
device and Meet goes black, which is worse than a meme.
EOF

#!/usr/bin/env python3
"""
doctor.py — check this machine can actually run the meme cam, before you are
sitting in a Meet wondering why the camera is black.

    python doctor.py

Every check prints ok / warn / FAIL and, when something is wrong, the one
command that fixes it. Exit code is 0 when nothing is blocking.

Stdlib only at import time: this has to run *before* the dependencies are
installed, so it can tell you that they aren't.
"""
import contextlib
import importlib
import importlib.util
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The calibration.json that ships in the repo. It is a real calibration — of
# somebody else's face — so it loads without complaint and silently measures
# every one of your expressions against a stranger's neutral.
SHIPPED_CALIB_SHA = "9daa39e0d253d36c02faa44ba3bee8c130291aad239ed0f965ff967cdf9fa166"

POSES = ["time_out", "heart", "cover_nose", "crashing_out", "dance", "nose_closed", "flirty",
         "hand_up", "tongue_out", "open_mouth", "disgusted", "talking_to_wall", "suspicious", "spin"]
MODELS = ["face_landmarker.task", "hand_landmarker.task", "pose_landmarker_lite.task"]

MAC, WIN = sys.platform == "darwin", os.name == "nt"
LINUX = sys.platform.startswith("linux")

_fails, _warns = [], []


def ok(msg):
    print(f"  ok    {msg}")


def warn(msg, fix=None):
    print(f"  warn  {msg}")
    if fix:
        print(f"        -> {fix}")
    _warns.append(msg)


def fail(msg, fix=None):
    print(f"  FAIL  {msg}")
    if fix:
        print(f"        -> {fix}")
    _fails.append(msg)


def head(title):
    print(f"\n{title}")


@contextlib.contextmanager
def quiet_stderr():
    """Silence OpenCV's native camera-probe chatter.

    Probing an index that doesn't exist makes OpenCV write C++ warnings
    straight to file descriptor 2, so setLogLevel can't reach them. Here they
    are expected on every machine and only make a healthy report look broken.
    """
    try:
        saved = os.dup(2)
    except OSError:
        yield                       # no fd 2 to redirect; nothing to do
        return
    try:
        with open(os.devnull, "w") as null:
            sys.stderr.flush()
            os.dup2(null.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)


# --------------------------------------------------------------------- python

def check_python():
    head("Python")
    v = sys.version_info
    tag = f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if v < (3, 9):
        fail(f"Python {tag} is too old", "install Python 3.12")
    elif v >= (3, 13):
        fail(f"Python {tag} — mediapipe 0.10.21 has no wheel for 3.13+",
             "install Python 3.12 and re-run setup")
    else:
        ok(f"Python {tag}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        ok(f"virtualenv active ({os.path.basename(sys.prefix)})")
    else:
        warn("not running inside a virtualenv",
             "source venv/bin/activate" if not WIN else r"venv\Scripts\activate")


# --------------------------------------------------------------- dependencies

def _version(mod):
    return getattr(mod, "__version__", "?")


def check_deps():
    head("Dependencies")
    mods = {}
    for name in ("cv2", "numpy", "mediapipe", "PIL", "pyvirtualcam"):
        try:
            mods[name] = importlib.import_module(name)
            ok(f"{name:13s} {_version(mods[name])}")
        except Exception as e:
            mods[name] = None
            fail(f"{name:13s} missing ({type(e).__name__})",
                 "pip install -r requirements.txt")

    # Installed-but-unusable and not-installed both surface as ImportError here,
    # and they need opposite advice, so check for the package itself first.
    if importlib.util.find_spec("pynput") is None:
        warn("pynput not installed — --hotkeys will be unavailable",
             "pip install pynput   (optional; typed commands still work)")
    else:
        try:
            importlib.import_module("pynput")
            ok("pynput        present (global hotkeys available)")
        except Exception as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            warn(f"pynput is installed but cannot start here ({first})",
                 "on macOS: System Settings > Privacy & Security > Accessibility > "
                 "allow your terminal. Typed commands work regardless.")

    # The pins in requirements.txt exist for a reason; a stray upgrade of one
    # of these is the classic way this install breaks at runtime, not install
    # time, which means it breaks while you are already in the meeting.
    np, cv = mods.get("numpy"), mods.get("cv2")
    if np is not None and _version(np)[:1].isdigit() and int(_version(np).split(".")[0]) >= 2:
        fail(f"numpy {_version(np)} — mediapipe 0.10.21 needs numpy<2",
             'pip install "numpy<2"')
    if cv is not None and _version(cv)[:1].isdigit() and int(_version(cv).split(".")[0]) >= 5:
        fail(f"opencv {_version(cv)} — pulls numpy 2 with it",
             'pip install "opencv-python<5" "opencv-contrib-python<5"')
    return mods


# -------------------------------------------------------------- repo contents

def check_files():
    head("Models and assets")
    mdir = os.path.join(HERE, "models")
    missing = [m for m in MODELS if not os.path.exists(os.path.join(mdir, m))]
    if missing:
        warn(f"{len(missing)} model file(s) not downloaded yet: {', '.join(missing)}",
             "they download automatically on first run (~17 MB)")
    else:
        ok(f"all {len(MODELS)} MediaPipe models present")

    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        fail("assets/ directory is missing", "re-clone the repo")
        return
    names = os.listdir(adir)
    gone = []
    for p in POSES:
        hit = any(os.path.splitext(f)[1].lower() in (".gif", ".png", ".jpg", ".jpeg")
                  and (os.path.splitext(f)[0] == p or os.path.splitext(f)[0].endswith("_" + p))
                  for f in names)
        if not hit:
            gone.append(p)
    if gone:
        warn(f"{len(gone)} pose(s) have no image, a grey placeholder shows instead: {', '.join(gone)}")
    else:
        ok(f"all {len(POSES)} pose images present")

    for mod in ("meme_control.py", "its_giving_v2.py"):
        if os.path.exists(os.path.join(HERE, mod)):
            ok(f"{mod} in place")
        else:
            fail(f"{mod} is missing", "re-clone the repo")


def check_calibration():
    head("Calibration")
    path = os.path.join(HERE, "calibration.json")
    if not os.path.exists(path):
        warn("no calibration.json — every expression will be harder to trigger than it should be",
             "python its_giving_v2.py --calibrate   (seven seconds)")
        return
    import hashlib
    raw = open(path, "rb").read()
    # Git on Windows (core.autocrlf=true) checks this file out with CRLF line
    # endings, which changes the hash. Compare the LF form, or the stranger's
    # calibration passes as "yours" on exactly the machines least likely to
    # have been calibrated.
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == SHIPPED_CALIB_SHA:
        warn("calibration.json is the one that SHIPPED WITH THE REPO — it is a "
             "stranger's face, not yours, and it loads without any warning",
             "python its_giving_v2.py --calibrate   (seven seconds, do this first)")
        return
    try:
        d = json.loads(raw)
        ok(f"calibrated {d.get('made', '?')} on {d.get('samples', '?')} frames — this one is yours")
    except ValueError:
        fail("calibration.json is not valid JSON", "python its_giving_v2.py --calibrate")


# -------------------------------------------------------------------- devices

def check_camera(mods):
    head("Webcam")
    cv2 = mods.get("cv2")
    if cv2 is None:
        fail("cannot test the camera without opencv")
        return
    found = []
    with quiet_stderr():
        for idx in range(4):
            cap = None
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    got, frame = cap.read()
                    if got and frame is not None:
                        h, w = frame.shape[:2]
                        found.append((idx, w, h))
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
    if not found:
        hint = ("System Settings > Privacy & Security > Camera: allow your terminal, then re-run"
                if MAC else "check the camera is plugged in and not held by another app")
        fail("no working camera on indexes 0-3", hint)
        return
    for idx, w, h in found:
        ok(f"camera {idx}: {w}x{h}" + ("   <- default" if idx == 0 else f"   (use --camera {idx})"))
    if found[0][0] != 0:
        warn(f"camera 0 did not work; pass --camera {found[0][0]}")


def check_virtualcam(mods):
    head("Virtual camera (what Meet will see)")
    pvc = mods.get("pyvirtualcam")
    if pvc is None:
        fail("pyvirtualcam not installed", "pip install pyvirtualcam")
        return
    try:
        cam = pvc.Camera(width=1280, height=720, fps=30)
        name = cam.device
        cam.close()
        ok(f"backend ready, device name: '{name}'")
        print(f"        pick '{name}' in Meet: settings > Video > Camera")
    except Exception as e:
        if MAC:
            fixit = "install OBS Studio from https://obsproject.com, open it once, quit it"
        elif WIN:
            fixit = "install OBS Studio, or run its virtual-camera installer"
        else:
            fixit = "sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback"
        fail(f"no virtual-camera backend ({type(e).__name__}: {e})", fixit)


def check_control():
    head("Arm / disarm layer")
    try:
        sys.path.insert(0, HERE)
        from meme_control import Controller
        c = Controller(POSES, log=lambda *a, **k: None)
        assert c.mode == "off", "default mode is not off"
        c.handle("manual")
        assert c.mode == "manual"
        c.handle("")                       # the panic path
        assert c.mode == "off", "blank Enter did not disarm"
        ok("controller imports and defaults to off; panic key works")
    except Exception as e:
        fail(f"meme_control is not usable ({type(e).__name__}: {e})")


# ----------------------------------------------------------------------- main

def main():
    print("it's giving — setup check")
    print(f"{platform.system()} {platform.release()}  |  {HERE}")

    check_python()
    mods = check_deps()
    check_files()
    check_calibration()
    check_control()
    check_camera(mods)
    check_virtualcam(mods)

    print("\n" + "-" * 62)
    if _fails:
        print(f"{len(_fails)} blocking problem(s). Fix those, then run this again.")
    elif _warns:
        print(f"Ready to run, with {len(_warns)} thing(s) worth sorting first (see 'warn' above).")
    else:
        print("Everything checks out.")

    if not _fails:
        print("\nNext:")
        print("  1. python its_giving_v2.py --calibrate     seven seconds of a bored face")
        print("  2. python its_giving_v2.py --hotkeys       starts OFF - plain webcam")
        print("  3. in Meet: settings > Video > Camera > the virtual camera named above")
        print("     and turn Meet's own effects off (... > Apply visual effects > None)")
        print("\n  Then type 'manual' and 'heart' in the terminal to see it work.")
        print("  Blank Enter is the panic button. 'auto 60' arms for a minute and disarms itself.")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())

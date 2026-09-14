#!/usr/bin/env python3
"""
meme_control.py — an arm / disarm layer for gazijarin/itsgiving.

Drop this next to its_giving_v2.py and apply the six edits in MEET_SETUP.md.

Three modes:

    off      Detection does not run at all. The virtual camera carries your
             plain webcam feed. Nothing can fire, because nothing is looking.
             This is the default, and it is what a professional Meet gets.

    manual   Detection runs (so the overlay knows where your head is) but
             nothing fires by itself. A meme appears only when you name it.

    auto     The original behaviour — poses fire on their own.

Commands reach it from two places, both optional:

    * the terminal you launched the script from (type, press Enter)
    * global hotkeys, if pynput is installed and --hotkeys was passed

Nothing here talks to the camera or the detectors. It only holds state and
answers two questions the main loop asks every frame: what mode am I in, and
is a meme being forced right now.
"""

import sys
import threading
import time

MODES = ("off", "manual", "auto")
# ctrl+alt+<key> fires the pose at the same position, matching the preview
# window's 1-9 0 - = [ ]. Plain characters on purpose: pynput's canonical()
# maps the Ctrl/Alt-mangled key back to its base character, so these match on
# Windows; virtual-key tokens (<77>) do not (verified with real hook events).
HOTKEY_MEMES = "1234567890-=[]"

POP_SECONDS = 2.5        # how long a named meme stays up
STICKY = float("inf")    # `hold <pose>` until `clear`

_LABEL = {
    "off":    "OFF  -  plain webcam",
    "manual": "MANUAL  -  memes on command only",
    "auto":   "AUTO  -  memes fire on their own",
}
_COLOUR = {              # BGR, for the preview badge
    "off":    (130, 130, 130),
    "manual": (0, 200, 255),
    "auto":   (40, 80, 255),
}


class Controller:
    """Mode state plus a forced-pose slot. Every public method is thread-safe."""

    def __init__(self, poses, mode="off", pop_seconds=POP_SECONDS, log=print):
        self.poses = list(poses)
        self.pop_seconds = pop_seconds
        self.log = log
        self.quit = False

        self._lock = threading.RLock()
        self._mode = mode if mode in MODES else "off"
        self._last_active = "auto" if self._mode == "off" else self._mode
        self._forced = None
        self._forced_until = 0.0
        self._expires_at = None       # deadline for a timed arm
        self._dirty = True            # mode changed; main loop should reset

    # ---------------------------------------------------------------- state

    @property
    def mode(self):
        with self._lock:
            return self._mode

    def seconds_left(self):
        """Seconds until a timed arm drops back to off, or None."""
        with self._lock:
            if self._expires_at is None:
                return None
            return max(0.0, self._expires_at - time.monotonic())

    def set_mode(self, mode, seconds=None):
        if mode not in MODES:
            return f"unknown mode {mode!r} — try: {', '.join(MODES)}"
        with self._lock:
            if mode != self._mode:
                self._dirty = True
            if mode != "off":
                self._last_active = mode
            self._mode = mode
            self._forced, self._forced_until = None, 0.0
            self._expires_at = (time.monotonic() + seconds) if (seconds and mode != "off") else None
        tail = f", back to off in {seconds:g}s" if seconds and mode != "off" else ""
        return f"[{mode}]{tail}"

    def toggle(self):
        """off <-> whichever active mode was used last."""
        return self.set_mode("off" if self.mode != "off" else self._last_active)

    def fire(self, pose, seconds=None, sticky=False):
        """Force a pose on screen. Works in manual and auto; ignored when off."""
        if pose not in self.poses:
            return f"no pose called {pose!r}"
        with self._lock:
            if self._mode == "off":
                return f"currently off — 'manual' first, then '{pose}'"
            self._forced = pose
            self._forced_until = STICKY if sticky else time.monotonic() + (seconds or self.pop_seconds)
        return f"-> {pose}" + ("  (held — 'clear' to drop it)" if sticky else "")

    def clear(self):
        with self._lock:
            self._forced, self._forced_until = None, 0.0
        return "cleared"

    def take_forced(self, now):
        """The pose to show this frame, or None. Called once per frame."""
        with self._lock:
            if self._forced and now < self._forced_until:
                return self._forced
            self._forced = None
            return None

    def tick(self, now):
        """Expire a timed arm. Called once per frame."""
        with self._lock:
            if self._expires_at is not None and now >= self._expires_at:
                self._expires_at = None
                self._mode = "off"
                self._dirty = True
                expired = True
            else:
                expired = False
        if expired:
            self.log("\n[timer expired] back to off — plain webcam.")

    def consume_dirty(self):
        """True once after each mode change, so the loop can drop stale state."""
        with self._lock:
            was, self._dirty = self._dirty, False
            return was

    # ------------------------------------------------------------- commands

    def _match(self, word):
        """Pose by exact name, by unique prefix, or by 1-based number."""
        if word.isdigit():
            i = int(word) - 1
            return (self.poses[i], None) if 0 <= i < len(self.poses) else (None, f"pick 1-{len(self.poses)}")
        if word in self.poses:
            return word, None
        hits = [p for p in self.poses if p.startswith(word)]
        if len(hits) == 1:
            return hits[0], None
        if hits:
            return None, "ambiguous: " + ", ".join(hits)
        return None, f"no pose called {word!r} — try 'list'"

    def handle(self, line):
        """Parse one command line. Returns a string to show the user."""
        parts = line.strip().lower().split()

        # A bare Enter is the panic button: smash it and you are a plain webcam.
        if not parts:
            return self.set_mode("off")

        cmd, rest = parts[0], parts[1:]
        seconds = None
        if rest and rest[-1].replace(".", "", 1).isdigit():
            seconds = float(rest[-1])
            rest = rest[:-1]

        if cmd in ("off", "safe", "panic", "stop", "p"):
            return self.set_mode("off")
        if cmd in ("on", "auto", "a"):
            return self.set_mode("auto", seconds)
        if cmd in ("manual", "man", "m"):
            return self.set_mode("manual", seconds)
        if cmd in ("toggle", "t"):
            return self.toggle()
        if cmd in ("clear", "x"):
            return self.clear()
        if cmd in ("hold", "h"):
            if not rest:
                return "hold what? e.g. 'hold heart'"
            pose, err = self._match(rest[0])
            return err or self.fire(pose, sticky=True)
        if cmd in ("list", "ls", "poses"):
            return "\n".join(f"  {i + 1:>2}  {p}" for i, p in enumerate(self.poses))
        if cmd in ("status", "s", "?"):
            left = self.seconds_left()
            return f"[{self.mode}]" + (f"  {left:.0f}s left" if left else "")
        if cmd in ("help", "commands"):
            return self.help_text()
        if cmd in ("quit", "exit", "q"):
            self.quit = True
            return "quitting"

        pose, err = self._match(cmd)
        if err:
            return err
        return self.fire(pose, seconds)

    def help_text(self):
        return (
            "\n  commands (type here, Enter)        keys (preview window)\n"
            "  ---------------------------        ---------------------\n"
            "  off | panic | <blank Enter>        space  -> off\n"
            "  manual [secs]                      n      -> manual\n"
            "  auto | on [secs]                   m      -> toggle\n"
            "  heart | 2 | crash  fire a meme     1-9 0 - = [ ]  fire a meme\n"
            "  hold heart / clear                 d      -> HUD\n"
            "  list | status | help | quit        q      -> quit\n"
            "\n  'auto 45' arms for 45 seconds, then drops back to off by itself.\n"
        )

    # -------------------------------------------------------------- inputs

    def start_console(self):
        """Read commands from the terminal on a daemon thread."""
        def loop():
            while not self.quit:
                line = sys.stdin.readline()
                if not line:          # stdin closed
                    return
                try:
                    self.log(self.handle(line))
                except Exception as e:      # a bad command must never kill the camera
                    self.log(f"! {e}")
        threading.Thread(target=loop, daemon=True, name="meme-console").start()

    def start_hotkeys(self):
        """Global hotkeys, so you never have to leave the Meet tab."""
        # pynput raises ImportError both when it isn't installed and when it is
        # installed but can't reach a display, so the exception type alone can't
        # tell you which. Ask whether the package exists first, or you end up
        # telling someone to install what they already have.
        import importlib.util
        if importlib.util.find_spec("pynput") is None:
            self.log("hotkeys need pynput:  pip install pynput   (typed commands still work)")
            return False
        try:
            from pynput import keyboard
        except Exception as e:
            # Installed, but no display or the OS refused input monitoring.
            # Never fatal: the camera matters more than the shortcut.
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            self.log(f"hotkeys unavailable ({first}) - typed commands still work")
            return False
        binds = {
            "<ctrl>+<alt>+m": lambda: self.log(self.handle("toggle")),
            "<ctrl>+<alt>+n": lambda: self.log(self.handle("manual")),
            "<ctrl>+<alt>+.": lambda: self.log(self.handle("off")),
        }
        # Fire a meme without leaving the Meet tab.
        for ch, pose in zip(HOTKEY_MEMES, self.poses):
            binds[f"<ctrl>+<alt>+{ch}"] = lambda p=pose: self.log(self.fire(p))
        try:
            hk = keyboard.GlobalHotKeys(binds)
            hk.daemon = True
            hk.start()
        except Exception as e:
            self.log(f"hotkeys unavailable ({e})")
            return False
        self.log("hotkeys: ctrl+alt+M toggle   ctrl+alt+N manual   ctrl+alt+.  off\n"
                 "         ctrl+alt+1-9 0 - = [ ]  fire a meme (works while Meet has focus)")
        return True


def draw_badge(img, ctl):
    """Bottom bar on the *preview* only — never on the frame sent to Meet."""
    import cv2
    mode = ctl.mode
    label = _LABEL[mode]
    left = ctl.seconds_left()
    if left:
        label += f"   ({left:.0f}s)"
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, H - 36), (W, H), (0, 0, 0), -1)
    cv2.rectangle(img, (0, H - 36), (10, H), _COLOUR[mode], -1)
    cv2.putText(img, label, (22, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, _COLOUR[mode], 2)

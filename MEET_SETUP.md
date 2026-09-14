# Safe-mode meme cam for Google Meet

Adds an arm / disarm layer to `gazijarin/itsgiving` so the virtual camera can stay
running all day without ever putting a meme on your face unless you ask for one.

**The design decision that matters:** don't stop the script for serious meetings.
If you quit it mid-call, the camera device disappears and Meet shows a black
rectangle, which is worse than a meme. Instead the script keeps publishing — it
just publishes your untouched webcam feed until you arm it.

| mode | detectors | what Meet sees |
|---|---|---|
| `off` *(default)* | not running at all | your plain webcam |
| `manual` | running | your face, plus a meme **only** when you name one |
| `auto` | running | original behaviour — poses fire on their own |

In `off` the frame is a straight pass-through. Not "memes suppressed" — the
MediaPipe calls are skipped entirely, so there is no code path that can draw
anything. That is the property you want before an interview.

---

## 1. Install

The arm/disarm layer is already integrated in this repo — there is nothing to
copy in. One command does the whole install:

```bash
./setup.sh                                          # macOS / Linux
powershell -ExecutionPolicy Bypass -File setup.ps1  # Windows
```

It finds a usable Python (3.9–3.12; mediapipe has no wheel for 3.13+), builds
the virtualenv, installs the pinned dependencies, runs `doctor.py`, and offers
to calibrate. If anything is wrong it names the command that fixes it.

Run `python doctor.py` any time something misbehaves — it is the fastest way to
find out whether the problem is the camera, the backend, the pins, or the
calibration.

Section 2 below records the edits that were applied to `its_giving_v2.py`, for
anyone who wants to replay them against a fresh clone of the upstream repo.

Virtual-camera backend, once per machine:

| OS | do this |
|---|---|
| macOS | install [OBS Studio](https://obsproject.com), open it once, quit it |
| Windows | install OBS Studio, or run its virtual-camera installer |
| Linux | `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback` |

Calibrate once (seven seconds of a bored face):

```bash
python its_giving_v2.py --calibrate
```

---

## 2. The edits applied to `its_giving_v2.py`

*(Already applied here. This section is the record, not a to-do list.)*

### Edit 1 — import (top of file, after the mediapipe imports)

Find:

```python
from mediapipe.tasks.python import vision
```

Add below it:

```python
from meme_control import Controller, draw_badge
```

### Edit 2 — two new flags (in `main()`, in the argparse block)

Find:

```python
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
```

Add below it:

```python
    ap.add_argument("--mode", default="off", choices=("off", "manual", "auto"),
                    help="starting state (default: off — plain webcam)")
    ap.add_argument("--hotkeys", action="store_true",
                    help="global hotkeys via pynput, so you needn't leave the Meet tab")
```

### Edit 3 — build the controller (just before the main loop)

Find this line:

```python
    print("Running. Focus the preview window: q quit, d HUD, c recalibrate, 1-9 0 - = [ ] test a pose")
```

Replace it with:

```python
    ctl = Controller(POSES, mode=args.mode)
    ctl.start_console()
    if args.hotkeys:
        ctl.start_hotkeys()
    print(ctl.help_text())
    print(f"Running in [{ctl.mode}]. Type a command here, or use the keys in the preview window.")
```

### Edit 4 — gate the detectors (the big one)

Inside `while True:`, find the block that starts:

```python
            ts = clock.next()
```

…and ends with:

```python
            else:
                shown = None
```

Replace that **whole** block with:

```python
            now = time.monotonic()
            ctl.tick(now)
            if ctl.consume_dirty():          # mode just changed — drop stale state
                motion, shown, hold = Motion(), None, 0
                arm = {p: 0 for p in POSES}

            face, hands, body, m = None, [], None, {}
            raw, dbg = None, {}

            if ctl.mode == "off":
                # No detection, no overlay. `frame` goes to the virtual camera
                # exactly as the webcam produced it.
                shown, hold = None, 0
            else:
                ts = clock.next()
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fr = face_det.detect_for_video(mp_img, ts)
                hr = hand_det.detect_for_video(mp_img, ts)
                pr = pose_det.detect_for_video(mp_img, ts)

                face = Face(fr.face_landmarks[0],
                            fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                    if fr.face_landmarks else None
                hands = [Hand(h, W, H) for h in hr.hand_landmarks]
                body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None

                m = measure(face, base) if face is not None else {}
                tongue = tongue_score(frame, face, hands,
                                      over("tongue_jaw", m, "z_jaw", "jaw")) if face is not None else 0.0
                gesture = motion.update(hands, face)
                raw, dbg = decide(face, hands, body, tongue, gesture, m)

                fired = None
                for p in POSES:
                    arm[p] = arm[p] + 1 if raw == p else 0
                    if ctl.mode == "auto" and raw == p and arm[p] >= ARM.get(p, 3):
                        fired = p
                fired = ctl.take_forced(now) or fired   # a named meme always wins

                if fired:
                    if fired != shown:
                        shown_since = now
                    shown, hold = fired, HOLD_FRAMES
                elif hold > 0:
                    hold -= 1
                else:
                    shown = None
```

Note what stays unchanged below it: the `if vcam: vcam.send(frame)` block. In
`off` mode `frame` was never written to, so the send is a pass-through.

### Edit 5 — always show the mode on the preview

Find:

```python
            preview = frame
            if show_hud:
                preview = frame.copy()
                draw_hud(preview, shown, raw, dbg, face, hands, body, base)
```

Replace with:

```python
            preview = frame.copy()
            if show_hud:
                draw_hud(preview, shown, raw, dbg, face, hands, body, base)
            draw_badge(preview, ctl)        # preview only — never sent to Meet
```

The badge is drawn on a *copy*, after `vcam.send()`. Nobody in the call sees it.

### Edit 6 — keys

Find:

```python
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
```

Replace with:

```python
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or ctl.quit:
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("m"):
                print(ctl.handle("toggle"))
            elif key == ord("n"):
                print(ctl.handle("manual"))
            elif key == ord(" "):
                print(ctl.handle("off"))
```

Then, at the bottom of the same key block, find:

```python
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
```

Replace with:

```python
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                ctl.fire(POSES[TEST_KEYS.index(chr(key))])
```

(The old `forced, forced_until = None, 0.0` initialiser above the loop is now
unused. Harmless either way.)

---

## 3. Run it

```bash
python its_giving_v2.py --hotkeys            # starts OFF
python its_giving_v2.py --mode manual        # starts armed, command-only
python its_giving_v2.py --size 640x480       # if your laptop fans spin up
```

It prints the device name. Point Meet at that.

---

## 4. Commands

Type these into the terminal that is running the script:

| command | effect |
|---|---|
| *(blank Enter)* | **panic — straight to off.** Smash it. |
| `off` / `panic` | plain webcam |
| `manual` | armed, command-only |
| `auto` | poses fire on their own |
| `auto 45` | auto for 45 seconds, then back to off **by itself** |
| `heart`, `crash`, `2` | fire that meme for ~2.5s (prefix or number both work) |
| `hold heart` / `clear` | stick one up until cleared |
| `list` / `status` / `help` | … |
| `quit` | shut down cleanly |

Keys in the preview window: `space` off, `n` manual, `m` toggle, `d` HUD,
`1-9 0 - = [ ]` fire a meme, `q` quit.

Global hotkeys with `--hotkeys` (work while Meet has focus, so you never
leave the call):
`ctrl+alt+M` toggle · `ctrl+alt+N` manual · `ctrl+alt+.` off ·
`ctrl+alt+1-9 0 - = [ ]` fire that meme (same order as `list`).

The timed arm is the feature to actually use. `auto 60` before a call with
friends means it cannot possibly still be armed an hour later when your
Extended Essay supervisor dials in.

---

## 5. Google Meet specifics

1. **Start the script before opening the Meet tab.** Chrome enumerates cameras
   when the tab gets media permission. Meet handles a late-arriving device
   better than Zoom does, but starting first avoids the question entirely.
2. Join, then **⋮ → Settings → Video → Camera → "OBS Virtual Camera"**. Meet
   remembers this per browser profile, so it is a one-time choice.
3. **Turn Meet's own effects off.** ⋮ → Apply visual effects → None. Background
   blur runs *after* your feed arrives, so it will happily blur a meme into
   mush, and it costs you a lot of CPU on top of MediaPipe.
4. **Mirroring.** Your self-view is mirrored; other people see the unmirrored
   feed. Only matters if a meme has text on it.
5. **Test on yourself first.** Open a Meet with no one else in it, arm `auto`,
   and pull faces. Meet's self-view is what the room gets.
6. **CPU.** Three MediaPipe models at 1280x720 is real work. `--size 640x480`
   roughly quarters it, and Meet downscales you anyway.

### Two honest warnings

**It will fire on things you didn't mean.** `talking_to_wall` triggers on hands
moving in frame, `suspicious` on a turned head plus a squint, `open_mouth` on a
yawn. That is the whole reason `manual` exists — in `manual` you get the
tracking without the surprises, which is the mode to use for anything where
being funny is a bonus rather than the point.

**Check the room before you arm it.** School and university Meets get recorded,
and a recording outlives the joke. Default-off plus `auto 60` is the habit that
keeps this a good idea. For anything that goes near an admissions officer or a
recommender, leave it off — your camera should be boring and your work
shouldn't be.

---

## 6. If something breaks

| symptom | cause |
|---|---|
| Meet shows no "OBS Virtual Camera" | OBS never opened once / script started after the tab |
| black frame in Meet | script quit — restart it, the device vanishes with the process |
| `ImportError: pyvirtualcam` | `pip install pyvirtualcam` inside the venv |
| hotkeys silent on macOS | System Settings → Privacy & Security → Accessibility → allow your terminal |
| MediaPipe aborts on start | you unpinned a requirement — `pip install -r requirements.txt` |
| everything fires at once | calibrated mid-expression — `python its_giving_v2.py --calibrate` again |
| nothing fires in `auto` | check `POSES` order before touching thresholds; first match wins |

# It's giving...

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/aa5ed48f-70c2-4022-ac7f-87a4c3066a24" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/c764c5eb-c17a-47f2-b4b0-49153c8cb3c0" width="100%"></td>
  </tr>
</table>

Pull a face at your webcam. It works out *which* face, and drops the matching
meme over your head, scaled to follow you around the frame. You can extend and
add more memes to your heart's desire.

Point Zoom at its virtual camera and the whole call sees it.

```bash
python its_giving.py              # preview + virtual camera
python its_giving.py --no-vcam    # preview only
```

Fourteen reactions: time out, heart hands, hands over face, crashing out,
dancing, nose pinch, flirty, hand up, tongue out, gasp, disgust, talking to the
wall, side-eye, and spinning.

There's a second file, `its_giving_v2.py`, which is the same thing with the
expression thresholds calibrated to *your* face instead of to a number I
guessed. 

---

## Setup

One command. It finds a usable Python, builds the virtualenv, installs the
pinned dependencies, checks the machine, and offers to calibrate:

```bash
./setup.sh                                          # macOS / Linux
powershell -ExecutionPolicy Bypass -File setup.ps1  # Windows
```

Or by hand:

```bash
python3.12 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
python doctor.py
```

Python 3.9–3.12 (mediapipe 0.10.21 has no wheel for 3.13+). Three MediaPipe
models (~17 MB) download themselves on first run.

**`python doctor.py` is the thing to run when something is wrong.** It checks
the Python version, the dependency pins, the models and assets, whose face the
calibration belongs to, which camera indexes actually work, and whether a
virtual-camera backend exists — and prints the command that fixes each one.

**Don't unpin the dependencies.** MediaPipe 0.10.30+ (including 1.0.x) ships
macOS wheels that abort the moment they open a detector, so it's held at
0.10.21. That build needs NumPy 1.x, and OpenCV 5 needs NumPy 2 — and 0.10.21
asks for an *unpinned* `opencv-contrib-python`, which quietly drags OpenCV 5 and
therefore NumPy 2 back in. That's why the OpenCV pins are in there even though
nothing in the code cares. Unpin one and you have to unpin all three.

---

## Running it

```bash
source venv/bin/activate              # every new terminal
python its_giving_v2.py --calibrate   # once, seven seconds
python its_giving_v2.py               # starts OFF - plain webcam
```

**Calibrate before you rely on it.** A `calibration.json` ships in this repo and
it is somebody else's resting face. It is valid JSON, so it loads without any
warning and quietly measures your expressions against a stranger's neutral.
`doctor.py` flags it.

**It starts with the memes off.** The virtual camera runs and carries your
ordinary face; nothing fires until you arm it. See
[Arming and disarming](#arming-and-disarming) below, or `MEET_SETUP.md` for the
full Google Meet walkthrough.

| key | does |
|---|---|
| `q` | quit |
| `d` | toggle the HUD |
| `c` | recalibrate |
| `space` | off — plain webcam |
| `n` | manual — armed, command-only |
| `m` | toggle off / last armed mode |
| `1`–`9` `0` `-` `=` `[` `]` | force a reaction on screen for ~2.5 seconds |

---

## Using it in meetings

The virtual camera is on by default, and Zoom, Meet, Teams, Discord and OBS all
treat it as a normal webcam.

**1. Install a backend** (once):

| OS | do this |
|---|---|
| macOS | install [OBS Studio](https://obsproject.com), open it once, quit it |
| Windows | install OBS Studio, or run its virtual-camera installer |
| Linux | `sudo apt install v4l2loopback-dkms` then `sudo modprobe v4l2loopback` |

**2. Run it.** It prints the device it's publishing to:

```
Virtual camera: 'OBS Virtual Camera'  <- pick this camera in Zoom / Meet
```

**3. Pick that device** in your meeting app — Zoom: Settings → Video → Camera.
Meet, Teams and Discord all have the same setting under Video.

**Start this before your meeting app.** Most of them scan for cameras once at
launch and won't notice a device that appeared later.

**Don't quit it for a serious meeting.** Killing the script removes the camera
device and your meeting app shows a black rectangle, which is worse than a meme.
Leave it running and set it to `off` instead — that is what the modes are for.

---

## Arming and disarming

Three modes. It starts in `off`.

| mode | detectors | what the call sees |
|---|---|---|
| `off` *(default)* | not running at all | your plain webcam |
| `manual` | running | your face, plus a meme **only** when you name one |
| `auto` | running | reactions fire on their own |

In `off` the frame is a straight pass-through: the MediaPipe calls are skipped
entirely, so there is no code path that can draw anything. Type commands into
the terminal running the script:

| command | effect |
|---|---|
| *(blank Enter)* | **panic — straight to off.** Smash it. |
| `off` / `panic` | plain webcam |
| `manual` | armed, command-only |
| `auto` | reactions fire on their own |
| `auto 45` | auto for 45 seconds, then back to off **by itself** |
| `heart`, `crash`, `2` | fire that reaction (prefix or number both work) |
| `hold heart` / `clear` | stick one up until cleared |
| `list` / `status` / `help` / `quit` | … |

`--mode manual` or `--mode auto` changes what it starts in. `--hotkeys` adds
global shortcuts via pynput (`ctrl+alt+M` toggle, `ctrl+alt+N` manual,
`ctrl+alt+.` off, `ctrl+alt+1-9 0 - = [ ]` fire that meme) so you needn't
leave the meeting tab.

The timed arm is the one to build a habit around. The realistic mistake isn't
forgetting to switch it on — it's forgetting it's still on two hours later, so
`auto 60` before a call with friends means it cannot still be armed when
someone who matters dials in.

Two honest warnings. **`auto` misfires**: `talking_to_wall` triggers on hands
moving in frame and `suspicious` on a turned head plus a squint, so an ordinary
explaining-something gesture can set it off — `manual` is the mode for anything
where funny is a bonus rather than the point. And **check the room before you
arm it**: calls get recorded, and a recording outlives the joke.

---

## The reactions

| pose | do this |
|---|---|
| `time_out` | referee's T — one hand flat on top, one vertical underneath |
| `heart` | two hands, index tips together, thumb tips together |
| `cover_nose` | both hands over your nose and mouth |
| `crashing_out` | both hands to your head, mouth open |
| `dance` | both hands up behind your head, mouth closed |
| `nose_closed` | pinch your nose shut |
| `flirty` | one index fingertip on your lips |
| `hand_up` | one open palm up beside your head |
| `tongue_out` | tongue out, mouth open |
| `open_mouth` | jaw drops |
| `disgusted` | scrunch your nose, or brows down and frown |
| `talking_to_wall` | hands in frame, gesturing away |
| `suspicious` | turn your head and squint |
| `spin` | leave the frame entirely |

Assets live in `assets/`, named after the pose — `heart.jpeg`, `spin.gif`.
Swap in your own by dropping a file with the right name; JPEG, PNG and animated
GIF all work, alpha channels composite properly, and GIF frame timings are read
from the file. A missing asset gets you a red placeholder, not a crash.

---
## Making it yours

### Swapping a meme (~30 seconds)

Drop a file in `assets/` named after the pose — `heart.png` replaces the heart
reaction. JPEG, PNG and animated GIF all work; transparency composites properly
and GIF timings are read from the file. A `something_` prefix is ignored, so
`2019_heart.jpeg` still counts. Press that pose's test key to check it sits
right on your head.

### Adding a pose

**1.** Drop `assets/thinking.png` in place.

**2.** Add the name to `POSES`. The list is checked top to bottom and the first
match wins, so put it above anything it might be mistaken for.

**3.** Add a branch to `decide()`:

```python
    for h in hands:
        if near(h.palm, face.chin, 0.5) and not h.open:
            return "thinking", d
```

You have `face` (`.nose` `.chin` `.mouth` `.w` `.h`, `.b("jawOpen")` for any
blendshape), `hands` (`.palm` `.thumb` `.index`, `.open`), `body`
(`.elbows_up`), `m` for expressions in sigma, and `near(a, b, k)` for "within k
face widths" — which is what keeps it working at any distance from the camera.

**4.** Give it an `ARM` count if it's twitchy, then tune it against the HUD.
Getting it to fire is easy; the work is *stopping* doing it, doing everything
nearby that might be confused with it, and watching the number stay low.

If your pose needs an expression channel that isn't measured yet, add it to `Z`,
`FLOOR` and `measure()`, then put it in `draw_hud()` - you can't tune a number
you can't see.

### Two gotchas

`TEST_KEYS` has one key per pose, matched by position. Adding a fifteenth pose
is fine (it just gets no test key), but removing one without removing a key
crashes when that key is pressed.

If a new pose never fires, check the `POSES` order before you touch any
threshold. Something earlier matching first is the usual cause, and lowering
`Z` can't fix it.

---

## How it works

```
camera frame
     |
 1.  MediaPipe    face: 478 landmarks + 52 blendshapes
                  hands: 2 x 21 points
                  body: shoulders, elbows, wrists
     |
 2.  Measures     face-relative geometry, tongue colour, hand speed
     |
 3.  Baseline     expressions re-expressed in sigma above YOUR neutral face
     |
 4.  decide()     one ordered pass -- first pose that matches wins
     |
 5.  arm / hold   must persist N frames to fire, lingers 10 frames after
     |
  overlay         scaled to your face, alpha-composited, GIFs animated
```

### Normalising away the camera

Landmarks come out as pixel coordinates, which depend on how far you're sitting
from the lens. So nothing is compared in pixels: every distance is divided by
the width of your face box first. `near(hand.index, face.mouth, 0.22)` means
"within 22% of a face width", and that means the same thing at 40 cm and at a
metre and a half. Hand speed gets the same treatment — face-widths per frame.

Head turn is the nose's position between the two edges of your face: 0 facing
the camera, about 0.4 in full profile. Already a ratio, so already scale-free.

### Why fixed thresholds don't work, and what to do instead

This is the interesting part.

MediaPipe's blendshape values are **not zero when your face is at rest**, and
the offset is very personal. Some faces idle at `jawOpen` 0.02; others sit at
0.19 doing nothing. If your mouth naturally turns up you can read `mouthSmile`
0.3 while thinking about absolutely nothing.

So `jawOpen > 0.5` is not one threshold — it's a different threshold for every
face that meets it. Too eager for some, physically unreachable for others. Any
constant you pick is a compromise between people, and no individual user is the
average of those people.

v2 fixes this by measuring your own neutral first. Seven seconds of a bored face
records the **mean and the standard deviation** of all 52 channels, and from
then on every expression is scored as:

```
z = (what the channel reads now - your resting mean) / your resting wobble
```

"6 sigma above your neutral jaw" means the same thing on every face. "Above 0.5"
doesn't. Same two gestures, on a face that idles low and sits still versus one
that idles high and fidgets:

```
                        still face        loose face
resting                 z  +0.1           z  +0.1        both quiet
gasp                    z +38.7           z  +8.7        both fire
nose scrunch            z +19.3           z  +5.5        both fire
```

---

## What's where

```
its_giving_v2.py   the calibrated version — the one to use
its_giving.py      v1: same poses, fixed thresholds
meme_control.py    the off / manual / auto layer and its commands
doctor.py          checks this machine can run it, and says what to fix
setup.sh           one-command install (macOS / Linux)
setup.ps1          one-command install (Windows)
MEET_SETUP.md      Google Meet walkthrough, and what to do when it breaks
calibration.json   your neutral face (made by --calibrate, gitignored)
requirements.txt   pinned on purpose — read the comments before changing them
assets/            the memes, named after their pose
models/            MediaPipe .task files (downloaded on first run)
```

Inside the file: `POSES` / `Z` / `FLOOR` / `ARM` is the tuning block,
`Baseline` and `run_calibration()` are the seven-second sit-still, `measure()`
turns a face into sigma-above-your-neutral, and `decide()` is the ordered pose
checks.

Want to change **what sets off what**? `decide()`.
Want to change **how easily it goes off**? `Z`, `FLOOR` and `ARM`.

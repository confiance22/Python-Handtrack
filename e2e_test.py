"""End-to-end headless test: runs main.py with fake camera + fake hands.

Patches cv2 (camera/imshow/waitKey) and replaces the HandLandmarker
factory with a fake that returns synthetic hand landmarks. Hand poses
change over time (pinch pinky <-> release) so the gesture system cycles
through all 18 filters while a portal polygon stays open, exercising
every apply_filter() branch (including GALAXY with the real segmenter).
Also presses the screenshot / record / full-frame / help keys to exercise
the capture and mode features.

Run:  venv/Scripts/python.exe e2e_test.py
"""
import os
import runpy
import shutil
import sys

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480
FILTERS = [
    "MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR",
    "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY",
    "RAINBOW", "KALEIDOSCOPE", "SWIRL", "BULGE", "VHS",
    "WATERCOLOR", "EMBOSS",
]
FRAMES_PER_POSE = 4

_frame_counter = {"n": 0}


class FakeVideoCapture:
    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def read(self):
        frame = np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        cv2.circle(frame, (FRAME_W // 2, FRAME_H // 2), 100, (200, 150, 120), -1)
        return True, frame

    def release(self):
        pass


class LM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def make_hand(cx, cy, pinch_pinky):
    """21-landmark hand; thumb(4)+index(8) always pinched at (cx,cy).

    If pinch_pinky is True, pinky(20) is brought close to the thumb to
    trigger the 'change filter' gesture.
    """
    lms = [LM(cx, cy) for _ in range(21)]
    lms[4] = LM(cx, cy)                # thumb tip
    lms[8] = LM(cx + 0.05, cy + 0.02)  # index tip (portal corner)
    if pinch_pinky:
        lms[20] = LM(cx + 0.03, cy + 0.01)  # pinky ~ thumb -> change filter
    else:
        lms[20] = LM(cx + 0.35, cy + 0.15)  # far away
    return lms


class FakeLandmarker:
    @staticmethod
    def create_from_options(*a, **k):
        return FakeLandmarker()

    def detect_for_video(self, image, ts):
        n = _frame_counter["n"]
        phase = (n // FRAMES_PER_POSE) % 2
        pinch_pinky = (phase == 0)
        return FakeResults([make_hand(0.35, 0.40, pinch_pinky),
                            make_hand(0.70, 0.40, pinch_pinky)])


class FakeResults:
    def __init__(self, hand_landmarks):
        self.hand_landmarks = hand_landmarks


class FakeKeyQueue:
    """Scheduled key presses; everything else is 'no key'. Quits at the end."""

    def __init__(self):
        # Key events scheduled at specific frame numbers (exercise features)
        self.schedule = {
            8: ord('h'),    # toggle help panel off
            30: ord('s'),   # screenshot
            50: ord('r'),   # start recording
            110: ord('f'),  # full-frame mode on
            160: ord('f'),  # full-frame mode off
            200: ord('r'),  # stop recording
            230: ord('t'),  # toggle trails
        }
        self.remaining = 260

    def poll(self):
        self.remaining -= 1
        n = _frame_counter["n"]
        _frame_counter["n"] += 1
        if self.remaining <= 0:
            return ord('q')
        return self.schedule.get(n, 0)


# Track which filters were actually rendered (via the on-screen label)
_filters_seen = []
_orig_put_text = cv2.putText


def tracking_put_text(img, text, *a, **k):
    label = "Filter Aktif:"
    if isinstance(text, str) and text.startswith(label):
        name = text[len(label):].strip()
        if name not in _filters_seen:
            _filters_seen.append(name)
    return _orig_put_text(img, text, *a, **k)


# --- Patch cv2 ------------------------------------------------------------
cv2.VideoCapture = FakeVideoCapture
cv2.imshow = lambda *a, **k: None
cv2.destroyAllWindows = lambda: None
cv2.putText = tracking_put_text
_key_queue = FakeKeyQueue()
cv2.waitKey = lambda ms: _key_queue.poll()

# --- Patch the HandLandmarker factory used by main.py ---------------------
import mediapipe as mp  # noqa: E402
mp.tasks.vision.HandLandmarker.create_from_options = FakeLandmarker.create_from_options

# --- Run the actual app ---------------------------------------------------
try:
    runpy.run_path("main.py", run_name="__main__")
finally:
    cv2.destroyAllWindows = lambda: None  # app already cleaned up

# --- Assertions -----------------------------------------------------------
missing = [f for f in FILTERS if f not in _filters_seen]
print()
print(f"Filters rendered: {len(_filters_seen)}/{len(FILTERS)}")
print(f"  seen: {_filters_seen}")
if missing:
    print(f"[FAIL] Filters never reached: {missing}")
    sys.exit(1)

shots = [f for f in os.listdir("captures") if f.endswith(".png")] if os.path.isdir("captures") else []
recs = [f for f in os.listdir("recordings") if f.endswith(".avi")] if os.path.isdir("recordings") else []
print(f"Screenshots saved: {len(shots)} | Recordings saved: {len(recs)}")
if not shots:
    print("[FAIL] No screenshot was produced by the 's' key.")
    sys.exit(1)
if not recs:
    print("[FAIL] No video was produced by the 'r' key.")
    sys.exit(1)
if not os.path.getsize(os.path.join("recordings", recs[0])) > 0:
    print("[FAIL] Recorded video file is empty.")
    sys.exit(1)

# clean up test artifacts
shutil.rmtree("captures", ignore_errors=True)
shutil.rmtree("recordings", ignore_errors=True)

print(f"[PASS] main.py ran end-to-end through {len(FILTERS)} filters")
print("       with portal + gestures + full-frame + screenshot + recording, no crash.")

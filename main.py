"""
RETROLENS PRO - Interactive Hand-Tracking Camera Filters & Portal
=================================================================
Full-screen revamp: 24+ filters, scalable UI, animated portal, glow trails,
hand skeleton, full-frame mode, screenshots, video recording, filter carousel,
FPS meter, settings panel, particle system, and gesture controls.

Controls
--------
Gestures:
  - Pinch thumb + index on BOTH hands ............ open/close portal
  - Thumb + pinky (single hand) .................. next filter
  - Both index fingertips close .................. prev filter

Keys:
  q  quit              s  screenshot             r  record video
  f  toggle fullscreen F  force windowed          t  toggle trails/skeleton
  h  help panel        g  settings panel         n / ->  next filter
  p / <-  prev filter  c  clear trails           m  mirror toggle
  1-9  quick filter    ESC exit fullscreen
"""

import math
import os
import time
import json

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------
#  Config
# --------------------------------------------------------------------------
class Config:
    FILTERS = [
        "MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR",
        "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY",
        "RAINBOW", "KALEIDOSCOPE", "SWIRL", "BULGE", "VHS",
        "WATERCOLOR", "EMBOSS", "EDGE", "POSTERIZE", "SOLARIZE",
        "CHROMATIC", "CYBERPUNK", "RETROWAVE",
    ]

    GESTURE_DISTANCE = 45          # px threshold for pinch
    PORTAL_SMOOTHING = 0.60        # EMA factor (0 = none, 1 = heavy)
    TRAIL_LIFETIME = 1.0           # seconds
    MAX_TRAIL_POINTS = 500
    REC_FPS = 25.0
    GALAXY_SIZE = (1920, 1080)
    UI_SCALE = 1.0                 # set by screen size
    MIRROR = True

    SETTINGS_FILE = "settings.json"

    @classmethod
    def load(cls):
        if os.path.exists(cls.SETTINGS_FILE):
            try:
                with open(cls.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cls.MIRROR = data.get("mirror", cls.MIRROR)
                cls.TRAIL_LIFETIME = data.get("trail_lifetime", cls.TRAIL_LIFETIME)
                cls.PORTAL_SMOOTHING = data.get("portal_smoothing", cls.PORTAL_SMOOTHING)
            except Exception:
                pass

    @classmethod
    def save(cls):
        data = {
            "mirror": cls.MIRROR,
            "trail_lifetime": cls.TRAIL_LIFETIME,
            "portal_smoothing": cls.PORTAL_SMOOTHING,
        }
        try:
            with open(cls.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


Config.load()

# Hand topology (MediaPipe 21-point landmark connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)
THUMB = 4
INDEX = 8
PINKY = 20

# --------------------------------------------------------------------------
#  MediaPipe models
# --------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
))

segmenter = None
try:
    segmenter = ImageSegmenter.create_from_options(ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path="selfie_segmenter.tflite"),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True,
    ))
except Exception as e:
    print("Segmenter tidak tersedia (selfie_segmenter.tflite):", e)

# --------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------
def make_galaxy_background():
    w, h = Config.GALAXY_SIZE
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (20, 8, 35)
    rng = np.random.default_rng(42)
    for _ in range(1200):
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        bg[sy, sx] = (255, 255, 255)
    for _ in range(150):
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(bg, (sx, sy), int(rng.integers(2, 7)),
                   (int(rng.integers(150, 255)), int(rng.integers(100, 255)), 255), -1)
    # nebula clouds
    for _ in range(8):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        radius = int(rng.integers(100, 400))
        color = (int(rng.integers(0, 80)), int(rng.integers(0, 60)), int(rng.integers(120, 200)))
        cv2.circle(bg, (cx, cy), radius, color, -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    return bg


def hsv_palette(n=180, sat=255, val=255):
    px = np.zeros((1, n, 3), dtype=np.uint8)
    px[0, :, 0] = np.arange(n) % 180
    px[0, :, 1] = sat
    px[0, :, 2] = val
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in px) for px in bgr]


PALETTE = hsv_palette()
_remap_cache = {}


def _remap_maps(name, w, h):
    key = (name, w, h)
    if key in _remap_cache:
        return _remap_cache[key]
    cx, cy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    if name == "SWIRL":
        radius = min(w, h) * 0.75
        factor = np.clip((radius - r) / radius, 0, 1) ** 2
        t2 = theta + 2.5 * factor
        mx, my = cx + r * np.cos(t2), cy + r * np.sin(t2)
    elif name == "BULGE":
        max_r = np.sqrt(cx * cx + cy * cy)
        factor = np.clip(1 - r / max_r, 0, 1) ** 2
        scale = 1 - 0.45 * factor
        mx, my = cx + dx * scale, cy + dy * scale
    elif name == "KALEIDOSCOPE":
        segments = 8
        wedge = 2 * np.pi / segments
        seg_idx = np.floor(theta / wedge)
        theta_folded = np.where(seg_idx % 2 == 0,
                                np.mod(theta, wedge),
                                wedge - np.mod(theta, wedge))
        mx, my = cx + r * np.cos(theta_folded), cy + r * np.sin(theta_folded)
    else:
        raise ValueError(name)

    _remap_cache[key] = (mx.astype(np.float32), my.astype(np.float32))
    return _remap_cache[key]


def _watercolor(roi):
    small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    res = cv2.bilateralFilter(small, 9, 75, 75)
    res = cv2.medianBlur(res, 5)
    return cv2.resize(res, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)


def _vhs(roi, frame_index):
    h, w = roi.shape[:2]
    out = roi.copy()
    shift = 5
    if w > shift:
        out[:, shift:, 2] = roi[:, :-shift, 2]
        out[:, :-shift, 0] = roi[:, shift:, 0]
    out[::2, :] = (out[::2, :].astype(np.float32) * 0.6).astype(np.uint8)
    noise = np.random.randint(0, 35, (h, w, 1), dtype=np.uint8)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3 and h > 20:
        band = int(np.random.randint(0, h - 10))
        bh = int(np.random.randint(4, 16))
        out[band:band + bh] = np.roll(out[band:band + bh],
                                      int(np.random.randint(-20, 21)), axis=1)
    cv2.putText(out, f"PLAY {frame_index // 30:06d}", (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _emboss(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    emb = cv2.filter2D(gray, cv2.CV_32F, kernel)
    out = np.clip(gray.astype(np.float32) + emb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _rainbow(roi, frame_index):
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_grad = (np.arange(w)[None, :] * 180.0 / max(w, 1)) + (frame_index * 1.5)
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_grad, 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _edge(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(roi, 0.3, edges_bgr, 0.7, 0)


def _posterize(roi, levels=6):
    div = 256 // levels
    return (roi // div) * div + div // 2


def _solarize(roi, threshold=128):
    mask = roi > threshold
    return np.where(mask, 255 - roi, roi)


def _chromatic(roi, shift=8):
    out = roi.copy()
    h, w = roi.shape[:2]
    if w > shift * 2:
        out[:, shift:, 2] = roi[:, :-shift, 2]
        out[:, :-shift, 0] = roi[:, shift:, 0]
    return out


def _cyberpunk(roi):
    out = cv2.applyColorMap(roi, cv2.COLORMAP_WINTER)
    return cv2.addWeighted(roi, 0.5, out, 0.5, 0)


def _retrowave(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + 30, 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_filter(roi, filter_name, frame_index=0, x=0, y=0,
                 mask_person=None, frame_galaxy=None):
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    if filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (31, 31), 0)
    if filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]], dtype=np.float32)
        return np.clip(cv2.transform(roi, kernel), 0, 255).astype(np.uint8)
    if filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [147, 20, 255]
        return filtered
    if filter_name == "PIXELATE":
        hr, wr = roi.shape[:2]
        if hr > 8 and wr > 8:
            small = cv2.resize(roi, (wr // 12, hr // 12), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (wr, hr), interpolation=cv2.INTER_NEAREST)
        return roi
    if filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    if filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cv2.bitwise_not(gray), (25, 25), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    if filter_name == "GLITCH":
        wr = roi.shape[1]
        shift = max(6, wr // 18)
        glitch = roi.copy()
        if wr > shift:
            glitch[:, :-shift, 2] = roi[:, shift:, 2]
            glitch[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch
    if filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        return cv2.dilate(edges_bgr, np.ones((3, 3), np.uint8), iterations=1)
    if filter_name == "GALAXY":
        if mask_person is not None and frame_galaxy is not None:
            bh, bw = roi.shape[:2]
            roi_mask = mask_person[y:y + bh, x:x + bw]
            roi_galaxy = frame_galaxy[y:y + bh, x:x + bw]
            filtered = roi.copy()
            filtered[roi_mask == 0] = roi_galaxy[roi_mask == 0]
            return filtered
        return roi
    if filter_name == "RAINBOW":
        return _rainbow(roi, frame_index)
    if filter_name in ("KALEIDOSCOPE", "SWIRL", "BULGE"):
        mx, my = _remap_maps(filter_name, roi.shape[1], roi.shape[0])
        return cv2.remap(roi, mx, my, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    if filter_name == "VHS":
        return _vhs(roi, frame_index)
    if filter_name == "WATERCOLOR":
        return _watercolor(roi)
    if filter_name == "EMBOSS":
        return _emboss(roi)
    if filter_name == "EDGE":
        return _edge(roi)
    if filter_name == "POSTERIZE":
        return _posterize(roi)
    if filter_name == "SOLARIZE":
        return _solarize(roi)
    if filter_name == "CHROMATIC":
        return _chromatic(roi)
    if filter_name == "CYBERPUNK":
        return _cyberpunk(roi)
    if filter_name == "RETROWAVE":
        return _retrowave(roi)
    return roi


def draw_hand_skeleton(img, hand_landmarks, w, h, frame_index):
    for lms in hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            hue = (frame_index * 3 + a * 25) % 180
            cv2.line(img, pts[a], pts[b], PALETTE[hue], 3, cv2.LINE_AA)
        for idx in FINGERTIPS:
            hue = (frame_index * 3 + idx * 25) % 180
            cv2.circle(img, pts[idx], 6, PALETTE[hue], -1, cv2.LINE_AA)
            cv2.circle(img, pts[idx], 10, PALETTE[hue], 2, cv2.LINE_AA)


def draw_animated_border(img, pts, frame_index):
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        nd = max(3, int(dist // 15))
        offset = int(frame_index * 0.3) % nd
        for d in range(nd):
            k0 = (d + offset) / nd
            k1 = (d + 1 + offset) / nd
            p0 = (int(a[0] + (b[0] - a[0]) * k0), int(a[1] + (b[1] - a[1]) * k0))
            p1 = (int(a[0] + (b[0] - a[0]) * k1), int(a[1] + (b[1] - a[1]) * k1))
            hue = int((frame_index * 4 + i * 40) % 180)
            cv2.line(img, p0, p1, PALETTE[hue], 4, cv2.LINE_AA)


def draw_particles(img, particles, frame_index):
    for p in particles[:]:
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1
        if p["life"] <= 0:
            particles.remove(p)
            continue
        alpha = p["life"] / p["max_life"]
        size = max(1, int(4 * alpha))
        hue = int((frame_index + p["hue_off"]) % 180)
        cv2.circle(img, (int(p["x"]), int(p["y"])), size, PALETTE[hue], -1, cv2.LINE_AA)


def spawn_particles(particles, x, y, count=6, hue_off=0):
    for _ in range(count):
        particles.append({
            "x": x, "y": y,
            "vx": np.random.uniform(-4, 4),
            "vy": np.random.uniform(-4, 4),
            "life": int(np.random.uniform(15, 40)),
            "max_life": 40,
            "hue_off": hue_off,
        })


def draw_carousel(img, active, frame_w, frame_index):
    chip_w, chip_h, gap = 130, 30, 10
    step = chip_w + gap
    max_visible = max(1, (frame_w - 40) // step)
    start = max(0, min(active - max_visible // 2,
                       len(Config.FILTERS) - max_visible))
    scale = Config.UI_SCALE
    y0 = int(img.shape[0] - chip_h * scale - 16 * scale)
    for i in range(start, min(start + max_visible, len(Config.FILTERS))):
        x0 = int(20 * scale + (i - start) * step * scale)
        cw = int(chip_w * scale)
        ch = int(chip_h * scale)
        if i == active:
            cv2.rectangle(img, (int(x0 - 4 * scale), int(y0 - 4 * scale)),
                          (int(x0 + cw + 4 * scale), int(y0 + ch + 4 * scale)),
                          PALETTE[int(frame_index * 4) % 180], 3, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch),
                      (60, 60, 80) if i != active else (0, 170, 255), -1, cv2.LINE_AA)
        color = (255, 255, 255) if i != active else (0, 0, 0)
        cv2.putText(img, Config.FILTERS[i], (int(x0 + 8 * scale), int(y0 + ch - 6 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, 2, cv2.LINE_AA)


def draw_help(img, frame_index):
    lines = [
        "RETROLENS PRO - KONTROL / CONTROLS",
        "Cubit jempol+telunjuk 2 tangan .... buka/tutup portal",
        "Jempol+kelingking ................. filter berikutnya",
        "2 telunjuk dekat .................. filter sebelumnya",
        "n / ->  next        p / <-  prev     1-9 quick filter",
        "f  fullscreen      F  windowed      t  trails/skeleton",
        "s  screenshot      r  record video  c  clear particles",
        "h  help            g  settings      m  mirror toggle",
        "ESC  exit fullscreen               q  keluar / quit",
    ]
    scale = Config.UI_SCALE
    ox, oy = int(20 * scale), int(img.shape[0] - 230 * scale)
    w_box, h_box = int(520 * scale), int(215 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + w_box, oy + h_box), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.rectangle(img, (ox, oy), (ox + w_box, oy + h_box), (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (int(ox + 12 * scale), int(oy + 24 * scale + i * 22 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 200, 255) if i == 0 else (255, 255, 255), 2, cv2.LINE_AA)


def draw_settings(img, frame_index):
    lines = [
        "SETTINGS",
        f"Mirror:        {'ON' if Config.MIRROR else 'OFF'}  (m)",
        f"Trails life:   {Config.TRAIL_LIFETIME:.1f}s",
        f"Portal smooth: {Config.PORTAL_SMOOTHING:.2f}",
        "",
        "m  toggle mirror",
        "g  close settings",
    ]
    scale = Config.UI_SCALE
    ox, oy = int(20 * scale), int(80 * scale)
    w_box, h_box = int(340 * scale), int(200 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + w_box, oy + h_box), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.rectangle(img, (ox, oy), (ox + w_box, oy + h_box), (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (int(ox + 12 * scale), int(oy + 26 * scale + i * 24 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                    (0, 200, 255) if i == 0 else (255, 255, 255), 2, cv2.LINE_AA)


def draw_toast(img, text, frame_index):
    scale = Config.UI_SCALE
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, 2)
    tx = int((img.shape[1] - tw) // 2)
    ty = int(100 * scale)
    cv2.rectangle(img, (tx - 12, ty - 10), (tx + tw + 12, ty + th + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale,
                (0, 200, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
#  App state
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Tidak dapat membuka kamera.")
    raise SystemExit(1)

# Try to request a high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

galaxy_bg = make_galaxy_background()
print("RETROLENS PRO dimulai. Tekan 'q' untuk keluar, 'h' bantuan, 'g' pengaturan.")

current_filter = 0
gesture_triggered = False
full_frame = False
fullscreen = True
trails_enabled = True
skeleton_enabled = True
help_visible = False
settings_visible = False
recording = False
writer = None
rec_start = 0.0
frame_index = 0
mirror = Config.MIRROR

trail_points = []
particles = []
smoothed_pts = None
fps = 0.0
prev_time = time.time()
toast_text = ""
toast_until = 0.0
flash_until = 0.0

os.makedirs("captures", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

WINDOW_NAME = "RETROLENS PRO - Hand Tracking Filters"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show_toast(text, seconds=2.0):
    global toast_text, toast_until
    toast_text = text
    toast_until = time.time() + seconds


def toggle_fullscreen(on):
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)


try:
    while True:
        success, img = cap.read()
        if not success:
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        if mirror:
            img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_index += 1

        frame_galaxy = galaxy_bg[:h, :w]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = time.time_ns() // 1_000_000

        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        filter_name = Config.FILTERS[current_filter]

        mask_person = None
        if filter_name == "GALAXY" and segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h),
                                             interpolation=cv2.INTER_NEAREST)

        # ---------------- gestures ----------------
        change_filter = False
        prev_filter = False
        portal_open = False
        if results.hand_landmarks:
            hands = results.hand_landmarks
            # two-hand pinch -> portal
            if len(hands) >= 2:
                i0 = hands[0][INDEX]
                i1 = hands[1][INDEX]
                pt0 = (int(i0.x * w), int(i0.y * h))
                pt1 = (int(i1.x * w), int(i1.y * h))
                if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < Config.GESTURE_DISTANCE:
                    portal_open = True
            # single hand gestures
            for lms in hands:
                thumb = lms[THUMB]
                pinky = lms[PINKY]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                px, py = int(pinky.x * w), int(pinky.y * h)
                if math.hypot(tx - px, ty - py) < Config.GESTURE_DISTANCE:
                    change_filter = True
                # both index tips close (within same hand across hands handled above)
                idx = lms[INDEX]
                ix, iy = int(idx.x * w), int(idx.y * h)
                # check against other hand index if present
                for other in hands:
                    if other is lms:
                        continue
                    oi = other[INDEX]
                    ox_, oy_ = int(oi.x * w), int(oi.y * h)
                    if math.hypot(ix - ox_, iy - oy_) < Config.GESTURE_DISTANCE:
                        prev_filter = True

        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(Config.FILTERS)
                show_toast(f"Filter: {Config.FILTERS[current_filter]}")
                gesture_triggered = True
        elif prev_filter:
            if not gesture_triggered:
                current_filter = (current_filter - 1) % len(Config.FILTERS)
                show_toast(f"Filter: {Config.FILTERS[current_filter]}")
                gesture_triggered = True
        else:
            gesture_triggered = False

        # ---------------- portal ----------------
        pts_portal = []
        if results.hand_landmarks and portal_open:
            for lms in results.hand_landmarks:
                for i in (THUMB, INDEX):
                    lm = lms[i]
                    pts_portal.append([int(lm.x * w), int(lm.y * h)])

        portal_poly = None
        if not full_frame and len(pts_portal) == 4:
            pts_portal.sort(key=lambda p: p[1])
            top_pts, bottom_pts = pts_portal[:2], pts_portal[2:]
            top_pts.sort(key=lambda p: p[0])
            bottom_pts.sort(key=lambda p: p[0])
            raw_poly = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]],
                                dtype=np.float32)
            if smoothed_pts is None:
                smoothed_pts = raw_poly.copy()
            else:
                s = Config.PORTAL_SMOOTHING
                smoothed_pts = (1 - s) * smoothed_pts + s * raw_poly
            portal_poly = smoothed_pts.astype(np.int32)
        else:
            smoothed_pts = None

        # ---------------- trails & particles ----------------
        if trails_enabled and results.hand_landmarks:
            for lms in results.hand_landmarks:
                for i in FINGERTIPS:
                    lm = lms[i]
                    trail_points.append((int(lm.x * w), int(lm.y * h), now))
            cutoff = now - Config.TRAIL_LIFETIME
            while trail_points and trail_points[0][2] < cutoff:
                trail_points.pop(0)
            if len(trail_points) > Config.MAX_TRAIL_POINTS:
                del trail_points[:len(trail_points) - Config.MAX_TRAIL_POINTS]

            if trail_points:
                overlay = np.zeros_like(img)
                for tx, ty, born in trail_points:
                    age = now - born
                    alpha = max(0.0, 1 - age / Config.TRAIL_LIFETIME)
                    hue = int(born * 300) % 180
                    cv2.circle(overlay, (tx, ty), max(1, int(4 + 8 * alpha)),
                               PALETTE[hue], -1)
                glow = cv2.GaussianBlur(overlay, (0, 0), 9)
                img = cv2.addWeighted(img, 1.0, glow, 0.8, 0)

        if skeleton_enabled and results.hand_landmarks:
            draw_hand_skeleton(img, results.hand_landmarks, w, h, frame_index)

        # spawn particles at fingertips
        if results.hand_landmarks and np.random.rand() < 0.3:
            for lms in results.hand_landmarks:
                tip = lms[INDEX]
                px_, py_ = int(tip.x * w), int(tip.y * h)
                spawn_particles(particles, px_, py_, count=2, hue_off=frame_index)
        draw_particles(img, particles, frame_index)

        # ---------------- apply filter ----------------
        if full_frame:
            img = apply_filter(img, filter_name, frame_index, 0, 0,
                               mask_person, frame_galaxy)
        elif portal_poly is not None:
            x, y, bw, bh = cv2.boundingRect(portal_poly)
            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)
            if bw > 0 and bh > 0:
                roi = img[y:y + bh, x:x + bw].copy()
                filtered_roi = apply_filter(roi, filter_name, frame_index,
                                            x, y, mask_person, frame_galaxy)
                mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.fillPoly(mask, [portal_poly - [x, y]], 255)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                img[y:y + bh, x:x + bw] = np.where(mask_3ch == 255,
                                                   filtered_roi, roi)
                draw_animated_border(img, portal_poly, frame_index)
                cv2.putText(img, f"PORTAL: {filter_name}",
                            (int(portal_poly[0][0]), int(portal_poly[0][1] - 15 * Config.UI_SCALE)),
                            cv2.FONT_HERSHEY_PLAIN, 1.5 * Config.UI_SCALE,
                            (255, 255, 255), 2, cv2.LINE_AA)
                spawn_particles(particles, int(portal_poly[0][0]), int(portal_poly[0][1]),
                                count=1, hue_off=frame_index)

        # ---------------- overlays ----------------
        scale = Config.UI_SCALE

        # FPS
        cv2.putText(img, f"{fps:4.1f} FPS", (int(w - 130 * scale), int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), 2, cv2.LINE_AA)

        mode_txt = "FULL-FRAME" if full_frame else ("FULLSCREEN" if fullscreen else "PORTAL")
        cv2.putText(img, f"Mode: {mode_txt}", (int(10 * scale), int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Filter Aktif: {filter_name}", (int(10 * scale), int(62 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 2, cv2.LINE_AA)

        if recording:
            cv2.circle(img, (int(w - 140 * scale), int(58 * scale)),
                       int(8 * scale), (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, f"REC {int(now - rec_start):02d}s",
                        (int(w - 126 * scale), int(64 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 255), 2, cv2.LINE_AA)

        draw_carousel(img, current_filter, w, frame_index)

        if help_visible:
            draw_help(img, frame_index)
        if settings_visible:
            draw_settings(img, frame_index)

        if toast_text and now < toast_until:
            draw_toast(img, toast_text, frame_index)

        if now < flash_until:
            alpha = max(0.0, (flash_until - now) / 0.25)
            img = cv2.addWeighted(img, 1 - alpha, np.full_like(img, 255), alpha, 0)

        if recording and writer is not None:
            writer.write(img)

        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKeyEx(1)
        if key == -1:
            key = 0
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = time.strftime("captures/retrolens_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, img)
            flash_until = now + 0.25
            show_toast(f"Screenshot: {fname}")
        elif key == ord('r'):
            if recording:
                if writer is not None:
                    writer.release()
                writer = None
                recording = False
                show_toast("Rekaman berhenti")
            else:
                fname = time.strftime("recordings/retrolens_%Y%m%d_%H%M%S.avi")
                writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"MJPG"),
                                         Config.REC_FPS, (w, h))
                if writer.isOpened():
                    recording = True
                    rec_start = now
                    show_toast("Merekam video...")
                else:
                    writer = None
                    show_toast("Gagal membuat file video")
        elif key == ord('f'):
            fullscreen = True
            full_frame = False
            toggle_fullscreen(True)
            show_toast("Fullscreen ON")
        elif key == ord('F'):
            fullscreen = False
            toggle_fullscreen(False)
            show_toast("Windowed mode")
        elif key == ord('t'):
            trails_enabled = not trails_enabled
            skeleton_enabled = trails_enabled
            show_toast(f"Trails/Skeleton: {'ON' if trails_enabled else 'OFF'}")
        elif key == ord('h'):
            help_visible = not help_visible
            settings_visible = False
        elif key == ord('g'):
            settings_visible = not settings_visible
            help_visible = False
        elif key == ord('m'):
            mirror = not mirror
            Config.MIRROR = mirror
            Config.save()
            show_toast(f"Mirror: {'ON' if mirror else 'OFF'}")
        elif key == ord('c'):
            trail_points.clear()
            particles.clear()
            show_toast("Partikel dibersihkan")
        elif key == 27:  # ESC
            if fullscreen:
                fullscreen = False
                toggle_fullscreen(False)
                show_toast("Keluar fullscreen")
            else:
                break
        elif key == ord('n') or key == 2555904:
            current_filter = (current_filter + 1) % len(Config.FILTERS)
            show_toast(f"Filter: {Config.FILTERS[current_filter]}")
        elif key == ord('p') or key == 2424832:
            current_filter = (current_filter - 1) % len(Config.FILTERS)
            show_toast(f"Filter: {Config.FILTERS[current_filter]}")
        elif 49 <= key <= 57:  # 1-9
            idx = key - 49
            if idx < len(Config.FILTERS):
                current_filter = idx
                show_toast(f"Filter: {Config.FILTERS[current_filter]}")

finally:
    if writer is not None:
        writer.release()
    Config.save()
    cap.release()
    cv2.destroyAllWindows()

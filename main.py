"""
RETROLENS - Interactive Hand-Tracking Camera Filters & Portal
=============================================================
Enhanced version: 18 filters, fingertip glow trails, hand skeleton,
full-frame mode, screenshots, video recording, animated portal, FPS meter
and a filter carousel UI.

Controls
--------
  Gestures:
    - Pinch thumb + index on BOTH hands ........ open the filter portal
    - Thumb + pinky (or both index tips close) . next filter
  Keys:
    q  quit                s  screenshot          r  record video
    f  full-frame mode     t  toggle trails      h  help panel
    n / ->  next filter    p / <-  prev filter
"""
import math
import os
import time

import cv2
import mediapipe as mp
import numpy as np

# --------------------------------------------------------------------------
#  Config
# --------------------------------------------------------------------------
FILTERS = [
    "MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR",
    "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY",
    "RAINBOW", "KALEIDOSCOPE", "SWIRL", "BULGE", "VHS",
    "WATERCOLOR", "EMBOSS",
]

GESTURE_DISTANCE = 40      # px threshold for pinch gestures
PORTAL_SMOOTHING = 0.55    # EMA factor for portal corners (0=no smoothing)
TRAIL_LIFETIME = 0.9       # seconds a trail point lives
REC_FPS = 20.0
GALAXY_SIZE = (1920, 1080)

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
    print("Segmenter tidak tersedia (pastikan file selfie_segmenter.tflite ada)", e)

# --------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------
def make_galaxy_background():
    """Procedurally generate a 1920x1080 starfield."""
    w, h = GALAXY_SIZE
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (30, 10, 40)  # deep space purple
    for _ in range(800):
        sx, sy = np.random.randint(0, w), np.random.randint(0, h)
        bg[sy, sx] = (255, 255, 255)
    for _ in range(100):
        sx, sy = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(bg, (sx, sy), np.random.randint(2, 6),
                   (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)
    return bg


def hsv_palette(n=180, sat=255, val=255):
    """Precompute n HSV->BGR colors (fast hue cycling)."""
    px = np.zeros((1, n, 3), dtype=np.uint8)
    px[0, :, 0] = np.arange(n) % 180
    px[0, :, 1] = sat
    px[0, :, 2] = val
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0]
    return [tuple(int(c) for c in px) for px in bgr]


PALETTE = hsv_palette()

_remap_cache = {}


def _remap_maps(name, w, h):
    """Build (and cache) remap maps for distortion filters."""
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
        t2 = theta + 2.2 * factor
        mx, my = cx + r * np.cos(t2), cy + r * np.sin(t2)
    elif name == "BULGE":
        max_r = np.sqrt(cx * cx + cy * cy)
        factor = np.clip(1 - r / max_r, 0, 1) ** 2
        scale = 1 - 0.4 * factor
        mx, my = cx + dx * scale, cy + dy * scale
    elif name == "KALEIDOSCOPE":
        segments = 6
        wedge = 2 * np.pi / segments
        seg_idx = np.floor(theta / wedge)
        theta_folded = np.where(seg_idx % 2 == 0,
                                np.mod(theta, wedge),
                                wedge - np.mod(theta, wedge))
        mx, my = cx + r * np.cos(theta_folded), cy + r * np.sin(theta_folded)
    else:
        raise ValueError(name)

    maps = (mx.astype(np.float32), my.astype(np.float32))
    _remap_cache[key] = maps
    return maps


def _watercolor(roi):
    small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    res = cv2.bilateralFilter(small, 11, 60, 60)
    res = cv2.medianBlur(res, 5)
    return cv2.resize(res, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)


def _vhs(roi, frame_index):
    h, w = roi.shape[:2]
    out = roi.copy()
    # chromatic aberration
    shift = 6
    if w > shift:
        out[:, shift:, 2] = roi[:, :-shift, 2]
        out[:, :-shift, 0] = roi[:, shift:, 0]
    # scanlines
    out[::2, :] = (out[::2, :].astype(np.float32) * 0.65).astype(np.uint8)
    # noise
    noise = np.random.randint(0, 40, (h, w, 1), dtype=np.uint8)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # horizontal jitter band
    if np.random.rand() < 0.35 and h > 20:
        band = np.random.randint(0, h - 10)
        bh = np.random.randint(4, 18)
        out[band:band + bh] = np.roll(out[band:band + bh],
                                      np.random.randint(-20, 21), axis=1)
    # tracking timestamp
    cv2.putText(out, f"PLAY {frame_index // 30:06d}", (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _emboss(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    emb = cv2.filter2D(gray, cv2.CV_32F, kernel)
    out = np.clip(gray.astype(np.float32) + emb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _rainbow(roi, frame_index):
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_grad = (np.arange(w)[None, :] * 180.0 / max(w, 1)) + (frame_index * 1.2)
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_grad, 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_filter(roi, filter_name, frame_index=0, x=0, y=0,
                 mask_person=None, frame_galaxy=None):
    """Apply one filter to a region of interest. Returns a new BGR image."""
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    if filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    if filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(roi, kernel), 0, 255).astype(np.uint8)
    if filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]   # orange
        filtered[mask_c == 0] = [147, 20, 255]    # pink
        return filtered
    if filter_name == "PIXELATE":
        hr, wr = roi.shape[:2]
        if hr > 10 and wr > 10:
            small = cv2.resize(roi, (wr // 10, hr // 10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (wr, hr), interpolation=cv2.INTER_NEAREST)
    if filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    if filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cv2.bitwise_not(gray), (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    if filter_name == "GLITCH":
        wr = roi.shape[1]
        shift = max(5, wr // 20)
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
    return roi


def draw_hand_skeleton(img, hand_landmarks, w, h, frame_index):
    """Glowing skeleton + joints for every detected hand."""
    for lms in hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            hue = (frame_index * 2 + a * 30) % 180
            cv2.line(img, pts[a], pts[b], PALETTE[hue], 2)
        for idx in FINGERTIPS:
            hue = (frame_index * 2 + idx * 30) % 180
            cv2.circle(img, pts[idx], 5, PALETTE[hue], -1)
            cv2.circle(img, pts[idx], 9, PALETTE[hue], 1)


def draw_animated_border(img, pts, frame_index):
    """Rotating dashed glow border around the portal polygon."""
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        nd = max(2, int(dist // 18))
        offset = int(frame_index * 0.25) % nd
        for d in range(nd):
            k0 = (d + offset) / nd
            k1 = (d + 1 + offset) / nd
            p0 = (int(a[0] + (b[0] - a[0]) * k0), int(a[1] + (b[1] - a[1]) * k0))
            p1 = (int(a[0] + (b[0] - a[0]) * k1), int(a[1] + (b[1] - a[1]) * k1))
            hue = int((frame_index * 3 + i * 45) % 180)
            cv2.line(img, p0, p1, PALETTE[hue], 3)


def draw_carousel(img, active, frame_w, frame_index):
    """Bottom filter carousel; scrolls to keep the active filter visible."""
    chip_w, chip_h, gap = 108, 24, 6
    step = chip_w + gap
    max_visible = max(1, (frame_w - 20) // step)
    start = max(0, min(active - max_visible // 2, len(FILTERS) - max_visible))
    y0 = img.shape[0] - chip_h - 8
    for i in range(start, min(start + max_visible, len(FILTERS))):
        x0 = 10 + (i - start) * step
        if i == active:
            cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + chip_w + 2, y0 + chip_h + 2),
                          PALETTE[int(frame_index * 3) % 180], 2)
        cv2.rectangle(img, (x0, y0), (x0 + chip_w, y0 + chip_h),
                      (70, 70, 90) if i != active else (0, 170, 255), -1)
        color = (255, 255, 255) if i != active else (0, 0, 0)
        cv2.putText(img, FILTERS[i], (x0 + 6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1)


def draw_help(img):
    lines = [
        "KONTROL / CONTROLS:",
        "Cubit jempol+telunjuk 2 tangan .... buka portal",
        "Jempol+kelingking / 2 telunjuk dekat . filter berikutnya",
        "n / ->  filter berikutnya      p / <-  filter sebelumnya",
        "f  mode layar penuh (full-frame)  t  trail & kerangka tangan",
        "s  screenshot                    r  rekam video",
        "h  tutup panel ini               q  keluar",
    ]
    ox, oy = 10, img.shape[0] - 190
    cv2.rectangle(img, (ox, oy), (ox + 430, oy + 150), (0, 0, 0), -1)
    cv2.rectangle(img, (ox, oy), (ox + 430, oy + 150), (255, 255, 255), 1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (ox + 12, oy + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if i else (0, 200, 255), 1)


# --------------------------------------------------------------------------
#  App state
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat membuka kamera. Pastikan webcam tersedia.")
    exit(1)

galaxy_bg = make_galaxy_background()
print("Membuka Kamera... Tekan tombol 'q' untuk keluar, 'h' untuk bantuan.")

current_filter = 0
gesture_triggered = False
full_frame = False
trails_enabled = True
help_visible = True
recording = False
writer = None
rec_start = 0.0
frame_index = 0

trail_points = []          # (x, y, born_time)
smoothed_pts = None        # EMA-smoothed portal corners
fps = 0.0
prev_time = time.time()
toast = None               # (text, until)
flash_until = 0.0

os.makedirs("captures", exist_ok=True)
os.makedirs("recordings", exist_ok=True)


def show_toast(text, seconds=1.8):
    global toast
    toast = (text, time.time() + seconds)


# --------------------------------------------------------------------------
#  Main loop
# --------------------------------------------------------------------------
try:
    while True:
        success, img = cap.read()
        if not success:
            break

        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_index += 1

        frame_galaxy = galaxy_bg[:h, :w]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = time.time_ns() // 1_000_000

        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        filter_name = FILTERS[current_filter]

        mask_person = None
        if filter_name == "GALAXY" and segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h),
                                             interpolation=cv2.INTER_NEAREST)

        # -------------------------------------------------- gestures ------
        change_filter = False
        if results.hand_landmarks:
            if len(results.hand_landmarks) >= 2:
                idx0, idx1 = results.hand_landmarks[0][8], results.hand_landmarks[1][8]
                pt0 = (int(idx0.x * w), int(idx0.y * h))
                pt1 = (int(idx1.x * w), int(idx1.y * h))
                if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < GESTURE_DISTANCE:
                    change_filter = True
            for lms in results.hand_landmarks:
                thumb = lms[4]
                pinky = lms[20]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                px, py = int(pinky.x * w), int(pinky.y * h)
                if math.hypot(tx - px, ty - py) < GESTURE_DISTANCE:
                    change_filter = True

        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(FILTERS)
                show_toast(f"Filter: {FILTERS[current_filter]}", 1.2)
                gesture_triggered = True
        else:
            gesture_triggered = False

        # ------------------------------------------------ portal / frame --
        pts_portal = []
        if results.hand_landmarks:
            for lms in results.hand_landmarks:
                for i in (4, 8):
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
            # EMA smoothing of corners
            if smoothed_pts is None:
                smoothed_pts = raw_poly.copy()
            else:
                smoothed_pts = (1 - PORTAL_SMOOTHING) * smoothed_pts + PORTAL_SMOOTHING * raw_poly
            portal_poly = smoothed_pts.astype(np.int32)
        else:
            smoothed_pts = None

        # trails (even in full-frame mode)
        if trails_enabled:
            for lms in results.hand_landmarks:
                for i in FINGERTIPS:
                    lm = lms[i]
                    trail_points.append((int(lm.x * w), int(lm.y * h), now))
            cutoff = now - TRAIL_LIFETIME
            while trail_points and trail_points[0][2] < cutoff:
                trail_points.pop(0)
            if len(trail_points) > 400:
                del trail_points[:len(trail_points) - 400]

            if trail_points:
                overlay = np.zeros_like(img)
                for tx, ty, born in trail_points:
                    age = now - born
                    alpha = max(0.0, 1 - age / TRAIL_LIFETIME)
                    hue = int(born * 300) % 180
                    cv2.circle(overlay, (tx, ty), max(1, int(3 + 7 * alpha)),
                               PALETTE[hue], -1)
                glow = cv2.GaussianBlur(overlay, (0, 0), 7)
                img = cv2.addWeighted(img, 1.0, glow, 0.85, 0)

        # apply filter: full-frame or inside the portal
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
                img[y:y + bh, x:x + bw] = np.where(mask_3ch == 255, filtered_roi, roi)
                draw_animated_border(img, portal_poly, frame_index)
                cv2.putText(img, f"PORTAL: {filter_name}",
                            (portal_poly[0][0], portal_poly[0][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

        # ------------------------------------------------ overlays --------
        if trails_enabled and results.hand_landmarks:
            draw_hand_skeleton(img, results.hand_landmarks, w, h, frame_index)

        # FPS meter
        cv2.putText(img, f"{fps:4.1f} FPS", (w - 95, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # mode + active filter label
        mode_txt = "FULL-FRAME" if full_frame else "PORTAL"
        cv2.putText(img, f"Mode: {mode_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Filter Aktif: {filter_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # recording indicator
        if recording:
            cv2.circle(img, (w - 130, 58), 8, (0, 0, 255), -1)
            cv2.putText(img, f"REC {int(now - rec_start):02d}s", (w - 118, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        draw_carousel(img, current_filter, w, frame_index)

        if help_visible:
            draw_help(img)

        # toast message
        if toast and now < toast[1]:
            txt, _ = toast
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = (w - tw) // 2
            cv2.rectangle(img, (tx - 10, 66), (tx + tw + 10, 96), (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 255), 2)

        # screenshot flash
        if now < flash_until:
            alpha = max(0.0, (flash_until - now) / 0.25)
            img = cv2.addWeighted(img, 1 - alpha, np.full_like(img, 255), alpha, 0)

        if recording and writer is not None:
            writer.write(img)

        cv2.imshow("RETROLENS Pake Python", img)

        key = cv2.waitKeyEx(1)  # extended codes keep arrow keys distinct
        if key == -1:
            key = 0
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = time.strftime("captures/retrolens_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, img)
            flash_until = now + 0.25
            show_toast(f"Tersimpan: {fname}")
        elif key == ord('r'):
            if recording:
                writer.release()
                writer = None
                recording = False
                show_toast("Rekaman berhenti")
            else:
                fname = time.strftime("recordings/retrolens_%Y%m%d_%H%M%S.avi")
                writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"MJPG"),
                                         REC_FPS, (w, h))
                if writer.isOpened():
                    recording = True
                    rec_start = now
                    show_toast("Merekam video...")
                else:
                    writer = None
                    show_toast("Gagal membuat file video")
        elif key == ord('f'):
            full_frame = not full_frame
            show_toast("Mode: LAYAR PENUH" if full_frame else "Mode: PORTAL")
        elif key == ord('t'):
            trails_enabled = not trails_enabled
            show_toast("Trail: AKTIF" if trails_enabled else "Trail: NONAKTIF")
        elif key == ord('h'):
            help_visible = not help_visible
        elif key == ord('n') or key == 2555904:  # 'n' or RIGHT arrow
            current_filter = (current_filter + 1) % len(FILTERS)
            show_toast(f"Filter: {FILTERS[current_filter]}")
        elif key == ord('p') or key == 2424832:  # 'p' or LEFT arrow
            current_filter = (current_filter - 1) % len(FILTERS)
            show_toast(f"Filter: {FILTERS[current_filter]}")
finally:
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

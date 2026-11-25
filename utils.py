"""
utils.py
========
Shared helper functions used by every script in the project:
  * folder / file paths
  * loading the Haar Cascade classifier
  * frame pre-processing (resize, BGR->RGB, grayscale, blur)
  * attendance CSV handling (no duplicates per day)
  * drawing helpers (boxes, labels, header panel)
  * FPS counter and a simple MotionDetector (bonus topic)
"""

import csv
import importlib
import importlib.util
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. PATHS  (everything is relative to the folder that contains this file)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

CASCADE_FILE = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
ENCODINGS_FILE = os.path.join(MODELS_DIR, "encodings.pickle")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "Attendance.csv")

CSV_COLUMNS = ["Name", "Date", "Time", "Status"]


def ensure_folders():
    """Create every project folder if it does not exist yet."""
    for folder in (DATASET_DIR, MODELS_DIR, ATTENDANCE_DIR, SCREENSHOT_DIR):
        os.makedirs(folder, exist_ok=True)


# --------------------------------------------------------------------------
# 2. LOADING LIBRARIES / MODELS
# --------------------------------------------------------------------------
_face_lib = None


def load_face_lib():
    """
    Import the *real* `face_recognition` library.

    WHY THIS EXISTS: the lab requires a project file called face_recognition.py,
    but that name is identical to the library name. If we simply wrote
    `import face_recognition`, Python could import OUR file instead of the
    library. So we temporarily remove the project folder from sys.path,
    import the library, and then restore sys.path.
    """
    global _face_lib
    if _face_lib is not None:
        return _face_lib

    saved_path = list(sys.path)
    saved_module = sys.modules.pop("face_recognition", None)
    here = os.path.normcase(os.path.abspath(BASE_DIR))
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p or os.getcwd())) != here]
    try:
        _face_lib = importlib.import_module("face_recognition")
    except ImportError:
        sys.exit("ERROR: the 'face_recognition' library is not installed.\n"
                 "Run:  pip install -r requirements.txt  (see README.md for dlib help)")
    finally:
        sys.path = saved_path
        if saved_module is not None:
            sys.modules["face_recognition"] = saved_module
    return _face_lib


def load_local_module(filename, alias):
    """
    Load one of OUR project files (e.g. face_recognition.py) under a safe
    alias so it never clashes with the real library of the same name.
    """
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_cascade():
    """
    Load the Haar Cascade classifier used for face DETECTION.
    Looks in models/ first, then falls back to the copy bundled with OpenCV.
    """
    candidates = [CASCADE_FILE]
    if hasattr(cv2, "data"):
        candidates.append(os.path.join(cv2.data.haarcascades,
                                       "haarcascade_frontalface_default.xml"))
    for path in candidates:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    sys.exit("ERROR: haarcascade_frontalface_default.xml not found.\n"
             "Copy it into the models/ folder (see README.md).")


# --------------------------------------------------------------------------
# 3. IMAGE OPERATIONS
# --------------------------------------------------------------------------
def preprocess_frame(frame, scale=0.5):
    """
    Prepare one webcam frame for detection / recognition.

    Returns:
        small_bgr : resized colour frame
        gray      : grayscale + light blur + histogram equalisation (for Haar)
        rgb       : resized frame converted BGR -> RGB (for face_recognition)
    """
    # RESIZE: smaller frames = faster processing
    small_bgr = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # GRAYSCALE: Haar Cascades only need intensity information
    gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

    # BLUR: removes sensor noise that causes false detections
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # HISTOGRAM EQUALISATION: improves contrast in dark / bright rooms
    gray = cv2.equalizeHist(gray)

    # BGR -> RGB: OpenCV uses BGR but face_recognition (dlib) expects RGB
    rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    return small_bgr, gray, rgb


def is_blurry(image_bgr, threshold=60.0):
    """Variance of the Laplacian: low value = blurry image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# --------------------------------------------------------------------------
# 4. ATTENDANCE (CSV)  -  one record per person per day
# --------------------------------------------------------------------------
def init_attendance_file():
    """Create Attendance.csv with a header row if it does not exist."""
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)


class AttendanceManager:
    """Keeps today's attendance in memory and appends new rows to the CSV."""

    def __init__(self):
        init_attendance_file()
        self.today = None
        self.marked = {}          # name -> time string (today only)
        self._load_today()

    def _load_today(self):
        """Read the CSV with pandas and keep only today's rows."""
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.marked = {}
        try:
            df = pd.read_csv(ATTENDANCE_FILE, dtype=str)
            for _, row in df[df["Date"] == self.today].iterrows():
                self.marked[row["Name"]] = row["Time"]
        except (pd.errors.EmptyDataError, KeyError):
            pass

    def is_marked(self, name):
        self._check_new_day()
        return name in self.marked

    def mark(self, name):
        """
        Mark `name` present. Returns True only if this is the FIRST time today
        (so duplicates are silently ignored).
        """
        self._check_new_day()
        if name in self.marked:
            return False
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, self.today, time_str, "Present"])
        self.marked[name] = time_str
        return True

    def _check_new_day(self):
        """If the program runs past midnight, start a fresh day."""
        if datetime.now().strftime("%Y-%m-%d") != self.today:
            self._load_today()

    def present_count(self):
        return len(self.marked)


# --------------------------------------------------------------------------
# 5. DRAWING HELPERS  (Image Analysis & Transformation)
# --------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE, BLACK = (0, 200, 0), (0, 0, 220), (0, 215, 255), (255, 255, 255), (0, 0, 0)


def put_text_bg(img, text, org, scale=0.55, color=WHITE, bg=BLACK, thickness=1):
    """Draw text on a filled rectangle so it is readable on any background."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 8, y + base), bg, -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_face_box(img, box, name, status, confidence=None, color=GREEN):
    """Bounding box + name label + attendance status under the box."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label = name if confidence is None else f"{name} ({confidence:.0f}%)"
    put_text_bg(img, label, (x, max(y - 8, 20)), 0.6, WHITE, color)
    if status:
        put_text_bg(img, status, (x, y + h + 22), 0.55, BLACK, color)


def draw_header(img, registered, present, fps, motion_text):
    """Top information bar: title, date, time, counters, FPS."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 58), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)      # semi-transparent bar
    now = datetime.now()
    cv2.putText(img, "Smart Face Attendance", (10, 22), FONT, 0.65, RED, 2, cv2.LINE_AA)
    cv2.putText(img, now.strftime("Date: %Y-%m-%d   Time: %H:%M:%S"), (10, 48),
                FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
    right = f"Registered: {registered}  Present: {present}  FPS: {fps:.0f}"
    (w, _), _ = cv2.getTextSize(right, FONT, 0.5, 1)
    cv2.putText(img, right, (img.shape[1] - w - 10, 22), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    (w, _), _ = cv2.getTextSize(motion_text, FONT, 0.5, 1)
    cv2.putText(img, motion_text, (img.shape[1] - w - 10, 48), FONT, 0.5, GREEN, 1, cv2.LINE_AA)


def draw_present_list(img, marked):
    """Small list of today's attendance (latest 5) at the top-left."""
    y = 85
    put_text_bg(img, "Present today:", (10, y), 0.5, WHITE, (60, 60, 60))
    for name, t in list(marked.items())[-5:]:
        y += 24
        put_text_bg(img, f"{name}  {t}", (10, y), 0.48, WHITE, (0, 120, 0))


# --------------------------------------------------------------------------
# 6. FPS COUNTER
# --------------------------------------------------------------------------
class FPSCounter:
    """Smoothed frames-per-second value (exponential moving average)."""

    def __init__(self):
        self.last = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        instant = 1.0 / max(now - self.last, 1e-6)
        self.last = now
        self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
        return self.fps


# --------------------------------------------------------------------------
# 7. MOTION DETECTION (bonus)  -  frame differencing
# --------------------------------------------------------------------------
class MotionDetector:
    """
    Detects movement by comparing each frame with the previous one.

    Steps:  grayscale -> Gaussian blur -> absdiff -> threshold -> dilate
            -> find contours -> keep contours bigger than `min_area`.

    Face recognition is only run while motion is "active", which saves CPU
    when nobody is in front of the camera.
    """

    def __init__(self, min_area=500, hold_seconds=3.0):
        self.min_area = min_area
        self.hold_seconds = hold_seconds
        self.prev = None
        self.last_motion = time.time()

    def detect(self, gray):
        """Returns (is_active, list_of_motion_boxes)."""
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)       # ignore tiny noise
        if self.prev is None or self.prev.shape != blurred.shape:
            self.prev = blurred
            return True, []
        diff = cv2.absdiff(self.prev, blurred)               # what changed?
        self.prev = blurred
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)      # fill small holes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > self.min_area]
        if boxes:
            self.last_motion = time.time()
        return self.is_active(), boxes

    def keep_alive(self):
        """Call when a face is seen so a person sitting still is not ignored."""
        self.last_motion = time.time()

    def is_active(self):
        return (time.time() - self.last_motion) < self.hold_seconds

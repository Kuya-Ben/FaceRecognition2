"""
face_recognition.py
===================
Detection + recognition logic used by main.py.

  1. Haar Cascade  -> finds WHERE the faces are (bounding boxes)
  2. face_recognition (dlib) -> finds WHO the face is (128-D encoding match)

NOTE: this file has the same name as the library. We never import the library
by name here; utils.load_face_lib() does it safely.
"""

import math
import os
import pickle

import numpy as np

import utils


def detect_faces_haar(cascade, gray):
    """
    Haar Cascade face detection on a grayscale image.
    Returns a list of (x, y, w, h) boxes.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # image pyramid step (smaller = slower, more accurate)
        minNeighbors=5,       # higher = fewer false positives
        minSize=(40, 40),     # ignore tiny detections
    )
    return [tuple(int(v) for v in f) for f in faces]


def distance_to_confidence(distance, threshold=0.5):
    """Convert a dlib face distance into a 0..1 'confidence' value."""
    rng = 1.0 - threshold
    linear = (1.0 - distance) / (rng * 2.0)
    if distance > threshold:
        return max(0.0, linear)
    return min(1.0, linear + ((1.0 - linear) * math.pow((linear - 0.5) * 2, 0.2)))


class FaceRecognizer:
    """Loads the trained encodings and identifies faces."""

    def __init__(self, tolerance=0.5):
        self.lib = utils.load_face_lib()
        self.tolerance = tolerance          # lower = stricter matching
        self.names = []
        self.encodings = []
        self.load()

    def load(self):
        """Read models/encodings.pickle created by train_faces.py."""
        if not os.path.exists(utils.ENCODINGS_FILE):
            raise SystemExit("ERROR: no trained model found.\n"
                             "Run capture_faces.py and then train_faces.py first.")
        with open(utils.ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.encodings = data["encodings"]

    @property
    def registered_people(self):
        """Unique registered names."""
        return sorted(set(self.names))

    def recognize(self, rgb_small, boxes):
        """
        For every Haar box, compute the face encoding and compare it with all
        known encodings.  Returns a list of (name, confidence_percent).
        """
        if not boxes:
            return []
        # Haar gives (x, y, w, h); face_recognition wants (top, right, bottom, left)
        locations = [(y, x + w, y + h, x) for (x, y, w, h) in boxes]
        encodings = self.lib.face_encodings(rgb_small, known_face_locations=locations)

        results = []
        for enc in encodings:
            if not self.encodings:
                results.append(("Unknown", 0.0))
                continue
            distances = self.lib.face_distance(self.encodings, enc)
            best = int(np.argmin(distances))                 # closest known face
            if distances[best] <= self.tolerance:
                conf = distance_to_confidence(distances[best], self.tolerance) * 100
                results.append((self.names[best], conf))
            else:
                results.append(("Unknown", 0.0))
        return results


if __name__ == "__main__":
    rec = FaceRecognizer()
    print("Registered people:", rec.registered_people)
    print("Total encodings  :", len(rec.encodings))

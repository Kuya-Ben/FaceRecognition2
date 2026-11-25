"""
train_faces.py  -  STEP 2: "train" the recognizer
=================================================
Reads every image in dataset/<person>/ , converts it into a 128-number face
encoding with the face_recognition library (dlib deep-learning model) and
saves all encodings + names in models/encodings.pickle .

(There is no long training loop: the deep model is pre-trained, we only
 extract and store the encodings of YOUR registered people.)

Usage:   python train_faces.py
"""

import os
import pickle

import cv2

import utils

face_recognition = utils.load_face_lib()       # the real library (see utils.py)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    utils.ensure_folders()
    names, encodings = [], []

    people = sorted(d for d in os.listdir(utils.DATASET_DIR)
                    if os.path.isdir(os.path.join(utils.DATASET_DIR, d)))
    if not people:
        raise SystemExit("dataset/ is empty. Run capture_faces.py first.")

    for person in people:
        folder = os.path.join(utils.DATASET_DIR, person)
        display_name = person.replace("_", " ")
        count = 0
        for file in sorted(os.listdir(folder)):
            if not file.lower().endswith(IMAGE_EXT):
                continue
            img = cv2.imread(os.path.join(folder, file))
            if img is None:
                continue

            # IMAGE OPERATIONS: shrink big images, then BGR -> RGB
            if img.shape[1] > 800:
                scale = 800 / img.shape[1]
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Find the face; our saved crops are already faces, so if the HOG
            # detector fails we treat the whole image as the face.
            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                h, w = rgb.shape[:2]
                locations = [(0, w, h, 0)]
            # keep only the biggest face in the image
            locations = [max(locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))]

            enc = face_recognition.face_encodings(rgb, known_face_locations=locations)
            if enc:
                encodings.append(enc[0])
                names.append(display_name)
                count += 1

        print(f"  {display_name:<20} {count} encodings")
        if count < 5:
            print(f"    WARNING: only {count} usable images - capture more for better accuracy")

    if not encodings:
        raise SystemExit("No faces could be encoded.")

    with open(utils.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)
    print(f"\nSaved {len(encodings)} encodings for {len(set(names))} people -> {utils.ENCODINGS_FILE}")


if __name__ == "__main__":
    main()

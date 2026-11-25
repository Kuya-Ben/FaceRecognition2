"""
capture_faces.py  -  STEP 1: register a new person
==================================================
Usage:
    python capture_faces.py --name "Ben Jasper Solitario"
    python capture_faces.py --name Maria --count 30 --camera 0

The webcam opens, a Haar Cascade finds your face and a cropped photo is saved
automatically every half second into dataset/<Name>/ .
Tips: move your head slightly (left/right/up/down), change expression, and
keep good lighting.  Press  q  to stop early.
"""

import argparse
import os
import time

import cv2

import utils

# Load our own detection helpers under a safe alias (see utils.load_local_module)
recog = utils.load_local_module("face_recognition.py", "fa_recognition")


def main():
    parser = argparse.ArgumentParser(description="Capture face images for one person")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--count", type=int, default=25, help="Number of images to save")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between saves")
    args = parser.parse_args()

    utils.ensure_folders()
    # Folder name: spaces -> underscores (shown back as spaces in the app)
    folder_name = args.name.strip().replace(" ", "_")
    person_dir = os.path.join(utils.DATASET_DIR, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    saved = len(os.listdir(person_dir))          # continue numbering if re-run

    cascade = utils.load_cascade()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("ERROR: cannot open webcam. Try --camera 1")

    print(f"Capturing {args.count} images for '{args.name}'. Press 'q' to stop.")
    taken, last_save = 0, 0.0

    while taken < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                       # mirror view
        scale = 0.5
        small, gray, _ = utils.preprocess_frame(frame, scale)
        boxes = recog.detect_faces_haar(cascade, gray)

        message = "No face detected" if not boxes else "Face OK - hold still / move head slowly"
        if len(boxes) > 1:
            message = "Only ONE person please"

        if len(boxes) == 1:
            x, y, w, h = [int(v / scale) for v in boxes[0]]   # back to full-size coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), utils.GREEN, 2)

            # Save a cropped face (with 20% margin) every `interval` seconds
            if time.time() - last_save > args.interval:
                m = int(0.2 * w)
                y0, y1 = max(y - m, 0), min(y + h + m, frame.shape[0])
                x0, x1 = max(x - m, 0), min(x + w + m, frame.shape[1])
                crop = frame[y0:y1, x0:x1]
                # Skip blurry images (Laplacian variance) - they hurt accuracy
                if crop.size and not utils.is_blurry(crop):
                    crop = cv2.resize(crop, (300, 300))
                    saved += 1
                    taken += 1
                    cv2.imwrite(os.path.join(person_dir, f"{folder_name}_{saved:03d}.jpg"), crop)
                    last_save = time.time()
                    print(f"  saved {taken}/{args.count}")

        utils.put_text_bg(frame, f"{args.name}: {taken}/{args.count}", (10, 30), 0.7)
        utils.put_text_bg(frame, message, (10, 62), 0.55, utils.BLACK, utils.YELLOW)
        cv2.imshow("Capture Faces (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {taken} new images stored in {person_dir}")
    print("Next step:  python train_faces.py")


if __name__ == "__main__":
    main()

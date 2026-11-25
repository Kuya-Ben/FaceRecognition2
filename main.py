"""
main.py  -  STEP 3: run the live attendance system
==================================================
Usage:
    python main.py
    python main.py --camera 1 --tolerance 0.45 --no-motion

Keys:   q = quit    s = save screenshot    m = motion gate on/off    r = reload model
"""

import argparse
import os
from datetime import datetime

import cv2

import utils

# Our own recognition module (loaded under an alias, see utils.py)
recog = utils.load_local_module("face_recognition.py", "fa_recognition")

SCALE = 0.5            # processing scale (0.5 = half resolution)
CONFIRM_FRAMES = 3     # same person must be seen N times in a row before marking


def main():
    parser = argparse.ArgumentParser(description="Smart Face Attendance System")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="match strictness: lower = stricter (default 0.5)")
    parser.add_argument("--skip", type=int, default=2,
                        help="run recognition every N frames (higher = faster)")
    parser.add_argument("--no-motion", action="store_true",
                        help="disable the motion-detection gate")
    args = parser.parse_args()

    utils.ensure_folders()
    cascade = utils.load_cascade()                       # Haar Cascade
    recognizer = recog.FaceRecognizer(args.tolerance)    # trained encodings
    attendance = utils.AttendanceManager()               # CSV + duplicate check
    motion = utils.MotionDetector()
    fps_counter = utils.FPSCounter()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("ERROR: cannot open webcam. Try --camera 1")

    use_motion = not args.no_motion
    streak = {}               # name -> consecutive recognitions
    last_results = []         # [(box, name, confidence)] reused on skipped frames
    toast, toast_until = "", 0.0
    frame_no = 0
    print("System running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame not received - exiting.")
            break
        frame = cv2.flip(frame, 1)                       # mirror view feels natural
        frame_no += 1
        small, gray, rgb = utils.preprocess_frame(frame, SCALE)

        # ---- (Bonus) MOTION DETECTION -------------------------------------
        if use_motion:
            active, motion_boxes = motion.detect(gray)
            for (mx, my, mw, mh) in motion_boxes:        # thin yellow motion boxes
                p1 = (int(mx / SCALE), int(my / SCALE))
                p2 = (int((mx + mw) / SCALE), int((my + mh) / SCALE))
                cv2.rectangle(frame, p1, p2, utils.YELLOW, 1)
            motion_text = "Motion: ACTIVE" if active else "Motion: IDLE (waiting)"
        else:
            active, motion_text = True, "Motion gate: OFF"

        # ---- DETECT (Haar) + RECOGNISE (face_recognition) ------------------
        if not active:
            last_results, streak = [], {}
        elif frame_no % args.skip == 0:
            boxes = recog.detect_faces_haar(cascade, gray)
            if boxes:
                motion.keep_alive()
            people = recognizer.recognize(rgb, boxes)
            # scale boxes back to the full-size frame
            last_results = [(tuple(int(v / SCALE) for v in b), n, c)
                            for b, (n, c) in zip(boxes, people)]

            # ---- ATTENDANCE LOGIC ------------------------------------------
            new_streak = {}
            for _, name, _ in last_results:
                if name == "Unknown":
                    continue
                new_streak[name] = streak.get(name, 0) + 1
                if new_streak[name] >= CONFIRM_FRAMES and attendance.mark(name):
                    toast = f"Attendance marked: {name}"
                    toast_until = datetime.now().timestamp() + 3
                    print(f"[{datetime.now():%H:%M:%S}] {name} -> Present")
            streak = new_streak

        # ---- DRAW RESULTS --------------------------------------------------
        for box, name, conf in last_results:
            if name == "Unknown":
                utils.draw_face_box(frame, box, "Unknown", "Not registered", None, utils.RED)
            elif attendance.is_marked(name):
                utils.draw_face_box(frame, box, name,
                                    f"Present  {attendance.marked[name]}", conf, utils.GREEN)
            else:
                utils.draw_face_box(frame, box, name, "Verifying...", conf, utils.YELLOW)

        fps = fps_counter.tick()
        utils.draw_header(frame, len(recognizer.registered_people),
                          attendance.present_count(), fps, motion_text)
        utils.draw_present_list(frame, attendance.marked)
        if datetime.now().timestamp() < toast_until:
            utils.put_text_bg(frame, toast, (frame.shape[1] // 2 - 130, frame.shape[0] - 30),
                              0.7, utils.WHITE, (0, 140, 0))
        utils.put_text_bg(frame, "q:quit  s:screenshot  m:motion  r:reload",
                          (10, frame.shape[0] - 8), 0.45, utils.WHITE, (60, 60, 60))

        cv2.imshow("Smart Face Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = os.path.join(utils.SCREENSHOT_DIR,
                                f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(path, frame)
            print("Screenshot saved:", path)
        elif key == ord("m"):
            use_motion = not use_motion
        elif key == ord("r"):
            recognizer.load()
            print("Model reloaded.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session finished. {attendance.present_count()} people present today.")


if __name__ == "__main__":
    main()

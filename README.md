# Smart Face Attendance Monitoring System (OpenCV)
install    **python -m pip install opencv-python numpy pandas**
download the **haarcascade**


train face 
**python -m pip install setuptools**
**python -m pip install dlib-bin**
**python -m pip install face_recognition --no-deps**
**python -m pip install face_recognition_models Click Pillow**

check **python -c "import dlib, face_recognition; print('OK')"**
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip install "opencv-python<5"
## 1. Project Overview
A live-webcam attendance system. It **detects** faces with a Haar Cascade, **recognizes**
registered people with the `face_recognition` library, and writes **Name, Date, Time, Status**
to `attendance/Attendance.csv` – **once per person per day**.
A bonus **motion-detection gate** only runs the (heavier) recognition when something moves.

## 2. Features
- Live webcam feed with bounding boxes, name, confidence %, and attendance status
- Register unlimited people (one folder each)
- Duplicate-proof: each person marked once per day (also across program restarts)
- Header overlay: date, time, registered count, present count, FPS
- "Present today" list on screen and "Attendance marked" toast
- Motion detection (frame differencing) – toggle with `m`
- Screenshot key (`s`), model reload (`r`)
- Blur-checking while capturing, confirmation over 3 frames to avoid false marks

## 3. CV Topics Used
| Topic | Where |
|---|---|
| Haar Cascade Classifier | `face_recognition.py -> detect_faces_haar()` |
| Image Operations (resize, BGR->RGB, grayscale, blur, equalizeHist) | `utils.py -> preprocess_frame()` |
| Image Analysis & Transformation (boxes, labels, overlays, date/time) | `utils.py -> draw_*()` |
| Motion Detection (bonus) | `utils.py -> MotionDetector` |

## 4. Installation
Python 3.8 – 3.12 recommended.
```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
**`dlib` needs CMake and a C++ compiler:**
- Windows: install *Visual Studio Build Tools* ("Desktop development with C++") and `pip install cmake`
  (or use a pre-built dlib wheel matching your Python version)
- Ubuntu: `sudo apt install build-essential cmake`
- macOS: `brew install cmake`

## 5. Required Libraries
`opencv-python`, `numpy`, `pandas`, `dlib`, `face_recognition`, `setuptools`
(standard library: `datetime`, `os`, `csv`, `pickle`, `argparse`)

### Haar Cascade file
`models/haarcascade_frontalface_default.xml` – if missing, download it from the OpenCV repo
(`data/haarcascades/`). The code also falls back to the copy bundled in `cv2.data.haarcascades`
that `opencv-python` installs.

## 6. How to Register Faces
```bash
python capture_faces.py --name "John Smith"
python capture_faces.py --name "Maria Lopez" --count 30
```
Look at the camera, slowly turn your head, change expression. Images are saved in
`dataset/John_Smith/`. Repeat for each person. Use good, even lighting; one person in view.

## 7. How to Train
```bash
python train_faces.py
```
Creates `models/encodings.pickle`. **Re-run it whenever you add or remove people.**

## 8. How to Run
```bash
python main.py
python main.py --camera 1 --tolerance 0.45 --no-motion
```
| Key | Action |
|---|---|
| q | quit |
| s | save screenshot to `screenshots/` |
| m | motion gate ON/OFF |
| r | reload trained model |

Options: `--tolerance` (lower = stricter, default 0.5), `--skip` (recognize every N frames, default 2).

## 9. Folder Structure
```
FaceAttendance/
├── dataset/            <- one sub-folder of face photos per person
├── models/             <- Haar XML + encodings.pickle (created by training)
├── attendance/         <- Attendance.csv
├── screenshots/        <- saved with the 's' key
├── main.py             <- live attendance app
├── capture_faces.py    <- register faces
├── train_faces.py      <- build encodings
├── face_recognition.py <- Haar detection + recognition class
├── utils.py            <- helpers, CSV, drawing, motion, FPS
├── requirements.txt
├── README.md
└── documentation.pdf
```
> **Important:** `face_recognition.py` has the same name as the library. The code handles this
> through `utils.load_face_lib()` and `utils.load_local_module()`. Don't `import face_recognition`
> directly in your own files in this folder.

## 10. Sample Outputs
**Console**
```
System running. Press 'q' to quit.
[09:02:14] John Smith -> Present
[09:03:41] Maria Lopez -> Present
Session finished. 2 people present today.
```
**attendance/Attendance.csv**
```
Name,Date,Time,Status
John Smith,2026-11-24,09:02:14,Present
Maria Lopez,2026-11-24,09:03:41,Present
```
**Window:** green box + `John Smith (91%)` + `Present 09:02:14`; red box for `Unknown`; header with
date/time, `Registered: 2  Present: 2  FPS: 18`.

## 11. Troubleshooting
| Problem | Fix |
|---|---|
| Camera won't open | try `--camera 1`; close Zoom/Teams |
| Person shown as Unknown | capture more images (30+), better light, raise `--tolerance` to 0.55 |
| Wrong person recognized | lower `--tolerance` to 0.45, recapture, retrain |
| Low FPS | raise `--skip` to 3, keep `--no-motion` off |
| `dlib` install fails | install CMake + C++ build tools (section 4) |

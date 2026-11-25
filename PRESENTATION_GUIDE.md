# 15-Minute Presentation Plan
| Min | Content |
|---|---|
| 0-1 | Title, problem: manual roll-call is slow and proxy attendance is easy |
| 1-3 | Objectives + architecture diagram: Webcam -> preprocess -> motion gate -> Haar detect -> encode/match -> CSV |
| 3-6 | CV techniques: Haar (integral image, AdaBoost, cascade stages), image ops, drawing/overlays, motion differencing - show the code for each (< 10 lines each) |
| 6-11 | LIVE DEMO: (1) `capture_faces.py` register a classmate (2) `train_faces.py` (3) `main.py`: green box, "Present", then show a second pass does NOT add a row (4) show unknown person = red box (5) show motion IDLE/ACTIVE with `m` (6) open Attendance.csv |
| 11-13 | Accuracy, limitations, future work |
| 13-15 | Q&A |

**Before the demo:** train beforehand with 2-3 people, test lighting, keep a recorded video as backup,
delete today's rows from the CSV so the demo shows a fresh mark, take screenshots with `s`.

# Q&A
1. **Why Haar AND face_recognition?** Haar is a fast detector (where is a face); face_recognition uses a deep network to turn a face into a 128-D vector (who is it). Using Haar to supply boxes avoids running dlib's slower detector.
2. **How do you prevent duplicates?** `AttendanceManager` loads today's rows from the CSV into a dictionary; `mark()` returns False if the name exists. Because it reads the CSV at start-up, duplicates are prevented after restarts too, and it resets at midnight.
3. **What does `tolerance` mean?** The maximum Euclidean distance between two 128-D encodings to count as the same person. Lower = stricter (fewer false accepts, more Unknowns). 0.6 is the library default; we use 0.5.
4. **Can it be fooled by a photo?** Yes - there is no liveness detection. Fixes: blink detection via facial landmarks (MediaPipe), depth camera, or asking for a random head-turn.
5. **How does motion detection help?** Frame differencing (absdiff -> threshold -> contours) shows whether anything moves. Recognition is skipped when idle, saving CPU. A face in view keeps it active so a seated person isn't ignored.

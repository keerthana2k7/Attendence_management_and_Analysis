import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition
from ultralytics import YOLO

# -----------------------
# Load YOLO face model
# -----------------------
model = YOLO("yolov8n-face.pt")

# -----------------------
# Load dataset
# -----------------------
known_encodings = []
known_names = []

dataset_path = "dataset"

for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    for img in os.listdir(student_folder):
        path = os.path.join(student_folder, img)

        image0 = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image0)

        if len(encs) > 0:
            known_encodings.append(encs[0])
            known_names.append(student_name)

print("All Students Encoded Successfully")

# -----------------------
# Load group image
# -----------------------
image_path = "group.jpg"
image = cv2.imread(image_path)

if image is None:
    print("group.jpg not found")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------
# YOLO face detection
# -----------------------
results = model(rgb)

face_locations = []

h, w, _ = rgb.shape

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes.xyxy:

        x1, y1, x2, y2 = map(int, box)

        # small padding for better crop
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        top = y1
        right = x2
        bottom = y2
        left = x1

        face_locations.append((top, right, bottom, left))

# -----------------------
# Face encodings from YOLO boxes
# -----------------------
face_encodings = face_recognition.face_encodings(
    rgb, face_locations
)

marked_students = set()

# create csv if not exists
if not os.path.exists("attendance.csv"):
    pd.DataFrame(
        columns=["Name", "Time", "Status"]
    ).to_csv("attendance.csv", index=False)

# -----------------------
# Recognition
# -----------------------
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    name = "Unknown"

    if len(known_encodings) > 0:

        distances = face_recognition.face_distance(
            known_encodings, face_encoding
        )

        best_index = np.argmin(distances)

        # tighter threshold to avoid wrong person
        if distances[best_index] < 0.47:
            name = known_names[best_index]

            if name not in marked_students:
                now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                pd.DataFrame(
                    [[name, now, "PRESENT"]],
                    columns=["Name", "Time", "Status"]
                ).to_csv(
                    "attendance.csv",
                    mode="a",
                    header=False,
                    index=False
                )

                marked_students.add(name)
                print(name, "Marked Present")

    # draw box
    cv2.rectangle(image, (left, top), (right, bottom),
                  (0, 255, 0), 2)

    cv2.putText(image, name, (left, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

# -----------------------
# Show result
# -----------------------
cv2.imshow("YOLO Face Attendance", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

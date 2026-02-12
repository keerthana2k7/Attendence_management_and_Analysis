import face_recognition
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------
# Load Dataset (All Students)
# -----------------------

known_encodings = []
known_names = []

dataset_path = "dataset"

for student_name in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.isdir(student_folder):
        continue

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(student_name)

print("All Students Encoded Successfully")

# -----------------------
# Load Group Image
# -----------------------

group_image = cv2.imread("group.jpg")

if group_image is None:
    print("group.jpg not found")
    exit()

rgb_group = cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(
    rgb_group,
    number_of_times_to_upsample=2,
    model="hog"
)

face_encodings = face_recognition.face_encodings(
    rgb_group, face_locations
)

marked_students = set()

# Create CSV with header if not exists
if not os.path.exists("attendance.csv"):
    df = pd.DataFrame(columns=["Name", "Time", "Status"])
    df.to_csv("attendance.csv", index=False)

# -----------------------
# Compare & Mark Attendance
# -----------------------

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    name = "Unknown"

    face_distances = face_recognition.face_distance(
        known_encodings, face_encoding
    )

    if len(face_distances) > 0:

        best_match_index = np.argmin(face_distances)

        # strict threshold
        if face_distances[best_match_index] < 0.48:
            name = known_names[best_match_index]

            if name not in marked_students:

                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

                df = pd.DataFrame(
                    [[name, dt_string, "PRESENT"]],
                    columns=["Name", "Time", "Status"]
                )

                df.to_csv("attendance.csv", mode="a",
                          header=False, index=False)

                marked_students.add(name)
                print(f"{name} Marked Present")

    cv2.rectangle(group_image, (left, top), (right, bottom),
                  (0, 255, 0), 2)

    cv2.putText(group_image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2)

# -----------------------
# Show Result
# -----------------------
# -----------------------
# Mark absent students
# -----------------------

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

for student in set(known_names):

    if student not in marked_students:

        df = pd.DataFrame(
            [[student, dt_string, "ABSENT"]],
            columns=["Name", "Time", "Status"]
        )

        df.to_csv("attendance.csv", mode="a",
                  header=False, index=False)

        print(f"{student} Marked Absent")


cv2.imshow("Group Attendance Result", group_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import os
from ultralytics import YOLO

# -----------------------
# Student name
# -----------------------
name = "Keerthana"   # change each time

dataset_path = f"dataset/{name}"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# -----------------------
# Load YOLO face model
# -----------------------
model = YOLO("yolov8n-face.pt")

cap = cv2.VideoCapture(0)

count = 0

print("Press 's' to save face, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    face_boxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

    cv2.imshow("Face Capture (YOLO)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(face_boxes) > 0:

        # take the largest face
        face_boxes = sorted(
            face_boxes,
            key=lambda b: (b[2]-b[0])*(b[3]-b[1]),
            reverse=True
        )

        x1, y1, x2, y2 = face_boxes[0]

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size != 0:
            img_path = f"{dataset_path}/{count}.jpg"
            cv2.imwrite(img_path, face_crop)
            print(f"Saved {img_path}")
            count += 1

    if key == ord('q') or count == 30:
        break

cap.release()
cv2.destroyAllWindows()

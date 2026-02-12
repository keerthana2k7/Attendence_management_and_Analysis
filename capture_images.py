import cv2
import os

name = "Keerthana"
dataset_path = f"dataset/{name}"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Press S to Save Image", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_path = f"{dataset_path}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved {count}.jpg")
        count += 1

    if key == ord('q') or count == 30:
        break

cap.release()
cv2.destroyAllWindows()

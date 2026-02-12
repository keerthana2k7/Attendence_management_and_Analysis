import cv2
from ultralytics import YOLO

# Load YOLO face detection model
model = YOLO("yolov8n.pt")


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame, conf=0.5)

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("Face Detection - YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

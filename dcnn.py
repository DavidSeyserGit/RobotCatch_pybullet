import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Download model from github
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on frame
    results = model(frame)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Only process if the detected object is a person
            if class_name == 'person':
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence
                conf = float(box.conf[0])
                
                # Calculate center coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Person {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Display coordinates on the frame
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(frame, coord_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Print coordinates to console
                print(f"Person detected at coordinates: ({center_x}, {center_y})")

    # Display the frame
    cv2.imshow('YOLOv8 Person Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
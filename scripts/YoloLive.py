import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("../Models/best.pt")  # Pretrained YOLO model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Predict the frame using the model

    # Process the results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        masks = result.masks  # Segmentation masks
        keypoints = result.keypoints  # Keypoints (pose)
        obb = result.obb  # Oriented bounding boxes (OBB)

        # If boxes are present, plot them on the frame
        if boxes:
            for i, box in enumerate(boxes.xyxy):  # Iterate through the boxes
                x1, y1, x2, y2 = map(int, box)  # Convert to integers for drawing
                label_idx = int(boxes.cls[i])  # Get the class index for this box
                label = result.names[label_idx]  # Get the corresponding label

                # Get the confidence score for the current box
                conf_score = boxes.conf[i] if boxes.conf is not None else 0.0

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Display the label and confidence score
                label_text = f'{label} {conf_score:.2f}'  # Label and confidence score
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, label_text, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

        result.plot()  # This function automatically plots results

    cv2.imshow("frame", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

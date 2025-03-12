from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # pretrained YOLO11n model

images = ["rock-paper-scissors.v1i.yolov11/valid/images/0098_png.rf.0d15a8c9eb7df0e4231c66b129f1635a.jpg"]
results = model(images)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
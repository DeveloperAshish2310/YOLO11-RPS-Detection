from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
# Anotating.v1i.yolov11
# Train the model
results = model.train(data="rock-paper-scissors.v1i.yolov11/data.yaml", epochs=200, imgsz=640)

# saving the model
save_path = "Model-trained/model-v11.pt"  # Define the path where the model will be saved

# Create the directory if it does not exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model.save(save_path)  # Save the trained model to a local file

# YOLO 11 - Rock Paper Scissors Detection

## Overview

This project leverages YOLO 11 to detect and classify hand gestures (Rock, Paper, Scissors) in real-time. It supports multiple hands simultaneously and works with live video feeds.

## Project Structure

- **models/**
  - `best_model.pt` (Best-trained YOLO 11 model)
  - `last_model.pt` (Last-trained YOLO 11 model)
- **scripts/**
  - `YoloTrain.py` - Train the YOLO 11 model on Rock Paper Scissors dataset.
  - `YoloTest.py` - Test the trained YOLO model on a single image.
  - `YoloLive.py` - Run the trained YOLO model in real-time using a webcam.

## Installation

Ensure you have the necessary dependencies installed:

```bash
pip install ultralytics opencv-python torch torchvision
```

## Training the Model

To train the model using your dataset, run:

```bash
python scripts/YoloTrain.py
```

## Testing on an Image

To test the trained model on a single image:

Change Images Path in Python File
```python
images = ["<image_path>"]
```
Run Python File
```bash
python scripts/YoloTest.py
```

## Running in Real-Time

To detect hand gestures in real-time:

```bash
python scripts/YoloLive.py
```

## Model Performance

The best model (`best_model.pt`) provides the highest accuracy, while `last_model.pt` is the most recent checkpoint.

## Contributions

Feel free to contribute by improving the dataset, optimizing training, or adding new gesture recognition features.

## License

This project is open-source and free to use under the Apache License.

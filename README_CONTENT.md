# AI Drone Racing Gate Detection using YOLOv8

## Project Overview

This project trains a **YOLOv8 object detection model** to detect **drone racing gates** using a custom dataset exported from Roboflow.
The model is trained in **Google Colab** and can later be used for **autonomous drone navigation and computer vision pipelines**.

The dataset contains labeled images of drone racing gates and is divided into:

* Train dataset
* Validation dataset
* Test dataset

The trained model can be used for:

* Real-time drone gate detection
* Autonomous drone racing
* Robotics perception pipelines
* Simulation environments like **AirSim**

---

# Dataset

Dataset Source:
https://app.roboflow.com/javeds-workspace/airsim-drone-racing-lab-gates-ua7be

Dataset Structure (YOLO Format)

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

Each label file contains:

```
class_id x_center y_center width height
```

Coordinates are normalized between **0 and 1**.

---

# Environment

The project uses **Google Colab GPU environment**.

Required libraries:

* Python
* Ultralytics YOLOv8
* Roboflow

---

# Step 1: Open Google Colab

Open Google Colab and enable GPU:

Runtime → Change Runtime Type → GPU

---

# Step 2: Install Dependencies

```python
!pip install ultralytics
!pip install roboflow
```

---

# Step 3: Import Libraries

```python
from ultralytics import YOLO
from roboflow import Roboflow
```

---

# Step 4: Download Dataset from Roboflow

Replace `YOUR_API_KEY` with your Roboflow API key.

```python
rf = Roboflow(api_key="YOUR_API_KEY")

project = rf.workspace("javeds-workspace").project("airsim-drone-racing-lab-gates-ua7be")

dataset = project.version(1).download("yolov8")
```

Dataset will be downloaded into:

```
/content/airsim-drone-racing-lab-gates-ua7be-1/
```

---

# Step 5: Load YOLOv8 Model

We use the **YOLOv8 Nano model** for fast training.

```python
model = YOLO("yolov8n.pt")
```

Other available models:

```
yolov8n.pt
yolov8s.pt
yolov8m.pt
yolov8l.pt
yolov8x.pt
```

---

# Step 6: Train the Model

```python
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

Training results will be saved in:

```
runs/detect/train/
```

---

# Step 7: Evaluation Metrics

During training YOLOv8 automatically computes:

Example Output:

```
A YOLOv8n model was trained for 25 epochs using the prepared dataset, achieving strong performance metrics:
    *   mAP50:** 0.96 (on training set) and 0.959 (on validation set).
    *   mAP50-95:** 0.825 (on training set) and 0.825 (on validation set).
    *   Precision:** 0.853 (on validation set).
    *   Recall:** 0.965 (on validation set).
```

These metrics evaluate how accurately the model detects drone racing gates.

---

# Step 8: Test the Model

Run inference on a test image.

```python
model = YOLO("runs/detect/train/weights/best.pt") # expert model

results = model.predict(
    source="/content/test_image.jpg",
    conf=0.25,
    save=True
)
```

Output will be saved in:

```
runs/detect/predict/
```

---

# Step 9: Download the Trained Model

The best trained model is located at:

```
runs/detect/train/weights/best.pt
```

Download it using:

```python
from google.colab import files
files.download('runs/detect/train/weights/best.pt')
```

---

# Example Detection

Input:

Drone racing image

Output:

Detected gate with bounding box and confidence score.

---

# Applications

This model can be integrated into:

Drone Vision Pipeline

```
Drone Camera
      ↓
Gate Detection (YOLOv8)
      ↓
Pose Estimation (PnP)
      ↓
State Estimation (Kalman Filter)
      ↓
Trajectory Planning
      ↓
Flight Controller
```

---

# Future Improvements

* Train using larger YOLOv8 models
* Increase dataset size
* Implement real-time detection
* Integrate with ROS2
* Deploy on NVIDIA Jetson or drone onboard computer

---



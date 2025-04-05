# 🧠 Human and Dog Breed Detection System 🐶

This project uses **YOLOv12n** to perform real-time detection of humans and dog breeds. When a human is detected, their face is extracted and checked against a local `.pkl` face database to identify if the person is missing. For dogs, the system detects and identifies their breed and checks if this type of dog has been reported missing.

## 📸 Features

- 🔍 **YOLOv12n** for fast, lightweight object detection
- 🧑‍🤝‍🧑 **Human detection** and **face recognition** using a prebuilt face embedding database
- 🐕 **Dog detection** with optional breed classification
- ⚠️ **Alerts** when a missing person is detected
- 🗃️ Pickle-based face recognition system (efficient and lightweight)
- 🖼️ Real-time or batch image/video processing


## Sources:
- Dog Breed Detection Database and base model: https://www.kaggle.com/code/chg0901/dog-breed-detection
- YOLOv12n: https://github.com/sunsmarterjie/yolov12
- Face recognition: https://www.youtube.com/watch?v=iBomaK2ARyI

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (compatible with YOLOv12n)
- OpenCV
- `face_recognition` or `dlib` (for facial encoding)
- `pickle` (standard Python library)

Install dependencies:

```bash
pip install -r requirements.txt



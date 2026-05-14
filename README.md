# CS231 – Vietnamese License Plate Recognition Final Project

This project presents a Vietnamese License Plate Recognition (VLPR) system developed for the course **CS231 – Introduction to Computer Vision**. The system automatically detects and recognizes Vietnamese vehicle license plates from images and video streams using deep learning techniques. The project follows a two-stage pipeline: (1) License Plate Detection and (2) Character Detection and Recognition. YOLOv5 is used as the primary object detection framework for both stages. The system supports both one-line and two-line Vietnamese license plates and can perform real-time recognition through webcam input.

## Features

- Vietnamese license plate detection using YOLOv5
- Character recognition from detected plates
- Support for one-line and two-line license plates
- Real-time webcam inference
- Image-based inference
- Training notebooks for custom datasets

## Project Structure

```bash
├── model/                      # Pretrained models
├── result/                     # Output results and demo images
├── test_image/                 # Sample test images
├── training/                   # Training notebooks
│   ├── Plate_detection.ipynb
│   └── Letter_detection.ipynb
├── webcam.py                   # Webcam inference
├── lp_image.py                 # Image inference
├── requirement.txt             # Required dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/CS231-License-Plate-Recognition-Final-Project.git
cd CS231-License-Plate-Recognition-Final-Project
```

Install dependencies:

```bash
pip install -r requirement.txt
```

Download the compatible YOLOv5 version from the following link:

https://drive.google.com/file/d/1g1u7M4NmWDsMGOppHocgBKjbwtDA-uIu/view?usp=sharing

After downloading:

1. Extract the YOLOv5 folder
2. Copy the folder into the project directory

## Pretrained Models

Pretrained models are provided in the `model/` directory, including:

- License plate detection model
- Character detection model

## Running the Project

### Webcam Inference

```bash
python webcam.py
```

### Image Inference

```bash
python lp_image.py -i test_image/3.jpg
```

### Notebook Demonstration

Open the notebook below to understand the complete recognition pipeline and model workflow:

```bash
LP_recognition.ipynb
```

## Dataset

This project uses two datasets for the two-stage recognition pipeline.

### License Plate Detection Dataset

https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing

### Character Detection Dataset

https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing

Special thanks to Mì AI and winter2897 for sharing and contributing parts of the dataset resources.

## Training

Training notebooks are available in the `training/` directory.

### Train License Plate Detection Model

```bash
training/Plate_detection.ipynb
```

### Train Character Detection Model

```bash
training/Letter_detection.ipynb
```

## Methodology

### Stage 1 – License Plate Detection

YOLOv5 is trained to detect Vietnamese license plates from input images. The detected plate region is cropped for the next stage.

### Stage 2 – Character Detection and Recognition

A second YOLOv5 model detects individual characters inside the cropped plate image. Characters are sorted based on spatial position and combined to reconstruct the final license plate text.

## Results

### Sample Detection Result

![Demo](result/image.jpg)

### Real-Time Recognition Demo

![Video Demo](result/video.gif)

## Technologies Used

- Python
- OpenCV
- PyTorch
- YOLOv5
- NumPy
- Jupyter Notebook / Google Colab

## Course Information

- Course: CS231 – Introduction to Computer Vision
- Project Type: Final Project
- Topic: Vietnamese License Plate Recognition using Deep Learning

## Contributors

- Hoàng Nguyễn
- CS231 Final Project Team

## References

- YOLOv5 – Ultralytics
- OpenCV Documentation
- PyTorch Documentation
- Vietnamese License Plate datasets from community contributors

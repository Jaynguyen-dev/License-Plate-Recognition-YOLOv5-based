# CS231 – Vietnamese License Plate Recognition Final Project

This project presents a Vietnamese License Plate Recognition (VLPR) system developed for the course **CS231 – Introduction to Computer Vision**. The system automatically detects and recognizes Vietnamese vehicle license plates from images and video streams using deep learning techniques. The project implements a multi-stage pipeline, starting with License Plate Detection using YOLOv5. For the Optical Character Recognition stage, two distinct approaches are implemented and explored: (1) bounding-box-based character detection using a second YOLOv5 model, and (2) a sequence-based Optical Character Recognition (OCR) pipeline using a Layout CNN and CRNN. The system supports both one-line and two-line Vietnamese license plates and can perform real-time recognition through webcam input.

## Features

- Vietnamese license plate detection using YOLOv5
- Two approaches for character recognition: YOLOv5 character detection vs. CRNN-based OCR sequence recognition
- Support for one-line and two-line license plates
- Real-time webcam inference
- Image-based inference
- Training notebooks and scripts for custom datasets

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
## Dataset

This project uses two datasets for the two-stage recognition pipeline.

### License Plate Detection Dataset

https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing

### Character Detection Dataset

https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing
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

This project implements two distinct approaches for recognizing characters on Vietnamese license plates after the initial plate detection stage.

### Stage 1: License Plate Detection (Common)

Both approaches begin by using a YOLOv5 object detection model (`LP_detector.pt`) to locate and crop the Vietnamese license plate from the input image or video stream.

### Approach 1: Two-Stage YOLOv5 Object Detection (Heuristic-based Sorting)

This approach treats Character Recognition as an object detection problem followed by heuristic spatial sorting.

1. **Character Detection:** A second YOLOv5 model (`LP_ocr.pt`) trained specifically on alphanumeric classes detects individual characters within the cropped license plate, yielding bounding boxes and class labels.
2. **Layout Analysis (Linear Regression):** The system calculates the center point of each character's bounding box. It finds the leftmost and rightmost points and computes a linear equation. If any character's center deviates significantly from this line, the plate is classified as a "two-line" format; otherwise, it is a "one-line" format.
3. **Character Sorting & Reconstruction:**
   - **One-line plates:** Bounding boxes are simply sorted from left to right based on their x-coordinates.
   - **Two-line plates:** Characters are split into a top row and a bottom row based on whether their y-coordinate is above or below the mean y-coordinate of all characters. Each row is then sorted from left to right. The final license plate string is formed by concatenating the top row characters, a hyphen (`-`), and the bottom row characters.

### Approach 2: Deep Learning OCR Pipeline (CRNN & CTC)

This approach uses sequence modeling to read the entire license plate text without needing character-level bounding boxes.

1. **Layout Classification (CNN):** The cropped plate image is passed through a custom 4-layer Convolutional Neural Network (`LayoutCNN`) that acts as an image classifier. Utilizing Convolutional layers, Batch Normalization, and Adaptive Average Pooling, it predicts the plate format: "one-line" or "two-line".
2. **Line Splitting:** If the CNN classifies the plate as "two-line", an image processing algorithm horizontally splits the plate image to isolate the upper character line from the lower character line.
3. **Sequence Feature Extraction (CNN + BiLSTM):** Each line image is fed into a Convolutional Recurrent Neural Network (`CRNN`). 
   - The CNN backbone extracts deep spatial feature maps from the image.
   - These feature maps are reshaped into a sequence and passed into a 2-layer Bidirectional LSTM. The BiLSTM captures the sequential context of the characters from both directions.
4. **CTC Decoding:** The output logits from the RNN are decoded using Connectionist Temporal Classification (CTC) greedy decoding. This directly translates the unsegmented image sequence into the final predicted text string.
5. **Post-processing & Validation:** The recognized text from both lines is concatenated. A `PlateValidator` applies format constraints based on a predefined metadata set to autocorrect typical OCR misclassifications (e.g., confusing `8` and `B` or `0` and `D`) according to standard Vietnamese license plate syntax rules.

## Results

### Sample Detection Result

![Demo](result/output0.png)
![Demo](result/output2.png)
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

- Nguyễn Việt Hoàng - 24520561
- Đặng Ngọc Trường Chinh - 23520186

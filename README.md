# Real-time-jewelry-tracker
Real-time ring detection on finger joints using MediaPipe landmarks and a custom-trained CNN classifier. Built for jewelry visualization and hand tracking applications.

## 💡 Overview

This project detects rings on the **INDEX**, **MIDDLE**, and **RING** fingers in real-time using a webcam. It combines:
-  **MediaPipe** for hand and joint landmark detection
-  A **Convolutional Neural Network (CNN)** trained to classify if a ring is present in a cropped patch around a finger joint
- Designed for use in fashion-tech, augmented content creation, and jewelry styling previews.



## ⚙️ Setup Instructions

1. **Clone the Repository**
2. Install Dependencies
   pip install -r requirements.txt

##  Dataset Structure
   
   data/cropped_rings_labeled/
├── ring/
├── no_ring/


   

##  🏋️ Train the Model

1. Open main.py and set:  TRAIN_MODE = True
2. Run the script


##  🏋️ Run Real-Time Ring Detection
1. Once training is done: Set TRAIN_MODE = False in main.py
2. Run the script
3. Allow webcam access.

   

# The program will:

Detect hands via MediaPipe

Crop image patches at key finger joints

Predict ring presence with the CNN

Overlay labels (RING / NO RING) on the video

Press Q to quit the video feed.


# Real-time-jewelry-tracker
Real-time ring detection on finger joints using MediaPipe landmarks and a custom-trained CNN classifier. Built for jewelry visualization and hand tracking applications.

## 💡 Overview

This project detects rings on the **INDEX**, **MIDDLE**, and **RING** fingers in real-time using a webcam. It combines:
-  **MediaPipe** for hand and joint landmark detection
-  A **Convolutional Neural Network (CNN)** trained to classify if a ring is present in a cropped patch around a finger joint
- Designed for use in fashion-tech, augmented content creation, and jewelry styling previews.



## ⚙️ Setup Instructions

- **Clone the Repository**
- Install Dependencies
   pip install -r requirements.txt


## 📁 Dataset

Training data is based on images from:
- [Hands and Palm Images Dataset on Kaggle](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset)

Additional images and augmentations were generated to improve ring/no-ring balance and simulate real-world variations.


###  Dataset Structure
   
   data/cropped_rings_labeled/
├── ring/
├── no_ring/


   

##  🏋️ Train the Model

- Open main.py and set:  TRAIN_MODE = True
- Run the script


##  🏋️ Run Real-Time Ring Detection
- Once training is done: Set TRAIN_MODE = False in main.py
- Run the script
- Allow webcam access.

   

##  How It Works

The program will:

- Detect hands via MediaPipe

- Crop image patches at key finger joints

- Predict ring presence with the CNN

- Overlay labels (RING / NO RING) on the video

-  Press Q to quit the video feed.


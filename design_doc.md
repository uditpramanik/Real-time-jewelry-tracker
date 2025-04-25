
---

### 3. **`design_doc.md` (Project Reasoning + Thought Process)**




# Jewelry Detection Challenge

## 👩‍💻 Assumptions

- Anna wears rings on three fingers: index, middle, and ring.
- Only **rings** are tracked for now.
- The goal is to build a simple, real-time system to detect rings from webcam video using computer vision.

---

## 🎯 Project Scope

- Build a CNN-based ring classifier trained on labeled patches.
- Use MediaPipe Hands to detect finger landmarks and crop finger patches.
- Run real-time classification of ring presence at specific finger joints.

---

## 🧠 Thought Process & Design

### Why MediaPipe?
MediaPipe Hands gives reliable and efficient 21-point hand landmarks, making it easy to crop patches for ring detection without needing expensive hand segmentation or pose models.
This project was based off on a seperate project of mine where I used MediaPipe Hands extentensively for Gesture Recognition

### Why a CNN Classifier?
We framed the ring detection as a binary classification problem:
- Input: cropped image around a joint
- Output: ring / no ring

CNNs are lightweight, easy to train, and sufficient for this image-level task.

---

## 🔁 Training

- Used ImageDataGenerator to apply Data Augmentation for better results:
  - Rotation, shift, shear, zoom, flip, brightness
- Model: 2 Conv2D layers + MaxPool + Dense + Dropout
- Trained for 100 epochs with `ModelCheckpoint` for best accuracy

---

## 🎥 Inference Pipeline

1. Capture webcam video
2. Detect hand landmarks using MediaPipe
3. For each joint (index, middle, ring):
   - Crop a 64×64 patch around the joint
   - Normalize and pass to classifier
   - Predict “RING” or “NO RING”
   - Display result live on video feed

---

## 🔍 Observations / Failures

- Small patches sometimes miss the ring if hand is tilted or occluded
- Classifier fails with motion blur or low light
- Using pretrained classifiers like ResNet-50 or others worsens performance
- Future improvement: multi-frame voting or object detector (e.g., YOLO for better segmentation & classification)

---

## ✨ What I Would Do With More Time

- Add **earring detection** using face landmarks
- Integrate a **3D hand model** using Blender or NeRF for smooth jewelry placement
- Cluster ring styles using image features (ResNet) and recommend similar designs
- Add a multi-label classifier for ring **type/style/color**

---

## 📌 Key Takeaways

This project demonstrates:
- Landmark-based tracking
- Real-time inference via webcam
- CNN image classification on specific body parts

It’s a modular and extensible approach.

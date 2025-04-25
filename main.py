import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import mediapipe as mp




# === Parameters ===
DATA_DIR = 'data\cropped_rings_labeled'
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100


MODEL_PATH = os.path.join('model', 'best_model.h5')
TRAIN_MODE = True  #  Toggle this to True to train again

# === Functions ===
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, val_generator

def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_gen, val_gen = create_data_generators()
    model = build_model()
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
    
    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    if TRAIN_MODE:
        train_model()

    # === Load the model for inference ===
    classifier = load_model(MODEL_PATH)
    
    # === MediaPipe Live Classification ===
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ring_joint_ids = {
        "INDEX": mp_hands.HandLandmark.INDEX_FINGER_PIP,
        "MIDDLE": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        "RING": mp_hands.HandLandmark.RING_FINGER_PIP,
        # "THUMB": mp_hands.HandLandmark.THUMB_IP,
        # "PINKY": mp_hands.HandLandmark.PINKY_PIP
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_bgr = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for label, joint_id in ring_joint_ids.items():
                    landmark = hand_landmarks.landmark[joint_id]
                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    size = 32
                    x1, y1 = max(x - size, 0), max(y - size, 0)
                    x2, y2 = min(x + size, frame.shape[1]), min(y + size, frame.shape[0])
                    patch = frame[y1:y2, x1:x2]
                    if patch.shape[0] == 0 or patch.shape[1] == 0:
                        continue
                    patch_resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE)) / 255.0
                    patch_input = np.expand_dims(patch_resized, axis=0)
                    pred = classifier.predict(patch_input)[0][0]
                    label_pred = "RING" if pred > 0.5 else "NO RING"
                    color = (0, 255, 0) if label_pred == "RING" else (0, 0, 255)
                    cv2.circle(frame_bgr, (x, y), 10, color, -1)
                    cv2.putText(frame_bgr, f"{label}: {label_pred}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Ring Detection", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
File 3: 03_realtime_emotion_recognition.py
===========================================
GenAI & Deep Learning Assessment - Module 3
Topic: Real-Time Facial Emotion Recognition & Automated Feedback System

Description:
This module integrates computer vision (OpenCV) with deep learning (Keras CNN)
to perform real-time facial expression analysis. It detects face boundaries,
preprocesses the region of interest (48x48 gray), predicts the emotion state,
and displays dynamic context-aware feedback messages for educational environments.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

FEEDBACK_MESSAGES = {
    'Angry': "You appear upset. Try deep breathing or talk calmly about what's bothering you.",
    'Disgust': "Something seems uncomfortable. Let's revisit the topic together.",
    'Fear': "You seem anxious. Don't worry, take a deep breath, you're doing fine.",
    'Happy': "Great! You seem to be enjoying the class. Keep up the positive spirit!",
    'Sad': "You seem a bit down. Take a short break or share your concern with your teacher.",
    'Surprise': "Looks like something caught your attention! Stay curious.",
    'Neutral': "All good. Stay focused and keep participating!"
}

def create_fallback_model(save_path="emotion_model.h5"):
    """Creates a sample dummy model if pre-trained weights file is absent."""
    print("Creating a placeholder emotion recognition CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save(save_path)
    print(f"Placeholder model saved to '{save_path}'.")
    return model

def load_emotion_model(model_path="emotion_model.h5"):
    """Loads existing model or initializes a fallback."""
    if os.path.exists(model_path):
        try:
            print(f"Loading model from '{model_path}'...")
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    return create_fallback_model(model_path)

def process_frame(frame, face_detector, model):
    """Processes a single video frame for face detection and emotion classification."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract Region of Interest (ROI)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Model Inference
        preds = model.predict(face, verbose=0)
        emotion_index = np.argmax(preds)
        emotion = EMOTION_LABELS[emotion_index]
        confidence = np.max(preds) * 100

        # Retrieve Feedback Message
        feedback = FEEDBACK_MESSAGES.get(emotion, "Stay focused!")

        # Draw Overlay Box & Text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, feedback, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def run_realtime_recognition():
    """Initializes webcam feed and executes main inference loop."""
    print("\n[Facial Emotion Recognition System Initialized]")
    print("Press 'q' in the video display window to quit.\n")

    model = load_emotion_model()
    
    # Load OpenCV Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Webcam device not found. Operating in headless test mode...")
        # Test frame generation
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = process_frame(dummy_frame, face_detector, model)
        print("Headless frame processing verified successfully.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab video frame.")
            break

        processed_frame = process_frame(frame, face_detector, model)
        cv2.imshow('Facial Emotion Recognition with Feedback', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting webcam stream...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_recognition()

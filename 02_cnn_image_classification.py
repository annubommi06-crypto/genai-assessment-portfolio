"""
File 2: 02_cnn_image_classification.py
======================================
GenAI & Deep Learning Assessment - Module 2
Topic: Convolutional Neural Network (CNN) Image Classifier & Metrics Evaluation

Description:
This script builds a robust Convolutional Neural Network (CNN) using Keras,
processes 2D image representations, handles normalization and data split,
trains the model, and computes detailed performance metrics including
Confusion Matrix and Classification Report (Precision, Recall, F1-score).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Constructs a 2D Convolutional Neural Network."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.3),

        # Dense Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="fc1"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", name="output")
    ], name="CNN_Image_Classifier")

    return model

def load_data():
    """Loads and formats dataset into (Height, Width, Channels)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape for 2D Conv layers: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1).astype("float32") / 255.0
    x_test  = np.expand_dims(x_test, axis=-1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)

def evaluate_model_performance(model, x_test, y_test, class_names=None):
    """Computes predictions, confusion matrix, and classification report."""
    print("\nCalculating Test Predictions & Evaluation Metrics...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix (Ensures correct tick placement)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d", xticks_rotation=45)
    plt.title("CNN Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=300)
    print("Saved confusion matrix plot to 'cnn_confusion_matrix.png'.")
    plt.close()

def main():
    print("[1/4] Loading Dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("[2/4] Building CNN Model...")
    model = build_cnn_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("[3/4] Training Model...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.15,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    print("[4/4] Evaluating Performance...")
    evaluate_model_performance(model, x_test, y_test)

    # Save model
    model.save("cnn_image_classifier.keras")
    print("Saved model to 'cnn_image_classifier.keras'.")

if __name__ == "__main__":
    main()

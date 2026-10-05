"""
File 1: 01_mlp_mnist_classification.py
======================================
GenAI & Deep Learning Assessment - Module 1
Topic: Multi-Layer Perceptron (MLP) for Handwritten Digit Classification (MNIST)

Description:
This script loads the MNIST dataset, reshapes and normalizes the image vectors,
constructs a deep Sequential MLP with Dropout regularization, compiles it with the
Adam optimizer, trains it using early stopping and learning rate reduction callbacks,
evaluates test performance, and exports the model artifact.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_and_preprocess_data():
    """Loads and normalizes MNIST dataset."""
    print("[1/5] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten 28x28 images into 784-dimensional vectors
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # Split validation set from training set
    val_size = 10000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"Training samples:   {x_train.shape[0]}")
    print(f"Validation samples: {x_val.shape[0]}")
    print(f"Test samples:       {x_test.shape[0]}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_mlp_model(input_dim=784, num_classes=10, dropout_rate=0.3):
    """Builds a Multi-Layer Perceptron model."""
    print("[2/5] Constructing MLP Architecture...")
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu", name="hidden_layer_1"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation="relu", name="hidden_layer_2"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.5),
        layers.Dense(64, activation="relu", name="hidden_layer_3"),
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="MLP_MNIST_Classifier")

    return model

def main():
    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # 2. Build Model
    model = build_mlp_model()
    model.summary()

    # 3. Compile Model
    print("[3/5] Compiling Model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 4. Define Callbacks & Train
    print("[4/5] Training Model with Callbacks...")
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate & Export Model
    print("[5/5] Evaluating on Test Dataset...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n==========================================")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"==========================================\n")

    output_path = "mlp_mnist_model.keras"
    model.save(output_path)
    print(f"Model successfully saved to '{output_path}'.")

if __name__ == "__main__":
    main()

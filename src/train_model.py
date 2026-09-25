import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==============================
# Configuration
# ==============================

DATASET_DIR = "dataset"
MODEL_DIR = "models"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


# ==============================
# Create model directory
# ==============================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# Load training dataset
# ==============================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


# ==============================
# Load validation dataset
# ==============================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


# ==============================
# Show classes
# ==============================

print("\nClasses:")
print(train_dataset.class_names)


# ==============================
# Optimize data pipeline
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==============================
# Build CNN model
# ==============================

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==============================
# Display model
# ==============================

model.summary()


# ==============================
# Compile model
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# Callbacks
# ==============================

checkpoint = ModelCheckpoint(
    "models/face_mask_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==============================
# Train
# ==============================

print("\nStarting CNN training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ==============================
# Evaluate
# ==============================

loss, accuracy = model.evaluate(
    validation_dataset
)

print("\n==============================")
print("Training completed")
print("==============================")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# ==============================
# Save final model
# ==============================

# ==============================
# Save training history
# ==============================

import json

history_data = {
    "accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"],
    "loss": history.history["loss"],
    "val_loss": history.history["val_loss"]
}

with open(
    "results/training_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(history_data, file)

print("Training history saved.")

model.save(
    "models/face_mask_cnn_final.keras"
)

print("\nModel saved successfully.")
print("models/face_mask_cnn.keras")
print("models/face_mask_cnn_final.keras")
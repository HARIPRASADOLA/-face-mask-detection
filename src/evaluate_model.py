import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==============================
# Configuration
# ==============================

DATASET_DIR = "dataset"
MODEL_PATH = "models/face_mask_cnn.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


# ==============================
# Load trained model
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


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
    label_mode="binary",
    shuffle=False
)


class_names = validation_dataset.class_names

print("\nClasses:")
print(class_names)


# ==============================
# Get actual labels
# ==============================

y_true = []

for images, labels in validation_dataset:
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true).astype(int)


# ==============================
# Get predictions
# ==============================

print("\nGenerating predictions...")

y_probability = model.predict(
    validation_dataset,
    verbose=1
)

y_probability = y_probability.flatten()

y_pred = (y_probability >= 0.5).astype(int)


# ==============================
# Accuracy
# ==============================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ==============================
# Classification Report
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# ==============================
# Save evaluation results
# ==============================

os.makedirs("results", exist_ok=True)

with open(
    "results/evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("FACE MASK DETECTION - MODEL EVALUATION\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

    file.write("Classification Report\n")
    file.write("-" * 50 + "\n")

    file.write(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    file.write("\nConfusion Matrix\n")
    file.write("-" * 50 + "\n")

    file.write(str(cm))


print("\nEvaluation report saved:")
print("results/evaluation_report.txt")
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ==========================================
# Configuration
# ==========================================

TEST_DIR = r"C:\Users\olaplus\Downloads\face_mask_dataset\Face Mask Dataset\Test"

MODEL_PATH = "models/face_mask_cnn.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# ==========================================
# Load trained model
# ==========================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# Load ORIGINAL TEST dataset
# ==========================================

print("\nLoading original test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# Optimize dataset
# ==========================================

test_dataset = test_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)


# ==========================================
# Actual labels
# ==========================================

print("\nReading actual labels...")

y_true = []

for images, labels in test_dataset:
    y_true.extend(
        labels.numpy().flatten()
    )

y_true = np.array(y_true).astype(int)


# ==========================================
# Predictions
# ==========================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
)

probabilities = probabilities.flatten()

y_pred = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n========================================")
print("FINAL TEST DATASET EVALUATION")
print("========================================")

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# Classification Report
# ==========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ==========================================
# Save results
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    "results/test_evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FACE MASK DETECTION\n"
    )

    file.write(
        "FINAL TEST DATASET EVALUATION\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(report)

    file.write(
        "\nConfusion Matrix\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        str(cm)
    )


print("\n========================================")
print("Evaluation completed successfully.")
print("========================================")

print(
    "\nReport saved:"
)

print(
    "results/test_evaluation_report.txt"
)
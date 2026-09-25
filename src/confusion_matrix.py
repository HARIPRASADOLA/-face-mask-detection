import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ==========================================
# Configuration
# ==========================================

TEST_DIR = r"C:\Users\olaplus\Downloads\face_mask_dataset\Face Mask Dataset\Test"

MODEL_PATH = "models/face_mask_cnn.keras"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# ==========================================
# Load model
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")


# ==========================================
# Load test dataset
# ==========================================

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
# Predictions
# ==========================================

print("\nGenerating predictions...")

y_true = []

for images, labels in test_dataset:
    y_true.extend(
        labels.numpy().flatten()
    )

y_true = np.array(y_true).astype(int)


probabilities = model.predict(
    test_dataset,
    verbose=1
)

probabilities = probabilities.flatten()

y_pred = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# Create results directory
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================
# Plot confusion matrix
# ==========================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Face Mask Detection - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# ==========================================
# Add values
# ==========================================

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=200
)

plt.show()

print(
    "\nConfusion matrix saved:"
)

print(
    "results/confusion_matrix.png"
)
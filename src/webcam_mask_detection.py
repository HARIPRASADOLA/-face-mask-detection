import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/face_mask_cnn.keras"

IMG_SIZE = (128, 128)

CONFIDENCE_THRESHOLD = 0.50


# ==========================================
# Load CNN model
# ==========================================

print("Loading face mask model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# Load OpenCV face detector
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load OpenCV face cascade."
    )


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )

print("Webcam opened successfully.")
print("Press Q to quit.")


# ==========================================
# Real-time detection
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break


    # ------------------------------
    # Convert to grayscale
    # ------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ------------------------------
    # Detect faces
    # ------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # ------------------------------
    # Process each face
    # ------------------------------

    for (x, y, w, h) in faces:

        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue


        # ------------------------------
        # Prepare image for CNN
        # ------------------------------

        face_resized = cv2.resize(
            face,
            IMG_SIZE
        )

        face_rgb = cv2.cvtColor(
            face_resized,
            cv2.COLOR_BGR2RGB
        )

        face_array = (
            np.expand_dims(
                face_rgb,
                axis=0
            ).astype("float32")
        )


        # ------------------------------
        # CNN prediction
        # ------------------------------

        probability = float(
            model.predict(
                face_array,
                verbose=0
            )[0][0]
        )


        # Dataset class order from Keras
        #
        # 0 = WithMask
        # 1 = WithoutMask
        #
        # Therefore:
        # lower probability = WithMask
        # higher probability = WithoutMask

        if probability >= CONFIDENCE_THRESHOLD:

            label = "WITHOUT MASK"

            confidence = probability

        else:

            label = "WITH MASK"

            confidence = 1.0 - probability


        confidence_percent = (
            confidence * 100
        )


        # ------------------------------
        # Bounding box
        # ------------------------------

        color = (0, 255, 0)

        if label == "WITHOUT MASK":

            color = (0, 0, 255)


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )


        # ------------------------------
        # Label
        # ------------------------------

        text = (
            f"{label}: "
            f"{confidence_percent:.1f}%"
        )


        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # ==================================
    # Display webcam
    # ==================================

    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    # ==================================
    # Quit with Q
    # ==================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Webcam detection stopped.")
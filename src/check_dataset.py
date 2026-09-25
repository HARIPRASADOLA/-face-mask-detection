import os
from PIL import Image

DATASET_DIR = "dataset"

classes = [
    "with_mask",
    "without_mask"
]

total = 0

for class_name in classes:

    folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.exists(folder):
        print(f"ERROR: Missing folder: {folder}")
        continue

    image_count = 0
    invalid_count = 0

    for filename in os.listdir(folder):

        file_path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(file_path):
            continue

        try:
            with Image.open(file_path) as image:
                image.verify()

            image_count += 1

        except Exception:
            invalid_count += 1
            print(f"Invalid image: {file_path}")

    print()
    print(f"Class: {class_name}")
    print(f"Valid images: {image_count}")
    print(f"Invalid images: {invalid_count}")

    total += image_count

print()
print("=" * 40)
print(f"Total valid images: {total}")
print("=" * 40)
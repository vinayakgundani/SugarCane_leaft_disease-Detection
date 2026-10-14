import os
import shutil
import random

# Original dataset
source_dir = "Dataset"

# New dataset folders
output_dir = "Dataset_Split"

# Split percentages
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

# Make the result reproducible
random.seed(42)

# Disease classes
classes = os.listdir(source_dir)

for class_name in classes:

    source_class_dir = os.path.join(source_dir, class_name)

    # Get only files
    images = [
        file for file in os.listdir(source_class_dir)
        if os.path.isfile(os.path.join(source_class_dir, file))
    ]

    # Shuffle images randomly
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    # Create folders
    for split in ["train", "validation", "test"]:

        split_dir = os.path.join(output_dir, split, class_name)

        os.makedirs(split_dir, exist_ok=True)

    # Copy images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(output_dir, "train", class_name, image)
        )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(output_dir, "validation", class_name, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(source_class_dir, image),
            os.path.join(output_dir, "test", class_name, image)
        )

    print(
        class_name,
        "→ Train:", len(train_images),
        "Validation:", len(validation_images),
        "Test:", len(test_images)
    )

print("\nDataset splitting completed!")
import os
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "sugarcane_disease_model_v5.keras"

TEST_DIR = "Age_Dataset_Split/test"

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

CLASS_NAMES = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]

AGE_VALUES = {
    "5_months": 5.0,
    "8_months": 8.0,
    "12_months": 12.0
}


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading V5 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA FOR ONE AGE GROUP
# ============================================================

def load_age_test_data(age_folder):

    age_value = AGE_VALUES[age_folder]

    age_path = os.path.join(
        TEST_DIR,
        age_folder
    )

    image_paths = []
    labels = []
    ages = []


    # --------------------------------------------------------
    # Read every disease folder
    # --------------------------------------------------------

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            age_path,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                "WARNING: Folder not found:",
                class_path
            )

            continue


        for filename in os.listdir(class_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):

                image_paths.append(
                    os.path.join(
                        class_path,
                        filename
                    )
                )

                labels.append(
                    class_index
                )

                ages.append(
                    age_value
                )


    return (
        image_paths,
        np.array(labels),
        np.array(ages, dtype=np.float32)
    )


# ============================================================
# LOAD AN IMAGE
# ============================================================

def load_image(image_path):

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# ============================================================
# PREDICT ONE AGE GROUP
# ============================================================

def evaluate_age_group(age_folder):

    print()
    print("----------------------------------------")
    print(f"Evaluating: {age_folder}")
    print("----------------------------------------")


    # --------------------------------------------------------
    # Load paths and labels
    # --------------------------------------------------------

    image_paths, y_true, ages = load_age_test_data(
        age_folder
    )


    print(
        "Test images:",
        len(image_paths)
    )


    if len(image_paths) == 0:

        print(
            "No images found for",
            age_folder
        )

        return None


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    all_predictions = []


    for start in range(
        0,
        len(image_paths),
        BATCH_SIZE
    ):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        batch_ages = ages[
            start:start + BATCH_SIZE
        ]


        # Load images

        batch_images = []

        for path in batch_paths:

            image = load_image(
                path
            )

            batch_images.append(
                image
            )


        batch_images = tf.stack(
            batch_images
        )


        # Age shape:
        #
        # [5, 5, 5, ...]
        #
        # becomes:
        #
        # [[5],
        #  [5],
        #  [5], ...]

        batch_ages = batch_ages.reshape(
            (-1, 1)
        )


        # ----------------------------------------------------
        # V5 has TWO inputs
        # ----------------------------------------------------

        predictions = model.predict(

            {
                "image": batch_images,

                "age": batch_ages
            },

            verbose=0

        )


        predicted_classes = np.argmax(
            predictions,
            axis=1
        )


        all_predictions.extend(
            predicted_classes
        )


    y_pred = np.array(
        all_predictions
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    correct = np.sum(
        y_true == y_pred
    )

    total = len(
        y_true
    )

    accuracy = (
        correct / total
    ) * 100


    print(
        f"Accuracy: {accuracy:.2f}%"
    )


    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print()
    print("Disease-wise performance:")

    print(
        f"{'Disease':<12}"
        f"{'Correct':<10}"
        f"{'Total':<10}"
        f"{'Recall':<10}"
    )

    print(
        "-" * 42
    )


    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        actual = np.sum(
            y_true == class_index
        )

        correct_class = np.sum(
            (y_true == class_index)
            &
            (y_pred == class_index)
        )


        if actual > 0:

            recall = (
                correct_class / actual
            ) * 100

        else:

            recall = 0


        print(
            f"{class_name:<12}"
            f"{correct_class:<10}"
            f"{actual:<10}"
            f"{recall:.2f}%"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion = np.zeros(
        (
            len(CLASS_NAMES),
            len(CLASS_NAMES)
        ),
        dtype=int
    )


    for actual, predicted in zip(
        y_true,
        y_pred
    ):

        confusion[
            actual,
            predicted
        ] += 1


    print()
    print("Confusion Matrix:")

    print(
        f"{'Actual / Pred':<15}",
        end=""
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:<10}",
            end=""
        )

    print()


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:<15}",
            end=""
        )

        for j in range(
            len(CLASS_NAMES)
        ):

            print(
                f"{confusion[i][j]:<10}",
                end=""
            )

        print()


    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion": confusion
    }


# ============================================================
# EVALUATE ALL AGE GROUPS
# ============================================================

results = {}


for age_folder in AGE_VALUES:

    results[age_folder] = evaluate_age_group(
        age_folder
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

print()
print()
print("================================================")
print("OVERALL V5 AGE-WISE EVALUATION")
print("================================================")


total_correct = 0
total_images = 0


for age_folder, result in results.items():

    if result is None:
        continue


    total_correct += result["correct"]

    total_images += result["total"]


    print(
        f"{age_folder:<12}"
        f"Accuracy: {result['accuracy']:.2f}% "
        f"({result['correct']}/{result['total']})"
    )


# ------------------------------------------------------------
# Overall accuracy
# ------------------------------------------------------------

if total_images > 0:

    overall_accuracy = (
        total_correct /
        total_images
    ) * 100

else:

    overall_accuracy = 0


print()
print(
    "Overall Test Accuracy:"
)

print(
    f"{overall_accuracy:.2f}%"
)


print()
print("================================================")
print("EVALUATION COMPLETE")
print("================================================")
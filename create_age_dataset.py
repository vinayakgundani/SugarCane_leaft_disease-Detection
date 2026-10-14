import os
import shutil
import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

SOURCE_DIR = "Dataset"
OUTPUT_DIR = "Age_Dataset"

AGES = [
    "5_months",
    "8_months",
    "12_months"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CALCULATE VISUAL LEAF SIZE
# ============================================================

def calculate_leaf_score(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize only for analysis
    image = cv2.resize(image, (500, 500))

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # Detect green/leaf-like pixels
    # --------------------------------------------------------

    lower_green = np.array([20, 20, 20])
    upper_green = np.array([100, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Remove tiny noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Largest detected leaf-like region
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest_contour
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    image_area = 500 * 500

    # Percentage of image occupied by leaf
    area_ratio = area / image_area

    # Leaf dimensions
    normalized_width = width / 500
    normalized_height = height / 500

    # --------------------------------------------------------
    # Aspect/shape information
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

    else:

        circularity = 0


    # --------------------------------------------------------
    # Combined visual size score
    # --------------------------------------------------------

    score = (
        area_ratio * 0.60
        +
        normalized_width * 0.20
        +
        normalized_height * 0.20
    )

    return score


# ============================================================
# CREATE AGE DATASET
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print("ERROR: Dataset folder not found.")

    print(
        "Run this file from your "
        "Sugarcane-Disease-Detection folder."
    )

    exit()


# Delete previous generated dataset

if os.path.exists(OUTPUT_DIR):

    print("Removing previous Age_Dataset...")

    shutil.rmtree(
        OUTPUT_DIR
    )


total_images = 0


age_counts = {
    "5_months": 0,
    "8_months": 0,
    "12_months": 0
}


# ============================================================
# PROCESS EACH DISEASE SEPARATELY
# ============================================================

for disease in sorted(
    os.listdir(SOURCE_DIR)
):

    disease_path = os.path.join(
        SOURCE_DIR,
        disease
    )

    if not os.path.isdir(
        disease_path
    ):
        continue


    print()
    print(
        "======================================"
    )

    print(
        "Processing:",
        disease
    )

    print(
        "======================================"
    )


    image_scores = []


    # --------------------------------------------------------
    # Calculate score for every image
    # --------------------------------------------------------

    for image_name in os.listdir(
        disease_path
    ):

        if not image_name.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue


        image_path = os.path.join(
            disease_path,
            image_name
        )


        score = calculate_leaf_score(
            image_path
        )


        if score is not None:

            image_scores.append(
                (
                    image_name,
                    score
                )
            )


    # --------------------------------------------------------
    # Sort from smallest-looking to largest-looking
    # --------------------------------------------------------

    image_scores.sort(
        key=lambda x: x[1]
    )


    total = len(
        image_scores
    )


    if total == 0:

        print(
            "No readable images found."
        )

        continue


    # --------------------------------------------------------
    # Divide into three groups
    # --------------------------------------------------------

    first_end = total // 3

    second_end = (
        2 * total
    ) // 3


    groups = {

        "5_months":
            image_scores[
                :first_end
            ],

        "8_months":
            image_scores[
                first_end:second_end
            ],

        "12_months":
            image_scores[
                second_end:
            ]
    }


    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    for age in AGES:

        destination = os.path.join(
            OUTPUT_DIR,
            age,
            disease
        )

        os.makedirs(
            destination,
            exist_ok=True
        )


        selected_images = groups[
            age
        ]


        for image_name, score in selected_images:

            source_file = os.path.join(
                disease_path,
                image_name
            )

            destination_file = os.path.join(
                destination,
                image_name
            )


            shutil.copy2(
                source_file,
                destination_file
            )


            total_images += 1

            age_counts[age] += 1


        # ----------------------------------------------------
        # Show group information
        # ----------------------------------------------------

        scores = [
            score
            for _, score
            in selected_images
        ]


        if len(scores) > 0:

            print(
                f"{age:10s} : "
                f"{len(selected_images):3d} images | "
                f"score "
                f"{min(scores):.3f}"
                f" - "
                f"{max(scores):.3f}"
            )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print(
    "======================================"
)

print(
    "AGE DATASET CREATED"
)

print(
    "======================================"
)

print(
    "Total images:",
    total_images
)

print()

print(
    "5 months :",
    age_counts[
        "5_months"
    ]
)

print(
    "8 months :",
    age_counts[
        "8_months"
    ]
)

print(
    "12 months:",
    age_counts[
        "12_months"
    ]
)

print()
print(
    "Output:",
    OUTPUT_DIR
)

print(
    "======================================"
)
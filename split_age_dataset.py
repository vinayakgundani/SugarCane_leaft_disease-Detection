import os
import shutil
import random

# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = "Age_Dataset_Split"

AGE_GROUPS = [
    "5_months",
    "8_months",
    "12_months"
]

DISEASE_CLASSES = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)


# =========================================================
# SPLIT EACH AGE + DISEASE GROUP
# =========================================================

for age in AGE_GROUPS:

    for disease in DISEASE_CLASSES:

        # -------------------------------------------------
        # SOURCE
        # -------------------------------------------------

        source_path = os.path.join(
            BASE_DIR,
            age,
            disease
        )

        if not os.path.exists(source_path):

            print(
                "WARNING: Folder not found:",
                source_path
            )

            continue


        # -------------------------------------------------
        # GET IMAGES
        # -------------------------------------------------

        files = [
            f
            for f in os.listdir(source_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]


        random.shuffle(files)

        total = len(files)

        # -------------------------------------------------
        # CALCULATE SPLIT
        # -------------------------------------------------

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train_files = files[
            :train_count
        ]

        validation_files = files[
            train_count:
            train_count + val_count
        ]

        test_files = files[
            train_count + val_count:
        ]


        # -------------------------------------------------
        # DESTINATIONS
        # -------------------------------------------------

        split_data = {

            "train": train_files,

            "validation": validation_files,

            "test": test_files

        }


        # -------------------------------------------------
        # COPY FILES
        # -------------------------------------------------

        for split_name, split_files in split_data.items():

            destination_path = os.path.join(
                BASE_DIR,
                split_name,
                age,
                disease
            )

            os.makedirs(
                destination_path,
                exist_ok=True
            )


            for filename in split_files:

                source_file = os.path.join(
                    source_path,
                    filename
                )

                destination_file = os.path.join(
                    destination_path,
                    filename
                )

                shutil.copy2(
                    source_file,
                    destination_file
                )


        # -------------------------------------------------
        # PRINT RESULT
        # -------------------------------------------------

        print(
            f"{age} | {disease} | "
            f"Total: {total} | "
            f"Train: {len(train_files)} | "
            f"Validation: {len(validation_files)} | "
            f"Test: {len(test_files)}"
        )


print("\n======================================")
print("AGE DATASET SPLIT COMPLETE")
print("======================================")
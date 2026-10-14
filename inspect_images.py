import os
from PIL import Image

dataset_path = "Dataset"

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    print("\nClass:", class_name)

    count = 0

    for filename in os.listdir(class_path):

        image_path = os.path.join(class_path, filename)

        try:
            image = Image.open(image_path)

            if count < 3:
                print(
                    filename,
                    "→",
                    image.size,
                    "→",
                    image.mode
                )

            count += 1

        except Exception as e:
            print("Problem with:", filename)
            print(e)

    print("Total valid images:", count)
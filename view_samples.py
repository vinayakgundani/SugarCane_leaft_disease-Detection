import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "Dataset"

classes = os.listdir(dataset_path)

plt.figure(figsize=(15, 4))

for i, class_name in enumerate(classes):

    class_path = os.path.join(dataset_path, class_name)

    image_name = os.listdir(class_path)[0]

    image_path = os.path.join(class_path, image_name)

    image = Image.open(image_path)

    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()
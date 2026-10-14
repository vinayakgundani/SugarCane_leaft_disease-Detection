import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Settings
# -----------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dir = "Dataset_Split/test"

# -----------------------------
# 2. Load test dataset
# -----------------------------

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

print("Classes:")
print(class_names)

# -----------------------------
# 3. Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "sugarcane_disease_model_v3.keras"
)

# -----------------------------
# 4. Get predictions
# -----------------------------

true_labels = []
predicted_labels = []

for images, labels in test_data:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted_classes)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# -----------------------------
# 5. Create confusion matrix
# -----------------------------

confusion_matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=len(class_names)
)

print("\nConfusion Matrix:")
print(confusion_matrix.numpy())

# -----------------------------
# 6. Display confusion matrix
# -----------------------------

plt.figure(figsize=(8, 6))

plt.imshow(confusion_matrix.numpy())

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Sugarcane Disease Confusion Matrix")

# Show numbers inside cells
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            confusion_matrix[i, j].numpy(),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()
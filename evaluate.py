import tensorflow as tf

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

# -----------------------------
# 3. Load V3 trained model
# -----------------------------

model = tf.keras.models.load_model(
    "sugarcane_disease_model_v3.keras"
)

# -----------------------------
# 4. Evaluate model
# -----------------------------

test_loss, test_accuracy = model.evaluate(test_data)

# -----------------------------
# 5. Display result
# -----------------------------

print("\nTest Results")
print("------------")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")
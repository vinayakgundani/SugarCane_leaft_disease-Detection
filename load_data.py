import tensorflow as tf

# Paths
train_dir = "Dataset_Split/train"
validation_dir = "Dataset_Split/validation"
test_dir = "Dataset_Split/test"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load validation data
validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load test data
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(train_data.class_names)

print("\nNumber of classes:")
print(len(train_data.class_names))
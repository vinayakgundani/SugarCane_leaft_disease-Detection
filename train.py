import tensorflow as tf

# -----------------------------
# 1. Settings
# -----------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 10

# -----------------------------
# 2. Dataset paths
# -----------------------------

train_dir = "Dataset_Split/train"
validation_dir = "Dataset_Split/validation"

# -----------------------------
# 3. Load datasets
# -----------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# 4. Data augmentation
# -----------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# -----------------------------
# 5. Load pretrained model
# -----------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# 6. Build our model
# -----------------------------

model = tf.keras.Sequential([
    data_augmentation,

    # MobileNetV2 preprocessing
    tf.keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# -----------------------------
# 7. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 8. Train
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# -----------------------------
# 9. Save model
# -----------------------------

model.save("sugarcane_disease_model_v2.keras")

print("\nTraining completed!")
print("Model saved as sugarcane_disease_model.keras")
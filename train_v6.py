
import os
import random
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIGURATION ----------------

DATASET_DIR = Path("Age_Dataset_Split")
MODEL_PATH = "sugarcane_disease_model_v6.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
AGE_NAMES = {
    "5_months": 5,
    "8_months": 8,
    "12_months": 12,
}

EPOCHS_HEAD = 15
EPOCHS_FINE_TUNE = 25
FINE_TUNE_LAYERS = 60

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- LOAD FILE PATHS ----------------

def collect_images(split):
    paths = []
    labels = []
    ages = []

    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing folder: {split_dir.resolve()}"
        )

    for age_folder, age_value in AGE_NAMES.items():
        for label, class_name in enumerate(CLASS_NAMES):
            folder = split_dir / age_folder / class_name

            if not folder.exists():
                print(f"Warning: missing folder {folder}")
                continue

            for file in folder.iterdir():
                if file.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }:
                    paths.append(str(file))
                    labels.append(label)
                    ages.append([float(age_value)])

    if not paths:
        raise ValueError(f"No images found under {split_dir.resolve()}")

    return (
        np.array(paths),
        np.array(labels, dtype=np.int32),
        np.array(ages, dtype=np.float32),
    )


train_paths, train_labels, train_ages = collect_images("train")
val_paths, val_labels, val_ages = collect_images("validation")
test_paths, test_labels, test_ages = collect_images("test")

print("\nDataset counts")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

print("\nTraining images per class:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:10s}: {np.sum(train_labels == i)}")

# ---------------- CLASS WEIGHTS ----------------
# Gives greater training importance to classes with fewer examples.

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=train_labels,
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:10s}: {class_weights[i]:.3f}")

# ---------------- TF.DATA PIPELINE ----------------

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, age, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    # EfficientNetB0 includes its own input rescaling.
    # Keep pixel values in the 0-255 range.
    image = tf.cast(image, tf.float32)

    inputs = {
        "image": image,
        "age": age,
    }

    return inputs, label


def make_dataset(paths, ages, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, ages, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(
    train_paths, train_ages, train_labels, training=True
)

val_ds = make_dataset(
    val_paths, val_ages, val_labels
)

test_ds = make_dataset(
    test_paths, test_ages, test_labels
)

# ---------------- BUILD V6 MODEL ----------------

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="image_augmentation",
)

image_input = tf.keras.Input(
    shape=(224, 224, 3),
    name="image",
)

age_input = tf.keras.Input(
    shape=(1,),
    name="age",
)

augmented_image = augmentation(image_input)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False

image_features = base_model(
    augmented_image,
    training=False,
)

image_features = tf.keras.layers.GlobalAveragePooling2D()(
    image_features
)

image_features = tf.keras.layers.BatchNormalization()(
    image_features
)

image_features = tf.keras.layers.Dropout(0.35)(
    image_features
)

image_features = tf.keras.layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(image_features)

image_features = tf.keras.layers.Dropout(0.25)(
    image_features
)

# Age remains an explicit input to the model.
normalized_age = tf.keras.layers.Rescaling(
    scale=1.0 / 12.0,
    name="age_normalization",
)(age_input)

age_features = tf.keras.layers.Dense(
    32,
    activation="relu",
)(normalized_age)

age_features = tf.keras.layers.Dense(
    16,
    activation="relu",
)(age_features)

combined = tf.keras.layers.Concatenate()(
    [image_features, age_features]
)

combined = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(combined)

combined = tf.keras.layers.Dropout(0.35)(combined)

output = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="disease",
)(combined)

model = tf.keras.Model(
    inputs={
        "image": image_input,
        "age": age_input,
    },
    outputs=output,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=loss_fn,
        metrics=["accuracy"],
    )


model.summary()

# ---------------- CALLBACKS ----------------

def make_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

# ---------------- PHASE 1: TRAIN CLASSIFIER HEAD ----------------

print("\n" + "=" * 55)
print("V6 PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 55)

compile_model(learning_rate=0.0005)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

# ---------------- PHASE 2: FINE-TUNE BACKBONE ----------------

print("\n" + "=" * 55)
print("V6 PHASE 2: FINE-TUNING EFFICIENTNETB0")
print("=" * 55)

base_model.trainable = True

# Fine-tune only the final portion of the backbone.
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
    # Keep BatchNormalization statistics stable on a small dataset.
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

compile_model(learning_rate=0.00001)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    class_weight=class_weights,
    callbacks=make_callbacks(),
)

# ---------------- FINAL EVALUATION ----------------
# Load the best checkpoint selected by validation loss.
# Test data is used only here.

print("\nLoading best V6 checkpoint...")
best_model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nV6 test accuracy: {test_accuracy * 100:.2f}%")
print(f"V6 test loss: {test_loss:.4f}")

predictions = best_model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification report:")
print(
    classification_report(
        test_labels,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

matrix = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=np.arange(len(CLASS_NAMES)),
)

print("\nConfusion matrix (rows=actual, columns=predicted):")
print("Classes:", CLASS_NAMES)
print(matrix)

print("\nAccuracy by crop age:")

# Evaluate each age group using its corresponding test images.
for age_folder, age_value in AGE_NAMES.items():
    indices = np.where(test_ages[:, 0] == age_value)[0]

    if len(indices) == 0:
        continue

    age_dataset = make_dataset(
        test_paths[indices],
        test_ages[indices],
        test_labels[indices],
    )

    age_predictions = best_model.predict(
        age_dataset,
        verbose=0,
    )

    age_predicted = np.argmax(age_predictions, axis=1)
    age_accuracy = np.mean(
        age_predicted == test_labels[indices]
    )

    print(
        f"{age_folder:12s}: "
        f"{age_accuracy * 100:.2f}% "
        f"({len(indices)} images)"
    )

print("\nV6 training and evaluation completed.")
print(f"Best model saved as: {MODEL_PATH}")

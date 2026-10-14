import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "Dataset_Split/train"
VAL_DIR = "Dataset_Split/validation"
TEST_DIR = "Dataset_Split/test"

MODEL_PATH = "sugarcane_disease_model_v3.keras"

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading training data...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("\nClasses:")
print(class_names)

print("\nNumber of classes:", len(class_names))

# ============================================================
# 3. PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# ============================================================
# 5. MOBILE NET V2
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze MobileNetV2
base_model.trainable = False

# ============================================================
# 6. BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.0001)
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ============================================================
# 7. PHASE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n========================================")
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("========================================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=8,
    callbacks=callbacks_phase1
)

# ============================================================
# 8. PHASE 2 - FINE TUNING
# ============================================================

print("\n========================================")
print("PHASE 2: FINE-TUNING MOBILENETV2")
print("========================================")

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze earlier layers.
# Only fine-tune the later layers.
fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
# This makes fine-tuning more stable on a small dataset.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_layers = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print("\nTrainable MobileNetV2 layers:", trainable_layers)

# Recompile after changing trainable layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks_phase2
)

# ============================================================
# 9. LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")

best_model = tf.keras.models.load_model(MODEL_PATH)

# ============================================================
# 10. FINAL TEST
# ============================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")

test_loss, test_accuracy = best_model.evaluate(
    test_data,
    verbose=1
)

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("Test Loss:", test_loss)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print("\nBest model saved as:")
print(MODEL_PATH)

print("\nTraining completed successfully!")
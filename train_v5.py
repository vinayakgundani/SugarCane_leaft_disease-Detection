import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5

TRAIN_DIR = "Age_Dataset_Split/train"
VAL_DIR = "Age_Dataset_Split/validation"

CLASS_NAMES = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow"
]

AGE_VALUES = {
    "5_months": 5.0,
    "8_months": 8.0,
    "12_months": 12.0
}


# ============================================================
# DATASET LOADER
# ============================================================

def create_dataset(directory, shuffle=True):

    image_paths = []
    labels = []
    ages = []

    # --------------------------------------------------------
    # Collect image paths, disease labels and age
    # --------------------------------------------------------

    for age_folder, age_value in AGE_VALUES.items():

        age_path = os.path.join(
            directory,
            age_folder
        )

        for class_index, class_name in enumerate(CLASS_NAMES):

            class_path = os.path.join(
                age_path,
                class_name
            )

            if not os.path.exists(class_path):

                print(
                    "WARNING: Folder not found:",
                    class_path
                )

                continue

            for filename in os.listdir(class_path):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                ):

                    image_paths.append(
                        os.path.join(
                            class_path,
                            filename
                        )
                    )

                    labels.append(
                        class_index
                    )

                    ages.append(
                        age_value
                    )


    # --------------------------------------------------------
    # Display dataset information
    # --------------------------------------------------------

    print()
    print("Loaded from:", directory)
    print("Total images:", len(image_paths))


    if len(image_paths) == 0:

        raise ValueError(
            f"No images found in {directory}. "
            "Check the folder structure."
        )


    # --------------------------------------------------------
    # Convert to TensorFlow tensors
    # --------------------------------------------------------

    image_paths = tf.constant(
        image_paths
    )

    labels = tf.constant(
        labels,
        dtype=tf.int32
    )

    ages = tf.constant(
        ages,
        dtype=tf.float32
    )


    # --------------------------------------------------------
    # Create TensorFlow dataset
    # --------------------------------------------------------

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
            ages
        )
    )


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True
        )


    # --------------------------------------------------------
    # Load and preprocess image
    # --------------------------------------------------------

    def load_image(path, label, age):

        # Read image file
        image = tf.io.read_file(
            path
        )

        # Decode JPEG/PNG
        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        # Resize to 224 × 224
        image = tf.image.resize(
            image,
            IMG_SIZE
        )

        # Convert to float
        image = tf.cast(
            image,
            tf.float32
        )


        # ----------------------------------------------------
        # Age shape
        #
        # Convert:
        # 5
        #
        # into:
        # [5]
        # ----------------------------------------------------

        age = tf.expand_dims(
            age,
            axis=-1
        )


        # ----------------------------------------------------
        # TWO MODEL INPUTS
        # ----------------------------------------------------

        inputs = {

            "image": image,

            "age": age

        }


        # ----------------------------------------------------
        # Return:
        #
        # inputs → image + age
        # label  → disease
        # ----------------------------------------------------

        return (
            inputs,
            label
        )


    # --------------------------------------------------------
    # Apply image loading
    # --------------------------------------------------------

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    dataset = dataset.batch(
        BATCH_SIZE
    )


    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_ds = create_dataset(
    TRAIN_DIR,
    shuffle=True
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

val_ds = create_dataset(
    VAL_DIR,
    shuffle=False
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# IMAGE INPUT
# ============================================================

image_input = layers.Input(
    shape=(224, 224, 3),
    name="image"
)


# ============================================================
# IMAGE AUGMENTATION
# ============================================================

x = data_augmentation(
    image_input
)


# ============================================================
# EFFICIENTNETB0
# ============================================================

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224, 224, 3)

)


# Initially freeze EfficientNet
base_model.trainable = False


# Extract image features

x = base_model(
    x,
    training=False
)


# Global average pooling

x = layers.GlobalAveragePooling2D()(
    x
)


# Dropout

x = layers.Dropout(
    0.30
)(
    x
)


# Dense image feature layer

x = layers.Dense(

    256,

    activation="relu",

    kernel_regularizer=
    tf.keras.regularizers.l2(
        0.0001
    )

)(
    x
)


# Another dropout

x = layers.Dropout(
    0.20
)(
    x
)


# ============================================================
# AGE INPUT
# ============================================================

age_input = layers.Input(
    shape=(1,),
    name="age"
)


# Normalize age

age = layers.Normalization(
    axis=None,
    name="age_normalization"
)(
    age_input
)


# Age feature extraction

age = layers.Dense(
    32,
    activation="relu"
)(
    age
)


age = layers.Dense(
    16,
    activation="relu"
)(
    age
)


# ============================================================
# COMBINE IMAGE + AGE
# ============================================================

combined = layers.Concatenate(
    name="image_age_combination"
)(

    [
        x,
        age
    ]

)


# Combined feature layer

combined = layers.Dense(

    128,

    activation="relu"

)(
    combined
)


# Dropout

combined = layers.Dropout(
    0.30
)(
    combined
)


# ============================================================
# DISEASE OUTPUT
# ============================================================

output = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    name="disease"

)(
    combined
)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(

    inputs={

        "image": image_input,

        "age": age_input

    },

    outputs=output

)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# PHASE 1
# ============================================================

print()
print("==============================")
print("PHASE 1: TRAINING CLASSIFIER")
print("==============================")


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# PHASE 1 CALLBACKS
# ============================================================

callbacks_phase1 = [

    ModelCheckpoint(

        "sugarcane_disease_model_v5_phase1.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max"

    ),

    EarlyStopping(

        monitor="val_accuracy",

        patience=4,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-6

    )

]


# ============================================================
# PHASE 1 TRAINING
# ============================================================

history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=12,

    callbacks=callbacks_phase1

)


# ============================================================
# PHASE 2
# FINE-TUNING
# ============================================================

print()
print("==============================")
print("PHASE 2: FINE-TUNING")
print("==============================")


# Unfreeze EfficientNet

base_model.trainable = True


# ------------------------------------------------------------
# Freeze all but last 50 layers
# ------------------------------------------------------------

for layer in base_model.layers[:-50]:

    layer.trainable = False


# ------------------------------------------------------------
# Keep BatchNormalization layers frozen
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COMPILE AGAIN WITH SMALL LEARNING RATE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# PHASE 2 CALLBACKS
# ============================================================

callbacks_phase2 = [

    ModelCheckpoint(

        "sugarcane_disease_model_v5.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max"

    ),

    EarlyStopping(

        monitor="val_accuracy",

        patience=5,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7

    )

]


# ============================================================
# PHASE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=20,

    callbacks=callbacks_phase2

)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================")
print("V5 TRAINING COMPLETE")
print("==============================")

print(
    "Model saved as:"
)

print(
    "sugarcane_disease_model_v5.keras"
)
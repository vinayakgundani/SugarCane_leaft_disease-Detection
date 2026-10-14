import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers
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

TRAIN_DIR = "Dataset_Split/train"

VAL_DIR = "Dataset_Split/validation"

MODEL_PATH = "sugarcane_disease_model_v4.keras"


# ============================================================
# LOAD ORIGINAL DISEASE DATASET
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=True,

    seed=42
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(

    VAL_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


print("\nClasses:")

print(
    train_dataset.class_names
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

])


# ============================================================
# IMAGE MODEL
# ============================================================

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224, 224, 3)

)


# Freeze initially

base_model.trainable = False


# ============================================================
# IMAGE INPUT
# ============================================================

image_input = tf.keras.Input(

    shape=(224, 224, 3),

    name="image"
)


x = data_augmentation(
    image_input
)


x = base_model(

    x,

    training=False

)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.30
)(x)


x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=regularizers.l2(0.0001)
)(x)


x = layers.Dropout(
    0.20
)(x)


# ============================================================
# AGE INPUT
# ============================================================

age_input = tf.keras.Input(

    shape=(1,),

    name="age"
)


# Normalize age

age = layers.Normalization(
    axis=None
)(age_input)


age = layers.Dense(
    32,
    activation="relu"
)(age)


age = layers.Dense(
    16,
    activation="relu"
)(age)


# ============================================================
# COMBINE IMAGE + AGE
# ============================================================

combined = layers.Concatenate()([

    x,

    age

])


combined = layers.Dense(

    128,

    activation="relu"

)(combined)


combined = layers.Dropout(
    0.30
)(combined)


# ============================================================
# DISEASE OUTPUT
# ============================================================

output = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    name="disease"

)(combined)


# ============================================================
# COMPLETE MODEL
# ============================================================

model = tf.keras.Model(

    inputs=[

        image_input,

        age_input

    ],

    outputs=output

)


model.summary()


# ============================================================
# IMPORTANT:
# DATASET ONLY CONTAINS IMAGES + DISEASE.
#
# For training we need an age value.
#
# For this experiment we use an age assigned to each
# training sample.
# ============================================================

def add_age_to_dataset(
    dataset
):

    # --------------------------------------------------------
    # For now, use a fixed age value during the baseline
    # training experiment.
    #
    # Later we will replace this with the actual age associated
    # with each image.
    # --------------------------------------------------------

    def transform(
        images,
        labels
    ):

        batch_size = tf.shape(
            images
        )[0]

        ages = tf.fill(

            [batch_size, 1],

            tf.constant(
                8.0
            )

        )

        return (

            {
                "image": images,

                "age": ages

            },

            labels

        )

    return dataset.map(
        transform
    )


train_dataset = add_age_to_dataset(
    train_dataset
)

validation_dataset = add_age_to_dataset(
    validation_dataset
)


# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# PHASE 1
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.001

    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]

)


callbacks = [

    ModelCheckpoint(

        MODEL_PATH,

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


print("\n===================================")

print(
    "PHASE 1"
)

print(
    "Training classification head"
)

print(
    "===================================")


model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=12,

    callbacks=callbacks

)


# ============================================================
# PHASE 2
# FINE-TUNING
# ============================================================

print("\n===================================")

print(
    "PHASE 2"
)

print(
    "Fine-tuning EfficientNetB0"
)

print(
    "===================================")


base_model.trainable = True


# Freeze early layers

for layer in base_model.layers[:-50]:

    layer.trainable = False


# Keep BatchNorm frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]

)


callbacks2 = [

    ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max"

    ),

    EarlyStopping(

        monitor="val_accuracy",

        patience=7,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-7

    )

]


model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=20,

    callbacks=callbacks2

)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model = tf.keras.models.load_model(

    MODEL_PATH

)


print("\n===================================")

print(
    "V4 TRAINING COMPLETE"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "===================================")

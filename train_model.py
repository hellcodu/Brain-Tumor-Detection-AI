import os
import json
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

EPOCHS = 15

DATASET_PATH = "dataset"

MODEL_DIR = "static/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "brain_tumor_mobilenetv2.keras"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"""
Dataset folder not found:

{DATASET_PATH}

Please download the Kaggle dataset and
place the class folders inside:

dataset/
"""
    )


# =========================================================
# TRAINING DATA GENERATOR
# =========================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.2,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1,

    horizontal_flip=True
)


# =========================================================
# VALIDATION DATA GENERATOR
# =========================================================

validation_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    validation_split=0.2
)


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("\nLoading training data...\n")


train_data = train_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True
)


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

print("\nLoading validation data...\n")


validation_data = validation_datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False
)


# =========================================================
# DISPLAY CLASS INFORMATION
# =========================================================

print("\n======================================")

print("CLASS MAPPING")

print("======================================")

print(
    train_data.class_indices
)

print(
    "\nNumber of classes:",
    train_data.num_classes
)

print(
    "Training images:",
    train_data.samples
)

print(
    "Validation images:",
    validation_data.samples
)


# =========================================================
# NUMBER OF CLASSES
# =========================================================

NUM_CLASSES = train_data.num_classes


# =========================================================
# LOAD MOBILENETV2
# =========================================================

print("\nLoading MobileNetV2...\n")


base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )
)


# =========================================================
# FREEZE PRETRAINED MODEL
# =========================================================

base_model.trainable = False


# =========================================================
# CREATE CLASSIFICATION HEAD
# =========================================================

x = base_model.output


x = GlobalAveragePooling2D()(x)


x = Dense(
    128,
    activation="relu"
)(x)


x = Dropout(
    0.4
)(x)


output = Dense(

    NUM_CLASSES,

    activation="softmax"

)(x)


# =========================================================
# CREATE FINAL MODEL
# =========================================================

model = Model(

    inputs=base_model.input,

    outputs=output
)


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\n======================================")

print("MODEL SUMMARY")

print("======================================\n")


model.summary()


# =========================================================
# CALLBACKS
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


callbacks = [

    early_stopping,

    model_checkpoint,

    reduce_lr

]


# =========================================================
# TRAIN MODEL
# =========================================================

print("\n======================================")

print("STARTING TRAINING")

print("======================================\n")


history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=callbacks
)


# =========================================================
# EVALUATE MODEL
# =========================================================

print("\n======================================")

print("EVALUATING MODEL")

print("======================================\n")


loss, accuracy = model.evaluate(

    validation_data,

    verbose=1
)


print("\n======================================")

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: "
    f"{loss:.4f}"
)

print("======================================")


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    MODEL_PATH
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


# =========================================================
# SAVE CLASS NAMES
# =========================================================

class_names = {

    str(index): class_name

    for class_name, index

    in train_data.class_indices.items()

}


CLASS_NAMES_PATH = os.path.join(

    MODEL_DIR,

    "class_names.json"

)


with open(

    CLASS_NAMES_PATH,

    "w"

) as file:

    json.dump(

        class_names,

        file,

        indent=4

    )


print(
    "\nClass names saved to:"
)

print(
    CLASS_NAMES_PATH
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_data = {

    "accuracy": [

        float(x)

        for x in
        history.history["accuracy"]

    ],

    "val_accuracy": [

        float(x)

        for x in
        history.history["val_accuracy"]

    ],

    "loss": [

        float(x)

        for x in
        history.history["loss"]

    ],

    "val_loss": [

        float(x)

        for x in
        history.history["val_loss"]

    ],

    "classes":
        train_data.class_indices,

    "train_images":
        train_data.samples,

    "validation_images":
        validation_data.samples,

    "epochs":
        len(
            history.history["accuracy"]
        )

}


with open(

    HISTORY_PATH,

    "w"

) as file:

    json.dump(

        history_data,

        file,

        indent=4

    )


print(
    "\nTraining history saved to:"
)

print(
    HISTORY_PATH
)


# =========================================================
# FINISHED
# =========================================================

print("\n======================================")

print("TRAINING COMPLETED")

print("======================================")

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nClasses:"
)

print(
    train_data.class_indices
)

print(
    "\nYou can now run:"
)

print(
    "python app.py"
)

print("======================================")
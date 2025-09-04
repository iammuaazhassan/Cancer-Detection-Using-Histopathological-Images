# =============================
# Cancer Detection CNN Training
# =============================

import sys
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# TensorFlow Import Check
# -----------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("❌ TensorFlow is not installed. Run:")
    print("   pip install tensorflow")
    sys.exit(1)

if sys.version_info >= (3, 12):
    print("⚠️ Warning: TensorFlow may not support Python >= 3.12. Use Python 3.10/3.11 instead.")

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -----------------------------
# Dataset Preparation
# -----------------------------
def prepare_dataset(base_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Splits dataset into train and validation folders.
    """
    for category in ['class0', 'class1']:
        source_path = os.path.join(base_dir, category)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"❌ Missing folder: {source_path}")

        files = os.listdir(source_path)
        random.shuffle(files)
        split = int(split_ratio * len(files))
        train_files = files[:split]
        val_files = files[split:]

        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        # Copy only if train/val is empty
        if not os.listdir(os.path.join(train_dir, category)):
            for f in train_files:
                shutil.copy(os.path.join(source_path, f),
                            os.path.join(train_dir, category, f))

        if not os.listdir(os.path.join(val_dir, category)):
            for f in val_files:
                shutil.copy(os.path.join(source_path, f),
                            os.path.join(val_dir, category, f))


# Paths
base_dir = r"D:\UNIVERSITY\INTERNSHIP\WEEK-2\dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# If train/val not already created
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("📂 Preparing dataset...")
    prepare_dataset(base_dir, train_dir, val_dir, split_ratio=0.8)


# -----------------------------
# Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)


# -----------------------------
# Model Definition
# -----------------------------
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3),
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # Binary classification
    ])
    return model


model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]


# -----------------------------
# Training
# -----------------------------
print("🚀 Starting training...")
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks
)





# -----------------------------
# Final Evaluation
# -----------------------------
loss, acc = model.evaluate(val_gen)
print(f"✅ Final Validation Accuracy: {acc * 100:.2f}%")

# Save results to file for GitHub
with open("results.txt", "w") as f:
    f.write(f"Final Validation Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Final Validation Loss: {loss:.4f}\n")





base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

vgg_model = Model(inputs=base_model.input, outputs=predictions)
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
vgg_model.summary()

history_vgg = vgg_model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)


import matplotlib.pyplot as plt

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.legend()

plt.savefig("training_results.png")
plt.show()


# Plot training history for VGG16
plt.figure(figsize=(12,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history_vgg.history['accuracy'], label='train acc')
plt.plot(history_vgg.history['val_accuracy'], label='val acc')
plt.title('VGG16 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history_vgg.history['loss'], label='train loss')
plt.plot(history_vgg.history['val_loss'], label='val loss')
plt.title('VGG16 Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Final evaluation
loss, acc = vgg_model.evaluate(val_gen)
print(f"Validation Accuracy: {acc*100:.2f}%")

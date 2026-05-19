import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# SETTINGS
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_1 = 12          # phase 1
EPOCHS_2 = 8           # phase 2 fine-tune
DATASET_PATH = "dataset"
MODEL_OUT = "best_model.keras"

# ==============================
# DATA (IMPORTANT: use preprocess_input)
# ==============================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class Mapping:", train_data.class_indices)

# ==============================
# MODEL
# ==============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Phase 1: freeze base
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
ckpt = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)

# ==============================
# TRAIN PHASE 1
# ==============================
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_1,
    callbacks=[early_stop, reduce_lr, ckpt]
)

# ==============================
# TRAIN PHASE 2 (Fine-tune last 30 layers)
# ==============================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_2,
    callbacks=[early_stop, reduce_lr, ckpt]
)

# ==============================
# SAVE FINAL MODEL
# ==============================
model.save(MODEL_OUT)
print(f"Saved final model: {MODEL_OUT}")

# ==============================
# PLOT
# ==============================
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

plt.plot(acc)
plt.plot(val_acc)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.show()

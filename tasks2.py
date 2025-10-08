"""
End-to-end TensorFlow image classification (CIFAR-10)

Features:
- Downloads CIFAR-10 via tf.keras.datasets
- Preprocessing + data augmentation
- Builds a flexible CNN (ResNet-like with residual blocks)
- Training with callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping)
- Evaluation and example inference function
- Save & load model
- Optional small Flask app snippet (commented) for serving predictions

Requirements:
- Python 3.8+
- pip install tensorflow flask numpy matplotlib

Run:
python tf_image_classification_endtoend.py

"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------
# Config / Hyperparams
# ----------------------

DEFAULTS = {
    "batch_size": 128,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "model_dir": "saved_model",
    "img_height": 32,
    "img_width": 32,
    "num_classes": 10,
}

# ----------------------
# Helper: Residual block
# ----------------------

def residual_block(x, filters, downsample=False, name=None):
    stride = 2 if downsample else 1
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# ----------------------
# Model builder
# ----------------------

def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)

    # Stem
    x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual stack
    x = residual_block(x, 64, downsample=False)
    x = residual_block(x, 64, downsample=False)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128, downsample=False)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256, downsample=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="simple_resnet_cifar10")
    return model

# ----------------------
# Data: load + augmentation
# ----------------------

def prepare_datasets(batch_size=128, buffer_size=5000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Create tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Augmentation pipeline
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ], name="data_augmentation")

    def _augment(x, y):
        x = tf.cast(x, tf.float32)
        x = data_augmentation(x)
        return x, y

    train_ds = (train_ds.shuffle(buffer_size)
                .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds

# ----------------------
# Training loop
# ----------------------

def train(args):
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    print("Preparing datasets...")
    train_ds, test_ds = prepare_datasets(batch_size=args.batch_size)

    print("Building model...")
    model = build_model(input_shape=(args.img_height, args.img_width, 3), num_classes=args.num_classes)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)

    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=[checkpoint_cb, reduce_lr, early_stop],
    )

    # Save final model (SavedModel format)
    final_path = os.path.join(model_dir, "final_saved_model")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Evaluate
    print("Evaluating on test set...")
    results = model.evaluate(test_ds)
    print("Test results (loss, accuracy):", results)

    return model, history

# ----------------------
# Inference utility
# ----------------------

def predict_image(model, image_array, class_names=None):
    """image_array: HxWxC uint8 or float32 array (single image)
    returns: (predicted_class_index, probability)
    """
    img = tf.cast(image_array, tf.float32)
    if img.ndim == 3:
        img = tf.expand_dims(img, 0)
    img = img / 255.0
    probs = model.predict(img)
    idx = int(tf.argmax(probs[0]))
    prob = float(tf.reduce_max(probs[0]))
    name = class_names[idx] if class_names is not None else str(idx)
    return idx, prob, name

# ----------------------
# CLI
# ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"]) 
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"]) 
    p.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"]) 
    p.add_argument("--model_dir", type=str, default=DEFAULTS["model_dir"]) 
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, history = train(args)

    # Example: load saved model and run a single prediction from the test set
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    sample = x_test[0]
    idx, prob, name = predict_image(model, sample, class_names)
    print(f"Sample predicted: {name} (index {idx}) with probability {prob:.4f}; ground-truth: {class_names[int(y_test[0])]}")


# ----------------------
# Optional: Small Flask app for inference (uncomment and install Flask)
# ----------------------
#
# from flask import Flask, request, jsonify
# import base64
# from PIL import Image
# import io
# 
# app = Flask(__name__)
# loaded_model = None
# 
# def load_model_for_serving(path="saved_model/final_saved_model"):
#     global loaded_model
#     loaded_model = keras.models.load_model(path)
# 
# def read_b64_image(b64_str):
#     decoded = base64.b64decode(b64_str)
#     img = Image.open(io.BytesIO(decoded)).convert('RGB')
#     img = img.resize((32, 32))
#     return np.array(img)
# 
# @app.route('/predict', methods=['POST'])
# def predict_route():
#     data = request.get_json()
#     img_b64 = data.get('image')
#     if img_b64 is None:
#         return jsonify({"error": "no image provided"}), 400
#     img_arr = read_b64_image(img_b64)
#     idx, prob, name = predict_image(loaded_model, img_arr, class_names)
#     return jsonify({"predicted_index": idx, "predicted_label": name, "probability": prob})
# 
# if __name__ == '__main__':
#     load_model_for_serving()
#     app.run(host='0.0.0.0', port=5000)

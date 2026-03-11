
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==========================================
#  CONFIG  — update paths to match your PC
# ==========================================

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20

TRAIN_DIR = r"C:\Users\matru\pp\processed\train"
VAL_DIR   = r"C:\Users\matru\pp\processed\val"
TEST_DIR  = r"C:\Users\matru\pp\processed\test"

MODEL_SAVE_PATH = "hydration_model.h5"
CLASS_NAMES     = ["Dehydrated", "Hydrated"]

# ==========================================
#  DATA GENERATORS
# ==========================================

def create_data_generators(batch_size=None):
    if batch_size is None:
        batch_size = BATCH_SIZE
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=25, zoom_range=0.2,
        horizontal_flip=True, brightness_range=[0.8, 1.2]
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='binary'
    )
    val_data = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='binary'
    )
    test_data = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return train_data, val_data, test_data

# ==========================================
#  MODEL
# ==========================================

def build_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_callbacks():
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    return [checkpoint, early_stop]

def train_model(model, train_data, val_data, callbacks):
    try:
        return model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
    except tf.errors.AbortedError:
        global BATCH_SIZE
        BATCH_SIZE = max(1, BATCH_SIZE // 2)
        train_data, val_data, _ = create_data_generators(batch_size=BATCH_SIZE)
        return model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")

def fine_tune_model(model, train_data, val_data):
    model.layers[0].trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print("\n🔥 Starting Fine Tuning...\n")
    model.fit(train_data, validation_data=val_data, epochs=10)

# ==========================================
#  XAI — LIME ONLY
# ==========================================

def make_predict_fn(model):
    def predict_fn(images):
        preds = model.predict(images, verbose=0)
        return np.hstack([1 - preds, preds])
    return predict_fn

def explain_with_lime(model, image_path, num_samples=1000, num_features=6, save_path=None):
    img     = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = img_to_array(img) / 255.0

    prob       = float(model.predict(np.expand_dims(img_arr, 0), verbose=0)[0][0])
    pred_idx   = int(prob >= 0.5)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob
    print(f"\n🔍 Prediction: {pred_label}  |  Confidence: {confidence:.1%}")

    explainer  = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_arr, make_predict_fn(model), top_labels=2, hide_color=0, num_samples=num_samples
    )
    top_label = explanation.top_labels[0]

    temp_all, mask_all = explanation.get_image_and_mask(top_label, positive_only=False, num_features=num_features, hide_rest=False)
    temp_pos, mask_pos = explanation.get_image_and_mask(top_label, positive_only=True,  num_features=num_features, hide_rest=True)
    temp_neg, mask_neg = explanation.get_image_and_mask(top_label, positive_only=False, negative_only=True, num_features=num_features, hide_rest=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img_arr);                              axes[0].set_title("Original Image");            axes[0].axis("off")
    axes[1].imshow(mark_boundaries(temp_all, mask_all)); axes[1].set_title("All Important Regions");     axes[1].axis("off")
    axes[2].imshow(mark_boundaries(temp_pos, mask_pos)); axes[2].set_title("✅ Supports Prediction");    axes[2].axis("off")
    axes[3].imshow(mark_boundaries(temp_neg, mask_neg)); axes[3].set_title("❌ Contradicts Prediction"); axes[3].axis("off")

    fig.suptitle(f"LIME Explanation  |  Prediction: {pred_label} ({confidence:.1%})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved → {save_path}")
    plt.show()
    return explanation

def explain_batch_lime(model, test_dir, n_samples=5, output_dir="xai_outputs", num_samples=1000, num_features=6):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, f))
    if not image_paths:
        print(f"⚠️  No images found in {test_dir}"); return
    chosen = np.random.choice(image_paths, size=min(n_samples, len(image_paths)), replace=False)
    for i, path in enumerate(chosen):
        print(f"\n[{i+1}/{len(chosen)}] {path}")
        base = os.path.splitext(os.path.basename(path))[0]
        explain_with_lime(model, path, num_samples=num_samples, num_features=num_features,
                          save_path=os.path.join(output_dir, f"{base}_lime.png"))
    print(f"\n✅ All saved to: {output_dir}/")

# ==========================================
#  MAIN
# ==========================================

def main():
    train_data, val_data, test_data = create_data_generators()
    model = build_model()
    callbacks = create_callbacks()
    train_model(model, train_data, val_data, callbacks)
    fine_tune_model(model, train_data, val_data)
    evaluate_model(model, test_data)
    explain_batch_lime(model, test_dir=TEST_DIR, n_samples=5, output_dir="xai_outputs")

def load_and_explain_lime(model_path=MODEL_SAVE_PATH, image_path=None):
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    if image_path:
        explain_with_lime(model, image_path, save_path="lime_result.png")

if __name__ == "__main__":
    main()
    # Already trained? Use this instead:
    # load_and_explain_lime("hydration_model.h5", "path/to/any_image.jpg")
"""Taş-Kağıt-Makas sınıflandırıcısını yerel bilgisayarda eğitir.

Notebook'taki en iyi yaklaşımın (MobileNetV2 + iki aşamalı transfer learning)
Colab'a bağımlı olmayan sürümüdür.

Kullanım:
    python src/train.py --data-dir veri_final_birlesik
    python src/train.py --data-dir veri_gercek --model cnn --epochs 20
"""

import argparse
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 150
CLASS_NAMES = ("paper", "rock", "scissors")


def parse_args():
    p = argparse.ArgumentParser(description="Taş-Kağıt-Makas modelini eğitir.")
    p.add_argument("--data-dir", required=True,
                   help="paper/ rock/ scissors/ alt klasörlerini içeren veri klasörü")
    p.add_argument("--model", choices=["mobilenet", "cnn"], default="mobilenet",
                   help="mobilenet: transfer learning (önerilen), cnn: sıfırdan eğitilen basit CNN")
    p.add_argument("--epochs", type=int, default=10,
                   help="1. aşama epoch sayısı (cnn modunda toplam epoch)")
    p.add_argument("--finetune-epochs", type=int, default=20,
                   help="2. aşama (ince ayar) epoch sayısı, sadece mobilenet için")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="models/rps_model.keras",
                   help="Eğitilen modelin kaydedileceği dosya")
    return p.parse_args()


def load_datasets(data_dir, batch_size):
    """Veriyi %80 eğitim / %20 doğrulama olarak böler."""
    common = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        validation_split=0.2,
        seed=123,
    )
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **common)

    autotune = tf.data.AUTOTUNE
    return (train_ds.cache().prefetch(autotune),
            val_ds.cache().prefetch(autotune))


def augmentation_block():
    """Notebook'taki agresif veri artırmanın Keras katmanı karşılığı."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),      # ~±30 derece
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
    ], name="augmentation")


def build_simple_cnn():
    """Projenin ilk fazındaki sıfırdan eğitilen CNN."""
    return keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        augmentation_block(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="rps_cnn")


def build_mobilenet():
    """MobileNetV2 tabanlı transfer learning modeli."""
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = augmentation_block()(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.Model(inputs, outputs, name="rps_mobilenetv2"), base


def main():
    args = parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(
            f"Veri klasörü bulunamadı: {data_dir}\n"
            "README'deki 'Veri setini hazırlama' adımlarını izleyin."
        )

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")
    train_ds, val_ds = load_datasets(data_dir, args.batch_size)

    metrics = [
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.F1Score(average="macro", name="f1_score"),
    ]
    callbacks = [
        keras.callbacks.ModelCheckpoint(out_path, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2,
                                          patience=3, verbose=1),
    ]

    if args.model == "cnn":
        model = build_simple_cnn()
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)
        model.summary()
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    else:
        model, base = build_mobilenet()
        model.summary()

        # AŞAMA 1: Sadece yeni eklenen sınıflandırıcı katmanları eğitilir.
        print("\n--- AŞAMA 1: Özellik çıkarımı ---")
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=metrics)
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

        # AŞAMA 2: Temel modelin son 40 katmanı çok düşük öğrenme oranıyla açılır.
        print("\n--- AŞAMA 2: İnce ayar (fine-tuning) ---")
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="categorical_crossentropy", metrics=metrics)
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.finetune_epochs, callbacks=callbacks)

    print(f"\nEğitim tamamlandı. En iyi model: {out_path}")


if __name__ == "__main__":
    main()

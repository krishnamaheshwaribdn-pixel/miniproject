import argparse
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path


DATA_DIR = Path(
    os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
) / "ChatbotGesture"
DEFAULT_MODEL_PATH = DATA_DIR / "sign_model.keras"
ISL_CLASSES = set("abcdefghijklmnopqrstuvwxyz")
ASL_CLASSES = {
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "del",
    "nothing",
    "space",
}


def find_training_directory(data_path):
    data_path = data_path.resolve()
    candidates = [data_path]
    if data_path.is_dir():
        candidates.extend(path for path in data_path.iterdir() if path.is_dir())
    for candidate in candidates:
        nested = candidate / "asl_alphabet_train"
        if nested.is_dir():
            candidate = nested
        if not candidate.is_dir():
            continue
        class_directories = {path.name for path in candidate.iterdir() if path.is_dir()}
        if class_directories == ISL_CLASSES or class_directories == ASL_CLASSES:
            return candidate
    raise FileNotFoundError(
        "Could not find an ISL dataset with exactly lowercase a-z folders or an "
        "ASL Alphabet dataset with A-Z, del, nothing, and space folders under "
        f"{data_path}."
    )


def extract_archive(archive_path, extraction_dir):
    extraction_dir.mkdir(parents=True, exist_ok=True)
    marker = extraction_dir / ".asl-alphabet-extracted"
    if marker.exists():
        return

    print(f"Extracting {archive_path} to {extraction_dir}...")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extraction_dir)
    marker.write_text("extracted\n", encoding="utf-8")


def build_model(tf, class_count, image_size):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(image_size, image_size, 3), name="camera_image")
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.02),
            layers.RandomTranslation(0.04, 0.04),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="normalize")(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax", name="sign_probabilities")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    parser = argparse.ArgumentParser(
        description="Train a camera sign classifier using an ISL or ASL alphabet dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="Extracted alphabet dataset folder or the asl_alphabet_train folder.",
    )
    source.add_argument(
        "--archive",
        type=Path,
        help="Downloaded alphabet dataset ZIP archive.",
    )
    parser.add_argument(
        "--webcam-dir",
        type=Path,
        help="Optional webcam sample folder with the same 26 lowercase letter classes.",
    )
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    if args.dataset_dir:
        dataset_path = args.dataset_dir
    else:
        dataset_path = DATA_DIR / "asl-alphabet"
        extract_archive(args.archive, dataset_path)

    training_path = find_training_directory(dataset_path)
    batch_size = args.batch_size
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        training_path,
        validation_split=0.1,
        subset="training",
        seed=2026,
        image_size=(args.image_size, args.image_size),
        batch_size=batch_size,
        label_mode="int",
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        training_path,
        validation_split=0.1,
        subset="validation",
        seed=2026,
        image_size=(args.image_size, args.image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    labels = train_dataset.class_names
    classes = set(labels)
    if classes not in (ISL_CLASSES, ASL_CLASSES):
        raise ValueError(
            "Dataset class folders must be lowercase a-z, or A-Z plus del, "
            "nothing, and space. "
            f"Found: {labels}"
        )
    print(f"Training {len(labels)} classes from {training_path}.")
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    webcam_validation_dataset = None
    webcam_training_path = None
    steps_per_epoch = None
    if args.webcam_dir:
        webcam_training_path = find_training_directory(args.webcam_dir)
        webcam_classes = {
            path.name
            for path in webcam_training_path.iterdir()
            if path.is_dir()
        }
        if webcam_classes != ISL_CLASSES:
            raise ValueError(
                "Webcam samples must have exactly the lowercase a-z folders. "
                f"Found: {sorted(webcam_classes)}"
            )
        webcam_counts = {
            label: sum(
                1
                for path in (webcam_training_path / label).iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for label in "abcdefghijklmnopqrstuvwxyz"
        }
        insufficient_labels = [
            label for label, count in webcam_counts.items() if count < 5
        ]
        if insufficient_labels:
            raise ValueError(
                "Capture at least 5 webcam samples for every letter before "
                "training. Too few samples for: "
                + ", ".join(insufficient_labels)
            )

        webcam_train_dataset = tf.keras.utils.image_dataset_from_directory(
            webcam_training_path,
            validation_split=0.2,
            subset="training",
            seed=2026,
            image_size=(args.image_size, args.image_size),
            batch_size=batch_size,
            label_mode="int",
            class_names=labels,
        )
        webcam_validation_dataset = tf.keras.utils.image_dataset_from_directory(
            webcam_training_path,
            validation_split=0.2,
            subset="validation",
            seed=2026,
            image_size=(args.image_size, args.image_size),
            batch_size=batch_size,
            label_mode="int",
            class_names=labels,
            shuffle=False,
        ).prefetch(tf.data.AUTOTUNE)
        train_count = sum(
            int(count * 0.8)
            for count in webcam_counts.values()
        )
        webcam_batches = webcam_train_dataset.unbatch().repeat()
        isl_batches = train_dataset.unbatch().repeat()
        train_dataset = tf.data.Dataset.sample_from_datasets(
            [isl_batches, webcam_batches],
            weights=[0.9, 0.1],
            seed=2026,
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        isl_train_count = sum(
            1
            for class_directory in training_path.iterdir()
            if class_directory.is_dir()
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        steps_per_epoch = math.ceil(
            max(isl_train_count / 0.9, train_count / 0.1) / batch_size
        )
        print(
            f"Fine-tuning with {sum(webcam_counts.values())} webcam images "
            f"from {webcam_training_path}; reserving 20% for webcam validation."
        )
    else:
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    training_validation_dataset = (
        webcam_validation_dataset
        if webcam_validation_dataset is not None
        else validation_dataset
    )

    model, base = build_model(tf, len(labels), args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    def early_stopping():
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=2,
                mode="max",
                restore_best_weights=True,
            ),
        ]

    model.fit(
        train_dataset,
        validation_data=training_validation_dataset,
        epochs=args.head_epochs,
        callbacks=early_stopping(),
        shuffle=False,
        steps_per_epoch=steps_per_epoch,
    )

    fine_tune_epochs = max(args.fine_tune_epochs, 0)
    if fine_tune_epochs:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_dataset,
            validation_data=training_validation_dataset,
            epochs=fine_tune_epochs,
            callbacks=early_stopping(),
            shuffle=False,
            steps_per_epoch=steps_per_epoch,
        )

    metrics = model.evaluate(validation_dataset, return_dict=True, verbose=1)
    webcam_metrics = None
    if webcam_validation_dataset is not None:
        webcam_metrics = model.evaluate(
            webcam_validation_dataset,
            return_dict=True,
            verbose=1,
        )
    model_path = args.model_out.expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="chatbot-sign-model-",
        dir=model_path.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        candidate_model = temporary_path / model_path.name
        candidate_labels = candidate_model.with_suffix(".labels.json")
        model.save(candidate_model)
        candidate_labels.write_text(
            json.dumps(labels, indent=2) + "\n",
            encoding="utf-8",
        )
        labels_path = model_path.with_suffix(".labels.json")
        candidate_labels.replace(labels_path)
        candidate_model.replace(model_path)

    report = {
        "model": str(model_path),
        "classes": labels,
        "validation": {key: float(value) for key, value in metrics.items()},
        "training_directory": str(training_path),
        "webcam_validation": (
            {key: float(value) for key, value in webcam_metrics.items()}
            if webcam_metrics is not None
            else None
        ),
        "webcam_training_directory": (
            str(webcam_training_path) if webcam_training_path is not None else None
        ),
    }
    report_path = model_path.with_suffix(".training.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved labels: {model_path.with_suffix('.labels.json')}")
    print(f"Validation metrics: {json.dumps(report['validation'])}")
    print(
        "Restart sign_server.py to use the trained model. "
        "Validation accuracy is not a guarantee of camera accuracy."
    )


if __name__ == "__main__":
    main()

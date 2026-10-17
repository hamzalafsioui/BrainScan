import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# Loading
def _clean_keras_config(config):
    """Recursively strip newer Keras parameters that older Keras versions reject."""
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for value in config.values():
            _clean_keras_config(value)
    elif isinstance(config, list):
        for item in config:
            _clean_keras_config(item)


def _load_keras_model_safely(model_path, compile=False):
    
    try:
        return tf.keras.models.load_model(model_path, compile=compile)
    except Exception as exc:
        model_str = str(model_path)
        if model_str.endswith(".keras") and "quantization_config" in str(exc):
            import tempfile
            import zipfile

            with zipfile.ZipFile(model_path, "r") as zin:
                file_contents = {item.filename: zin.read(item.filename) for item in zin.infolist()}

            if "config.json" in file_contents:
                config = json.loads(file_contents["config.json"].decode("utf-8"))
                _clean_keras_config(config)
                file_contents["config.json"] = json.dumps(config).encode("utf-8")

                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                        for filename, content in file_contents.items():
                            zout.writestr(filename, content)
                    return tf.keras.models.load_model(tmp_path, compile=compile)
                finally:
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink(missing_ok=True)
        raise exc


def load_model_and_classes(model_path, class_names_path):
    
    model = _load_keras_model_safely(
        model_path,
        compile=False
    )

    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    # Safety check: model output size must match the class list
    model_outputs = model.output_shape[-1]
    if model_outputs != len(class_names):
        raise ValueError(
            f"Model has {model_outputs} outputs but {len(class_names)} class names."
        )

    print(f"Model loaded: {model_path}")
    print(f"Classes: {class_names}")
    return model, class_names


def load_test_set(data_path):

    # Load only the test split
    with np.load(data_path, allow_pickle=False) as data:
        X_test = data["X_test"].astype("float32")
        y_test = data["y_test"].astype("float32")

    print(f"Test images: {X_test.shape}")
    print(f"Pixel range: {X_test.min():.1f} to {X_test.max():.1f}")
    return X_test, y_test


def load_history(history_path):
    # Load the saved training history

    with open(history_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Learning curves
def plot_learning_curves(history, save_path=None):

    # Plot accuracy and loss for training and validation
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Mark the best epoch (lowest validation loss)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    for ax in axes:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)

    fig.suptitle(f"Learning curves (best epoch: {best_epoch})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


# Predictions and metrics
def get_predictions(model, X_test, y_test, batch_size=32):

    # Return probabilities & predicted labels and true labels
    probabilities = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return probabilities, y_pred, y_true


def print_report(y_true, y_pred, probabilities, class_names):
    # Print accuracy & per-class metrics and ROC-AUC
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Samples : {len(y_true)}")
    print("-" * 55)
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    ))

    # ROC-AUC is only defined for 2 classes here
    if len(class_names) == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
        print(f"ROC-AUC: {auc:.4f}")

    print("=" * 55)
    return accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):

    # Show the confusion matrix with raw counts
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")

    # Write the count inside each cell
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

    return matrix


def print_medical_summary(matrix, class_names, positive_index=1):
    
    # Explain the confusion matrix in medical terms (binary case)
    # positive_index = the index of the 'tumor' class
    
    if matrix.shape != (2, 2):
        print("Medical summary is only available for 2 classes.")
        return

    negative_index = 1 - positive_index

    true_positive = matrix[positive_index, positive_index]
    false_negative = matrix[positive_index, negative_index]
    true_negative = matrix[negative_index, negative_index]
    false_positive = matrix[negative_index, positive_index]

    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)

    print("\n" + "=" * 55)
    print("MEDICAL READING OF THE RESULTS")
    print("=" * 55)
    print(f"Positive class: '{class_names[positive_index]}'")
    print(f"True positives  : {true_positive}  (tumor found correctly)")
    print(f"False negatives : {false_negative}  (MISSED tumor - most dangerous)")
    print(f"True negatives  : {true_negative}  (healthy found correctly)")
    print(f"False positives : {false_positive}  (false alarm)")
    print("-" * 55)
    print(f"Sensitivity (recall): {sensitivity:.4f}  -> ability to find tumors")
    print(f"Specificity         : {specificity:.4f}  -> ability to confirm healthy")
    print("=" * 55)



# Visual error analysis
def show_examples(X_test, y_true, y_pred, probabilities, class_names,
                  correct=True, num_images=8, save_path=None):
    
    # Display correctly or incorrectly classified images
    # Images are stored in BGR (OpenCV) so we flip the channels for display
    
    if correct:
        indices = np.where(y_pred == y_true)[0]
        title = "Correct predictions"
    else:
        indices = np.where(y_pred != y_true)[0]
        title = "Incorrect predictions"

    if len(indices) == 0:
        print(f"No examples found for: {title}")
        return

    selected = indices[:num_images]
    columns = 4
    rows = int(np.ceil(len(selected) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.4 * rows))
    axes = np.array(axes).reshape(-1)

    for position, index in enumerate(selected):
        image_rgb = X_test[index][..., ::-1]   # BGR -> RGB
        confidence = probabilities[index].max()

        axes[position].imshow(np.clip(image_rgb, 0, 1))
        axes[position].set_title(
            f"True: {class_names[y_true[index]]}\n"
            f"Pred: {class_names[y_pred[index]]} ({confidence:.0%})",
            fontsize=10,
            color="green" if correct else "red",
        )
        axes[position].axis("off")

    # Hide unused subplots
    for position in range(len(selected), len(axes)):
        axes[position].axis("off")

    fig.suptitle(f"{title} ({len(indices)} total)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def save_metrics(y_true, y_pred, class_names, accuracy, matrix, save_path):
    
    # Save the metrics to a JSON file for the report
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )

    results = {
        "test_accuracy": float(accuracy),
        "test_samples": int(len(y_true)),
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Metrics saved to: {save_path}")
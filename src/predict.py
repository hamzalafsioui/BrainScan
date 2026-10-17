import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "best_brainscan.keras"
DEFAULT_CLASSES_PATH = PROJECT_DIR / "models" / "class_names.json"


def _clean_keras_config(config):
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for value in config.values():
            _clean_keras_config(value)
    elif isinstance(config, list):
        for item in config:
            _clean_keras_config(item)


def _load_model_safely(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        if str(model_path).endswith(".keras") and "quantization_config" in str(exc):
            import tempfile
            import zipfile

            with zipfile.ZipFile(model_path, "r") as zin:
                files = {item.filename: zin.read(item.filename) for item in zin.infolist()}

            if "config.json" in files:
                cfg = json.loads(files["config.json"].decode("utf-8"))
                _clean_keras_config(cfg)
                files["config.json"] = json.dumps(cfg).encode("utf-8")

                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                    tmp_name = tmp.name

                try:
                    with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                        for fn, content in files.items():
                            zout.writestr(fn, content)
                    return tf.keras.models.load_model(tmp_name, compile=False)
                finally:
                    Path(tmp_name).unlink(missing_ok=True)
        raise exc


def load_model_and_classes(model_path=DEFAULT_MODEL_PATH,
                           class_names_path=DEFAULT_CLASSES_PATH):
   

    model_path = Path(model_path)
    class_names_path = Path(class_names_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_path}"
        )

    # compile=False is enough for prediction
    model = _load_model_safely(model_path)

    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model outputs does not match the class names."
        )

    return model, class_names


def prepare_image(image, model):
   

    if image is None:
        raise ValueError("The image could not be loaded.")

    # Model input shape: (None, height, width, channels)
    image_height = model.input_shape[1]
    image_width = model.input_shape[2]

    # OpenCV resize expects: (width, height)
    image = cv2.resize(image, (image_width, image_height))

    # Convert pixel values from [0, 255] to [0, 1]
    image = image.astype("float32") / 255.0

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    return image


def predict_bgr_image(image, model, class_names):

    # Predict an image already loaded with OpenCV and return a dictionary containing the predicted class

    prepared_image = prepare_image(image, model)

    probabilities = model.predict(
        prepared_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_by_class = {
        class_names[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }

    return {
        "class_index": predicted_index,
        "class_name": predicted_label,
        "confidence": confidence,
        "probabilities": probability_by_class,
    }


def predict_image(path,
                  model_path=DEFAULT_MODEL_PATH,
                  class_names_path=DEFAULT_CLASSES_PATH):
    
    # Predict an image using its file path.

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    model, class_names = load_model_and_classes(
        model_path,
        class_names_path
    )

    # IMREAD_COLOR always returns a three-channel BGR image
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    return predict_bgr_image(image, model, class_names)


def predict_image_bytes(image_bytes, model, class_names):

    # Predict an image received as bytes & this functions is used by streamlit without creating a temporary file

    byte_array = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(
        byte_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Invalid or corrupted image. Please upload a valid image."
        )

    return predict_bgr_image(image, model, class_names)


def main():
    # Run prediction from the terminal

    parser = argparse.ArgumentParser(
        description="Predict a brain MRI image."
    )
    parser.add_argument(
        "image",
        help="Path to the image"
    )

    args = parser.parse_args()

    result = predict_image(args.image)

    print("\nPrediction result")
    print("-" * 40)
    print(f"Class      : {result['class_name']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nProbabilities:")

    for class_name, probability in result["probabilities"].items():
        print(f"- {class_name}: {probability:.2%}")


if __name__ == "__main__":
    main()
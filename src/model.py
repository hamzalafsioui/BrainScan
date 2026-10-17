from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense, Rescaling
)
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape=(224,224,3),num_classes=2,learning_rate=1e-4):

    """
        - Conv2D : detect patterns (edges,shapes,tumors)
        - BatchNorm : stabilize + speed up training
        - MaxPool : reduce size, keep important info
        - Dropout : prevent overfitting
        - Dense : take the final decision

        - ReLu for hidden layers (fast,avoids vanishing gradient)
        - Softmax for output layer (gives probabilities that sum to 1) 
    """

    model = Sequential([
        Input(shape=input_shape),

        # --- Block 1: 32 filters ---
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),              # 224 -> 112

        # --- Block 2: 64 filters ---
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),              # 112 -> 56

        # --- Block 3: 128 filters ---
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),              # 56 -> 28
        Dropout(0.25),

        # --- Block 4: 128 filters ---
        Conv2D(128, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),              # 28 -> 14
        Dropout(0.25),

        # --- Classifier head ---
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),                      # strong dropout before output
        Dense(num_classes, activation="softmax")
    ], name="BrainScan_CNN")

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def show_architecture(model, save_path="../models/model_architecture.png"):
    model.summary()

    try:
        import os
        os.environ["PATH"] += os.pathsep + r"C:\Program Files (x86)\Graphviz\bin"

        from tensorflow.keras.utils import plot_model

        plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            dpi=70
        )

        print(f"Architecture saved to: {save_path}")

    except Exception as e:
        print(f"Could not generate architecture diagram: {e}")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical


RANDOM_SEED = 42
def shuffle_data(X,y,seed=RANDOM_SEED):

    indices  =np.arange(len(X))

    # shuffle the indices
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    print(f"data shuffled ({len(X)} samples)")

    # apply the same order

    return X[indices],y[indices]

def encode_labels(y_text,num_classes=None):

    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y_text) # 'no' -> 0, 'yes' -> 1
    y_onehot = to_categorical(y_int,num_classes) # 0 -> [1,0]

    print(f"labels encoded: {list(encoder.classes_)}")
    print(f"shape: {y_onehot.shape}")


    return y_onehot,encoder



def split_data(X,y,test_size=0.2,val_size=0.1,seed=RANDOM_SEED):

    # separate TEST set
    X_temp,X_test,y_temp,y_test = train_test_split(
        X,y,test_size=test_size,random_state=seed,stratify=y.argmax(axis=1) if y.ndim>1 else y
    )

    # Separate VALIDATION set from the rest
    val_ratio = val_size / (1-test_size)
    X_train,X_val,y_train,y_val = train_test_split(
        X_temp,y_temp,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_temp.argmax(axis=1) if y_temp.ndim > 1 else y_temp
    )

    print("\nData split:")
    print(f"  Train      : {len(X_train)} images")
    print(f"  Validation : {len(X_val)} images")
    print(f"  Test       : {len(X_test)} images")

    return X_train, X_val, X_test,y_train,y_val,y_test


def normalize(X):
    X_norm = X.astype("float32")/255.0
    print(f"Normalized => min={X_norm.min():.1f}, max={X_norm.max():.1f}")

    return X_norm

def get_class_weights(y_int):

    classes = np.unique(y_int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y = y_int
    )

    weight_dict = {int(c):float(w) for c,w in zip(classes,weights)}

    print(f"class weights: {weight_dict}")

    return weight_dict
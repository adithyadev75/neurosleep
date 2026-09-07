from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.models.cnn import build_cnn


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data") / "processed"

X_FILE = DATA_DIR / "X_multisubject.npy"
Y_FILE = DATA_DIR / "y_multisubject.npy"
SUBJECT_FILE = DATA_DIR / "subjects_multisubject.npy"

MODEL_DIR = Path("results") / "models"
METRICS_DIR = Path("results") / "metrics"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 30
BATCH_SIZE = 32


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n========== LOADING MULTI-SUBJECT DATASET ==========")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    subjects = np.load(SUBJECT_FILE)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"subjects shape: {subjects.shape}")

    print(f"Positive windows: {np.sum(y == 1)}")
    print(f"Negative windows: {np.sum(y == 0)}")

    unique_subjects = np.unique(subjects)

    print(f"Number of subjects: {len(unique_subjects)}")
    print(f"Subjects: {unique_subjects}")

    return X, y, subjects


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================
def subject_wise_split(X, y, subjects):

    print("\n========== SUBJECT-WISE SPLIT ==========")

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    # --------------------------------------------------------
    # 18 subjects -> training
    # 3 subjects  -> validation
    # 4 subjects  -> testing
    # --------------------------------------------------------

    train_subjects = shuffled_subjects[:18]

    val_subjects = shuffled_subjects[18:21]

    test_subjects = shuffled_subjects[21:25]

    train_mask = np.isin(
        subjects,
        train_subjects
    )

    val_mask = np.isin(
        subjects,
        val_subjects
    )

    test_mask = np.isin(
        subjects,
        test_subjects
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\nTraining subjects:")
    print(train_subjects)

    print("\nValidation subjects:")
    print(val_subjects)

    print("\nTest subjects:")
    print(test_subjects)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_subjects,
        val_subjects,
        test_subjects,
    )

# ============================================================
# SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    name,
    X,
    y
):

    positive = np.sum(y == 1)
    negative = np.sum(y == 0)

    print(
        f"{name}: "
        f"{len(y)} windows | "
        f"positive={positive} | "
        f"negative={negative}"
    )

    if len(y) > 0:

        print(
            f"         positive ratio="
            f"{positive / len(y) * 100:.2f}%"
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    X_train,
    X_val,
    X_test
):

    print("\n========== NORMALIZATION ==========")

    # IMPORTANT:
    # Calculate normalization statistics ONLY
    # from the training subjects.

    mean = np.mean(X_train)
    std = np.std(X_train)

    print(f"Training mean: {mean:.6f}")
    print(f"Training std:  {std:.6f}")

    X_train = (
        X_train - mean
    ) / (std + 1e-8)

    X_val = (
        X_val - mean
    ) / (std + 1e-8)

    X_test = (
        X_test - mean
    ) / (std + 1e-8)

    return (
        X_train,
        X_val,
        X_test
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y_train):

    positive_count = np.sum(
        y_train == 1
    )

    negative_count = np.sum(
        y_train == 0
    )

    total = (
        positive_count
        + negative_count
    )

    class_weight = {

        0: total / (
            2 * negative_count
        ),

        1: total / (
            2 * positive_count
        ),
    }

    return class_weight


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, subjects = load_data()

    # --------------------------------------------------------
    # Subject-wise split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_subjects,
        val_subjects,
        test_subjects,
    ) = subject_wise_split(
        X,
        y,
        subjects
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n========== DATASET SPLIT STATISTICS ==========")

    print_split_statistics(
        "Training",
        X_train,
        y_train
    )

    print_split_statistics(
        "Validation",
        X_val,
        y_val
    )

    print_split_statistics(
        "Test",
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = normalize_data(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # CNN input format
    #
    # samples × time × channels
    # --------------------------------------------------------

    X_train = X_train[
        ..., np.newaxis
    ]

    X_val = X_val[
        ..., np.newaxis
    ]

    X_test = X_test[
        ..., np.newaxis
    ]

    print(
        "\n========== FINAL INPUT SHAPES =========="
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    # --------------------------------------------------------
    # Build CNN
    # --------------------------------------------------------

    print(
        "\n========== BUILDING CNN =========="
    )

    model = build_cnn(
        input_shape=(
            X_train.shape[1],
            1
        )
    )

    model.summary()

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weight = calculate_class_weights(
        y_train
    )

    print(
        "\n========== CLASS WEIGHTS =========="
    )

    print(class_weight)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    best_model_path = (
    MODEL_DIR
    / "best_sleep_cnn_no_classweight.keras"
)

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\n========== TRAINING =========="
    )

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        

        callbacks=callbacks,

        verbose=1,

        shuffle=True
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print(
        "\n========== TEST EVALUATION =========="
    )

    test_results = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    for name, value in zip(
        model.metrics_names,
        test_results
    ):

        print(
            f"{name}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print(
        "\n========== CONFUSION MATRIX =========="
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(cm)

    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print(
        "        Pred 0   Pred 1"
    )

    print(
        f"Actual 0   {cm[0,0]:5d}    {cm[0,1]:5d}"
    )

    print(
        f"Actual 1   {cm[1,0]:5d}    {cm[1,1]:5d}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "No respiratory event",
                "Respiratory event"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # ROC-AUC and PR-AUC
    # --------------------------------------------------------

    if len(np.unique(y_test)) == 2:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.4f}"
        )

    else:

        print(
            "\nROC-AUC / PR-AUC cannot be "
            "calculated because the test set "
            "contains only one class."
        )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
    MODEL_DIR
    / "final_sleep_cnn_no_classweight.keras"
)

    model.save(
        final_model_path
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        final_model_path
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========== TRAINING COMPLETE =========="
    )

    print(
        f"Training subjects:   {train_subjects}"
    )

    print(
        f"Validation subject:  {val_subjects}"
    )

    print(
        f"Test subject:        {test_subjects}"
    )


if __name__ == "__main__":
    main()
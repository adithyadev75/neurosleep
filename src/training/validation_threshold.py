from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODEL_FILE = (
    Path("results")
    / "models"
    / "final_sleep_cnn_multisubject.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n========== LOADING DATA ==========")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    subjects = np.load(SUBJECT_FILE)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"subjects shape: {subjects.shape}")

    return X, y, subjects


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

def subject_wise_split(
    X,
    y,
    subjects
):

    print(
        "\n========== SUBJECT-WISE SPLIT =========="
    )

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    train_subjects = shuffled_subjects[:18]
    val_subjects = shuffled_subjects[18:21]
    test_subjects = shuffled_subjects[21:25]

    print(
        f"Training subjects:   {train_subjects}"
    )

    print(
        f"Validation subjects: {val_subjects}"
    )

    print(
        f"Test subjects:       {test_subjects}"
    )

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

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(
    X_train,
    X_val,
    X_test
):

    print(
        "\n========== NORMALIZATION =========="
    )

    mean = np.mean(X_train)
    std = np.std(X_train)

    print(
        f"Training mean: {mean:.6f}"
    )

    print(
        f"Training std:  {std:.6f}"
    )

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
# THRESHOLD METRICS
# ============================================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# FIND BEST VALIDATION THRESHOLD
# ============================================================

def find_best_threshold(
    y_val,
    val_probabilities
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VALIDATION THRESHOLD ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )

    print(
        "-" * 55
    )

    results = []

    for threshold in THRESHOLDS:

        (
            accuracy,
            precision,
            recall,
            f1
        ) = evaluate_threshold(
            y_val,
            val_probabilities,
            threshold
        )

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1
            )
        )

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

    best = max(
        results,
        key=lambda x: x[4]
    )

    (
        best_threshold,
        best_accuracy,
        best_precision,
        best_recall,
        best_f1
    ) = best

    print(
        "\n========== BEST VALIDATION THRESHOLD =========="
    )

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Accuracy:  {best_accuracy:.4f}"
    )

    print(
        f"Precision: {best_precision:.4f}"
    )

    print(
        f"Recall:    {best_recall:.4f}"
    )

    print(
        f"F1:        {best_f1:.4f}"
    )

    return best_threshold


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(
    y_test,
    probabilities,
    threshold
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 80
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(
        "\n========== CONFUSION MATRIX =========="
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


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, subjects = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = subject_wise_split(
        X,
        y,
        subjects
    )

    print(
        "\n========== SPLIT SIZES =========="
    )

    print(
        f"Training:   {len(y_train)}"
    )

    print(
        f"Validation: {len(y_val)}"
    )

    print(
        f"Test:       {len(y_test)}"
    )

    # --------------------------------------------------------
    # Normalize
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
    # Add channel dimension
    # --------------------------------------------------------

    X_val = X_val[
        ..., np.newaxis
    ]

    X_test = X_test[
        ..., np.newaxis
    ]

    print(
        f"\nValidation input: {X_val.shape}"
    )

    print(
        f"Test input:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print(
        "\n========== LOADING TRAINED CNN =========="
    )

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    print(
        f"Model loaded from: {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    print(
        "\n========== VALIDATION PREDICTIONS =========="
    )

    val_probabilities = model.predict(
        X_val,
        verbose=1
    ).ravel()

    print(
        f"Minimum probability: "
        f"{val_probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: "
        f"{val_probabilities.max():.6f}"
    )

    print(
        f"Mean probability: "
        f"{val_probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Find threshold ONLY using validation
    # --------------------------------------------------------

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    print(
        "\n========== TEST PREDICTIONS =========="
    )

    test_probabilities = model.predict(
        X_test,
        verbose=1
    ).ravel()

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    evaluate_test(
        y_test,
        test_probabilities,
        best_threshold
    )

    print(
        "\n========== COMPLETE =========="
    )

    print(
        "Threshold was selected using VALIDATION data."
    )

    print(
        "TEST data was used only for final evaluation."
    )


if __name__ == "__main__":
    main()
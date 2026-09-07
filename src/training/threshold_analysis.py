from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


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
# SAME SUBJECT-WISE SPLIT AS TRAINING
# ============================================================

def subject_wise_split(X, y, subjects):

    print("\n========== SUBJECT-WISE SPLIT ==========")

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    train_subjects = shuffled_subjects[:18]
    val_subjects = shuffled_subjects[18:21]
    test_subjects = shuffled_subjects[21:25]

    test_mask = np.isin(
        subjects,
        test_subjects
    )

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Training subjects:   {train_subjects}")
    print(f"Validation subjects: {val_subjects}")
    print(f"Test subjects:       {test_subjects}")

    print(f"\nTest windows: {len(y_test)}")
    print(
        f"Test positives: {np.sum(y_test == 1)}"
    )
    print(
        f"Test negatives: {np.sum(y_test == 0)}"
    )

    return X_test, y_test


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_test_data(
    X,
    subjects,
    X_test
):

    # Reproduce the SAME training split
    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    train_subjects = shuffled_subjects[:18]

    train_mask = np.isin(
        subjects,
        train_subjects
    )

    X_train = X[train_mask]

    # IMPORTANT:
    # Statistics come ONLY from training subjects.

    mean = np.mean(X_train)
    std = np.std(X_train)

    print("\n========== NORMALIZATION ==========")
    print(f"Training mean: {mean:.6f}")
    print(f"Training std:  {std:.6f}")

    X_test = (
        X_test - mean
    ) / (std + 1e-8)

    return X_test


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    y_true,
    probabilities
):

    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1"
    )

    print("-" * 55)

    results = []

    for threshold in thresholds:

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

    return results


# ============================================================
# BEST THRESHOLDS
# ============================================================

def print_best_thresholds(
    results
):

    print(
        "\n========== BEST THRESHOLDS =========="
    )

    best_f1 = max(
        results,
        key=lambda x: x[4]
    )

    best_recall = max(
        results,
        key=lambda x: x[3]
    )

    best_precision = max(
        results,
        key=lambda x: x[2]
    )

    print(
        "\nBest F1:"
    )

    print(
        f"Threshold: {best_f1[0]:.2f}"
    )

    print(
        f"Accuracy:  {best_f1[1]:.4f}"
    )

    print(
        f"Precision: {best_f1[2]:.4f}"
    )

    print(
        f"Recall:    {best_f1[3]:.4f}"
    )

    print(
        f"F1:        {best_f1[4]:.4f}"
    )

    print(
        "\nBest Recall:"
    )

    print(
        f"Threshold: {best_recall[0]:.2f}"
    )

    print(
        f"Accuracy:  {best_recall[1]:.4f}"
    )

    print(
        f"Precision: {best_recall[2]:.4f}"
    )

    print(
        f"Recall:    {best_recall[3]:.4f}"
    )

    print(
        f"F1:        {best_recall[4]:.4f}"
    )

    print(
        "\nBest Precision:"
    )

    print(
        f"Threshold: {best_precision[0]:.2f}"
    )

    print(
        f"Accuracy:  {best_precision[1]:.4f}"
    )

    print(
        f"Precision: {best_precision[2]:.4f}"
    )

    print(
        f"Recall:    {best_precision[3]:.4f}"
    )

    print(
        f"F1:        {best_precision[4]:.4f}"
    )


# ============================================================
# CONFUSION MATRIX AT SELECTED THRESHOLD
# ============================================================

def show_confusion_matrix(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print(
        f"\n========== CONFUSION MATRIX "
        f"AT THRESHOLD {threshold:.2f} =========="
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
        "\nClassification metrics:"
    )

    print(
        f"Accuracy:  "
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1:        "
        f"{f1_score(y_true, predictions, zero_division=0):.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y, subjects = load_data()

    # --------------------------------------------------------
    # Obtain SAME test set
    # --------------------------------------------------------

    X_test, y_test = subject_wise_split(
        X,
        y,
        subjects
    )

    # --------------------------------------------------------
    # Normalize using training statistics
    # --------------------------------------------------------

    X_test = normalize_test_data(
        X,
        subjects,
        X_test
    )

    # CNN input:
    # samples × time × channels

    X_test = X_test[
        ..., np.newaxis
    ]

    print(
        f"\nFinal test input shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Load trained CNN
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
    # Generate probabilities
    # --------------------------------------------------------

    print(
        "\n========== GENERATING PREDICTIONS =========="
    )

    probabilities = model.predict(
        X_test,
        verbose=1
    ).ravel()

    print(
        f"Minimum probability: {probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: {probabilities.max():.6f}"
    )

    print(
        f"Mean probability:    {probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Overall ranking metrics
    # --------------------------------------------------------

    print(
        "\n========== OVERALL MODEL PERFORMANCE =========="
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC:  "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    results = analyze_thresholds(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Best thresholds
    # --------------------------------------------------------

    print_best_thresholds(
        results
    )

    # --------------------------------------------------------
    # Show confusion matrix at 0.50
    # --------------------------------------------------------

    show_confusion_matrix(
        y_test,
        probabilities,
        0.50
    )

    # --------------------------------------------------------
    # Show confusion matrix at best F1
    # --------------------------------------------------------

    best_f1 = max(
        results,
        key=lambda x: x[4]
    )

    show_confusion_matrix(
        y_test,
        probabilities,
        best_f1[0]
    )


if __name__ == "__main__":
    main()
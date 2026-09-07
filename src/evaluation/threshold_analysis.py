from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODEL_PATH = (
    Path("results")
    / "models"
    / "final_sleep_cnn_multisubject.keras"
)

OUTPUT_DIR = (
    Path("results")
    / "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

THRESHOLDS = np.arange(
    0.05,
    0.91,
    0.05
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\n========== LOADING DATASET =========="
    )

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    subjects = np.load(SUBJECT_FILE)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"subjects shape: {subjects.shape}")

    return X, y, subjects


# ============================================================
# REPRODUCE THE EXACT SUBJECT SPLIT
# ============================================================

def subject_wise_split(
    X,
    y,
    subjects
):

    print(
        "\n========== REPRODUCING SUBJECT SPLIT =========="
    )

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()

    rng.shuffle(
        shuffled_subjects
    )

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

    print(
        f"Training subjects: {train_subjects}"
    )

    print(
        f"Validation subjects: {val_subjects}"
    )

    print(
        f"Test subjects: {test_subjects}"
    )

    print(
        f"\nTest samples: {len(y_test)}"
    )

    print(
        f"Test positive: {np.sum(y_test == 1)}"
    )

    print(
        f"Test negative: {np.sum(y_test == 0)}"
    )

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
        f"Training std: {std:.6f}"
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
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
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

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

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

    if (tn + fp) > 0:

        specificity = (
            tn / (tn + fp)
        )

    else:

        specificity = 0.0

    balanced_accuracy = (
        recall + specificity
    ) / 2

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, subjects = load_data()

    # --------------------------------------------------------
    # Reproduce exact split
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

    # --------------------------------------------------------
    # Normalize exactly like training
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

    X_test = X_test[
        ..., np.newaxis
    ]

    print(
        "\n========== TEST INPUT =========="
    )

    print(
        f"X_test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print(
        "\n========== LOADING TRAINED MODEL =========="
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Loaded model: {MODEL_PATH}"
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
        f"Minimum probability: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum probability: "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Mean probability:    "
        f"{probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Overall ranking performance
    # --------------------------------------------------------

    print(
        "\n========== OVERALL MODEL PERFORMANCE =========="
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
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        "Threshold | Accuracy | Precision | Recall | "
        "Specificity | F1 | Balanced Acc."
    )

    print(
        "-" * 100
    )

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_test,
            probabilities,
            threshold
        )

        results.append(metrics)

        print(
            f"{threshold:9.2f} | "
            f"{metrics['accuracy']:8.4f} | "
            f"{metrics['precision']:9.4f} | "
            f"{metrics['recall']:6.4f} | "
            f"{metrics['specificity']:11.4f} | "
            f"{metrics['f1']:5.4f} | "
            f"{metrics['balanced_accuracy']:13.4f}"
        )

    # --------------------------------------------------------
    # Find best thresholds
    # --------------------------------------------------------

    best_f1 = max(
        results,
        key=lambda x: x["f1"]
    )

    best_recall = max(
        results,
        key=lambda x: x["recall"]
    )

    best_balanced = max(
        results,
        key=lambda x: x["balanced_accuracy"]
    )

    best_precision = max(
        results,
        key=lambda x: x["precision"]
    )

    # --------------------------------------------------------
    # Print best thresholds
    # --------------------------------------------------------

    print(
        "\n========== BEST THRESHOLDS =========="
    )

    print(
        "\nBest F1:"
    )

    print(
        f"Threshold:        {best_f1['threshold']:.2f}"
    )

    print(
        f"Accuracy:         {best_f1['accuracy']:.4f}"
    )

    print(
        f"Precision:        {best_f1['precision']:.4f}"
    )

    print(
        f"Recall:           {best_f1['recall']:.4f}"
    )

    print(
        f"Specificity:      {best_f1['specificity']:.4f}"
    )

    print(
        f"F1:               {best_f1['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:{best_f1['balanced_accuracy']:.4f}"
    )

    print(
        "\nBest Recall:"
    )

    print(
        f"Threshold: {best_recall['threshold']:.2f}"
    )

    print(
        f"Precision: {best_recall['precision']:.4f}"
    )

    print(
        f"Recall:    {best_recall['recall']:.4f}"
    )

    print(
        f"F1:        {best_recall['f1']:.4f}"
    )

    print(
        "\nBest Balanced Accuracy:"
    )

    print(
        f"Threshold:        {best_balanced['threshold']:.2f}"
    )

    print(
        f"Accuracy:         {best_balanced['accuracy']:.4f}"
    )

    print(
        f"Precision:        {best_balanced['precision']:.4f}"
    )

    print(
        f"Recall:           {best_balanced['recall']:.4f}"
    )

    print(
        f"Specificity:      {best_balanced['specificity']:.4f}"
    )

    print(
        f"F1:               {best_balanced['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:"
        f"{best_balanced['balanced_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    for name, result in [
        ("0.50", calculate_metrics(
            y_test,
            probabilities,
            0.50
        )),
        (
            "BEST F1",
            best_f1
        ),
        (
            "BEST BALANCED ACCURACY",
            best_balanced
        ),
    ]:

        print(
            "\n" + "=" * 80
        )

        print(
            f"CONFUSION MATRIX — {name}"
        )

        print(
            "=" * 80
        )

        print(
            "[[TN  FP]"
        )

        print(
            " [FN  TP]]"
        )

        print(
            f"[[{result['tn']:4d} {result['fp']:4d}]"
        )

        print(
            f" [{result['fn']:4d} {result['tp']:4d}]]"
        )

        print(
            f"\nAccuracy:          "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Precision:         "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall:            "
            f"{result['recall']:.4f}"
        )

        print(
            f"Specificity:       "
            f"{result['specificity']:.4f}"
        )

        print(
            f"F1:                "
            f"{result['f1']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{result['balanced_accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    import csv

    output_file = (
        OUTPUT_DIR
        / "threshold_analysis_multisubject.csv"
    )

    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\nThreshold results saved to:"
    )

    print(
        output_file
    )

    print(
        "\n========== THRESHOLD ANALYSIS COMPLETE =========="
    )


if __name__ == "__main__":
    main()
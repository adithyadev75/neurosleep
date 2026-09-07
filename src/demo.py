"""
NeuroSleep - Trained CNN Demonstration

Run from the project root with:

    python -m src.demo

Optional:

    python -m src.demo --subject 200001 --start 3510.90

The demo:
1. Loads the saved CNN.
2. Loads an SHHS EDF.
3. Applies the same EEG preprocessing used during dataset creation.
4. Extracts one 30-second EEG window (3750 samples at 125 Hz).
5. Reproduces the training normalization statistics.
6. Produces a respiratory-event prediction.
7. Prints the prediction and confidence.
"""

from pathlib import Path
import argparse

import mne
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

EDF_DIR = Path(
    "data/raw/shhs/polysomnography/edfs/shhs1"
)

DATA_DIR = Path("data/processed")

X_FILE = DATA_DIR / "X_multisubject.npy"
SUBJECT_FILE = DATA_DIR / "subjects_multisubject.npy"

MODEL_FILE = (
    Path("results")
    / "models"
    / "final_sleep_cnn_multisubject.keras"
)


# ============================================================
# CONFIGURATION
# ============================================================

EEG_CHANNEL = "EEG"
SAMPLING_RATE = 125
WINDOW_SECONDS = 30
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS

SEED = 42

# The saved training pipeline used:
# 18 subjects for training, 3 for validation, 4 for testing.
TRAIN_SUBJECT_COUNT = 18


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run a live demonstration using the trained NeuroSleep CNN."
    )

    parser.add_argument(
        "--subject",
        default="200001",
        help="SHHS subject ID, e.g. 200001"
    )

    parser.add_argument(
        "--start",
        type=float,
        default=3510.90,
        help="Start time of the 30-second EEG window in seconds"
    )

    return parser.parse_args()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_for_demo(raw):

    """
    Use the same preprocessing function used by the dataset pipeline.
    """

    from src.data.preprocessing import preprocess_eeg

    raw = raw.copy()
    raw.pick([EEG_CHANNEL])

    return preprocess_eeg(raw)


# ============================================================
# REPRODUCE TRAINING NORMALIZATION
# ============================================================

def get_training_normalization():

    """
    Reproduce the normalization used by train.py.

    train.py calculates:

        mean = mean(X_train)
        std  = std(X_train)

    where X_train consists only of the training subjects.
    """

    if not X_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{X_FILE}"
        )

    if not SUBJECT_FILE.exists():
        raise FileNotFoundError(
            f"Subject mapping not found:\n{SUBJECT_FILE}"
        )

    X = np.load(X_FILE)
    subjects = np.load(SUBJECT_FILE)

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(SEED)

    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    train_subjects = shuffled_subjects[:TRAIN_SUBJECT_COUNT]

    train_mask = np.isin(
        subjects,
        train_subjects
    )

    X_train = X[train_mask]

    mean = np.mean(X_train)
    std = np.std(X_train)

    return mean, std, train_subjects


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_FILE}"
        )

    print("\n========== LOADING TRAINED MODEL ==========")
    print(f"Model: {MODEL_FILE}")

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD AND PREPROCESS EEG
# ============================================================

def load_eeg(subject_id):

    edf_file = (
        EDF_DIR
        / f"shhs1-{subject_id}.edf"
    )

    if not edf_file.exists():
        raise FileNotFoundError(
            f"EDF file not found:\n{edf_file}"
        )

    print("\n========== LOADING EEG ==========")
    print(f"Subject: {subject_id}")
    print(f"EDF: {edf_file}")

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=False,
        verbose=False
    )

    if EEG_CHANNEL not in raw.ch_names:
        raise ValueError(
            f"Channel '{EEG_CHANNEL}' not found.\n"
            f"Available channels: {raw.ch_names}"
        )

    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1] / 3600:.2f} hours")

    print("\n========== PREPROCESSING ==========")
    print("Band-pass filter: 0.3 - 35.0 Hz")

    processed = preprocess_for_demo(raw)

    print("Preprocessing complete.")

    return processed


# ============================================================
# EXTRACT WINDOW
# ============================================================

def extract_window(processed, start_time):

    sfreq = processed.info["sfreq"]

    if abs(sfreq - SAMPLING_RATE) > 1e-6:
        raise ValueError(
            f"Expected {SAMPLING_RATE} Hz but found {sfreq} Hz."
        )

    end_time = start_time + WINDOW_SECONDS

    duration = processed.times[-1]

    if start_time < 0:
        raise ValueError("Start time cannot be negative.")

    if end_time > duration:
        raise ValueError(
            f"Window ends at {end_time:.2f}s, "
            f"but recording ends at {duration:.2f}s."
        )

    start_sample = int(
        round(start_time * sfreq)
    )

    end_sample = (
        start_sample + WINDOW_SAMPLES
    )

    signal = processed.get_data(
        start=start_sample,
        stop=end_sample
    )[0]

    if len(signal) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"got {len(signal)}."
        )

    print("\n========== EEG WINDOW ==========")
    print(f"Start: {start_time:.2f} s")
    print(f"End:   {end_time:.2f} s")
    print(f"Duration: {WINDOW_SECONDS} s")
    print(f"Samples: {len(signal)}")

    return signal


# ============================================================
# PREDICTION
# ============================================================

def predict(model, signal, mean, std):

    print("\n========== NORMALIZATION ==========")
    print(f"Training mean: {mean:.6f}")
    print(f"Training std:  {std:.6f}")

    signal = (
        signal - mean
    ) / (std + 1e-8)

    # CNN input:
    # samples x time x channels
    X = signal.astype(
        np.float32
    )[np.newaxis, ..., np.newaxis]

    print(f"Model input shape: {X.shape}")

    probability = float(
        model.predict(
            X,
            verbose=0
        ).ravel()[0]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    return prediction, probability


# ============================================================
# MAIN DEMO
# ============================================================

def main():

    args = parse_arguments()

    print("\n" + "=" * 60)
    print("           NEUROSLEEP CNN DEMONSTRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Reproduce training normalization
    # --------------------------------------------------------

    mean, std, train_subjects = (
        get_training_normalization()
    )

    print("\nTraining subjects used for normalization:")
    print(train_subjects)

    # --------------------------------------------------------
    # Load EEG
    # --------------------------------------------------------

    processed = load_eeg(
        args.subject
    )

    # --------------------------------------------------------
    # Extract 30-second window
    # --------------------------------------------------------

    signal = extract_window(
        processed,
        args.start
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction, probability = predict(
        model,
        signal,
        mean,
        std
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("                 MODEL RESULT")
    print("=" * 60)

    print(
        f"Respiratory-event probability: "
        f"{probability:.4f}"
    )

    print(
        f"Respiratory-event probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Decision threshold: 50.00%"
    )

    if prediction == 1:

        print(
            "\nPREDICTION: RESPIRATORY EVENT"
        )

    else:

        print(
            "\nPREDICTION: NO RESPIRATORY EVENT"
        )

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

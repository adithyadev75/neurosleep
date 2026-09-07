from pathlib import Path

import mne
import numpy as np

from src.data.annotations import parse_annotations
from src.data.preprocessing import preprocess_eeg


# ============================================================
# PATHS
# ============================================================

EDF_DIR = Path(
    "data/raw/shhs/polysomnography/edfs/shhs1"
)

ANNOTATION_DIR = Path(
    "data/raw/shhs/polysomnography/annotations-events-nsrr/shhs1"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

EEG_CHANNEL = "EEG"
WINDOW_SECONDS = 30

RESPIRATORY_EVENTS = {
    "Hypopnea",
    "Obstructive apnea",
}


# ============================================================
# LABELING
# ============================================================

def window_contains_event(start, end, events):
    """
    Returns 1 if any respiratory event overlaps
    the EEG window, otherwise 0.
    """

    for event in events:
        event_start = event["start"]
        event_end = event["end"]

        # Check whether the intervals overlap
        if event_start < end and event_end > start:
            return 1

    return 0


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(edf_path, annotation_path):
    """
    Load one SHHS subject, preprocess C4/A1 EEG,
    divide into 30-second windows and assign
    respiratory-event labels.
    """

    subject_id = edf_path.stem.replace("shhs1-", "")

    print("\n" + "=" * 60)
    print(f"Processing subject: {subject_id}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load annotations
    # --------------------------------------------------------

    annotations = parse_annotations(annotation_path)

    respiratory_events = [
        event
        for event in annotations
        if event["name"] in RESPIRATORY_EVENTS
    ]

    print(f"Respiratory events: {len(respiratory_events)}")

    # --------------------------------------------------------
    # Load EEG
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose=False
    )

    if EEG_CHANNEL not in raw.ch_names:
        raise ValueError(
            f"{EEG_CHANNEL} channel not found in {edf_path}"
        )

    raw.pick([EEG_CHANNEL])

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    raw = preprocess_eeg(raw)

    signal = raw.get_data()[0]
    sfreq = raw.info["sfreq"]

    window_samples = int(WINDOW_SECONDS * sfreq)

    # --------------------------------------------------------
    # Extract windows
    # --------------------------------------------------------

    X = []
    y = []

    total_samples = len(signal)

    for start_sample in range(
        0,
        total_samples - window_samples + 1,
        window_samples
    ):

        end_sample = start_sample + window_samples

        start_time = start_sample / sfreq
        end_time = end_sample / sfreq

        window = signal[start_sample:end_sample]

        label = window_contains_event(
            start_time,
            end_time,
            respiratory_events
        )

        X.append(window)
        y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"Windows: {len(X)}")
    print(f"Positive: {np.sum(y == 1)}")
    print(f"Negative: {np.sum(y == 0)}")

    return X, y, subject_id


# ============================================================
# MAIN DATASET CREATION
# ============================================================

def main():

    edf_files = sorted(
        EDF_DIR.glob("*.edf")
    )

    if not edf_files:
        raise FileNotFoundError(
            f"No EDF files found in {EDF_DIR}"
        )

    all_X = []
    all_y = []
    all_subjects = []

    print(f"Found {len(edf_files)} EDF files")

    # --------------------------------------------------------
    # Match EDF with XML
    # --------------------------------------------------------

    for edf_path in edf_files:

        subject_id = edf_path.stem.replace(
            "shhs1-",
            ""
        )

        annotation_path = (
            ANNOTATION_DIR
            / f"shhs1-{subject_id}-nsrr.xml"
        )

        if not annotation_path.exists():

            print(
                f"WARNING: Annotation missing for "
                f"{subject_id}. Skipping."
            )

            continue

        X, y, subject_id = process_subject(
            edf_path,
            annotation_path
        )

        all_X.append(X)
        all_y.append(y)

        all_subjects.extend(
            [subject_id] * len(y)
        )

    # --------------------------------------------------------
    # Combine all subjects
    # --------------------------------------------------------

    if not all_X:
        raise RuntimeError(
            "No subjects were successfully processed."
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    subjects = np.asarray(all_subjects)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "X_multisubject.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y_multisubject.npy",
        y
    )

    np.save(
        OUTPUT_DIR / "subjects_multisubject.npy",
        subjects
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MULTI-SUBJECT DATASET COMPLETE")
    print("=" * 60)

    print(f"Subjects: {len(np.unique(subjects))}")
    print(f"Total windows: {len(X)}")
    print(f"Positive windows: {np.sum(y == 1)}")
    print(f"Negative windows: {np.sum(y == 0)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nWindows per subject:")

    for subject_id in np.unique(subjects):

        mask = subjects == subject_id

        print(
            f"  {subject_id}: "
            f"{np.sum(mask)} windows, "
            f"{np.sum(y[mask] == 1)} positive, "
            f"{np.sum(y[mask] == 0)} negative"
        )


if __name__ == "__main__":
    main()
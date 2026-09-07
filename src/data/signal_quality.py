from pathlib import Path

import numpy as np
import mne


EDF_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "edfs" / "shhs1" / "shhs1-200001.edf"
)

EEG_CHANNEL = "EEG"

LOW_CUTOFF = 0.3
HIGH_CUTOFF = 35.0


def print_statistics(name, data_uv):

    data_uv = np.asarray(data_uv)

    abs_data = np.abs(data_uv)

    print(f"\n========== {name} ==========")

    print(
        f"Samples: {len(data_uv):,}"
    )

    print(
        f"Mean: {np.mean(data_uv):.3f} µV"
    )

    print(
        f"Std: {np.std(data_uv):.3f} µV"
    )

    print(
        f"Minimum: {np.min(data_uv):.3f} µV"
    )

    print(
        f"Maximum: {np.max(data_uv):.3f} µV"
    )

    print(
        f"Median: {np.median(data_uv):.3f} µV"
    )

    print("\nAbsolute amplitude percentiles:")

    for percentile in [
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]:

        value = np.percentile(
            abs_data,
            percentile
        )

        print(
            f"|EEG| {percentile:5.1f}%: "
            f"{value:.3f} µV"
        )

    print("\nAmplitude threshold percentages:")

    for threshold in [
        25,
        50,
        75,
        100,
        125,
        150,
    ]:

        percentage = (
            np.sum(abs_data > threshold)
            / len(abs_data)
            * 100
        )

        print(
            f"|EEG| > {threshold:3d} µV: "
            f"{percentage:.4f}%"
        )


def main():

    if not EDF_FILE.exists():

        raise FileNotFoundError(
            f"EDF file not found:\n{EDF_FILE}"
        )

    print(
        "\n========== LOADING EEG =========="
    )

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=True,
        verbose=False
    )

    eeg = raw.copy().pick(
        [EEG_CHANNEL]
    )

    raw_data = (
        eeg.get_data()[0] * 1e6
    )

    print(
        f"Channel: {EEG_CHANNEL}"
    )

    print(
        f"Sampling frequency: "
        f"{eeg.info['sfreq']} Hz"
    )

    print(
        f"Total duration: "
        f"{eeg.times[-1] / 3600:.2f} hours"
    )

    # -----------------------------
    # Raw signal
    # -----------------------------

    print_statistics(
        "RAW EEG",
        raw_data
    )

    # -----------------------------
    # Filtered signal
    # -----------------------------

    filtered = eeg.copy()

    filtered.filter(
        l_freq=LOW_CUTOFF,
        h_freq=HIGH_CUTOFF,
        method="fir",
        phase="zero",
        verbose=False
    )

    filtered_data = (
        filtered.get_data()[0] * 1e6
    )

    print_statistics(
        "FILTERED EEG",
        filtered_data
    )

    print(
        "\n========== SIGNAL QUALITY ANALYSIS COMPLETE =========="
    )


if __name__ == "__main__":
    main()
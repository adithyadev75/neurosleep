from pathlib import Path

import numpy as np
import mne


EDF_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "edfs" / "shhs1" / "shhs1-200001.edf"
)

EEG_CHANNEL = "EEG"


def find_runs(mask):
    """
    Find consecutive True runs in a boolean array.

    Returns:
        List of (start_index, end_index) tuples.
    """

    mask = np.asarray(mask, dtype=bool)

    padded = np.concatenate(
        ([False], mask, [False])
    )

    changes = np.diff(
        padded.astype(np.int8)
    )

    starts = np.where(
        changes == 1
    )[0]

    ends = np.where(
        changes == -1
    )[0]

    return list(zip(starts, ends))


def analyze_exact_value(
    data_uv,
    value,
    sfreq
):

    mask = np.isclose(
        data_uv,
        value,
        atol=1e-6
    )

    indices = np.where(mask)[0]

    runs = find_runs(mask)

    print(
        f"\n========== EXACT VALUE: {value:.3f} µV =========="
    )

    print(
        f"Total samples: {len(indices):,}"
    )

    percentage = (
        len(indices)
        / len(data_uv)
        * 100
    )

    print(
        f"Percentage: {percentage:.6f}%"
    )

    print(
        f"Consecutive runs: {len(runs):,}"
    )

    if not runs:
        print("No consecutive runs found.")
        return

    durations = [
        (end - start) / sfreq
        for start, end in runs
    ]

    print(
        f"Longest run: "
        f"{max(durations):.6f} s"
    )

    print(
        f"Mean run: "
        f"{np.mean(durations):.6f} s"
    )

    print(
        f"Median run: "
        f"{np.median(durations):.6f} s"
    )

    print("\nLongest 10 runs:")

    longest = sorted(
        zip(runs, durations),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for index, ((start, end), duration) in enumerate(
        longest,
        start=1
    ):

        start_time = start / sfreq
        end_time = end / sfreq

        print(
            f"{index:02d}. "
            f"{start_time:.3f}s - "
            f"{end_time:.3f}s | "
            f"duration={duration:.6f}s | "
            f"samples={end-start}"
        )


def main():

    if not EDF_FILE.exists():
        raise FileNotFoundError(
            f"EDF file not found:\n{EDF_FILE}"
        )

    print(
        "\n========== LOADING EEG FOR ARTIFACT ANALYSIS =========="
    )

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=True,
        verbose=False
    )

    eeg = raw.copy().pick(
        [EEG_CHANNEL]
    )

    data_uv = (
        eeg.get_data()[0] * 1e6
    )

    sfreq = eeg.info["sfreq"]

    print(
        f"Channel: {EEG_CHANNEL}"
    )

    print(
        f"Sampling frequency: "
        f"{sfreq} Hz"
    )

    print(
        f"Total samples: "
        f"{len(data_uv):,}"
    )

    print(
        f"Duration: "
        f"{len(data_uv) / sfreq / 3600:.2f} hours"
    )

    # --------------------------------------------------
    # Exact upper boundary
    # --------------------------------------------------

    analyze_exact_value(
        data_uv,
        125.0,
        sfreq
    )

    # --------------------------------------------------
    # Exact lower boundary observed in this recording
    # --------------------------------------------------

    analyze_exact_value(
        data_uv,
        -124.02,
        sfreq
    )

    # --------------------------------------------------
    # Saturation-like thresholds
    # --------------------------------------------------

    print(
        "\n========== EXTREME AMPLITUDE RUNS =========="
    )

    for threshold in [
        100,
        110,
        120,
        124,
    ]:

        mask = (
            np.abs(data_uv) >= threshold
        )

        runs = find_runs(mask)

        total_samples = np.sum(mask)

        percentage = (
            total_samples
            / len(data_uv)
            * 100
        )

        if runs:

            durations = [
                (end - start) / sfreq
                for start, end in runs
            ]

            longest = max(durations)

        else:

            longest = 0.0

        print(
            f"|EEG| >= {threshold:3d} µV | "
            f"samples={total_samples:,} | "
            f"percentage={percentage:.6f}% | "
            f"runs={len(runs):,} | "
            f"longest={longest:.6f}s"
        )

    print(
        "\n========== ARTIFACT ANALYSIS COMPLETE =========="
    )


if __name__ == "__main__":
    main()
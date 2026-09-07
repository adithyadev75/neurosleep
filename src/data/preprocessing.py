

from pathlib import Path

import mne
import matplotlib.pyplot as plt


EDF_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "edfs" / "shhs1" / "shhs1-200001.edf"
)

EEG_CHANNEL = "EEG"

# Initial EEG filtering range
LOW_CUTOFF = 0.3
HIGH_CUTOFF = 35.0

# Event locations for visual inspection
EVENTS = {
    "short_hypopnea": {
        "start": 3510.90,
        "end": 3521.30,
    },
    "long_hypopnea": {
        "start": 14649.40,
        "end": 14754.30,
    },
    "obstructive_apnea": {
        "start": 8146.40,
        "end": 8185.80,
    },
}


def load_eeg():

    if not EDF_FILE.exists():
        raise FileNotFoundError(
            f"EDF file not found:\n{EDF_FILE}"
        )

    print("\n========== LOADING RAW EEG ==========")

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=False,
        verbose=False
    )

    print(f"Channel: {EEG_CHANNEL}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1] / 3600:.2f} hours")

    return raw


def filter_eeg(raw):

    print("\n========== FILTERING EEG ==========")
    print(
        f"Band-pass filter: "
        f"{LOW_CUTOFF} - {HIGH_CUTOFF} Hz"
    )

    eeg = raw.copy().pick(
        [EEG_CHANNEL]
    )

    eeg.load_data()

    eeg.filter(
        l_freq=LOW_CUTOFF,
        h_freq=HIGH_CUTOFF,
        method="fir",
        phase="zero",
        verbose=False
    )

    print("Filtering complete.")

    return eeg


def preprocess_eeg(raw):
    """
    Preprocess EEG for dataset creation.

    Steps:
    1. Select the EEG channel.
    2. Load the signal into memory.
    3. Apply the existing band-pass filter.
    4. Return the filtered EEG.
    """
    eeg = filter_eeg(raw)
    return eeg

    
def plot_raw_vs_filtered(
    raw,
    filtered,
    event_name,
    event_start,
    event_end,
    before=30,
    after=30
):

    plot_start = max(
        0,
        event_start - before
    )

    plot_end = event_end + after

    raw_segment = raw.copy().crop(
        tmin=plot_start,
        tmax=plot_end
    )

    filtered_segment = filtered.copy().crop(
        tmin=plot_start,
        tmax=plot_end
    )

    raw_data, raw_times = raw_segment.get_data(
        picks=[EEG_CHANNEL],
        return_times=True
    )

    filtered_data, filtered_times = (
        filtered_segment.get_data(
            picks=[EEG_CHANNEL],
            return_times=True
        )
    )

    raw_uv = raw_data[0] * 1e6
    filtered_uv = filtered_data[0] * 1e6

    raw_absolute_times = raw_times + plot_start
    filtered_absolute_times = (
        filtered_times + plot_start
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True
    )

    # -----------------------------
    # Raw EEG
    # -----------------------------

    axes[0].plot(
        raw_absolute_times,
        raw_uv,
        linewidth=0.7
    )

    axes[0].axvspan(
        event_start,
        event_end,
        alpha=0.25,
        label=event_name
    )

    axes[0].axvline(
        event_start,
        linestyle="--",
        linewidth=1
    )

    axes[0].axvline(
        event_end,
        linestyle="--",
        linewidth=1
    )

    axes[0].set_title(
        f"Raw C4/A1 EEG — {event_name}"
    )

    axes[0].set_ylabel(
        "EEG (µV)"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.2
    )

    # -----------------------------
    # Filtered EEG
    # -----------------------------

    axes[1].plot(
        filtered_absolute_times,
        filtered_uv,
        linewidth=0.7
    )

    axes[1].axvspan(
        event_start,
        event_end,
        alpha=0.25,
        label=event_name
    )

    axes[1].axvline(
        event_start,
        linestyle="--",
        linewidth=1
    )

    axes[1].axvline(
        event_end,
        linestyle="--",
        linewidth=1
    )

    axes[1].set_title(
        f"Filtered C4/A1 EEG — "
        f"{LOW_CUTOFF}-{HIGH_CUTOFF} Hz"
    )

    axes[1].set_xlabel(
        "Time (seconds)"
    )

    axes[1].set_ylabel(
        "EEG (µV)"
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.2
    )

    fig.suptitle(
        f"{event_name.replace('_', ' ').title()} "
        f"— Raw vs Filtered",
        fontsize=14
    )

    plt.tight_layout()

    return fig


def main():

    raw = load_eeg()

    filtered = filter_eeg(raw)

    output_dir = (
        Path("results")
        / "figures"
        / "preprocessing"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n========== GENERATING COMPARISON FIGURES =========="
    )

    for event_name, event in EVENTS.items():

        fig = plot_raw_vs_filtered(
            raw=raw,
            filtered=filtered,
            event_name=event_name,
            event_start=event["start"],
            event_end=event["end"],
        )

        output_file = (
            output_dir
            / f"{event_name}_raw_vs_filtered.png"
        )

        fig.savefig(
            output_file,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Saved: {output_file}"
        )

    print(
        "\n========== PREPROCESSING TEST COMPLETE =========="
    )


if __name__ == "__main__":
    main()
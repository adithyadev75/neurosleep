from pathlib import Path

import matplotlib.pyplot as plt
import mne


EDF_FILE = (
    Path("data")
    / "raw"
    / "shhs"
    / "polysomnography"
    / "edfs"
    / "shhs1"
    / "shhs1-200001.edf"
)

CHANNELS = ["EEG", "EEG(sec)"]

# Start 1 hour into the recording.
START_SECONDS = 3600

# Plot 5 minutes.
DURATION_SECONDS = 300


def main():
    print("Loading EEG segment...")

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=False,
        verbose=False,
    )

    raw_segment = raw.copy().crop(
        tmin=START_SECONDS,
        tmax=START_SECONDS + DURATION_SECONDS,
    ).pick(CHANNELS)

    data = raw_segment.get_data()
    times = raw_segment.times

    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Segment duration: {times[-1]:.2f} seconds")

    fig, axes = plt.subplots(
        len(CHANNELS),
        1,
        figsize=(14, 8),
        sharex=True,
    )

    for index, channel in enumerate(CHANNELS):
        axes[index].plot(times, data[index] * 1e6)
        axes[index].set_ylabel(f"{channel}\n(µV)")
        axes[index].grid(True)

    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "SHHS Single-Channel EEG Comparison\n"
        "Subject: shhs1-200001"
    )

    plt.tight_layout()

    output = Path("results") / "figures" / "eeg_channel_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output, dpi=150)
    plt.show()

    print(f"\nFigure saved to: {output}")


if __name__ == "__main__":
    main()
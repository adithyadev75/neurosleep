from pathlib import Path

import mne


def inspect_edf(edf_path: str | Path) -> None:
    """Inspect an SHHS EDF recording."""

    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose=False,
    )

    print("\n========== SHHS EDF INFORMATION ==========")
    print(f"File: {edf_path.name}")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1] / 3600:.2f} hours")
    print(f"Number of samples: {raw.n_times}")

    print("\n========== CHANNELS ==========")

    for i, channel in enumerate(raw.ch_names):
        print(f"{i:02d}: {channel}")

    print("\n========== EEG CHANNEL METADATA ==========")

    eeg_channels = ["EEG", "EEG(sec)"]

    for channel in eeg_channels:

        if channel not in raw.ch_names:
            print(f"{channel}: NOT FOUND")
            continue

        index = raw.ch_names.index(channel)
        channel_info = raw.info["chs"][index]

        print(f"\nChannel: {channel}")
        print(f"Index: {index}")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"Unit: {channel_info['unit']}")
        print(f"Unit multiplier: {channel_info['unit_mul']}")
        print(f"Calibration: {channel_info['cal']}")


if __name__ == "__main__":

    edf_file = (
        Path("data")
        / "raw"
        / "shhs"
        / "polysomnography"
        / "edfs"
        / "shhs1"
        / "shhs1-200001.edf"
    )

    inspect_edf(edf_file)
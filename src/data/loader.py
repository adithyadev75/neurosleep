```python
"""
SHHS EEG data loading utilities.

This module will contain functions for:
- Reading SHHS EDF recordings
- Listing available signals
- Selecting a single EEG channel
- Loading signal data for analysis
"""

from pathlib import Path

import mne


def inspect_edf(edf_path: str | Path) -> None:
    """
    Print basic information about an SHHS EDF recording.

    Parameters
    ----------
    edf_path:
        Path to the EDF recording.
    """
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose=False,
    )

    print("\n=== EDF Information ===")
    print(f"File: {edf_path.name}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")

    print("\n=== Available Channels ===")
    for index, channel in enumerate(raw.ch_names):
        print(f"{index:02d}: {channel}")


if __name__ == "__main__":
    print("SHHS EDF loader ready.")
```

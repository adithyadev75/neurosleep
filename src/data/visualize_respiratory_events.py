from pathlib import Path
import xml.etree.ElementTree as ET

import mne
import matplotlib.pyplot as plt


EDF_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "edfs" / "shhs1"
    / "shhs1-200001.edf"
)

XML_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "annotations-events-nsrr" / "shhs1"
    / "shhs1-200001-nsrr.xml"
)

EEG_CHANNEL = "EEG"


RESPIRATORY_EVENTS = {
    "Hypopnea",
    "Obstructive apnea",
}


def strip_namespace(tag):
    return tag.split("}")[-1]


def parse_respiratory_events(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    events = []

    for element in root.iter():

        if strip_namespace(element.tag) != "ScoredEvent":
            continue

        event = {}

        for child in element:

            tag = strip_namespace(child.tag)

            if child.text:
                event[tag] = child.text.strip()

        concept = event.get("EventConcept", "")
        event_name = concept.split("|")[0].strip()

        if event_name in RESPIRATORY_EVENTS:

            start = float(event["Start"])
            duration = float(event["Duration"])

            events.append({
                "event": event_name,
                "start": start,
                "end": start + duration,
                "duration": duration,
            })

    return events


def plot_event(raw, event, before=30, after=30):

    event_start = event["start"]
    event_end = event["end"]

    plot_start = max(0, event_start - before)
    plot_end = event_end + after

    print(
        f"\nPlotting {event['event']}: "
        f"{event_start:.2f}s - {event_end:.2f}s"
    )

    data, times = raw.copy().crop(
        tmin=plot_start,
        tmax=plot_end
    ).get_data(
        picks=[EEG_CHANNEL],
        return_times=True
    )

    # Convert volts to microvolts
    eeg_uv = data[0] * 1e6

    absolute_times = times + plot_start

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    ax.plot(
        absolute_times,
        eeg_uv,
        linewidth=0.8
    )

    # Respiratory event
    ax.axvspan(
        event_start,
        event_end,
        alpha=0.25,
        label=event["event"]
    )

    ax.axvline(
        event_start,
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        event_end,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"{event['event']} — C4/A1 EEG\n"
        f"{event_start:.1f}s–{event_end:.1f}s "
        f"({event['duration']:.1f}s)"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("EEG (µV)")

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    return fig


def main():

    if not EDF_FILE.exists():
        raise FileNotFoundError(
            f"EDF file not found:\n{EDF_FILE}"
        )

    if not XML_FILE.exists():
        raise FileNotFoundError(
            f"XML file not found:\n{XML_FILE}"
        )

    print("\n========== LOADING EEG ==========")

    raw = mne.io.read_raw_edf(
        EDF_FILE,
        preload=False,
        verbose=False
    )

    print(
        f"Channel: {EEG_CHANNEL}"
    )

    print(
        f"Sampling frequency: "
        f"{raw.info['sfreq']} Hz"
    )

    events = parse_respiratory_events(
        XML_FILE
    )

    print(
        f"Respiratory events: "
        f"{len(events)}"
    )

    # -------------------------------------------------
    # Select representative events
    # -------------------------------------------------

    short_hypopnea = next(
        event for event in events
        if event["event"] == "Hypopnea"
        and event["duration"] < 12
    )

    long_hypopnea = max(
        (
            event for event in events
            if event["event"] == "Hypopnea"
        ),
        key=lambda x: x["duration"]
    )

    obstructive_apnea = next(
        event for event in events
        if event["event"] == "Obstructive apnea"
    )

    selected_events = [
        (
            "short_hypopnea",
            short_hypopnea
        ),
        (
            "long_hypopnea",
            long_hypopnea
        ),
        (
            "obstructive_apnea",
            obstructive_apnea
        ),
    ]

    output_dir = (
        Path("results")
        / "figures"
        / "respiratory_events"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Generate plots
    # -------------------------------------------------

    for name, event in selected_events:

        fig = plot_event(
            raw,
            event,
            before=30,
            after=30
        )

        output_file = (
            output_dir
            / f"{name}.png"
        )

        fig.savefig(
            output_file,
            dpi=200
        )

        print(
            f"Saved: {output_file}"
        )

        plt.close(fig)

    print(
        "\n========== COMPLETE =========="
    )


if __name__ == "__main__":
    main()
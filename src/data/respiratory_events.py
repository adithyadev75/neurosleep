from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import statistics


XML_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "annotations-events-nsrr" / "shhs1"
    / "shhs1-200001-nsrr.xml"
)

RESPIRATORY_EVENTS = {"Hypopnea", "Obstructive apnea"}


def strip_namespace(tag: str) -> str:
    return tag.split("}")[-1]


def parse_respiratory_events(xml_file: str | Path):
    xml_file = Path(xml_file)

    if not xml_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_file}"
        )

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
                "duration": duration,
                "end": start + duration,
                "signal_location": event.get(
                    "SignalLocation",
                    "N/A"
                ),
            })

    return events


def print_duration_statistics(events):

    durations = [
        event["duration"]
        for event in events
    ]

    if not durations:
        print("No respiratory events found.")
        return

    print("\n========== DURATION STATISTICS ==========")

    print(f"Minimum duration : {min(durations):.2f} s")
    print(f"Maximum duration : {max(durations):.2f} s")
    print(f"Mean duration    : {statistics.mean(durations):.2f} s")
    print(f"Median duration  : {statistics.median(durations):.2f} s")

    if len(durations) > 1:
        print(
            f"Std deviation    : "
            f"{statistics.stdev(durations):.2f} s"
        )

    sorted_durations = sorted(durations)

    q1 = statistics.median(
        sorted_durations[:len(sorted_durations) // 2]
    )

    if len(sorted_durations) % 2 == 0:
        upper_half = sorted_durations[
            len(sorted_durations) // 2:
        ]
    else:
        upper_half = sorted_durations[
            len(sorted_durations) // 2 + 1:
        ]

    q3 = statistics.median(upper_half)

    print(f"Q1 duration      : {q1:.2f} s")
    print(f"Q3 duration      : {q3:.2f} s")

    print(
        f"Total event time : "
        f"{sum(durations):.2f} s"
    )


def print_timeline_statistics(events):

    if not events:
        return

    starts = [event["start"] for event in events]
    ends = [event["end"] for event in events]

    print("\n========== TIMELINE ==========")

    print(
        f"Earliest event : "
        f"{min(starts):.2f} s"
    )

    print(
        f"Latest event   : "
        f"{max(ends):.2f} s"
    )

    print(
        f"Night span     : "
        f"{(max(ends) - min(starts)):.2f} s"
    )


def print_event_counts(events):

    print("\n========== EVENT COUNTS ==========")

    counts = Counter(
        event["event"]
        for event in events
    )

    for event_name, count in counts.items():
        print(
            f"{event_name}: {count}"
        )


def main():

    events = parse_respiratory_events(XML_FILE)

    print(
        "\n========== RESPIRATORY EVENT ANALYSIS =========="
    )

    print(
        f"Total respiratory events: {len(events)}"
    )

    print_event_counts(events)

    print_duration_statistics(events)

    print_timeline_statistics(events)

    print("\n========== FIRST 20 EVENTS ==========")

    for index, event in enumerate(
        events[:20],
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{event['event']} | "
            f"start={event['start']:.2f}s | "
            f"duration={event['duration']:.2f}s | "
            f"end={event['end']:.2f}s | "
            f"location={event['signal_location']}"
        )


if __name__ == "__main__":
    main()
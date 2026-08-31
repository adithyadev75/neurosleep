from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


def strip_namespace(tag: str) -> str:
    """Remove an XML namespace from a tag name."""
    return tag.split("}")[-1]


def inspect_annotations(xml_path: str | Path) -> None:
    """Inspect SHHS NSRR scored events."""

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_path}"
        )

    print("\n========== SHHS ANNOTATION FILE ==========")
    print(f"File: {xml_path.name}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    print(f"Root element: {root.tag}")

    scored_events = []

    for event in root.iter():
        if strip_namespace(event.tag) != "ScoredEvent":
            continue

        event_data = {}

        for child in event:
            tag = strip_namespace(child.tag)

            if child.text:
                event_data[tag] = child.text.strip()

        scored_events.append(event_data)

    print(f"\nTotal scored events: {len(scored_events)}")

    # ---------------------------------------------------------
    # Event concept counts
    # ---------------------------------------------------------

    concepts = Counter(
        event.get("EventConcept", "UNKNOWN")
        for event in scored_events
    )

    print("\n========== EVENT COUNTS ==========")

    for concept, count in concepts.most_common():
        print(f"{concept}: {count}")

    # ---------------------------------------------------------
    # Event types
    # ---------------------------------------------------------

    event_types = Counter(
        event.get("EventType", "UNKNOWN")
        for event in scored_events
    )

    print("\n========== EVENT TYPES ==========")

    for event_type, count in event_types.most_common():
        print(f"{event_type}: {count}")

    # ---------------------------------------------------------
    # Signal locations
    # ---------------------------------------------------------

    signal_locations = Counter()

    for event in root.iter():
        if strip_namespace(event.tag) == "SignalLocation":
            if event.text:
                signal_locations[event.text.strip()] += 1

    print("\n========== SIGNAL LOCATIONS ==========")

    for location, count in signal_locations.most_common():
        print(f"{location}: {count}")

    # ---------------------------------------------------------
    # Detailed event information
    # ---------------------------------------------------------

    print("\n========== FIRST 20 SCORED EVENTS ==========")

    for index, event in enumerate(scored_events[:20], start=1):

        print(f"\nEvent {index}")
        print(
            f"  Concept:   "
            f"{event.get('EventConcept', 'N/A')}"
        )
        print(
            f"  Type:      "
            f"{event.get('EventType', 'N/A')}"
        )
        print(
            f"  Start:     "
            f"{event.get('Start', 'N/A')}"
        )
        print(
            f"  Duration:  "
            f"{event.get('Duration', 'N/A')}"
        )
        print(
            f"  Locations: "
            f"{event.get('SignalLocation', 'N/A')}"
        )


if __name__ == "__main__":

    xml_file = (
        Path("data")
        / "raw"
        / "shhs"
        / "polysomnography"
        / "annotations-events-nsrr"
        / "shhs1"
        / "shhs1-200001-nsrr.xml"
    )

    inspect_annotations(xml_file)
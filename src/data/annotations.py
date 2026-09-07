from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


def strip_namespace(tag: str) -> str:
    """Remove an XML namespace from a tag name."""
    return tag.split("}")[-1]


def parse_annotations(xml_path: str | Path) -> list[dict]:
    """
    Parse SHHS NSRR XML annotations.

    Returns a list of scored events.
    Each event contains:
        EventConcept
        EventType
        Start
        Duration
        SignalLocation
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    scored_events = []

    for event in root.iter():

        if strip_namespace(event.tag) != "ScoredEvent":
            continue

        event_data = {}

        for child in event:
            tag = strip_namespace(child.tag)

            if child.text:
                event_data[tag] = child.text.strip()

        # Convert numeric fields where possible
        if "Start" in event_data:
            event_data["Start"] = float(event_data["Start"])

        if "Duration" in event_data:
            event_data["Duration"] = float(event_data["Duration"])

        scored_events.append(event_data)

    return scored_events


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

    scored_events = parse_annotations(xml_path)

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
def parse_annotations(xml_path: str | Path) -> list[dict]:
    """
    Parse SHHS NSRR annotation XML into a simple event list.

    Each event contains:
        name
        type
        start
        duration
        signal_location
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []

    for event in root.iter():

        if strip_namespace(event.tag) != "ScoredEvent":
            continue

        event_data = {}

        for child in event:
            tag = strip_namespace(child.tag)

            if child.text:
                event_data[tag] = child.text.strip()

        concept = event_data.get("EventConcept", "")

        # Convert SHHS annotation names into simple labels
        if concept.startswith("Hypopnea"):
            name = "Hypopnea"

        elif concept.startswith("Obstructive apnea"):
            name = "Obstructive apnea"

        elif concept.startswith("Arousal"):
            name = "Arousal"

        elif concept.startswith("Stage 1"):
            name = "Stage 1"

        elif concept.startswith("Stage 2"):
            name = "Stage 2"

        elif concept.startswith("Stage 3"):
            name = "Stage 3"

        elif concept.startswith("REM"):
            name = "REM"

        elif concept.startswith("Wake"):
            name = "Wake"

        elif concept.startswith("SpO2 desaturation"):
            name = "SpO2 desaturation"

        elif concept.startswith("SpO2 artifact"):
            name = "SpO2 artifact"

        elif concept.startswith("Recording Start"):
            name = "Recording Start Time"

        else:
            name = concept

        events.append({
            "name": name,
            "type": event_data.get("EventType", ""),
            "start": float(event_data.get("Start", 0)),
            "duration": float(event_data.get("Duration", 0)),
            "signal_location": event_data.get(
                "SignalLocation", ""
            ),
        })

    return events


def parse_annotations(xml_path: str | Path):
    """
    Parse SHHS NSRR annotations into a standardized format.

    Each event contains:
        name
        start
        duration
        end
        signal_location
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for element in root.iter():

        if strip_namespace(element.tag) != "ScoredEvent":
            continue

        event_data = {}

        for child in element:

            tag = strip_namespace(child.tag)

            if child.text:
                event_data[tag] = child.text.strip()

        concept = event_data.get("EventConcept", "")

        # SHHS concepts can appear as:
        # "Hypopnea|Hypopnea"
        # "Stage 2 sleep|2"
        name = concept.split("|")[0].strip()

        if "Start" not in event_data:
            continue

        if "Duration" not in event_data:
            continue

        start = float(event_data["Start"])
        duration = float(event_data["Duration"])

        annotations.append({
            "name": name,
            "start": start,
            "duration": duration,
            "end": start + duration,
            "signal_location": event_data.get(
                "SignalLocation",
                "N/A"
            ),
        })

    return annotations

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
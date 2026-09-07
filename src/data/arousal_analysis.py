from pathlib import Path
import xml.etree.ElementTree as ET
import statistics


XML_FILE = (
    Path("data")
    / "raw"
    / "shhs"
    / "polysomnography"
    / "annotations-events-nsrr"
    / "shhs1"
    / "shhs1-200001-nsrr.xml"
)


def strip_namespace(tag: str) -> str:
    return tag.split("}")[-1]


def get_arousals(xml_file: Path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    arousals = []

    for event in root.iter():

        if strip_namespace(event.tag) != "ScoredEvent":
            continue

        data = {}

        for child in event:
            tag = strip_namespace(child.tag)

            if child.text:
                data[tag] = child.text.strip()

        concept = data.get("EventConcept", "")

        if concept.startswith("Arousal|Arousal"):
            arousals.append(data)

    return arousals


def main():

    arousals = get_arousals(XML_FILE)

    starts = [
        float(event["Start"])
        for event in arousals
        if "Start" in event
    ]

    durations = [
        float(event["Duration"])
        for event in arousals
        if "Duration" in event
    ]

    print("\n========== SHHS AROUSAL ANALYSIS ==========")

    print(f"Number of arousals: {len(arousals)}")

    print("\n========== DURATION ==========")

    print(f"Minimum: {min(durations):.2f} seconds")
    print(f"Maximum: {max(durations):.2f} seconds")
    print(f"Mean:    {statistics.mean(durations):.2f} seconds")
    print(f"Median:  {statistics.median(durations):.2f} seconds")

    print("\n========== TIMING ==========")

    print(f"First arousal: {min(starts):.2f} seconds")
    print(f"Last arousal:  {max(starts):.2f} seconds")

    print("\n========== FIRST 10 AROUSALS ==========")

    for i, event in enumerate(arousals[:10], start=1):

        print(
            f"{i:02d}. "
            f"start={event.get('Start')} s, "
            f"duration={event.get('Duration')} s, "
            f"location={event.get('SignalLocation', 'N/A')}"
        )


if __name__ == "__main__":
    main()
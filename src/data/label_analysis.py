from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


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


def parse_events(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    arousals = []
    stages = []

    for element in root.iter():

        if strip_namespace(element.tag) != "ScoredEvent":
            continue

        event = {}

        for child in element:
            tag = strip_namespace(child.tag)

            if child.text:
                event[tag] = child.text.strip()

        concept = event.get("EventConcept", "")

        if concept.startswith("Arousal|Arousal"):
            arousals.append(event)

        elif concept.startswith("Wake|"):
            stages.append(("Wake", event))

        elif concept.startswith("Stage 1 sleep|"):
            stages.append(("Stage 1", event))

        elif concept.startswith("Stage 2 sleep|"):
            stages.append(("Stage 2", event))

        elif concept.startswith("Stage 3 sleep|"):
            stages.append(("Stage 3", event))

        elif concept.startswith("REM sleep|"):
            stages.append(("REM", event))

    return arousals, stages


def find_stage(time_seconds, stages):

    for stage_name, event in stages:

        start = float(event["Start"])
        duration = float(event["Duration"])
        end = start + duration

        if start <= time_seconds < end:
            return stage_name

    return "Unknown"


def main():

    arousals, stages = parse_events(XML_FILE)

    stage_counts = Counter()

    print("\n========== AROUSAL / SLEEP-STAGE ANALYSIS ==========")

    for arousal in arousals:

        start = float(arousal["Start"])
        duration = float(arousal["Duration"])

        midpoint = start + duration / 2

        stage = find_stage(midpoint, stages)

        stage_counts[stage] += 1

    print(f"\nTotal arousals: {len(arousals)}")

    print("\n========== AROUSALS BY SLEEP STAGE ==========")

    for stage, count in stage_counts.items():
        print(f"{stage}: {count}")

    print("\n========== PERCENTAGES ==========")

    total = len(arousals)

    for stage, count in stage_counts.items():
        percentage = (count / total) * 100
        print(f"{stage}: {percentage:.2f}%")


if __name__ == "__main__":
    main()
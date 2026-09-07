from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


XML_FILE = (
    Path("data") / "raw" / "shhs" / "polysomnography"
    / "annotations-events-nsrr" / "shhs1"
    / "shhs1-200001-nsrr.xml"
)


RESPIRATORY_EVENTS = {
    "Hypopnea",
    "Obstructive apnea",
}

SLEEP_STAGES = {
    "Wake": "Wake",
    "Stage 1 sleep": "Stage 1",
    "Stage 2 sleep": "Stage 2",
    "Stage 3 sleep": "Stage 3",
    "REM sleep": "REM",
}


def strip_namespace(tag: str) -> str:
    return tag.split("}")[-1]


def parse_events(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    respiratory_events = []
    stage_events = []

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

            respiratory_events.append({
                "event": event_name,
                "start": start,
                "end": start + duration,
                "duration": duration,
            })

        elif event_name in SLEEP_STAGES:

            start = float(event["Start"])
            duration = float(event["Duration"])

            stage_events.append({
                "stage": SLEEP_STAGES[event_name],
                "start": start,
                "end": start + duration,
                "duration": duration,
            })

    return respiratory_events, stage_events


def calculate_overlap(event_start, event_end, stage_start, stage_end):

    overlap_start = max(event_start, stage_start)
    overlap_end = min(event_end, stage_end)

    if overlap_end <= overlap_start:
        return 0.0

    return overlap_end - overlap_start


def assign_stage(event, stages):

    overlaps = []

    for stage in stages:

        overlap = calculate_overlap(
            event["start"],
            event["end"],
            stage["start"],
            stage["end"],
        )

        if overlap > 0:

            overlaps.append({
                "stage": stage["stage"],
                "overlap": overlap,
            })

    if not overlaps:
        return "Unknown", 0.0, []

    overlaps.sort(
        key=lambda x: x["overlap"],
        reverse=True
    )

    primary_stage = overlaps[0]["stage"]
    primary_overlap = overlaps[0]["overlap"]

    return primary_stage, primary_overlap, overlaps


def main():

    respiratory_events, stage_events = parse_events(XML_FILE)

    print("\n========== STAGE / RESPIRATORY ANALYSIS ==========")

    print(
        f"Respiratory events: {len(respiratory_events)}"
    )

    print(
        f"Sleep-stage annotations: {len(stage_events)}"
    )

    stage_counts = Counter()

    for event in respiratory_events:

        stage, overlap, all_overlaps = assign_stage(
            event,
            stage_events
        )

        stage_counts[stage] += 1

    print("\n========== RESPIRATORY EVENTS BY PRIMARY STAGE ==========")

    total = len(respiratory_events)

    for stage in [
        "Wake",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "REM",
        "Unknown",
    ]:

        count = stage_counts[stage]

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{stage:10s}: "
            f"{count:3d} events "
            f"({percentage:6.2f}%)"
        )

    print("\n========== FIRST 20 EVENT-STAGE ASSIGNMENTS ==========")

    for index, event in enumerate(
        respiratory_events[:20],
        start=1
    ):

        stage, overlap, all_overlaps = assign_stage(
            event,
            stage_events
        )

        print(
            f"{index:02d}. "
            f"{event['event']:18s} | "
            f"{event['start']:8.2f} - "
            f"{event['end']:8.2f}s | "
            f"duration={event['duration']:6.2f}s | "
            f"primary_stage={stage:10s} | "
            f"overlap={overlap:6.2f}s"
        )

        if len(all_overlaps) > 1:

            details = ", ".join(
                f"{x['stage']}={x['overlap']:.2f}s"
                for x in all_overlaps
            )

            print(
                f"    Stage overlaps: {details}"
            )


if __name__ == "__main__":
    main()
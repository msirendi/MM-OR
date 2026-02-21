#!/usr/bin/env python3
"""Analyze MM-OR scene-graph signals relevant to back-table inventory tracking."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    ACTIVE_USE_PREDICATES_BY_TOOL,
    HANDLING_PREDICATES,
    PREDICATE_IMPLIED_TOOLS,
    TABLE_ENTITIES,
    TOOL_ENTITIES,
    ensure_parent_dir,
    frame_to_int,
    load_relationship_samples,
    to_base_take_name,
)


def empty_tool_stats() -> dict:
    return {
        "mentions": 0,
        "active_use_events_direct": 0,
        "active_use_events_inferred": 0,
        "handling_events": 0,
        "first_frame": None,
        "last_frame": None,
        "frames_with_table_context": 0,
    }


def update_frame_bounds(stats: dict, frame_idx: int) -> None:
    if stats["first_frame"] is None or frame_idx < stats["first_frame"]:
        stats["first_frame"] = frame_idx
    if stats["last_frame"] is None or frame_idx > stats["last_frame"]:
        stats["last_frame"] = frame_idx


def summarize_tool_status(tool_stats: dict) -> str:
    if tool_stats["mentions"] == 0:
        return "not_observed"
    if (tool_stats["active_use_events_direct"] + tool_stats["active_use_events_inferred"]) > 0:
        return "used"
    if tool_stats["handling_events"] > 0:
        return "handled_no_active_use"
    return "context_only"


def build_markdown_summary(report: dict) -> str:
    lines = []
    lines.append("# Scene-Graph Signal Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(
        f"- MM-OR samples analyzed: {report['overview']['samples_total_mmor']}"
    )
    lines.append(
        f"- MM-OR samples with >=1 relationship: {report['overview']['samples_with_relationships']}"
    )
    lines.append(
        f"- Distinct MM-OR takes observed: {report['overview']['distinct_takes']}"
    )
    lines.append(
        f"- `instrument_table` entity mentions: {report['focus_entity_mentions']['instrument_table']}"
    )
    lines.append(
        f"- `secondary_table` entity mentions: {report['focus_entity_mentions']['secondary_table']}"
    )
    lines.append("")
    lines.append("## Top Table Triplets")
    for triplet, count in report["top_table_triplets"][:10]:
        lines.append(f"- {triplet}: {count}")
    lines.append("")
    lines.append("## Top Tool Triplets")
    for triplet, count in report["top_tool_triplets"][:15]:
        lines.append(f"- {triplet}: {count}")
    lines.append("")
    lines.append("## Global Tool Status Counts")
    for status, count in report["global_tool_status_counts"].items():
        lines.append(f"- {status}: {count}")
    lines.append("")
    lines.append("## Caveats")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MM-OR scene-graph signals for back-table use.")
    parser.add_argument(
        "--sg-data-dir",
        type=Path,
        default=Path("scene_graph_generation/data"),
        help="Directory containing relationships_*.json files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("use_cases/back_table_inventory/reports/scene_graph_signal_report.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("use_cases/back_table_inventory/reports/scene_graph_signal_report.md"),
        help="Output Markdown summary path.",
    )
    args = parser.parse_args()

    samples = load_relationship_samples(args.sg_data_dir, mmor_only=True)
    if not samples:
        raise SystemExit(f"No MM-OR samples found in {args.sg_data_dir}")

    entity_counts = Counter()
    predicate_counts = Counter()
    table_triplet_counts = Counter()
    tool_triplet_counts = Counter()

    take_to_samples = defaultdict(int)
    take_tool_stats = defaultdict(lambda: {tool: empty_tool_stats() for tool in TOOL_ENTITIES})

    samples_with_relationships = 0
    for sample in samples:
        rels = sample.get("relationships", [])
        if rels:
            samples_with_relationships += 1
        take = to_base_take_name(sample["take_name"])
        frame_idx = frame_to_int(sample["frame_id"])
        take_to_samples[take] += 1

        frame_has_table_relation = False
        tools_in_frame = set()
        for relation in rels:
            if len(relation) != 3:
                continue
            sub, obj, pred = relation
            entity_counts[sub] += 1
            entity_counts[obj] += 1
            predicate_counts[pred] += 1

            involves_table = (sub in TABLE_ENTITIES) or (obj in TABLE_ENTITIES)
            involves_tool = (sub in TOOL_ENTITIES) or (obj in TOOL_ENTITIES)

            if involves_table:
                frame_has_table_relation = True
                table_triplet_counts[f"{sub},{obj},{pred}"] += 1
            if involves_tool:
                tool_triplet_counts[f"{sub},{obj},{pred}"] += 1

            for implied_tool in PREDICATE_IMPLIED_TOOLS.get(pred, set()):
                implied_stats = take_tool_stats[take][implied_tool]
                implied_stats["active_use_events_inferred"] += 1
                update_frame_bounds(implied_stats, frame_idx)

            for tool in TOOL_ENTITIES:
                if sub == tool or obj == tool:
                    tools_in_frame.add(tool)
                    stats = take_tool_stats[take][tool]
                    stats["mentions"] += 1
                    update_frame_bounds(stats, frame_idx)
                    if pred in HANDLING_PREDICATES:
                        stats["handling_events"] += 1
                    if pred in ACTIVE_USE_PREDICATES_BY_TOOL.get(tool, set()):
                        stats["active_use_events_direct"] += 1

        if frame_has_table_relation:
            for tool in tools_in_frame:
                take_tool_stats[take][tool]["frames_with_table_context"] += 1

    per_take_tool_summary = {}
    global_tool_status_counts = Counter()
    for take, tool_dict in sorted(take_tool_stats.items()):
        per_take_tool_summary[take] = {}
        for tool, stats in tool_dict.items():
            status = summarize_tool_status(stats)
            global_tool_status_counts[status] += 1
            enriched = dict(stats)
            enriched["status"] = status
            per_take_tool_summary[take][tool] = enriched

    report = {
        "overview": {
            "samples_total_mmor": len(samples),
            "samples_with_relationships": samples_with_relationships,
            "distinct_takes": len(take_to_samples),
            "takes": sorted(take_to_samples.keys()),
        },
        "focus_entity_mentions": {
            "instrument_table": entity_counts.get("instrument_table", 0),
            "secondary_table": entity_counts.get("secondary_table", 0),
            "instrument": entity_counts.get("instrument", 0),
            "drill": entity_counts.get("drill", 0),
            "hammer": entity_counts.get("hammer", 0),
            "saw": entity_counts.get("saw", 0),
        },
        "top_predicates": predicate_counts.most_common(20),
        "top_table_triplets": table_triplet_counts.most_common(30),
        "top_tool_triplets": tool_triplet_counts.most_common(40),
        "global_tool_status_counts": dict(global_tool_status_counts),
        "per_take_tool_summary": per_take_tool_summary,
        "caveats": [
            "Scene graphs are class-level (not per physical item instance).",
            "Tool-to-table 'lyingOn' style relations are sparse/absent in provided MM-OR split JSONs.",
            "secondary_table appears in ontology but has zero mentions in current MM-OR scene-graph splits.",
            "Use these outputs for triage and candidate generation, then validate against images/segmentation.",
        ],
    }

    output_json = args.output_json
    ensure_parent_dir(output_json)
    with output_json.open("w") as f:
        json.dump(report, f, indent=2)

    output_md = args.output_md
    ensure_parent_dir(output_md)
    output_md.write_text(build_markdown_summary(report))

    print("Scene-graph back-table analysis complete")
    print(f"- Samples analyzed: {report['overview']['samples_total_mmor']}")
    print(f"- Distinct takes: {report['overview']['distinct_takes']}")
    print(f"- instrument_table mentions: {report['focus_entity_mentions']['instrument_table']}")
    print(f"- secondary_table mentions: {report['focus_entity_mentions']['secondary_table']}")
    print(f"- Wrote JSON: {output_json}")
    print(f"- Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()

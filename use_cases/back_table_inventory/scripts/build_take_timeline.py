#!/usr/bin/env python3
"""Build per-frame timeline for a single MM-OR take with back-table/tool signals."""

from __future__ import annotations

import argparse
import csv
import json
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
    safe_load_json,
    to_base_take_name,
)


def relation_to_str(relation) -> str:
    sub, obj, pred = relation
    return f"{sub},{obj},{pred}"


def tool_state_for_frame(relations: list, tool: str) -> str:
    tool_rels = [r for r in relations if len(r) == 3 and (r[0] == tool or r[1] == tool)]
    for _, _, pred in relations:
        if tool in PREDICATE_IMPLIED_TOOLS.get(pred, set()):
            return "active_use"
    if not tool_rels:
        return "none"
    for _, _, pred in tool_rels:
        if pred in ACTIVE_USE_PREDICATES_BY_TOOL.get(tool, set()):
            return "active_use"
    for _, _, pred in tool_rels:
        if pred in HANDLING_PREDICATES:
            return "handling"
    return "context"


def parse_next_action(next_action_value):
    if next_action_value is None:
        return "", ""
    if isinstance(next_action_value, list) and len(next_action_value) >= 2:
        return str(next_action_value[0]), str(next_action_value[1])
    if isinstance(next_action_value, str):
        return next_action_value, ""
    return "", ""


def parse_sterility(sterility_value):
    if sterility_value is None:
        return 0, ""
    if isinstance(sterility_value, list):
        if not sterility_value:
            return 0, ""
        details = " | ".join(
            " ".join(x) if isinstance(x, list) and len(x) == 3 else str(x)
            for x in sterility_value
        )
        return len(sterility_value), details
    return 0, str(sterility_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-take inventory timeline from MM-OR scene graphs.")
    parser.add_argument("--take", required=True, help="Take name without _MMOR suffix, e.g. 001_PKA")
    parser.add_argument("--sg-data-dir", type=Path, default=Path("scene_graph_generation/data"))
    parser.add_argument("--mmor-root", type=Path, default=Path("MM-OR_data"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Default: use_cases/back_table_inventory/reports/timelines/<take>.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="Summary JSON output path. Default: use_cases/back_table_inventory/reports/timelines/<take>.summary.json",
    )
    args = parser.parse_args()

    take = args.take
    output_csv = args.output_csv or Path(f"use_cases/back_table_inventory/reports/timelines/{take}.csv")
    output_summary = args.output_summary_json or Path(
        f"use_cases/back_table_inventory/reports/timelines/{take}.summary.json"
    )

    samples = load_relationship_samples(args.sg_data_dir, mmor_only=True)
    take_samples = [s for s in samples if to_base_take_name(s["take_name"]) == take]
    if not take_samples:
        raise SystemExit(f"No MM-OR scene-graph samples found for take: {take}")

    samples_sorted = sorted(take_samples, key=lambda s: frame_to_int(s["frame_id"]))

    next_action_json = safe_load_json(args.mmor_root / "take_timestamp_to_next_action" / f"{take}.json")
    robot_phase_json = safe_load_json(args.mmor_root / "take_timestamp_to_robot_phase" / f"{take}.json")
    sterility_json = safe_load_json(args.mmor_root / "take_timestamp_to_sterility_breach" / f"{take}.json")

    header = [
        "take",
        "split",
        "frame_id",
        "relationship_count",
        "table_relation_count",
        "tool_relation_count",
        "active_use_relation_count",
        "instrument_state",
        "drill_state",
        "hammer_state",
        "saw_state",
        "next_action",
        "next_action_seconds",
        "robot_phase",
        "sterility_breach_count",
        "sterility_breach_details",
        "table_relations",
        "tool_relations",
        "active_use_relations",
    ]

    per_tool_counts = {
        tool: {"active_use": 0, "handling": 0, "context": 0, "none": 0}
        for tool in TOOL_ENTITIES
    }
    relationship_total = 0
    active_use_total = 0

    rows = []
    for sample in samples_sorted:
        split = sample.get("_split", "")
        frame_id = sample["frame_id"]
        rels = [r for r in sample.get("relationships", []) if len(r) == 3]

        table_rels = [r for r in rels if (r[0] in TABLE_ENTITIES or r[1] in TABLE_ENTITIES)]
        tool_rels = [r for r in rels if (r[0] in TOOL_ENTITIES or r[1] in TOOL_ENTITIES)]
        active_rels = []
        for rel in rels:
            sub, obj, pred = rel
            if pred in PREDICATE_IMPLIED_TOOLS:
                active_rels.append(rel)
                continue
            for tool in TOOL_ENTITIES:
                if (sub == tool or obj == tool) and pred in ACTIVE_USE_PREDICATES_BY_TOOL.get(tool, set()):
                    active_rels.append(rel)
                    break

        relationship_total += len(rels)
        active_use_total += len(active_rels)

        tool_states = {tool: tool_state_for_frame(rels, tool) for tool in TOOL_ENTITIES}
        for tool, state in tool_states.items():
            per_tool_counts[tool][state] += 1

        next_action_val = next_action_json.get(frame_id) if isinstance(next_action_json, dict) else None
        next_action_str, next_action_seconds = parse_next_action(next_action_val)
        robot_phase = ""
        if isinstance(robot_phase_json, dict):
            robot_phase = str(robot_phase_json.get(frame_id, ""))
        sterility_val = sterility_json.get(frame_id) if isinstance(sterility_json, dict) else None
        sterility_count, sterility_details = parse_sterility(sterility_val)

        row = [
            take,
            split,
            frame_id,
            len(rels),
            len(table_rels),
            len(tool_rels),
            len(active_rels),
            tool_states["instrument"],
            tool_states["drill"],
            tool_states["hammer"],
            tool_states["saw"],
            next_action_str,
            next_action_seconds,
            robot_phase,
            sterility_count,
            sterility_details,
            " | ".join(relation_to_str(r) for r in table_rels),
            " | ".join(relation_to_str(r) for r in tool_rels),
            " | ".join(relation_to_str(r) for r in active_rels),
        ]
        rows.append(row)

    ensure_parent_dir(output_csv)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    summary = {
        "take": take,
        "frames": len(rows),
        "relationships_total": relationship_total,
        "active_use_relationships_total": active_use_total,
        "per_tool_state_counts": per_tool_counts,
        "labels_available": {
            "next_action": isinstance(next_action_json, dict),
            "robot_phase": isinstance(robot_phase_json, dict),
            "sterility_breach": isinstance(sterility_json, dict),
        },
    }

    ensure_parent_dir(output_summary)
    with output_summary.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Take timeline build complete")
    print(f"- Take: {take}")
    print(f"- Frames exported: {len(rows)}")
    print(f"- CSV: {output_csv}")
    print(f"- Summary JSON: {output_summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Heuristic candidate detection for unused/wasted tool classes per MM-OR take."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import (
    ACTIVE_USE_PREDICATES_BY_TOOL,
    HANDLING_PREDICATES,
    NEXT_ACTION_TO_TOOLS,
    PREDICATE_IMPLIED_TOOLS,
    TOOL_ENTITIES,
    ensure_parent_dir,
    frame_to_int,
    load_relationship_samples,
    safe_load_json,
    to_base_take_name,
)


def summarize_tool_for_take(
    seen_frames: set[int],
    handling_frames: set[int],
    active_direct_frames: set[int],
    active_inferred_frames: set[int],
    active_next_action_frames: set[int],
    trailing_seconds: int,
) -> dict:
    all_active = active_direct_frames.union(active_inferred_frames).union(active_next_action_frames)
    summary = {
        "seen_frames_count": len(seen_frames),
        "handling_frames_count": len(handling_frames),
        "active_frames_from_direct_tool_relations_count": len(active_direct_frames),
        "active_frames_from_predicate_inference_count": len(active_inferred_frames),
        "active_frames_from_next_action_count": len(active_next_action_frames),
        "first_seen_frame": min(seen_frames) if seen_frames else None,
        "last_seen_frame": max(seen_frames) if seen_frames else None,
        "last_active_frame": max(all_active) if all_active else None,
        "status": None,
        "candidate_reasons": [],
    }

    if not seen_frames:
        summary["status"] = "not_observed"
        return summary

    if not all_active:
        summary["status"] = "seen_not_used_candidate"
        summary["candidate_reasons"].append("seen_in_context_but_no_active_use_signal")
        return summary

    summary["status"] = "used"
    tail = summary["last_seen_frame"] - summary["last_active_frame"]
    if tail >= trailing_seconds:
        summary["candidate_reasons"].append(
            f"tool_visible_or_referenced_{tail}s_after_last_active_use"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candidate unused/wasted tool classes from MM-OR annotations.")
    parser.add_argument("--sg-data-dir", type=Path, default=Path("scene_graph_generation/data"))
    parser.add_argument("--mmor-root", type=Path, default=Path("MM-OR_data"))
    parser.add_argument("--take", type=str, default=None, help="Optional single take filter, e.g. 001_PKA.")
    parser.add_argument(
        "--trailing-seconds",
        type=int,
        default=120,
        help="Flag tools that remain visible/referenced this long after last active use.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("use_cases/back_table_inventory/reports/unused_tool_candidates.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    samples = load_relationship_samples(args.sg_data_dir, mmor_only=True)
    if args.take:
        samples = [s for s in samples if to_base_take_name(s["take_name"]) == args.take]
    if not samples:
        raise SystemExit("No MM-OR samples found for the requested scope.")

    take_frame_to_relationships = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        take = to_base_take_name(sample["take_name"])
        frame_idx = frame_to_int(sample["frame_id"])
        rels = [r for r in sample.get("relationships", []) if len(r) == 3]
        take_frame_to_relationships[take][frame_idx].extend(rels)

    report = {
        "scope": {
            "take_filter": args.take,
            "trailing_seconds": args.trailing_seconds,
        },
        "per_take": {},
        "global_candidates": [],
    }

    for take in sorted(take_frame_to_relationships.keys()):
        frame_to_rels = take_frame_to_relationships[take]
        next_action_json = safe_load_json(args.mmor_root / "take_timestamp_to_next_action" / f"{take}.json")

        per_tool_raw = {
            tool: {
                "seen_frames": set(),
                "handling_frames": set(),
                "active_direct_frames": set(),
                "active_inferred_frames": set(),
                "active_next_action_frames": set(),
            }
            for tool in TOOL_ENTITIES
        }

        for frame_idx in sorted(frame_to_rels.keys()):
            rels = frame_to_rels[frame_idx]
            for sub, obj, pred in rels:
                for tool in TOOL_ENTITIES:
                    if sub == tool or obj == tool:
                        per_tool_raw[tool]["seen_frames"].add(frame_idx)
                        if pred in HANDLING_PREDICATES:
                            per_tool_raw[tool]["handling_frames"].add(frame_idx)
                        if pred in ACTIVE_USE_PREDICATES_BY_TOOL.get(tool, set()):
                            per_tool_raw[tool]["active_direct_frames"].add(frame_idx)
                for implied_tool in PREDICATE_IMPLIED_TOOLS.get(pred, set()):
                    per_tool_raw[implied_tool]["active_inferred_frames"].add(frame_idx)

        if isinstance(next_action_json, dict):
            for frame_id, next_action_val in next_action_json.items():
                frame_idx = frame_to_int(frame_id)
                if not isinstance(next_action_val, list) or len(next_action_val) < 1:
                    continue
                action = str(next_action_val[0]).strip().lower()
                mapped_tools = NEXT_ACTION_TO_TOOLS.get(action, set())
                for tool in mapped_tools:
                    per_tool_raw[tool]["active_next_action_frames"].add(frame_idx)

        per_tool_summary = {}
        take_candidates = []
        for tool in TOOL_ENTITIES:
            raw = per_tool_raw[tool]
            summary = summarize_tool_for_take(
                seen_frames=raw["seen_frames"],
                handling_frames=raw["handling_frames"],
                active_direct_frames=raw["active_direct_frames"],
                active_inferred_frames=raw["active_inferred_frames"],
                active_next_action_frames=raw["active_next_action_frames"],
                trailing_seconds=args.trailing_seconds,
            )
            per_tool_summary[tool] = summary
            if summary["candidate_reasons"]:
                take_candidates.append({"tool": tool, **summary})
                report["global_candidates"].append({"take": take, "tool": tool, **summary})

        report["per_take"][take] = {
            "tool_summary": per_tool_summary,
            "candidate_count": len(take_candidates),
            "candidates": take_candidates,
            "labels_available": {
                "next_action": isinstance(next_action_json, dict),
            },
        }

    ensure_parent_dir(args.output_json)
    with args.output_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("Unused/wasted candidate analysis complete")
    print(f"- Takes analyzed: {len(report['per_take'])}")
    print(f"- Candidate entries: {len(report['global_candidates'])}")
    print(f"- Output: {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check MM-OR download/readiness for back-table inventory analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import MMOR_TAKE_NAMES, ensure_parent_dir, get_take_folder_name

EXPECTED_ZIPS = [
    "001_PKA.zip",
    "002_PKA.zip",
    "003_TKA.zip",
    "004_PKA.zip",
    "005_TKA.zip",
    "006_PKA.zip",
    "007_TKA.zip",
    "008_PKA.zip",
    "009_TKA.zip",
    "010_PKA.zip",
    "011_TKA.zip",
    "012_PKA.zip",
    "013_PKA.zip",
    "014_PKA.zip",
    "015-018_PKA.zip",
    "019-022_PKA.zip",
    "023-032_PKA.zip",
    "033_PKA.zip",
    "035_PKA.zip",
    "036_PKA.zip",
    "037_TKA.zip",
    "038_TKA.zip",
    "take_jsons.zip",
    "take_point_clouds_sparse.zip",
    "take_timestamp_to_next_action.zip",
    "take_timestamp_to_robot_phase.zip",
    "take_timestamp_to_sterility_breach.zip",
    "take_tracks.zip",
    "take_transcripts.zip",
    "screen_summaries.zip",
    "take_audios.zip",
    "4D-OR_pcd_sparse.zip",
]


def count_per_take_file(mmor_root: Path, rel_path_pattern: str) -> tuple[int, list[str]]:
    present = 0
    missing_examples = []
    for take in MMOR_TAKE_NAMES:
        take_folder = get_take_folder_name(take)
        rel_path = rel_path_pattern.format(take=take, take_folder=take_folder)
        full = mmor_root / rel_path
        if full.exists():
            present += 1
        elif len(missing_examples) < 5:
            missing_examples.append(str(rel_path))
    return present, missing_examples


def count_per_take_dir(mmor_root: Path, rel_path_pattern: str) -> tuple[int, list[str]]:
    present = 0
    missing_examples = []
    for take in MMOR_TAKE_NAMES:
        take_folder = get_take_folder_name(take)
        rel_path = rel_path_pattern.format(take=take, take_folder=take_folder)
        full = mmor_root / rel_path
        if full.exists() and full.is_dir():
            present += 1
        elif len(missing_examples) < 5:
            missing_examples.append(str(rel_path))
    return present, missing_examples


def build_report(repo_root: Path, mmor_root: Path, markers_dir: Path) -> dict:
    top_level_dirs = [
        "take_jsons",
        "take_point_clouds_sparse",
        "take_timestamp_to_next_action",
        "take_timestamp_to_robot_phase",
        "take_timestamp_to_sterility_breach",
        "take_tracks",
        "take_transcripts",
        "screen_summaries",
        "take_audios",
    ]
    optional_derived_dirs = [
        "take_audio_per_timepoint",
        "take_audio_embeddings_per_timepoint",
        "take_transcripts_per_timepoint",
        "take_segmasks_per_timepoint",
    ]

    top_level_status = {}
    for rel in top_level_dirs:
        p = mmor_root / rel
        top_level_status[rel] = {"exists": p.exists(), "is_dir": p.is_dir()}

    optional_status = {}
    for rel in optional_derived_dirs:
        p = mmor_root / rel
        optional_status[rel] = {"exists": p.exists(), "is_dir": p.is_dir()}

    per_take_checks = {
        "take_json": count_per_take_file(mmor_root, "take_jsons/{take}.json"),
        "next_action": count_per_take_file(mmor_root, "take_timestamp_to_next_action/{take}.json"),
        "robot_phase": count_per_take_file(mmor_root, "take_timestamp_to_robot_phase/{take}.json"),
        "sterility_breach": count_per_take_file(mmor_root, "take_timestamp_to_sterility_breach/{take}.json"),
        "tracks": count_per_take_file(mmor_root, "take_tracks/{take}.json"),
        "transcript_srt": count_per_take_file(mmor_root, "take_transcripts/{take}.srt"),
        "audio_mp3": count_per_take_file(mmor_root, "take_audios/{take}.mp3"),
        "take_folder": count_per_take_dir(mmor_root, "{take_folder}"),
        "colorimage": count_per_take_dir(mmor_root, "{take_folder}/colorimage"),
        "simstation": count_per_take_dir(mmor_root, "{take_folder}/simstation"),
        "trackercam": count_per_take_dir(mmor_root, "{take_folder}/trackercam"),
        "screen_summaries": count_per_take_dir(mmor_root, "screen_summaries/{take}"),
        "point_cloud_sparse": count_per_take_dir(mmor_root, "take_point_clouds_sparse/{take}"),
    }

    per_take_status = {}
    for key, (present, missing_examples) in per_take_checks.items():
        per_take_status[key] = {
            "present_takes": present,
            "total_takes": len(MMOR_TAKE_NAMES),
            "missing_examples": missing_examples,
        }

    marker_files = sorted(markers_dir.glob("*.downloaded")) if markers_dir.exists() else []
    zip_files_in_repo = sorted(repo_root.glob("*.zip"))
    report = {
        "repo_root": str(repo_root),
        "mmor_root": str(mmor_root),
        "markers_dir": str(markers_dir),
        "download_progress": {
            "markers_found": len(marker_files),
            "expected_zip_count": len(EXPECTED_ZIPS),
            "marker_completion_ratio": round(len(marker_files) / len(EXPECTED_ZIPS), 4) if EXPECTED_ZIPS else None,
            "zip_files_currently_present_in_repo_root": [p.name for p in zip_files_in_repo],
        },
        "top_level_status": top_level_status,
        "optional_derived_status": optional_status,
        "per_take_status": per_take_status,
    }
    return report


def print_report(report: dict) -> None:
    progress = report["download_progress"]
    print("MM-OR readiness check")
    print(f"- MM-OR root: {report['mmor_root']}")
    print(f"- Download markers: {progress['markers_found']} / {progress['expected_zip_count']}")
    if progress["zip_files_currently_present_in_repo_root"]:
        print(f"- ZIPs currently present in repo root: {', '.join(progress['zip_files_currently_present_in_repo_root'][:6])}"
              + (" ..." if len(progress["zip_files_currently_present_in_repo_root"]) > 6 else ""))
    print("- Top-level availability:")
    for key, status in sorted(report["top_level_status"].items()):
        mark = "OK" if status["exists"] else "MISSING"
        print(f"  - {key}: {mark}")
    print("- Per-take availability (present / total):")
    for key, status in sorted(report["per_take_status"].items()):
        print(f"  - {key}: {status['present_takes']} / {status['total_takes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MM-OR data readiness for back-table analysis.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--mmor-root", type=Path, default=Path("MM-OR_data"), help="MM-OR data root path.")
    parser.add_argument("--markers-dir", type=Path, default=Path("markers"), help="Download marker directory.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("use_cases/back_table_inventory/reports/data_readiness_report.json"),
        help="Path to write JSON report.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    mmor_root = args.mmor_root if args.mmor_root.is_absolute() else (repo_root / args.mmor_root)
    markers_dir = args.markers_dir if args.markers_dir.is_absolute() else (repo_root / args.markers_dir)

    report = build_report(repo_root=repo_root, mmor_root=mmor_root, markers_dir=markers_dir)
    print_report(report)

    output_path = args.output_json if args.output_json.is_absolute() else (repo_root / args.output_json)
    ensure_parent_dir(output_path)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"- Wrote report: {output_path}")


if __name__ == "__main__":
    main()


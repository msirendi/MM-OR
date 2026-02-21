#!/usr/bin/env python3
"""Shared utilities for back-table inventory analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

# MM-OR take names used by the official codebase.
MMOR_TAKE_NAMES: List[str] = [
    "001_PKA", "002_PKA", "003_TKA", "004_PKA", "005_TKA", "006_PKA", "007_TKA", "008_PKA",
    "009_TKA", "010_PKA", "011_TKA", "012_1_PKA", "012_2_PKA", "013_PKA", "014_PKA", "015_PKA",
    "016_PKA", "017_PKA", "018_1_PKA", "018_2_PKA", "019_PKA", "020_PKA", "021_PKA", "022_PKA",
    "023_PKA", "024_PKA", "025_PKA", "026_PKA", "027_PKA", "028_PKA", "029_PKA", "030_PKA",
    "031_PKA", "032_PKA", "033_PKA", "035_PKA", "036_PKA", "037_TKA", "038_TKA",
]

MMOR_TAKE_NAME_TO_FOLDER: Dict[str, str] = {
    "012_1_PKA": "012_PKA",
    "012_2_PKA": "012_PKA",
    "015_PKA": "015-018_PKA",
    "016_PKA": "015-018_PKA",
    "017_PKA": "015-018_PKA",
    "018_1_PKA": "015-018_PKA",
    "018_2_PKA": "015-018_PKA",
    "019_PKA": "019-022_PKA",
    "020_PKA": "019-022_PKA",
    "021_PKA": "019-022_PKA",
    "022_PKA": "019-022_PKA",
    "023_PKA": "023-032_PKA",
    "024_PKA": "023-032_PKA",
    "025_PKA": "023-032_PKA",
    "026_PKA": "023-032_PKA",
    "027_PKA": "023-032_PKA",
    "028_PKA": "023-032_PKA",
    "029_PKA": "023-032_PKA",
    "030_PKA": "023-032_PKA",
    "031_PKA": "023-032_PKA",
    "032_PKA": "023-032_PKA",
}

RELATIONSHIP_SPLIT_FILES: List[str] = [
    "relationships_train.json",
    "relationships_validation.json",
    "relationships_test.json",
]

TOOL_ENTITIES = ("instrument", "drill", "hammer", "saw")
TABLE_ENTITIES = ("instrument_table", "secondary_table")

HANDLING_PREDICATES = {"holding", "touching", "manipulating", "preparing"}

ACTIVE_USE_PREDICATES_BY_TOOL = {
    "instrument": {"cutting", "cleaning", "cementing", "suturing", "scanning"},
    "drill": {"drilling"},
    "hammer": {"hammering"},
    "saw": {"sawing", "cutting"},
}

# In MM-OR scene graphs, active predicates are often attached to surgeon/patient
# relations rather than explicit tool relations. This map captures implied tool use.
PREDICATE_IMPLIED_TOOLS = {
    "drilling": {"drill"},
    "sawing": {"saw"},
    "hammering": {"hammer"},
    "cutting": {"instrument", "saw"},
    "cementing": {"instrument", "hammer"},
    "suturing": {"instrument"},
    "cleaning": {"instrument"},
    "scanning": {"instrument"},
}

NEXT_ACTION_TO_TOOLS = {
    "drill": {"drill"},
    "saw": {"saw"},
    "hammer": {"hammer"},
    "cut": {"instrument", "saw"},
    "cement": {"instrument", "hammer"},
    "suture": {"instrument"},
    "clean": {"instrument"},
    "scan": {"instrument"},
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_base_take_name(take_name: str) -> str:
    if take_name.endswith("_MMOR"):
        return take_name[:-5]
    return take_name


def get_take_folder_name(take_name: str) -> str:
    return MMOR_TAKE_NAME_TO_FOLDER.get(take_name, take_name)


def frame_to_int(frame_id: str) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return int(frame_id.lstrip("0") or "0")


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def safe_load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def load_relationship_samples(
    sg_data_dir: Path,
    split_files: Iterable[str] = RELATIONSHIP_SPLIT_FILES,
    mmor_only: bool = True,
) -> List[dict]:
    samples: List[dict] = []
    for split_file in split_files:
        path = sg_data_dir / split_file
        if not path.exists():
            continue
        split_name = split_file.replace("relationships_", "").replace(".json", "")
        with path.open("r") as f:
            split_samples = json.load(f)
        for sample in split_samples:
            take_name = sample.get("take_name", "")
            if mmor_only and "_MMOR" not in take_name:
                continue
            sample["_split"] = split_name
            samples.append(sample)
    return samples

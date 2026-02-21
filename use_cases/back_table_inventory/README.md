# Back-Table Inventory Analysis Workspace

This workspace is for using MM-OR to:
- document which tool classes are used during a procedure,
- build per-take event timelines,
- flag candidate unused/wasted tool classes for review.

It is designed to work while `download_mm-or.sh` is still running.

## Included

- `scripts/check_data_readiness.py`
  - Checks which MM-OR assets are already available and which are still missing.
- `scripts/analyze_scene_graph_signals.py`
  - Audits MM-OR scene-graph annotations for tool/table relevance.
- `scripts/build_take_timeline.py`
  - Builds a per-frame timeline for a take (tool states + optional downstream labels).
- `scripts/detect_unused_tool_candidates.py`
  - Uses heuristics to flag candidate unused/wasted tool classes per take.
- `scripts/common.py`
  - Shared constants and helpers.

## Quick Start

From repo root:

```bash
python3 use_cases/back_table_inventory/scripts/check_data_readiness.py
python3 use_cases/back_table_inventory/scripts/analyze_scene_graph_signals.py
python3 use_cases/back_table_inventory/scripts/build_take_timeline.py --take 001_PKA
python3 use_cases/back_table_inventory/scripts/detect_unused_tool_candidates.py
```

Outputs are written under:
- `use_cases/back_table_inventory/reports/`
- Includes `review_notes.md` with use-case-focused findings from the local `.webarchive` paper copy.

## Paper Review Notes (from local `.webarchive`)

For this use case, the most relevant paper claims/details are:
- MM-OR includes multimodal recordings with:
  - multi-view RGB-D room cameras,
  - RGB detail views explicitly targeting areas including the instrument table,
  - a low-exposure tracker camera that preserves highlights/tool visibility,
  - audio + speech transcripts,
  - robot logs/screen summaries,
  - infrared tracking.
- Downstream labels include:
  - robot phase,
  - next action,
  - sterility breach.
- Reported dataset scale:
  - 92,983 total timepoints,
  - 25,277 annotated timepoints,
  - 17 full-length procedures + 22 short clips.

## Relevant MM-OR Data Assets for This Use Case

Prioritize these folders when fully downloaded:

1. Core supervision and procedure context
- `scene_graph_generation/data/relationships_{train,validation,test}.json`
- `MM-OR_data/take_timestamp_to_next_action/*.json`
- `MM-OR_data/take_timestamp_to_robot_phase/*.json`
- `MM-OR_data/take_timestamp_to_sterility_breach/*.json`

2. Visual evidence for back-table/tool states
- `MM-OR_data/<take>/colorimage/`
- `MM-OR_data/<take>/simstation/`
- `MM-OR_data/<take>/trackercam/`
- `MM-OR_data/<take>/segmentation_export_*` and `simstation_segmentation_export_*`
- `MM-OR_data/take_segmasks_per_timepoint/<take>/` (if generated)

3. Additional multimodal context
- `MM-OR_data/take_tracks/*.json`
- `MM-OR_data/screen_summaries/<take>/`
- `MM-OR_data/take_transcripts/*.srt`
- `MM-OR_data/take_audios/*.mp3`

## Important Limitations

- Scene graphs are class-level, not per physical instrument instance.
- In the provided MM-OR scene-graph JSONs, direct `secondary_table` evidence is absent.
- Tool-to-table relations (for example `instrument -> instrument_table -> lyingOn`) are sparse/absent in the provided splits.
- Some usage signals are inferred from predicates (for example `drilling` implies drill use) because active predicates are often surgeon→patient relations.
- Waste/unused outputs here are heuristic candidates for human review, not definitive labels.

# MM-OR Review Notes for Back-Table Inventory Use Case

Source reviewed:
- `MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgic.webarchive`

## What Is Directly Relevant

From the paper content in the archived file:
- MM-OR includes detail RGB views targeting key regions including instrument table context.
- It includes low-exposure tracker-cam footage intended to preserve bright surgical highlights/tool visibility.
- It provides scene graphs, panoptic segmentation, and downstream labels (next action, robot phase, sterility breach).
- Reported scale: 92,983 total timepoints, 25,277 annotated timepoints, 17 full-length takes + 22 short clips.

## Current Local Findings (Before Full Data Download Completes)

From the available annotation files under `scene_graph_generation/data`:
- MM-OR samples analyzed: 17,121
- MM-OR takes in these split files: 17
- `instrument_table` mentions: 22,202
- `secondary_table` mentions: 0
- Top table interactions are mostly staff-to-`instrument_table` (`closeTo`, `preparing`)
- Tool actions are often implied by predicates like `drilling`, `sawing`, `hammering` attached to surgeon/patient relations rather than explicit tool-object relations

Implication for this use case:
- Scene graphs are useful for procedure-level usage documentation and timeline signal extraction.
- They are not sufficient for definitive per-item waste accounting by themselves.
- Visual segmentation/tracking evidence should be added when available to confirm table-level leftovers.

## Recommended MM-OR Subsets to Prioritize Once Download Is Ready

Highest priority:
- `MM-OR_data/take_jsons/*.json`
- `MM-OR_data/take_timestamp_to_next_action/*.json`
- `MM-OR_data/take_timestamp_to_robot_phase/*.json`
- `MM-OR_data/take_timestamp_to_sterility_breach/*.json`
- `MM-OR_data/<take>/colorimage/`
- `MM-OR_data/<take>/simstation/`
- `MM-OR_data/<take>/trackercam/`
- `MM-OR_data/<take>/segmentation_export_*`
- `MM-OR_data/<take>/simstation_segmentation_export_*`

Secondary but useful:
- `MM-OR_data/take_tracks/*.json`
- `MM-OR_data/screen_summaries/<take>/`
- `MM-OR_data/take_transcripts/*.srt`
- `MM-OR_data/take_audios/*.mp3`


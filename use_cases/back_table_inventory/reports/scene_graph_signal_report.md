# Scene-Graph Signal Report

## Overview
- MM-OR samples analyzed: 17121
- MM-OR samples with >=1 relationship: 16991
- Distinct MM-OR takes observed: 17
- `instrument_table` entity mentions: 22202
- `secondary_table` entity mentions: 0

## Top Table Triplets
- nurse,instrument_table,closeTo: 7811
- nurse,instrument_table,preparing: 5853
- circulator,instrument_table,closeTo: 4596
- mps,instrument_table,closeTo: 2120
- assistant_surgeon,instrument_table,closeTo: 1214
- head_surgeon,instrument_table,closeTo: 382
- assistant_surgeon,instrument_table,preparing: 76
- mps,instrument_table,preparing: 36
- nurse,instrument_table,touching: 31
- anaesthetist,instrument_table,closeTo: 30

## Top Tool Triplets
- nurse,instrument,holding: 5883
- head_surgeon,instrument,holding: 2383
- assistant_surgeon,instrument,holding: 1803
- head_surgeon,saw,holding: 1704
- circulator,instrument,holding: 1062
- head_surgeon,drill,holding: 769
- mps,instrument,holding: 698
- assistant_surgeon,drill,holding: 420
- nurse,saw,holding: 233
- nurse,drill,holding: 176
- head_surgeon,hammer,holding: 173
- assistant_surgeon,saw,holding: 159
- nurse,hammer,holding: 79
- saw,mako_robot,closeTo: 59
- assistant_surgeon,hammer,holding: 27

## Global Tool Status Counts
- used: 58
- handled_no_active_use: 4
- not_observed: 6

## Caveats
- Scene graphs are class-level (not per physical item instance).
- Tool-to-table 'lyingOn' style relations are sparse/absent in provided MM-OR split JSONs.
- secondary_table appears in ontology but has zero mentions in current MM-OR scene-graph splits.
- Use these outputs for triage and candidate generation, then validate against images/segmentation.

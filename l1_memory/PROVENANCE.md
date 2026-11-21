# L1 source and scope

The algorithm modules in this directory were adapted from
`/home/lilixing/projects/LightNav-0-codex/src/lightnav/memory` at commit
`6aefffada23de3d4edd422c3c9289182067dbad2`. That repository is licensed
under Apache License 2.0; its license is included as `LICENSE.Apache-2.0`.

Copied modules: `memory_writer.py`, `local_graph_fusion.py`,
`local_graph_config.py`, `l1_prompt_organizer.py`, `yolo_scene_extractor.py`,
`place_recognition.py`, `working_memory.py`, `episodic_memory.py`,
`scene_graph.py`. The Python 3.8 helper `compat.py` is new. The copies came from the earlier
LightNav-UniNaVid-L3 vendor at commit `59882b9`, which changed only import
paths, a local instruction matching helper, and Python 3.8 string helpers.
The behavior of RGB detection, DINOv2 descriptor extraction, L1 event writing,
history retrieval, and fact selection remains in those modules.

`l1_integration/sbadir_backend.py` is a new adapter. It feeds the real RGB
observation into `MemoryWriter`, persists the resulting events to the existing
`V3MemoryStore`, calls both `L1PromptOrganizer` and `LocalGraphFusion`, and
exposes their selected facts with original evidence references. The adapter
does not inject either module's standalone prompt header.

`provenance.json` records the source and adapted SHA256 for each migrated file.

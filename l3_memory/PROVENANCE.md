# L3 module provenance

This package is a **vendored, import-only copy** of the frozen LightNav L3 map
and official-style prompt serializer. It is not a reimplementation.

- Source repository: `/home/lilixing/projects/LightNav-0-codex`
- Source branch: `experiment/l3-official-style-prompt`
- Source commit: `6aefffa` (`feat: add L3 official-style navigation prompt`)
- Copy date: 2026-11-18

## Files copied

| Vendored file | Source file | Change |
| --- | --- | --- |
| `conceptgraphs_lite.py` | `src/lightnav/memory/conceptgraphs_lite.py` | import path only |
| `spatial_graph_adapter.py` | `src/lightnav/memory/spatial_graph_adapter.py` | import paths only |
| `l3_prompt_organizer.py` | `src/lightnav/memory/l3_prompt_organizer.py` | import paths only |
| `l3_official_prompt.py` | `src/lightnav/memory/l3_official_prompt.py` | import paths only |
| `yolo_seg_extractor.py` | `src/lightnav/memory/yolo_seg_extractor.py` | none |
| `v3_store.py` | `src/lightnav/memory/v3_store.py` | import path only |
| `v3_schemas.py` | `src/lightnav/memory/v3_schemas.py` | none |
| `schemas.py` | `src/lightnav/memory/schemas.py` | none |
| `spatial_l3_config.py` | `src/lightnav/memory/spatial_l3_config.py` | import path only |

The only textual differences are:

```text
lightnav.memory.<module>  ->  l3_memory.<module>
lightnav.memory.local_graph_fusion._plain_label -> l3_memory.lexical._plain_label
lightnav.memory.v3_retrieval._terms            -> l3_memory.lexical._terms
```

`l3_memory/lexical.py` contains exactly those two helper functions copied from
the upstream modules. They were extracted so this package does not import the
L1 fusion module or the full V3 retrieval module. No logic was changed.

## Source hashes

`provenance.json` records the source and vendored SHA-256 for every file. The
tests recompute the vendored hashes and fail if a file drifts. The source hashes
are evidence of what was copied; the source repository is read-only for this
task.

## What was deliberately not copied

L1 prompt construction (`l1_prompt_organizer.py`, `local_graph_fusion.py`,
`joint_prompt.py`), L2 hierarchy, the V4 path, the LightNav policy/runtime, and
the LightNav model. This project only needs the L3 map, evidence records,
candidate retrieval, and the official-style serializer.

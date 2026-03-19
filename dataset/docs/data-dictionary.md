# Data dictionary

This document describes the main CSV schemas.

## `split_data/split_data_XX.csv`

Gaze sample stream for a single view instance.

Columns:

- `time`: timestamp (ms)
- `x`: x coordinate (pixels)
- `y`: y coordinate (pixels)

## `AOI.csv`

AOI definition table.

Columns:

- `x1,y1,x2,y2`: AOI rectangle bounds
- `isKeyAOI`: boolean-like flag
- `componentInfo`: string
- `view`: view name
- `task`: one of `S` (search), `R` (read), `O` (operate)
- `issue`: boolean-like flag

## `view_switch.csv`

Ordered view visitation list.

Columns (observed):

- `view`: view name (may repeat)

## `metrics.csv`

AOI-level metrics, one row per AOI per view instance.

Identification columns:

- `participant`
- `session`
- `view_instance`
- `view`
- `view_instance_id`

Key derived label columns:

- `isKeyAOI`: 0/1
- `issue`: 0/1
- `search`, `read`, `operate`: one-hot task indicators

Metric columns (examples):

- Fixation-based: `FD`, `AFD`, `NFD`, `NAFD`, …
- Saccade-based: `SD`, `ASD`, `NSD`, `NASD`, …
- Sequence: `ASW`, `NASW`, `ARG`, `NARG`, `AED`
- View sequence: `VSW`, `NVSW`, `VRG`, `NVRG`, `VED`
- Spatial: `CA`, `IOU`

The exact set of columns depends on the metric definitions used in this dataset.

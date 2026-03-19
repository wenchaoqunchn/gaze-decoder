# Dataset structure

## Root

- `README.md`: dataset entry
- `participants/`: per-participant data
- `docs/`: documentation

## `participants/Pxx/`

Each participant folder uses an anonymized ID (`P1`–`P20`).

### AOI definition

- `AOI.csv`
  - AOIs are defined per view (column `view`).
  - Bounding box fields: `x1,y1,x2,y2`.
  - Attributes: `isKeyAOI`, `componentInfo`, `task`, `issue`.

### View sequence

- `view_switch.csv`
  - Ordered view visit log.
  - Each row corresponds to a **view instance**.
  - The same `view` name may appear multiple times (revisits).

### Split gaze samples (per view instance)

- `split_data/`
  - `split_data_01.csv`, `split_data_02.csv`, …
  - Each file corresponds to the same row index in `view_switch.csv` (after excluding Calibration entries).
  - Columns: `time, x, y`.

### Metrics

- `metrics.csv`
  - AOI-level metrics.
  - One row = one AOI within one view instance.
  - Identification columns:
    - `participant`: `Pxx`
    - `session`: equals `participant` (one user record)
    - `view_instance`: `split_data_XX`
    - `view`: view name
    - `view_instance_id`: integer instance index

### Images

- `img/origin/`, `img/rawpoint/`, `img/scanpath/`

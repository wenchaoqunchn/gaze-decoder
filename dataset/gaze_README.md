# Eye-tracking Dataset

This folder contains an eye-tracking dataset organized for analysis.

## Dataset structure

- `P1/` … `P20/`: participant folders (anonymized IDs)
  - `AOI.csv`: AOI definitions (per view)
  - `raw_data.csv`: raw gaze stream (time, x, y)
  - `view_switch.csv`: ordered sequence of visited views (each row corresponds to one view instance)
  - `split_data/`: gaze samples split by view instance (`split_data_01.csv`, `split_data_02.csv`, …)
  - `metrics.csv`: AOI-level metrics
    - Notes:
      - **One row = one AOI within one view instance**
      - `view` may repeat across instances; use `view_instance` / `view_instance_id` to distinguish revisits
  - `img/`: visualization images (organized by type)
    - `origin/`, `rawpoint/`, `scanpath/`

- `docs/`
  - `index.md`: documentation entry
  - `dataset-structure.md`: detailed folder/file descriptions
  - `data-dictionary.md`: column definitions for key CSV files

## Notes on semantics (important)

- **Participant session**: the complete record under a participant folder (e.g., `P1/`).
- **View instance**: one `split_data_XX.csv` file, representing one visit to one view.
  - The `view` name can repeat in `view_switch.csv`; repeated occurrences are treated as different *instances*.

## Metrics

`metrics.csv` contains AOI-level metrics derived from the gaze data in `split_data/` and the AOI definitions in `AOI.csv`.


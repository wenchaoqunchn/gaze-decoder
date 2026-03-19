# user_aoi_labeled/ — Manually Labelled AOI Definitions

This directory contains the per-session AOI (Area of Interest) CSV files
produced during the manual labelling phase of the knowledge-base construction.

Each file covers one labelling session (`S1`–`S6`) in which a domain expert
mapped UI components to bounding boxes and assigned semantic attributes.

---

## Files

| File         | Description                                         |
| ------------ | --------------------------------------------------- |
| `AOI_S1.csv` | Labelling session 1 — homepage and overview views   |
| `AOI_S2.csv` | Labelling session 2 — resources and services views  |
| `AOI_S3.csv` | Labelling session 3 — seat-reservation wizard views |
| `AOI_S4.csv` | Labelling session 4 — refinement pass on all views  |
| `AOI_S5.csv` | Labelling session 5 — inter-rater validation pass   |
| `AOI_S6.csv` | Labelling session 6 — final merged output           |

---

## Column Schema

| Column          | Type   | Description                                                |
| --------------- | ------ | ---------------------------------------------------------- |
| `view`          | string | Page name (matches `view` in `dataset/P*/AOI.csv`)         |
| `x1`            | int    | Left edge of bounding box (pixels, 1920 × 1080)            |
| `y1`            | int    | Top edge of bounding box                                   |
| `x2`            | int    | Right edge of bounding box                                 |
| `y2`            | int    | Bottom edge of bounding box                                |
| `componentInfo` | string | Human-readable component name (e.g. `SearchButton`)        |
| `task`          | string | Primary task type: `S` (Search), `R` (Read), `O` (Operate) |
| `isKeyAOI`      | 0 / 1  | Whether this AOI is task-critical                          |
| `issue`         | 0 / 1  | Ground-truth usability-issue label                         |
| `src_file`      | string | Relative path of the Vue source file for this component    |
| `src_line`      | int    | Line number in `src_file` where the component is defined   |

---

## Notes

- Coordinates are in absolute pixels for a 1920 × 1080 screen.
  `process_aoi.py` normalises them to [0, 1] before they are used downstream.
- `src_file` and `src_line` are filled by `add_src_index.py` and are used by
  `extract_context_data.py` to locate the corresponding code snippet.
- The final merged AOI table (all views, all sessions de-duplicated) is
  `dataset/AOI.csv`.

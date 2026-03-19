# scripts/ — Dataset Pre-processing Scripts

These scripts transform the **raw session output** produced by the backend
(`app/backend/data/session_XX/raw_data.csv`) into the **structured per-participant
format** found in `dataset/P*/`.

Run them in the order shown below after collecting raw data from a session.

---

## Execution Order

```
1. inspect_data.py
2. clean_calibration.py
3. final_clean_split.py
4. renumber_split.py
5. process_images.py
6. check_timestamps.py
7. check_counts.py
```

---

## Script Reference

### 1 `inspect_data.py`

**Purpose**: Quick sanity check on a raw session folder.
Prints row counts, timestamp ranges, and flags any obvious data-quality issues
(gaps > 500 ms, out-of-bounds coordinates, duplicate timestamps).

**Input**: `raw_data.csv` in a session folder.

**Output**: Console report only (no files written).

---

### 2 `clean_calibration.py`

**Purpose**: Remove calibration-phase rows from `raw_data.csv`.
Calibration trials are identified by a reserved view name (`Calibration`)
in `view_switch.csv`.

**Input**: `raw_data.csv`, `view_switch.csv`.

**Output**: Overwrites `raw_data.csv` with calibration rows removed.

---

### 3 `final_clean_split.py`

**Purpose**: Split the cleaned `raw_data.csv` into per-view-instance files
(`split_data_01.csv`, `split_data_02.csv`, …).
Uses the timestamps in `view_switch.csv` to determine split boundaries.

**Input**: `raw_data.csv`, `view_switch.csv`.

**Output**: `split_data/split_data_XX.csv` (one file per view instance).

---

### 4 `renumber_split.py`

**Purpose**: Ensure split-data files are sequentially numbered (01, 02, …)
with no gaps, in case earlier cleaning steps removed some files.

**Input**: Existing `split_data/` directory.

**Output**: Renames files in-place.

---

### 5 `process_images.py`

**Purpose**: Generate gaze-overlay visualisations for each view instance.
Produces three image variants per view:

- `origin/` — plain screenshot
- `rawpoint/` — screenshot with raw gaze points overlaid
- `scanpath/` — screenshot with scanpath (fixation bubbles + saccade arrows)

**Input**: `split_data/`, `img/origin/` (screenshots from backend), `AOI.csv`.

**Output**: `img/rawpoint/`, `img/scanpath/`.

---

### 6 `check_timestamps.py`

**Purpose**: Verify that timestamps in all `split_data_XX.csv` files are
monotonically increasing and that no large gaps exist within a view instance.

**Input**: `split_data/` directory.

**Output**: Console report; exits non-zero if any check fails.

---

### 7 `check_counts.py`

**Purpose**: Verify that the number of `split_data_XX.csv` files matches
the number of non-calibration rows in `view_switch.csv`.

**Input**: `split_data/`, `view_switch.csv`.

**Output**: Console report; exits non-zero if counts do not match.

---

## Usage Example

```bash
# Process participant P21's raw session (adjust paths as needed)
SESSION=../P21

python inspect_data.py       $SESSION
python clean_calibration.py  $SESSION
python final_clean_split.py  $SESSION
python renumber_split.py     $SESSION
python process_images.py     $SESSION
python check_timestamps.py   $SESSION
python check_counts.py       $SESSION
```

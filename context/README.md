# context/ — Semantic Knowledge Base Construction Pipeline

This directory implements **Contribution 1** of the GazeDecoder paper:
*Knowledge-driven task context modeling and extraction.*

The pipeline maps each UI component to a context anchor, generates a structured
semantic description from the application source code, and encodes it as a pair
of dense embedding vectors stored in a pre-computed knowledge base `K`.

---

## Concepts

### Context Anchor

For each AOI component `c`:

```
anchor(c) = (id_c, loc_c)
```

where `id_c` is the component identifier and `loc_c = (x1, y1, x2, y2, view)`
is its bounding-box location on the screen.

### Semantic Description

`desc(c)` is a structured natural-language string that captures:

- The component's functional role on the page
- The page it belongs to and its parent container
- The task type it supports: Search (S), Read (R), or Operate (O)
- A code snippet from the Vue source that renders the component

### Knowledge Base

The knowledge base `K` is a JSON file (`context_features/complete_econtext.json`)
that maps each `(view, aoi_id)` pair to:

```json
{
  "aoi_id": 42,
  "view": "SeatSelect",
  "componentInfo": "SeatGrid",
  "desc": "...",
  "text_embedding": [0.123, ...],   // 384-dimensional, all-MiniLM-L6-v2
  "code_embedding": [0.456, ...]    // 384-dimensional, all-MiniLM-L6-v2
}
```

At inference time the GazeDecoder model retrieves the embedding pair
`(v_text_c, v_code_c)` for the AOI that receives the most fixation weight in
the current gaze window.

---

## Pipeline Steps

```
Step 1  add_src_index.py         Index Vue source files → line-number map
Step 2  extract_context_data.py  AOI labelling + code-snippet extraction
Step 3  process_aoi.py           Normalise AOI bounding boxes
Step 4  distribute_aoi.py        Assign AOIs to views; resolve overlaps
Step 5  build_context_features.py Build pseudo-DOM hierarchy; write desc(c)
Step 6  process_context_logic.py  Encode desc(c) with SentenceTransformer → K
```

---

## Script Reference

### `add_src_index.py`

Walks `frontend_src/src/` and builds an index mapping component names to
their source-file path and relevant line numbers.
Output is used by `extract_context_data.py` to locate code snippets.

### `extract_context_data.py`

Reads the per-user AOI CSV files in `user_aoi_labeled/` and the source index.
For each AOI it extracts a ±5-line code snippet centred on the component's
source line.
Output: `context_features/context_extraction_raw.json`.

### `process_aoi.py`

Normalises bounding-box coordinates to the range [0, 1] relative to the
1920 × 1080 screen resolution and filters out AOIs with zero area.

### `distribute_aoi.py`

Resolves AOI overlaps using an IoU threshold and assigns each AOI to exactly
one view.
Produces the merged `AOI.csv` at the dataset root.

### `build_context_features.py`

Builds a pseudo-DOM tree for each view by nesting AOIs based on containment.
Generates the `desc(c)` string for each node using the domain-knowledge
dictionary and the extracted code snippet.
Output: enriches `context_extraction_raw.json` with `desc` fields.

### `process_context_logic.py`

Loads `context_extraction_raw.json`, encodes each `desc(c)` and code snippet
with `all-MiniLM-L6-v2` (SentenceTransformer), and writes the final knowledge
base to `context_features/complete_econtext.json`.

### `dataset_loader_example.py` / `dataset_loader_units.py`

Utility helpers demonstrating how to load the knowledge base and match
it to gaze windows.
These are the same loaders used internally by `model/shared/dataset.py`.

---

## Directory Layout

```
context/
├── README.md                      This file
├── app-function-spec.md           Complete specification of all application pages and components
├── add_src_index.py               Step 1
├── extract_context_data.py        Step 2
├── process_aoi.py                 Step 3
├── distribute_aoi.py              Step 4
├── build_context_features.py      Step 5
├── process_context_logic.py       Step 6
├── dataset_loader_example.py      Usage example
├── dataset_loader_units.py        Unit helpers
├── user_aoi_labeled/              Manually labelled AOI CSV files (per session)
│   ├── README.md
│   └── AOI_S1.csv … AOI_S6.csv
├── frontend_src/                  Snapshot of Vue source used for code embedding
│   └── src/
└── context_features/              Pre-computed knowledge base K
    ├── README.md
    ├── context_extraction_raw.json
    └── complete_econtext.json
```

---

## Re-running the Pipeline

```bash
cd context

# Requires SentenceTransformers; install model dependencies first:
pip install -r ../model/requirements.txt

python add_src_index.py
python extract_context_data.py
python process_aoi.py
python distribute_aoi.py
python build_context_features.py
python process_context_logic.py
```

The final knowledge base is written to
`context_features/complete_econtext.json`.
The pre-computed version is already committed to this repository, so this step
is only needed if you modify the source application or AOI definitions.

---

## Configuration

All scripts accept their input/output paths as command-line arguments
(using `argparse`) so that no hard-coded paths are embedded in the code.
Run any script with `--help` to see available options.

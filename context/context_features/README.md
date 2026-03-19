# context_features/ — Pre-computed Knowledge Base K

This directory stores the pre-computed semantic knowledge base `K` that the
GazeDecoder model uses at inference time to retrieve component-level context
embeddings.

---

## Files

### `context_extraction_raw.json`

Intermediate output of `extract_context_data.py` and `build_context_features.py`.
Contains one entry per AOI with the raw extracted code snippet and the generated
`desc(c)` description string, before embedding.

Schema (one JSON object per AOI):

```json
{
  "aoi_id": 42,
  "view": "SeatSelect",
  "componentInfo": "SeatGrid",
  "task": "O",
  "isKeyAOI": 1,
  "issue": 1,
  "x1": 240, "y1": 180, "x2": 1680, "y2": 920,
  "src_file": "views/reserve/SeatSelect.vue",
  "src_line": 87,
  "code_snippet": "...",
  "desc": "SeatGrid on SeatSelect: interactive seat-selection grid ..."
}
```

### `complete_econtext.json`

The final knowledge base produced by `process_context_logic.py`.
This is the file consumed by `model/shared/dataset.py` when building feature
vectors for the model.

Schema (one JSON object per AOI):

```json
{
  "aoi_id": 42,
  "view": "SeatSelect",
  "componentInfo": "SeatGrid",
  "task": "O",
  "isKeyAOI": 1,
  "issue": 1,
  "x1": 240, "y1": 180, "x2": 1680, "y2": 920,
  "desc": "SeatGrid on SeatSelect: interactive seat-selection grid ...",
  "text_embedding": [0.123, -0.045, ...],
  "code_embedding": [0.456,  0.012, ...]
}
```

#### Embedding Details

| Field            | Model                                     | Dimension |
| ---------------- | ----------------------------------------- | --------- |
| `text_embedding` | `all-MiniLM-L6-v2` (SentenceTransformers) | 384       |
| `code_embedding` | `all-MiniLM-L6-v2` (SentenceTransformers) | 384       |

`text_embedding` encodes `desc(c)` — the natural-language functional description.
`code_embedding` encodes the Vue source snippet extracted from the component.
Together they form the 768-dimensional semantic context slice `[2:770]` of the
feature vector used by GazeDecoder.

---

## Retrieval at Inference Time

Given a gaze window `W`:

1. Identify the AOI `c*` that receives the highest fixation weight in `W`.
2. Look up `c*` in `K` by `(view, aoi_id)`.
3. Concatenate `[text_embedding, code_embedding]` → 768-d context vector.
4. Pass to the Input Injection Block (IIB) of GazeDecoder.

The lookup is an O(1) dictionary access; no online computation is required.

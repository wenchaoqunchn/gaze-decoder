# dataset/ — Eye-Tracking Dataset

This directory contains the complete eye-tracking dataset collected for the
GazeDecoder study.

## Summary

20 participants · 28 UI pages · 3 037 labelled gaze windows

---

## Study Design

| Attribute                    | Value                                                       |
| ---------------------------- | ----------------------------------------------------------- |
| Participants                 | 20 (anonymised P1–P20)                                      |
| System under test            | University library portal (see `app/`)                      |
| Task                         | Seat-reservation workflow (multi-step task flow)            |
| Eye-tracker                  | Tobii Eye Tracker 5 (30 Hz)                                 |
| Screen resolution            | 1920 × 1080                                                 |
| Labelling unit               | AOI × view instance                                         |
| Positive label (`issue = 1`) | AOI reported as problematic in the post-task interview      |
| Negative label (`issue = 0`) | AOI not reported as problematic                             |
| Total gaze windows           | 3 037                                                       |
| Window length                | 64 frames (\(\omega = 64\))                                 |
| Window stride                | 32 samples (50 % overlap)                                   |

### Labelling Protocol

Ground truth is derived from a **semi-structured interview** conducted
immediately after task completion. Participants were asked to (1) report any
frustrations or obstacles encountered on specific pages and (2) review their own
gaze visualizations (e.g., heatmaps) to surface additional issues.

Only usability issues **spontaneously reported by participants and confirmed
through the interview** are treated as ground truth. No issues were introduced
independently by the researchers.

At the dataset level, UI components (AOIs) associated with participant-reported
issues are labeled `issue = 1`; all other components are labeled `issue = 0`.

At the window level, sliding windows are constructed with \(\omega = 64\) frames
and \(\delta = 32\) frames (50% overlap). Windows that span a view transition are
truncated at the boundary so that each window corresponds to a single UI view.

---

## Directory Layout

```text
dataset/
├── README.md            This file
├── docs/
│   ├── index.md         Documentation entry point
│   ├── dataset-structure.md   Detailed folder / file descriptions
│   └── data-dictionary.md     Column definitions for all CSV files
├── scripts/             Pre-processing scripts (raw → structured)
│   └── README.md
├── AOI.csv              Global AOI definition table (all views, merged)
└── P1/ … P20/           Per-participant data
    ├── AOI.csv          AOI definitions for this participant's session
    ├── raw_data.csv     Raw gaze stream (time, x, y) — pre-split
    ├── view_switch.csv  Ordered view-visit log
    ├── metrics.csv      AOI-level derived metrics (one row per AOI per view instance)
    ├── img/
    │   ├── origin/      Original screenshots
    │   ├── rawpoint/    Gaze-point overlay images
    │   └── scanpath/    Scanpath visualisations
    └── split_data/
        ├── split_data_01.csv   Gaze samples for view-instance 01
        ├── split_data_02.csv   Gaze samples for view-instance 02
        └── …
```

---

## Key Concepts

### View Instance

A **view instance** is one continuous visit to one page.
The same page (e.g. `HomePage`) may be visited multiple times in a session;
each visit is a separate view instance and corresponds to one `split_data_XX.csv`
file.
View instances are indexed in the same order as rows in `view_switch.csv`
(after excluding calibration rows).

### AOI (Area of Interest)

An AOI is a rectangular bounding box on a specific page that corresponds to one
semantic UI component (e.g. the "Search" button, the carousel navigation arrow).
AOI definitions are stored in `AOI.csv` with columns `x1, y1, x2, y2, view,
componentInfo, task, isKeyAOI, issue`.

### Feature Vector

For each gaze window the model pipeline (in `model/`) constructs a **786-dimensional**
feature vector:

| Slice       | Dimension | Content                                                           |
| ----------- | --------- | ----------------------------------------------------------------- |
| `[0:2]`     | 2         | Normalised gaze coordinates (x, y)                                |
| `[2:770]`   | 768       | Semantic context: Text embedding (384-d) + Code embedding (384-d) |
| `[770:778]` | 8         | Layer-1 micro-window behavioural features                         |
| `[778:786]` | 8         | Layer-2 macro-window behavioural features                         |

---

## Ethics and Privacy

All participants provided written informed consent.
Participant IDs are anonymised (P1–P20); no personally identifiable information
is stored in this repository.
The study was conducted in accordance with the institutional ethics review
guidelines.



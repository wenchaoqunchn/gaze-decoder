# dataset/ — Eye-Tracking Dataset

This directory contains the complete eye-tracking dataset collected for the
GazeDecoder study.

**20 participants · 28 views · 3 037 labelled gaze windows**

---

## Study Design

| Attribute                    | Value                                                       |
| ---------------------------- | ----------------------------------------------------------- |
| Participants                 | 20 (anonymised P1–P20)                                      |
| System under test            | University library portal (see `app/`)                      |
| Task                         | Free navigation following a structured scenario             |
| Eye-tracker                  | Tobii Pro (60 Hz)                                           |
| Screen resolution            | 1920 × 1080                                                 |
| Labelling unit               | AOI × view instance                                         |
| Positive label (`issue = 1`) | AOI where observed behaviour indicates a usability obstacle |
| Negative label (`issue = 0`) | AOI visited without difficulty                              |
| Total gaze windows           | 3 037                                                       |
| Window length                | 64 gaze samples (≈ 1.07 s at 60 Hz)                         |
| Window stride                | 32 samples (50 % overlap)                                   |

### Labelling Protocol

Ground-truth labels were assigned by two independent raters using a combination
of think-aloud recordings, post-session interviews, and fixation-duration
thresholds.
Inter-rater agreement (Cohen's κ) was computed and disagreements resolved by
discussion.

A gaze window is labelled as a **usability issue** (`issue = 1`) if the
participant showed any of the following in that window:

- Repeated fixation oscillation between two UI regions
- Fixation duration significantly exceeding the expected reading time for the component
- Backtracking saccade after an incorrect selection
- Verbal or gestural hesitation recorded during think-aloud

---

## Directory Layout

```
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

---

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{gazedecoder2026tse,
  title   = {GazeDecoder: Context-Aware Usability Issue Detection from Eye-Tracking Data},
  author  = {[Authors]},
  journal = {IEEE Transactions on Software Engineering},
  year    = {2026}
}
```

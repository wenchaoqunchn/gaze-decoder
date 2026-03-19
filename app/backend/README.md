# backend/ — Eye-Tracker Control API (Flask)

The backend is a lightweight Flask application that acts as the bridge between
the Vue frontend and the Tobii eye-tracker hardware.
It manages participant sessions, controls gaze recording, and persists raw data
to disk for later processing.

---

## Tech Stack

- **Python 3.9+**
- **Flask** web framework
- **PyGazeAnalyser** (bundled in `lib/PyGazeAnalyser/`) — Tobii SDK wrapper
- **Tobii Pro SDK** DLLs (bundled in `lib/dlls/`) — Windows-only

---

## Setup

```bash
pip install -r requirements.txt
python run.py
```

Server starts at `http://localhost:5000`.

### Eye-Tracker Hardware

The backend is designed for Tobii Pro eye-trackers connected over USB.
If no tracker is connected, the server still starts but `/start` and `/stop`
will return an error.
Recorded sessions are written to `data/session_XX/`.

---

## API Reference

All endpoints accept and return JSON.

### `POST /init`

Initialise a new participant session.

**Request body**

```json
{ "focus_session": "P01" }
```

**Response**

```json
{ "message": "Session initialized. Focus session=P01." }
```

### `POST /start`

Begin gaze recording for the current session.

**Response** `200 OK` on success; `500` on tracker error.

### `POST /stop`

Stop gaze recording and flush the buffer to `raw_data.csv`.

### `POST /screenshot`

Capture the current screen and save it under
`data/session_XX/img/screenshot_NNN.png`.
Called automatically by the frontend on every page navigation.

### `POST /clear`

Delete all files and subdirectories in the current session's data folder.
Use with caution — this is irreversible.

### `GET /status`

Return the current session metadata and eye-tracker connection status.

---

## Directory Layout

```
backend/
├── run.py               Application entry point
├── requirements.txt     Python dependencies
├── app/
│   ├── __init__.py      create_app() factory
│   ├── config.py        Flask configuration (debug mode, data path, etc.)
│   └── routes.py        All HTTP route handlers
├── lib/
│   ├── README.md        Third-party library attributions
│   ├── __init__.py
│   ├── dlls/            Tobii Pro SDK native DLLs (Windows x64)
│   └── PyGazeAnalyser/  Open-source gaze-analysis helpers (BSD licence)
└── utils/
    ├── __init__.py
    ├── eye_tracking.py  EyeTracker class — SDK wrapper
    ├── session_tools.py SessionManager class — on-disk session lifecycle
    └── data_analysis.py DataAnalyzer class — real-time metric computation
```

---

## Data Output Format

Each session creates the following files:

```
data/session_XX/
├── raw_data.csv        Timestamped gaze coordinates (time, x, y)
└── img/
    └── screenshot_NNN.png   One screenshot per page visit
```

`raw_data.csv` is the input for the pre-processing pipeline in `dataset/scripts/`.

# app/ — Stimulus Web Application

This directory contains the library-portal web application used as the stimulus
system in the GazeDecoder eye-tracking study.
The application is a faithful simulation of a real university library website and
was intentionally designed with several navigational and interaction issues to
provide a controlled set of usability problems for the experiment.

---

## Architecture

```
app/
├── frontend/    Vue 3 (Vite) single-page application — the library website
└── backend/     Flask REST API — eye-tracker control + session management
```

The **frontend** renders the stimulus pages and communicates with the backend
to start/stop gaze recording and to request screenshots at each page transition.

The **backend** wraps the Tobii eye-tracker SDK (via PyGazeAnalyser) and manages
per-participant session data.

---

## Prerequisites

| Component                    | Version                   |
| ---------------------------- | ------------------------- |
| Node.js                      | ≥ 18                      |
| npm                          | ≥ 9                       |
| Python                       | ≥ 3.9                     |
| Tobii eye-tracker (optional) | Pro SDK compatible device |

A Tobii device is only required for live gaze recording.
The frontend can be run standalone for UI development and inspection.

---

## Running the Full System

### Step 1 — Start the backend

```bash
cd app/backend
pip install -r requirements.txt
python run.py
```

The Flask server starts on `http://localhost:5000` by default.

### Step 2 — Start the frontend

```bash
cd app/frontend
npm install
npm run dev
```

The Vite dev server starts on `http://localhost:5173` by default.
Open that URL in a browser to view the stimulus application.

---

## Application Pages

The application follows the view-naming convention used throughout the dataset.
Each `view` name maps directly to the `view` column in `dataset/P*/AOI.csv`.

| View name          | Description                                     |
| ------------------ | ----------------------------------------------- |
| `HomePage`         | Main portal: global nav, search, news, carousel |
| `Overview`         | Library information hub (side-nav container)    |
| `LibIntro`         | Library history and collection statistics       |
| `LeaderSpeech`     | Director's message                              |
| `LibRule`          | Rules and regulations                           |
| `ServiceTime`      | Opening-hours table for each library zone       |
| `ServiceOverview`  | Service catalogue with locations and contacts   |
| `LibLayout`        | Floor-plan carousel (F2 / F3 / F4)              |
| `Resources`        | Digital-resource navigation hub                 |
| `CoreJournal`      | Core-journal index guide (SCI, SSCI, CSSCI …)   |
| `EBook`            | E-book database list                            |
| `LibThesis`        | Institutional thesis repository                 |
| `CommonApp`        | Research-tool download page                     |
| `Copyright`        | Copyright notice                                |
| `ServicePage`      | Service catalogue landing page                  |
| `BookBorrow`       | Book borrowing rules                            |
| `CardProcess`      | Campus-card activation guide                    |
| `AncientRead`      | Rare-books reading-room rules                   |
| `DiscRequest`      | Accompanying-disc request                       |
| `DocumentTransfer` | Inter-library loan                              |
| `TechSearch`       | Sci-tech novelty search                         |
| `InfoTeaching`     | Literature-retrieval course guide               |
| `VolunteerTeam`    | Volunteer programme info                        |
| `SeatReserve`      | Seat-reservation entry page                     |
| `Reserve`          | Seat-reservation wizard container               |
| `FloorSelect`      | Step 1: choose floor                            |
| `TimeSelect`       | Step 2: choose date and time slot               |
| `SeatSelect`       | Step 3: choose a seat on the floor map          |
| `ReserveConfirm`   | Step 4: confirm reservation details             |

---

## Backend API Endpoints

| Method | Path          | Description                                   |
| ------ | ------------- | --------------------------------------------- |
| `POST` | `/init`       | Initialise a new participant session          |
| `POST` | `/start`      | Begin gaze recording                          |
| `POST` | `/stop`       | Stop gaze recording                           |
| `POST` | `/screenshot` | Capture current screen                        |
| `POST` | `/clear`      | Delete all data in the current session folder |
| `GET`  | `/status`     | Return current session and tracker status     |

See [`backend/app/routes.py`](backend/app/routes.py) for full request/response schemas.

---

## Data Output

Raw gaze data is written to `backend/data/session_XX/raw_data.csv` during a session.
After the session ends, the pre-processing pipeline in `dataset/scripts/` converts
this raw output into the structured per-participant format found in `dataset/P*/`.

# frontend/ — Stimulus Web Application (Vue 3)

The stimulus application is a Vue 3 single-page application built with Vite.
It renders a simulated university library portal that participants navigated
during the eye-tracking experiment.

---

## Tech Stack

- **Vue 3** (Composition API + `<script setup>`)
- **Vite** build tool
- **Vue Router** for client-side navigation
- **Element Plus** UI component library
- **Pinia** state management

---

## Setup

```bash
npm install
npm run dev        # development server at http://localhost:5173
npm run build      # production build → dist/
npm run preview    # preview production build
```

---

## Source Layout

```
src/
├── main.js              Application entry point; registers Vue, router, Pinia
├── App.vue              Root component — renders <router-view>
├── style.css            Global styles
├── AOISelector.vue      AOI overlay component for hitbox visualisation
├── getAOIInfo.js        Utility: fetch AOI bounding-box data from the backend
├── assets/              Static images (library photos, icons)
├── components/          Shared UI components
│   ├── MainHeader.vue   Global navigation bar
│   ├── NewsList.vue     Homepage news list widget
│   ├── PosterPlayer.vue Carousel / poster player widget
│   ├── SearchBox.vue    Resource-type search box
│   └── StepBar.vue      Reservation wizard progress indicator
├── router/              Vue Router configuration (all page routes)
└── views/               Page-level view components (one file per view)
    ├── HomePage.vue
    ├── MainLayout.vue
    ├── ContactUs.vue
    ├── SessionReady.vue   Pre-session calibration entry screen
    ├── SessionDone.vue    Post-session thank-you screen
    ├── calibration/       Calibration sub-views
    ├── overview/          Library overview sub-views
    ├── reserve/           Seat-reservation wizard views
    ├── resources/         Resource navigation views
    └── services/          Library services views
```

---

## View → Page Name Mapping

Each view component maps to a `view` name used in the dataset's `AOI.csv` and
`view_switch.csv` files.

| Component file                        | `view` name in dataset |
| ------------------------------------- | ---------------------- |
| `views/HomePage.vue`                  | `HomePage`             |
| `views/overview/LibIntro.vue`         | `LibIntro`             |
| `views/overview/LeaderSpeech.vue`     | `LeaderSpeech`         |
| `views/overview/LibRule.vue`          | `LibRule`              |
| `views/overview/ServiceTime.vue`      | `ServiceTime`          |
| `views/overview/ServiceOverview.vue`  | `ServiceOverview`      |
| `views/overview/LibLayout.vue`        | `LibLayout`            |
| `views/resources/CoreJournal.vue`     | `CoreJournal`          |
| `views/resources/EBook.vue`           | `EBook`                |
| `views/resources/LibThesis.vue`       | `LibThesis`            |
| `views/resources/CommonApp.vue`       | `CommonApp`            |
| `views/resources/Copyright.vue`       | `Copyright`            |
| `views/services/BookBorrow.vue`       | `BookBorrow`           |
| `views/services/CardProcess.vue`      | `CardProcess`          |
| `views/services/AncientRead.vue`      | `AncientRead`          |
| `views/services/DiscRequest.vue`      | `DiscRequest`          |
| `views/services/DocumentTransfer.vue` | `DocumentTransfer`     |
| `views/services/TechSearch.vue`       | `TechSearch`           |
| `views/services/InfoTeaching.vue`     | `InfoTeaching`         |
| `views/services/VolunteerTeam.vue`    | `VolunteerTeam`        |
| `views/services/SeatReserve.vue`      | `SeatReserve`          |
| `views/reserve/FloorSelect.vue`       | `FloorSelect`          |
| `views/reserve/TimeSelect.vue`        | `TimeSelect`           |
| `views/reserve/SeatSelect.vue`        | `SeatSelect`           |
| `views/reserve/ReserveConfirm.vue`    | `ReserveConfirm`       |

---

## Experiment-Specific Behaviour

The application has two additions beyond a normal website:

1. **Screenshot requests** — on every route change, the frontend sends a `POST /screenshot`
   request to the backend so that a screen capture is stored alongside the gaze data.

2. **AOI export** — `AOISelector.vue` can render a transparent overlay showing AOI
   bounding boxes; `getAOIInfo.js` serialises these coordinates for the knowledge-base
   pipeline in `context/`.

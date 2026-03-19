# Application Function Specification
## Beijing University of Technology Library Portal (EyeSeq Experiment Edition)

This document is the English translation of the original Chinese functional
specification (`软件功能文档.md`).
It describes every page and interaction in the stimulus application used in the
GazeDecoder eye-tracking study.

---

## 1. System Overview

The system is a simulated frontend portal for the Beijing University of Technology
(BJUT) library website, built specifically for the EyeSeq eye-tracking experiment.
It is implemented in Vue.js 3 with the Element Plus component library and provides
complete route navigation, static information pages, and specific interactive
features (such as seat reservation).

Two experiment-specific additions are integrated:

- **Automatic screenshot requests**: on every route transition, the frontend sends
  a `POST /api/switch` request to the backend to record the page and timestamp.
- **AOI coordinate export**: the `exportAllAOIInfo` utility extracts bounding-box
  coordinates of DOM elements marked with `aoi`, `aoim`, `aoip`, or `key-aoi`
  classes for use in the knowledge-base pipeline.

---

## 2. Module Specifications

### 2.1 Home Page — `[view: HomePage]`

**Global navigation bar**

- Entry points for all major modules: Home, Overview, Resources, Services, Contact Us.
- Displays the library logo.
- Shows today's opening hours (static: 7:30–22:30).

**Search function**

- Resource-type toggle: "Physical books/journals", "Electronic journals",
  "Chinese and foreign articles".
- Search box with a search button (simulated click, no backend search logic).

**Library photos carousel**

- Displays real photos of library zones (Chinese-language book stacks, central
  atrium, self-study rooms, etc.).
- Supports auto-play and manual navigation.

**News list**

- Displays notices, lecture announcements, and resource recommendations.
- Different item types are distinguished by colour-coded tags.

**Footer**

- Friendly links (e.g. National Library, CALIS) and copyright information.

**Background theme**

- Automatically switches between day and night backgrounds based on system time
  (6:00–18:00 = day; otherwise = night).

---

### 2.2 Library Overview — `[view: Overview]`

Static information hub with a side navigation bar.

| Sub-page             | View name         | Content                                                                 |
| -------------------- | ----------------- | ----------------------------------------------------------------------- |
| Library Introduction | `LibIntro`        | History, floor area, collection size, service model                     |
| Director's Message   | `LeaderSpeech`    | Library director's welcome message                                      |
| Rules & Regulations  | `LibRule`         | Detailed reader violation handling rules                                |
| Service Hours        | `ServiceTime`     | Table of opening hours for all zones (Reading Room, Stacks, Study Room) |
| Service Overview     | `ServiceOverview` | Table of services with locations and contact numbers                    |
| Library Layout       | `LibLayout`       | Floor-plan carousel for F2, F3, and F4                                  |

---

### 2.3 Resources — `[view: Resources]`

Navigation hub for digital and physical library resources.

| Sub-page           | View name     | Content                                                                             |
| ------------------ | ------------- | ----------------------------------------------------------------------------------- |
| Core Journal Guide | `CoreJournal` | Usage guide for SCI, SSCI, CSSCI and other index databases                          |
| E-books            | `EBook`       | List of purchased Chinese and foreign e-book databases (SuperStar, Springer, Ebsco) |
| Theses             | `LibThesis`   | Coverage and query URL for the institutional thesis repository                      |
| Software Tools     | `CommonApp`   | Download guide for NoteExpress, EndNote, Adobe Reader, CAJViewer                    |
| Copyright Notice   | `Copyright`   | Statement on electronic-resource copyright protection                               |

---

### 2.4 Services — `[view: ServicePage]`

Detailed descriptions of library services.

| Sub-page                      | View name          | Content                                                                     |
| ----------------------------- | ------------------ | --------------------------------------------------------------------------- |
| Book Borrowing                | `BookBorrow`       | Borrowing rules, self-service machine locations, reading-area notices       |
| Card Processing               | `CardProcess`      | Campus-card borrowing-permission activation guide for staff and students    |
| Rare Books Reading            | `AncientRead`      | Opening hours, reservation process, reading-room rules for rare books       |
| Disc Request                  | `DiscRequest`      | Retrieval methods for book-accompanying discs (database, management system) |
| Document Transfer / ILL       | `DocumentTransfer` | Obtaining documents not held by the library; contact information            |
| Sci-Tech Search               | `TechSearch`       | Novelty-search service qualifications, scope, and delegation process        |
| Literature Retrieval Teaching | `InfoTeaching`     | Retrieval-course setup for graduate students                                |
| Volunteer Programme           | `VolunteerTeam`    | Volunteer organisation info and QR-code registration                        |
| Seat Reservation Entry        | `SeatReserve`      | Introduction to the seat-reservation system; "Start Reservation" button     |

---

### 2.5 Seat Reservation System — `[view: Reserve]`

The core interactive module of the application; uses a four-step wizard pattern.
A step progress bar at the top shows the current step:
**Floor → Date/Time → Seat → Confirm**.

#### Step 1 — Floor Selection `[view: FloorSelect]`

- Displays cards for floors F2, F3, and F4.
- Each card shows a floor photograph and the book classification for that floor
  (F2 = Social Sciences, F3 = Science & Technology).
- Clicking a card selects the floor.

#### Step 2 — Date and Time Selection `[view: TimeSelect]`

- **Date picker**: calendar component; only future dates within the next 7 days
  are selectable (past dates are disabled).
- **Time picker**: start and end times within opening hours (7:30–22:30).
- Advisory text displays reservation rules and available hours.

#### Step 3 — Seat Selection `[view: SeatSelect]`

- **Seat layout**: seats are displayed as a grid.
- **Status indicators**:
  - Available — reading-lamp icon
  - Occupied — grey, non-clickable
  - Selected — green highlight
- **Interaction logic**:
  - Clicking an available seat selects it.
  - **Experiment variable (simulated usability issue)**: 30% of click events
    produce no response. On some clicks, the system selects an adjacent seat
    instead of the target seat. This behaviour is intentional and is designed to
    induce measurable usability issues for the eye-tracking experiment.

#### Step 4 — Confirmation `[view: InfoConfirm]`

- Summarises the chosen floor, date, time slot, and seat number.
- Prompts the user to verify the information carefully.
- Clicking "Submit" completes the reservation and navigates to `SessionDone`.

**Navigation controls**: Previous / Next–Submit buttons at the bottom support
movement between steps.

---

### 2.6 Contact Us — `[view: ContactUs]`

- General enquiries, group visits, and front-desk contact numbers.
- Detailed table of phone numbers and email addresses for each department
  (General Affairs, Reference, Cataloguing, etc.).

---

### 2.7 Experiment Control and Auxiliary Features

#### Session Configuration — `[view: SessionReady]`

Pre-experiment setup screen:

- Set the session index (participant ID).
- Toggle: disable all POST API requests (for UI-only testing).
- Toggle: export AOI coordinate data on page transitions.
- Button: enter the calibration flow, then proceed to the home page.

#### Calibration Module

Multiple calibration-point screens (C1, C2, C3) plus a preparation screen,
used for eye-tracker calibration before the experiment begins.

#### Session Done — `[view: SessionDone]`

Displayed after the reservation is completed.
Provides buttons for: End Experiment, Analyse Data, Return to Home.

---

## 3. Technical Notes

- **State management**: Pinia stores are used to share reservation-flow state
  (floor, time slot, seat) and experiment configuration across components.
- **Layout**: optimised for 1920 × 1080 desktop resolution.
- **Simulated data**: seat status and some book-classification data are provided
  as static front-end data.

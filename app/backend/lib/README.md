# lib/ — Third-Party Libraries

This directory contains two bundled third-party components required by the
eye-tracker backend.
They are included in the repository for reproducibility because they are either
platform-specific binaries or lightly patched versions of upstream packages.

---

## `dlls/` — Tobii Pro SDK Native Libraries

**Source**: Tobii Technology AB — Tobii Pro SDK for Python
**Platform**: Windows x64
**Licence**: Tobii Pro SDK licence agreement
([tobii.com/products/software/tobii-pro-sdk/](https://www.tobii.com/products/software/tobii-pro-sdk/))

These DLLs are the native runtime components of the Tobii Pro SDK.
They enable the Python layer (`utils/eye_tracking.py`) to communicate with
a connected Tobii Pro eye-tracker over USB.

> **Note**: The DLLs only work on Windows.
> On other platforms the server starts but gaze recording is unavailable.

---

## `PyGazeAnalyser/` — Gaze-Data Analysis Helpers

**Source**: Edwin Dalmaijer — PyGazeAnalyser
**Upstream**: [github.com/esdalmaijer/PyGazeAnalyser](https://github.com/esdalmaijer/PyGazeAnalyser)
**Licence**: GNU General Public Licence v3 (GPLv3)

PyGazeAnalyser provides fixation detection (dispersion-based and velocity-based
I-DT / I-VT algorithms) and basic scanpath utilities.
The copy here may include minor local modifications for compatibility with the
session data format used in this project.

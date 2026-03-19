import os
import shutil
import csv

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "user_aoi_labeled")
GAZE_DIR = os.path.join(BASE_DIR, "../gaze")


def distribute_aoi_files():
    # verify that the gaze data root exists before copying
    if not os.path.exists(GAZE_DIR):
        print(f"Gaze directory not found: {GAZE_DIR}")
        return

    # 1. Handle P1 – P6 (corresponding to labelled participants S1 – S6)
    for i in range(1, 7):
        user_id = f"S{i}"
        participant_id = f"P{i}"

        src_file = os.path.join(SOURCE_DIR, f"AOI_{user_id}.csv")
        dest_dir = os.path.join(GAZE_DIR, participant_id)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Destination filename: use AOI.csv so downstream scripts have a consistent name.
        # The participant folder already differentiates users.
        dest_file = os.path.join(dest_dir, "AOI.csv")

        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} -> {dest_file}")
        else:
            print(f"Warning: Source file {src_file} not found.")

    # 2. Handle P7, P8 (no-issue control group)
    # Use AOI_S1.csv as a structural template and reset all defect columns.
    template_file = os.path.join(SOURCE_DIR, "AOI_S1.csv")
    if not os.path.exists(template_file):
        print("Error: Template file for P7/P8 not found.")
        return

    # read template and clear defect-related columns
    clean_rows = []
    fieldnames = []

    with open(template_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # reset all defect label columns
            row["is_designed_defect"] = "False"
            row["is_reported_by_user"] = "False"
            row["issue"] = "False"
            if "mapped_defect_ids" in row:
                row["mapped_defect_ids"] = ""

            clean_rows.append(row)

    # write clean AOI files for P7 and P8
    for p_id in ["P7", "P8"]:
        dest_dir = os.path.join(GAZE_DIR, p_id)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_file = os.path.join(dest_dir, "AOI.csv")

        with open(dest_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)
            print(f"Generated clean AOI file -> {dest_file}")


if __name__ == "__main__":
    distribute_aoi_files()

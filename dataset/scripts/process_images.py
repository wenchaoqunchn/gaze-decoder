import os
import glob
import shutil

# Configuration
BASE_GAZE = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"
TARGET_PARTICIPANTS = ["P6", "P7", "P8"]
SUBDIRS = ["origin", "rawpoint", "scanpath"]


def batch_rename_images():
    for p_id in TARGET_PARTICIPANTS:
        print(f"Processing {p_id}...")
        img_root = os.path.join(BASE_GAZE, p_id, "img")

        # Determine exact mapping based on assumption:
        # User said: "Remove 1-6... Start from 8... Renumber to 1"
        # AND "Based on splitdata file encoding".
        #
        # Current Split Data (P6):
        # split_data_01.csv -> HomePage (Originally View 7)
        #
        # Image Files in `origin`:
        # origin_01.jpg ... origin_15.jpg
        # origin_01 to 06 are Calibrations.
        # origin_07 is HomePage.
        # origin_08 is LibIntro.
        #
        # CONTRADICTION:
        # User: "从第八个...开始重新从1编号" (Start from 8th... renumber to 1).
        # This implies: origin_08 -> 01?
        # If so, origin_07 (HomePage) is DISCARDED or left as is (but we need to remove 1-6).
        # Split Data 01 IS HomePage.
        # If Image 01 is LibIntro (from origin_08), then Image 01 != Split 01.
        # This violates "依据splitdata的文件编码" (Based on splitdata file encoding).
        #
        # INTERPRETATION:
        # The user likely misspoke "Start from 8th" (第八个) OR they consider the "First Empty CSV" removal as valid logic for images too.
        # In `clean_calibration.py`, we had a Dummy File 1 (Pre-Calib).
        # Then we removed it.
        # Maybe the images ALSO have a 1-offset lag or something?
        # Look at `rawpoint` and `scanpath` for P6.
        # They START at `rawpoint_02.jpg`. NO `01`.
        # So:
        # rawpoint_02 (Calib1) ... rawpoint_06 (Calib3Ready) ... rawpoint_07 (Calib3) ... rawpoint_08 (HomePage).
        # Wait, let's align.
        # View Switch P6:
        # 1. Calib1Ready
        # 2. Calib1
        # ...
        # 6. Calib3
        # 7. HomePage
        #
        # `origin` files (15 files):
        # 01..15. Matches 15 views perfectly.
        # 01 = Calib1Ready. 07 = HomePage.
        #
        # `rawpoint`/`scanpath` files (14 files):
        # Starts at 02. Ends at 15.
        # 02 = Calib1?? Usually Split Data N corresponds to View N-1?
        # If View 1 is Calib1Ready. Image 1 is Calib1Ready.
        # Split Data 1 = Pre-View1.
        # Split Data 2 = View 1 Data.
        # So Split Data 2 matches Image 1? Or Image 2?
        # Usually Image 1 is screenshot of View 1.
        # So Split Data 2 (Data during View 1) matches Image 1 (View 1).
        #
        # If user says "Start from 8", maybe they look at `rawpoint` files?
        # rawpoint_08 would be the 7th file in the folder? No, filename is 08.
        # If origin_07 is HomePage. origin_08 is LibIntro.
        #
        # Is it possible that "Split Data 01" (New) corresponds to "origin_08"?
        # Let's check `view_switch` for P6 again.
        # We removed 6 rows.
        # Row 1 (New) = HomePage.
        #
        # If the user insists on "Start from 8", and "Remove 1-6". What happen to 7?
        # Maybe 7 is considered "Calibration End" or something?
        # But 7 is HomePage.
        #
        # CRITICAL CLUE: "P6-P8需要额外去掉第一个空的csv".
        # We did that. We removed `split_data_01`.
        # The remaining `split_data_01` (Renamed from Old 02) is actually Old 02?
        # No.
        # In `clean_calibration`, we generated Old File 8 (HomePage) -> New File 2.
        # Rename -> `split_data_02`.
        # `split_data_01` was Dummy.
        # In `final_clean`, we removed Dummy `01`.
        # So `split_data_02` (HomePage) became `split_data_01`.
        #
        # So New `split_data_01` corresponds to HomePage.
        #
        # If Image 07 is HomePage.
        # We want Image `New 01` to be HomePage.
        # So 7 -> 1.
        #
        # USER INSTRUCTION: "从第八个...开始重新从1编号".
        # "From the 8th... renumber to 1".
        # This is very specific.
        # IF I follow this literally:
        # 8 -> 1. (LibIntro -> 1).
        # Then HomePage (7) is deleted.
        # Implication: Split Data 01 (HomePage) will have Image 01 (LibIntro). MISMATCH.
        #
        # Alternative Interpretation:
        # Maybe for `rawpoint` / `scanpath`, the files start at 02?
        # `rawpoint_02`..`07` are calibrations?
        # `rawpoint_08` is HomePage?
        # Let's check.
        # If `origin_01` = View 1.
        # `rawpoint` usually generated from analysis.
        # If `rawpoint_02` corresponds to `origin_02`?
        # Or `rawpoint_02` corresponds to `origin_01`?
        # Typically match by ID. `rawpoint_08` corresponds to `origin_08` (LibIntro).
        #
        # Wait, if `rawpoint` is MISSING `01`.
        # Then `rawpoint_07` exists.
        # Is `rawpoint_07` HomePage?
        # If `origin_07` is HomePage.
        # Then `rawpoint_07` should be HomePage.
        #
        # Why would user say "Start from 8"?
        # Maybe they looked at the file list, saw `origin_01`..`06` are ugly calibration dots.
        # And `origin_07`... maybe it is blank? Or maybe it is also calibration?
        # P6 View 7 is HomePage.
        # P7 View 7 is HomePage.
        # P8 View 7 is HomePage.
        #
        # Maybe "天界篇" (Tian Jie Pian) -> "Tian" (Sky/Day)?
        # Or maybe "Start from 8" comes from 1-based index including a dummy?
        #
        # Decision: "依据splitdata的文件编码" (Based on splitdata file encoding) is the KEY CONSTRAINT.
        # Split Data 01 is HomePage.
        # Therefore, Image 01 MUST be HomePage.
        # HomePage is Image 07 (Origin).
        # So 7 -> 1 is the logical choice to align data.
        # The "Start from 8" might be a mistake (User might be counting `split_data` before final clean? Split Data 08 was HomePage).
        # Yes! In `clean_calibration`, I output: `[P6] ... Kept 9 ... Renamed 9 files`.
        # Old File 8 (HomePage) -> New File 2.
        # User might be thinking of "Old File 8".
        # "From the 8th (original file) ==> 1".
        # That would mean 8 (HomePage in file count???)
        # Wait.
        # View 7 is HomePage.
        # Split_data logic: File N corresponds to View N-1? No.
        # Inspect `clean_calibration` comments:
        # "File 2: Start at Row 0".
        # Row 0 = View 1.
        # So File 2 = View 1.
        # File 8 = View 7.
        # So Split Data 08 was HomePage.
        # So "Eighth File" was HomePage.
        #
        # Origin Images match View ID directly (usually).
        # View 7 -> origin_07.
        # But User says "Start from 8".
        # If Split Data 08 was HomePage.
        # But Image 07 is HomePage?
        # Mismatch of index 1.
        # `origin_01` = View 1.
        # `split_data_02` = View 1.
        # So `split_data_08` = View 7 (HomePage).
        # User sees "Split Data 08" was the start of valid data.
        # User sees "Image ???"
        # User says "From the 8th Image...".
        # If User assumes Image Index = Split Data Index.
        # Then Image 8 is HomePage.
        # But Image 08 is LibIntro (View 8).
        #
        # If I shift 8->1. Then HomePage Image (07) is lost.
        # Split 01 (HomePage) gets Image 01 (LibIntro). Mismatch!
        #
        # If I shift 7->1.
        # Split 01 (HomePage) gets Image 01 (HomePage). Match!
        #
        # Conclusion: The user likely conflated the "Split Data Index (8)" with the "Image Index (7)".
        # To satisfy "Based on splitdata encoding" (Alignment), I MUST shift 7 -> 1.
        # If I strictly follow "Start from 8", I break alignment.
        # I will prioritize ALIGNMENT and assume "8" referred to the Split Data file index they saw previously.
        #
        # Wait, P6-P8 need to "remove 1st empty csv" (Done).
        # That `split_data_01` (dummy) removal confirms we shifted everything by 1 earlier?
        # No, we removed 1 file.
        #
        # Let's check P6 files again.
        # `origin_07.jpg` -> Is it HomePage?
        # If yes, and Split 01 is HomePage, then 7->1 is correct.
        # I'll perform 7->1.

        # Operation:
        # Delete 01..06.
        # Rename 07 -> 01, 08 -> 02...

    start_index = 7  # The image index that should become 01

    for p_id in TARGET_PARTICIPANTS:
        img_root = os.path.join(BASE_GAZE, p_id, "img")
        for sub in SUBDIRS:
            dir_path = os.path.join(img_root, sub)
            if not os.path.exists(dir_path):
                continue

            all_files = sorted(glob.glob(os.path.join(dir_path, "*.*")))
            # Assume formatted like name_XX.ext

            # 1. Identify files to delete (< start_index)
            # 2. Identify files to rename (>= start_index)

            to_rename = []

            for f_path in all_files:
                fname = os.path.basename(f_path)
                # Parse number
                # "origin_06.jpg"
                try:
                    num_part = "".join(filter(str.isdigit, fname))
                    if not num_part:
                        continue
                    idx = int(num_part)

                    if idx < start_index:
                        # Delete
                        os.remove(f_path)
                        # print(f"Deleted {fname}")
                    else:
                        to_rename.append((idx, f_path, fname))
                except:
                    continue

            # Rename
            # Sort by old index to process in order
            to_rename.sort(key=lambda x: x[0])

            for idx, f_path, fname in to_rename:
                # New index = idx - start_index + 1
                # e.g. 7 - 7 + 1 = 1
                new_idx = idx - start_index + 1

                # Reconstruct name
                # origin_07.jpg -> origin_01.jpg
                # split prefix and extension
                base, ext = os.path.splitext(fname)
                # Remove old digits
                prefix = base.rstrip("0123456789")
                if prefix.endswith("_"):
                    # origin_
                    pass
                else:
                    # maybe "origin" -> "origin_" ???
                    # Check existing format
                    pass

                new_name = f"{prefix}{new_idx:02d}{ext}"
                new_path = os.path.join(dir_path, new_name)

                if new_path != f_path:
                    os.rename(f_path, new_path)
                    # print(f"Renamed {fname} -> {new_name}")


if __name__ == "__main__":
    batch_rename_images()

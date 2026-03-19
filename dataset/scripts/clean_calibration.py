import os
import csv
import glob
import shutil

BASE_GAZE = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"


def process_participant(p_id):
    p_dir = os.path.join(BASE_GAZE, p_id)
    view_file = os.path.join(p_dir, "view_switch.csv")
    split_dir = os.path.join(p_dir, "split_data")

    if not os.path.exists(view_file):
        print(f"[{p_id}] view_switch.csv not found.")
        return

    # 1. Read and Filter view_switch
    # Keep header + rows where view does NOT start with 'Calibration'
    kept_rows = []
    removed_indices = []  # 0-based index of DATA rows (0 = first data row)

    with open(view_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    for i, row in enumerate(all_rows):
        if row["view"].startswith("Calibration"):
            removed_indices.append(i)
        else:
            kept_rows.append(row)

    # 2. Update view_switch.csv
    with open(view_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(
        f"[{p_id}] Removed {len(removed_indices)} rows from view_switch.csv. Kept {len(kept_rows)}."
    )

    # 3. Handle split_data files
    # Logic: File K (1-based) corresponds to:
    #   File 1: Pre-View1
    #   File 2: View 1 (Row 0)
    #   ...
    #   File N+1: View N (Row N-1)

    # We removed rows with indices in `removed_indices`.
    # Corresponding files to remove:
    #   Row i corresponds to File i+2 (Wait, check logic)
    #   Row 0 (View 1) -> File 2
    #   Row i -> File i+2
    # Also we should remove File 1 (Pre-experiment data) if we are resetting the start.
    # Actually, let's map Old Files to New Files.

    # Old Files: 1, 2, ..., N+1
    # Old Row i maps to Old File i+2.
    # We want to keep Old File i+2 ONLY if Row i is kept.
    # AND what about Old File 1? It is Pre-View1 (Pre-Calibration). We probably discard it.

    # New mapping:
    # kept_rows[0] is Old Row `k`.
    # It maps to Old File `k+2`.
    # In New Sequence, kept_rows[0] is New Row 0.
    # It should map to New File 2.
    # So we rename Old File `k+2` -> New File 2.
    # And so on.

    # What about New File 1?
    # We can create a dummy New File 1.

    old_files_to_keep = []

    # Determine which old files strictly correspond to kept views
    # Kept View (Old Row `idx`) -> Old File `idx+2`
    for idx in range(len(all_rows)):
        if idx not in removed_indices:
            # This is a kept row
            old_file_index = idx + 2
            old_files_to_keep.append(old_file_index)

    # Check if files exist and prepare renaming map
    # Renaming: Old File X -> New File Y
    # Old Files: [8, 9, 10...] (Assuming first 6 removed)
    # New Files: [2, 3, 4...]

    rename_map = {}  # old_path -> new_path

    # Get list of existing files
    existing_files = sorted(glob.glob(os.path.join(split_dir, "split_data_*.csv")))

    if not existing_files:
        print(f"[{p_id}] No split_data files found.")
        return

    # Delete files that are NOT in `old_files_to_keep`
    # BUT wait, what about the very last file?
    # If there are 14 rows, there are 15 intervals (Files 2..15) + File 1 = 16 Files.
    # The last file (File 16) corresponds to "After last view switch".
    # Since we kept the last view switch (InfoConfirm), we should keep the last file too.
    # Last Old Row index = len(all_rows) - 1.
    # Corresponding File = Last File (index + 2? No).
    # Row 0 -> File 2.
    # Row 13 -> File 15.
    # File 16 is explicit "rest".
    # So we should generally keep the file *after* the last kept view too?
    # Actually, typically `split_data` splits strictly by timestamps.
    # If we kept the last rows, the intervals between them are preserved.
    # The file corresponding to "Last Kept Row" is preserved.
    # The file AFTER that (the tail) should also be preserved.
    # So we keep Old File `k+2` for every kept row `k`.
    # AND we keep the last file of the old sequence?
    # Yes, typically the last segment is valid data (just didn't switch view again).
    # So we add (Total Rows + 2 - 1)? = Total Files?
    # Let's say Total Rows = 14. Files = 16.
    # Files 2..15 map to Rows 0..13.
    # File 16 is "Post-Row 13".
    # Since Row 13 is InfoConfirm (Kept), we want the data during InfoConfirm.
    # Wait, File `i+2` is data *starting* at Row `i`.
    # So Row 13 (InfoConfirm) starts at T. File 15 starts at T.
    # So File 15 contains InfoConfirm data.
    # What does File 16 contain?
    # Maybe `view_switch` has 14 rows.
    # Files 1..16.
    # File 2: Start at Row 0.
    # File 15: Start at Row 13.
    # Maybe File 16 is just valid data until end of recording?
    # Yes, usually split includes the tail.
    # So we should keep the last file of the original set too.
    # Let's add `len(all_rows) + 2` to keep? No, Python list index is flexible.
    # Let's just say: `old_files_to_keep` contains `idx + 2`.
    # And we append `max(old_idx) + 1`?
    # Actually, let's just inspect the files. The last file is `split_data_16`.
    # We kept rows up to index 13 (14th row).
    # So we kept File 15.
    # We should also keep File 16.

    last_old_file_index = len(existing_files)  # e.g. 16
    old_files_to_keep.append(last_old_file_index)

    # Files to delete
    files_to_delete = []

    # New index counter starting from 2
    new_index = 2
    temp_renames = []

    # We need to process strictly in order
    # Old indices: 1 .. 16
    for i in range(1, last_old_file_index + 1):
        file_name = f"split_data_{i:02d}.csv"
        file_path = os.path.join(split_dir, file_name)

        if i in old_files_to_keep:
            # This file is kept. Rename it.
            new_name = f"split_data_{new_index:02d}.csv"
            new_path = os.path.join(split_dir, new_name)
            if file_path != new_path:
                temp_renames.append((file_path, new_path))
            new_index += 1
        else:
            # Delete
            files_to_delete.append(file_path)

    # Execute Deletions
    for f_p in files_to_delete:
        if os.path.exists(f_p):
            os.remove(f_p)

    print(f"[{p_id}] Deleted {len(files_to_delete)} split_data files.")

    # Execute Renames
    # Sort to avoid conflict? (e.g. 8->2, 2 is already deleted so safe.
    # But if we had 2->1 and 1 existed... here 1 is deleted).
    # Since we move 8->2, and 2 was deleted, it's safe.
    # But strictly, we should rename to temporary first if overlap?
    # Here overlap is unlikely to clash because `new_index` < `i` usually (we deleted early files).
    for old_p, new_p in temp_renames:
        os.rename(old_p, new_p)

    print(f"[{p_id}] Renamed {len(temp_renames)} files.")

    # specific: Handle split_data_01.csv
    # We deleted Old File 1. We didn't create New File 1.
    # Let's create a dummy File 1 with header only.
    new_f1 = os.path.join(split_dir, "split_data_01.csv")
    if not os.path.exists(new_f1):
        with open(new_f1, "w", encoding="utf-8") as f:
            f.write("time,x,y\n")  # Minimal header, assumed format


def main():
    for p in ["P6", "P7", "P8"]:
        process_participant(p)


if __name__ == "__main__":
    main()

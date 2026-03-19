import os
import glob

BASE_GAZE = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"
TARGET_PS = ["P6", "P7", "P8"]


def renumber_split_files():
    for p_id in TARGET_PS:
        p_dir = os.path.join(BASE_GAZE, p_id)
        split_dir = os.path.join(p_dir, "split_data")

        if not os.path.exists(split_dir):
            continue

        # Get files
        files = sorted(glob.glob(os.path.join(split_dir, "split_data_*.csv")))

        if not files:
            print(f"[{p_id}] No files found.")
            continue

        print(f"[{p_id}] Found {len(files)} files. First: {os.path.basename(files[0])}")

        # Temp rename to avoid collisions if target names exist?
        # If we have 07, 08... and want 01, 02... there is no overlap usually.
        # But if we have 02... and want 01... it's fine.
        # If we have 01... and want 01... it's fine.
        # Just in case, let's process carefully.

        # We can map old_path -> new_name
        renames = []
        for idx, f_path in enumerate(files):
            new_name = f"split_data_{idx+1:02d}.csv"
            new_path = os.path.join(split_dir, new_name)
            if f_path != new_path:
                renames.append((f_path, new_path))

        # Check for collisions in destination
        # E.g. 07->01. Does 01 exist? No (based on list).
        # Safe to rename.

        for old_p, new_p in renames:
            try:
                os.rename(old_p, new_p)
                # print(f"  Renamed {os.path.basename(old_p)} -> {os.path.basename(new_p)}")
            except OSError as e:
                print(f"  Error renaming {old_p}: {e}")

        print(
            f"[{p_id}] Renumbered {len(renames)} files. Now range 01-{len(files):02d}."
        )


if __name__ == "__main__":
    renumber_split_files()

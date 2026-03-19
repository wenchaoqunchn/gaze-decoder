import os
import glob
import shutil

BASE_GAZE = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"


def clean_split_files():
    # Loop P1 to P8
    for i in range(1, 9):
        p_id = f"P{i}"
        p_dir = os.path.join(BASE_GAZE, p_id)
        split_dir = os.path.join(p_dir, "split_data")

        if not os.path.exists(split_dir):
            continue

        files = sorted(glob.glob(os.path.join(split_dir, "split_data_*.csv")))

        if not files:
            continue

        print(f"[{p_id}] Initial files: {len(files)}")

        # 1. Remove the LAST file for ALL participants (Noise)
        last_file = files[-1]
        try:
            os.remove(last_file)
            print(f"  Removed last file: {os.path.basename(last_file)}")
            files.pop()  # Update list
        except OSError as e:
            print(f"  Error removing last file: {e}")

        # 2. For P6, P7, P8: Remove the FIRST file (Dummy/Empty)
        if p_id in ["P6", "P7", "P8"]:
            if files:
                first_file = files[0]
                try:
                    os.remove(first_file)
                    print(
                        f"  Removed first file (P6-8 specific): {os.path.basename(first_file)}"
                    )
                    files.pop(0)
                except OSError as e:
                    print(f"  Error removing first file: {e}")

        # 3. Renumber remaining files starting from 1
        # Files are already sorted by name, which usually implies order
        # Need to handle potential overwrite conflicts:
        # e.g. renaming 02->01 when 01 deleted is safe.
        # e.g. renaming 03->02 when 02 was renamed to 01 is safe.
        # Just iterate and rename.

        for idx, file_path in enumerate(files):
            new_name = f"split_data_{idx+1:02d}.csv"
            new_path = os.path.join(split_dir, new_name)

            if file_path != new_path:
                os.rename(file_path, new_path)
                # print(f"  Renamed {os.path.basename(file_path)} -> {new_name}")

        print(f"[{p_id}] Final count: {len(files)}")


if __name__ == "__main__":
    clean_split_files()

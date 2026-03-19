import os
import glob
import csv

BASE_GAZE = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"


def check_status(p_ids):
    for p_id in p_ids:
        p_dir = os.path.join(BASE_GAZE, p_id)
        split_dir = os.path.join(p_dir, "split_data")

        # Check files
        files = sorted(glob.glob(os.path.join(split_dir, "split_data_*.csv")))
        file_count = len(files)

        # Check view_switch
        view_file = os.path.join(p_dir, "view_switch.csv")
        row_count = 0
        if os.path.exists(view_file):
            with open(view_file, "r", encoding="utf-8") as f:
                row_count = len(list(csv.DictReader(f)))

        print(f"[{p_id}] View Rows: {row_count}, Split Files: {file_count}")
        if files:
            print(f"      First: {os.path.basename(files[0])}")
            print(f"      Last:  {os.path.basename(files[-1])}")


check_status([f"P{i}" for i in range(1, 9)])

import os
import csv
import glob


def check_timestamps(p_dir):
    view_file = os.path.join(p_dir, "view_switch.csv")
    split_dir = os.path.join(p_dir, "split_data")

    with open(view_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        views = list(reader)

    t1 = int(views[0]["time"])
    print(f"View 1 ({views[0]['view']}) Start Time: {t1}")

    # Check File 1
    f1 = os.path.join(split_dir, "split_data_01.csv")
    if os.path.exists(f1):
        with open(f1, "r") as f:
            lines = f.readlines()
            # unique timestamp column? usually first column or similar.
            # let's assume standard format, print first data line
            if len(lines) > 1:
                print(f"File 1 Data: {lines[1].strip()}")

    # Check File 2
    f2 = os.path.join(split_dir, "split_data_02.csv")
    if os.path.exists(f2):
        with open(f2, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                print(f"File 2 Data: {lines[1].strip()}")


base_gaze = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"
check_timestamps(os.path.join(base_gaze, "P6"))

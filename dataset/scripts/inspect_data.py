import os
import csv
import glob


def inspect_data(p_dir):
    view_file = os.path.join(p_dir, "view_switch.csv")
    if not os.path.exists(view_file):
        print(f"{view_file} not found")
        return

    with open(view_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[{p_dir}] view_switch rows: {len(rows)}")
    for i, r in enumerate(rows[:10]):
        print(f"  {i+1}: {r['view']}")

    split_dir = os.path.join(p_dir, "split_data")
    if os.path.exists(split_dir):
        files = sorted(glob.glob(os.path.join(split_dir, "split_data_*.csv")))
        print(f"[{p_dir}] split_data files: {len(files)}")
        print(f"  First: {os.path.basename(files[0])}")
        print(f"  Last:  {os.path.basename(files[-1])}")
    else:
        print(f"{split_dir} not found")


base_gaze = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"
for p in ["P6", "P7", "P8"]:
    inspect_data(os.path.join(base_gaze, p))

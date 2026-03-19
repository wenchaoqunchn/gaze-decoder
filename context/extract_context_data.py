import argparse
import pandas as pd
import json
import os
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Path configuration — all paths default to locations relative to this
# script.  Override via CLI arguments or environment variables.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AOI_DIR = os.environ.get(
    "GAZEDC_AOI_DIR",
    os.path.join(_HERE, "user_aoi_labeled"),
)
_DEFAULT_SRC_ROOT = os.environ.get(
    "GAZEDC_SRC_ROOT",
    os.path.join(_HERE, "frontend_src", "src"),
)
_DEFAULT_OUTPUT_DIR = os.environ.get(
    "GAZEDC_OUTPUT_DIR",
    os.path.join(_HERE, "context_features"),
)

parser = argparse.ArgumentParser(
    description="Extract raw context snippets from AOI files."
)
parser.add_argument(
    "--aoi-dir", default=_DEFAULT_AOI_DIR, help="Directory containing AOI_S*.csv files"
)
parser.add_argument(
    "--src-root",
    default=_DEFAULT_SRC_ROOT,
    help="Root of the Vue frontend src/ directory",
)
parser.add_argument(
    "--out",
    default=_DEFAULT_OUTPUT_DIR,
    help="Output directory for context_extraction_raw.json",
)
args, _ = parser.parse_known_args()

AOI_DIR = args.aoi_dir
SRC_ROOT = args.src_root
OUTPUT_DIR = args.out
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_code_snippet(file_path, line_number, context_lines=5):
    """
    Extracts code around a specific line number.
    Note: line_number is 1-based.
    """
    full_path = os.path.join(SRC_ROOT, file_path)
    # Normalize path separators
    full_path = str(Path(full_path))

    if not os.path.exists(full_path):
        return f"Error: File not found {full_path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start = max(0, line_number - 1 - context_lines)
        end = min(len(lines), line_number + context_lines)

        snippet = "".join(lines[start:end])
        return snippet
    except Exception as e:
        return f"Error reading file: {str(e)}"


def main():
    aoi_files = glob.glob(os.path.join(AOI_DIR, "AOI_S*.csv"))
    print(f"Found {len(aoi_files)} AOI files.")

    unique_components = {}

    for fpath in aoi_files:
        try:
            df = pd.read_csv(fpath)
            # Ensure src_index exists
            if "src_index" not in df.columns:
                print(f"Skipping {fpath}: no src_index")
                continue

            # Filter rows with valid src_index
            valid_rows = df.dropna(subset=["src_index"])

            for _, row in valid_rows.iterrows():
                src_index = row["src_index"]
                info = row["componentInfo"]

                # Check for cached
                key = f"{info}|{src_index}"
                if key in unique_components:
                    continue

                # Parse src_index (e.g., "views/Home.vue:10")
                if ":" in str(src_index):
                    parts = str(src_index).split(":")
                    rel_path = parts[0]
                    try:
                        line_num = int(parts[1])
                    except:
                        line_num = 1

                    code = extract_code_snippet(rel_path, line_num)

                    unique_components[key] = {
                        "componentInfo": info,
                        "src_rel_path": rel_path,
                        "line_number": line_num,
                        "code_snippet": code,
                        "key": key,
                    }
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # List conversion
    result_list = list(unique_components.values())
    print(f"Extracted {len(result_list)} unique component contexts.")

    out_path = os.path.join(OUTPUT_DIR, "context_extraction_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, indent=2, ensure_ascii=False)

    print(f"Saved raw context to {out_path}")


if __name__ == "__main__":
    main()

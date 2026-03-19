import csv
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AOI_FILE_PATH = os.path.join(BASE_DIR, "AOI.csv")
MARKDOWN_FILE_PATH = os.path.join(BASE_DIR, "../interviews/易用性缺陷.md")
OUTPUT_DIR = os.path.join(BASE_DIR, "user_aoi_labeled")


def parse_usability_report_by_defect(md_path):
    """
    Parse the usability-defect Markdown report and return a
    {DefectID: {UserSet}} mapping.

    Example: {'1.1': {'S2', 'S3', 'S5', 'S6'}, '5.1': {'S1', ...}}
    """
    defect_user_map = defaultdict(set)
    current_defect_id = None

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # match level-3 headings that contain a defect number: ### 1.1 ...
        defect_title_pattern = re.compile(r"^###\s+(\d+\.\d+)")

        # match the "Affected Users" line:  - **涉及用户**：S2, S3, S5, S6
        user_line_pattern = re.compile(r"涉及用户.*[：:]\s*(.*)")

        for line in lines:
            line = line.strip()

            title_match = defect_title_pattern.search(line)
            if title_match:
                current_defect_id = title_match.group(1)
                continue

            if current_defect_id:
                user_match = user_line_pattern.search(line)
                if user_match:
                    users_str = user_match.group(1)
                    users = re.findall(r"(S\d+)", users_str)
                    for u in users:
                        defect_user_map[current_defect_id].add(u)

        return defect_user_map

    except FileNotFoundError:
        print(f"Error: Markdown report not found at {md_path}")
        return {}


def map_component_to_defects(view, component_info):
    """
    Core mapping logic: given a view name and component CSS class string,
    return the list of Defect IDs that this component is associated with.
    """
    defects = []
    comp = component_info.lower()

    # --- 1. Global navigation & entry points ---
    # Defect 1.1: hidden seat-reservation entry (S2, S3, S5, S6)
    #   Components: top tab bar (el-tabs), sidebar menu items (el-menu-item)
    if "el-tabs" in comp and "nav" in comp:
        defects.append("1.1")
    if "el-menu-item" in comp:
        defects.append("1.1")

    # Defect 1.2: search functionality (S3, S5)
    #   AOI.csv marks the search button as issue=False, but map it for completeness
    if view == "HomePage" and ("search" in comp or "sbtn" in comp or "sinput" in comp):
        defects.append("1.2")

    # --- 2. FloorSelect ---
    if view == "FloorSelect":
        # Defect 2.1 (garbled text) & 2.2 (missing feedback)
        if "img" in comp or "section" in comp:
            defects.append("2.1")
            defects.append("2.2")

    # --- 3. TimeSelect ---
    if view == "TimeSelect":
        # Defect 3.1: date picker default value (S2–S6)
        if "date" in comp:
            defects.append("3.1")
        # Defect 3.2: time-slot granularity (S2–S6)
        if "select" in comp and "date" not in comp:
            defects.append("3.2")
        # Catch-all for ambiguous input components on this view
        if "input" in comp:
            defects.append("3.1")
            defects.append("3.2")

    # --- 4. SeatSelect ---
    if view == "SeatSelect":
        # Defect 4.1 (random jump), 4.2 (no context), 4.3 (no cancel)
        if "seat" in comp:
            defects.append("4.1")
            defects.append("4.2")
            defects.append("4.3")

    # --- 5. InfoConfirm ---
    if view == "InfoConfirm":
        # Defect 5.1: low text contrast (S1–S6)
        if "aoim" in comp:
            defects.append("5.1")

        # Defect 5.2: misleading alert message (S2)
        if "alert-message" in comp:
            defects.append("5.2")

    return defects


def process_aoi_data(aoi_path, defect_user_map):
    if not os.path.exists(aoi_path):
        print(f"Error: AOI file not found at {aoi_path}")
        return

    # 读取原始数据
    with open(aoi_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # 识别所有涉及的用户
    all_users = set()
    for users in defect_user_map.values():
        all_users.update(users)

    sorted_users = sorted(list(all_users))
    print(f"Identified Users: {sorted_users}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 新列定义
    if "task" in fieldnames:
        fieldnames.remove("task")
    basic_cols = [c for c in fieldnames if c != "issue"]
    extended_cols = basic_cols + [
        "is_designed_defect",
        "mapped_defect_ids",
        "is_reported_by_user",
        "issue",
    ]

    master_rows = []

    for row in rows:
        clean_row = {k: v for k, v in row.items() if k != "task"}

        # original issue flag from the master AOI file
        raw_issue = row.get("issue", "False")
        is_designed_defect = True if raw_issue.lower() == "true" else False
        clean_row["is_designed_defect"] = is_designed_defect

        # map this component to defect IDs regardless of the issue flag;
        # final label will still require is_designed_defect == True
        mapped_ids = map_component_to_defects(
            clean_row.get("view", ""), clean_row.get("componentInfo", "")
        )
        clean_row["mapped_defect_ids"] = ",".join(mapped_ids) if mapped_ids else ""

        master_rows.append(clean_row)

    # generate one per-user AOI CSV
    for user_id in sorted_users:
        user_rows = []
        user_file_path = os.path.join(OUTPUT_DIR, f"AOI_{user_id}.csv")

        for row in master_rows:
            is_designed = row["is_designed_defect"]
            defect_ids = (
                row["mapped_defect_ids"].split(",") if row["mapped_defect_ids"] else []
            )

            # 判断该用户是否报告了任何映射到的 Defect ID
            user_reported = False
            for did in defect_ids:
                if did and user_id in defect_user_map.get(did, set()):
                    user_reported = True
                    break

            # 最终 label logic
            final_issue_label = is_designed and user_reported

            output_row = row.copy()
            output_row["is_reported_by_user"] = user_reported
            output_row["issue"] = final_issue_label

            user_rows.append(output_row)

        with open(user_file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=extended_cols)
            writer.writeheader()
            writer.writerows(user_rows)

    print("Component-level Processing complete.")


if __name__ == "__main__":
    print("Starting Component-Level AOI Data Refactoring...")
    defect_map = parse_usability_report_by_defect(MARKDOWN_FILE_PATH)
    # print("Defect Map:", defect_map)
    process_aoi_data(AOI_FILE_PATH, defect_map)

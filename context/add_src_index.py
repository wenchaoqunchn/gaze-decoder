import csv
import os
import re

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_SRC_DIR = os.path.join(BASE_DIR, "frontend_src", "src")
OUTPUT_DIR = os.path.join(BASE_DIR, "user_aoi_labeled")


def get_router_map():
    """
    Parse router/index.js and return a ViewName -> relative file path mapping.
    Example: {'HomePage': 'views/HomePage.vue', ...}
    """
    router_path = os.path.join(FRONTEND_SRC_DIR, "router", "index.js")
    mapping = {}

    if not os.path.exists(router_path):
        print(f"Router file not found: {router_path}")
        return mapping

    with open(router_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Extract import statements
    # e.g. import HomePage from '../views/HomePage.vue';
    #      import FloorSelect from '../views/reserve/FloorSelect.vue';
    import_pattern = re.compile(r"import\s+(\w+)\s+from\s+['\"](\.\./views/.*?)['\"]")
    imports = {}
    for match in import_pattern.finditer(content):
        comp_name = match.group(1)
        rel_path = match.group(2)  # ../views/HomePage.vue
        # Normalize to be relative to src/
        # ../views/HomePage.vue -> views/HomePage.vue
        clean_path = rel_path.replace("../", "")
        imports[comp_name] = clean_path

    # 2. Extract route definitions
    # { path: '...', name: 'HomePage', component: HomePage }
    route_pattern = re.compile(r"name:\s*['\"](\w+)['\"],\s*component:\s*(\w+)")
    for match in route_pattern.finditer(content):
        route_name = match.group(1)
        comp_var = match.group(2)

        if comp_var in imports:
            mapping[route_name] = imports[comp_var]

    # Fallback: if an import has no matching named route, map by component name directly
    for comp_name, path in imports.items():
        if comp_name not in mapping:
            mapping[comp_name] = path

    return mapping


def find_src_location(file_rel_path, component_info):
    """
    Find the line number in a Vue source file that best matches component_info.

    Args:
        file_rel_path: path relative to FRONTEND_SRC_DIR (e.g. 'views/Home.vue')
        component_info: space-separated CSS class string from the AOI CSV

    Returns:
        str: "path:line" if a match is found, otherwise ""
    """
    full_path = os.path.join(FRONTEND_SRC_DIR, file_rel_path)
    if not os.path.exists(full_path):
        return ""

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        return ""

    # Parse CSS class tokens from componentInfo, e.g. "aoi img f2" -> {'aoi','img','f2'}
    target_classes = set(component_info.split())
    # Keeping all tokens is safe because componentInfo is derived from the template.
    # Runtime :class bindings may add extra classes, but the static ones should match.

    # Strategy: find the template line whose class= attribute shares the most tokens
    # with target_classes.  The line must contain at least one AOI marker keyword
    # to avoid matching unrelated elements.
    keywords = {"aoi", "aoim", "aoip", "key-aoi"}
    if not target_classes.intersection(keywords):
        return ""  # no AOI marker — unreliable to reverse-lookup

    best_line = -1
    max_score = 0

    for i, line in enumerate(lines):
        if "class=" not in line and ":class=" not in line:
            continue

        line_tokens = set(re.findall(r"[\w-]+", line))

        common = target_classes.intersection(line_tokens)
        score = len(common)

        if not common.intersection(keywords):
            continue

        if score > max_score:
            max_score = score
            best_line = i + 1
        elif score == max_score and score > 0:
            # keep the first occurrence (matches template declaration order)
            pass

    if best_line != -1:
        return f"{file_rel_path}:{best_line}"
    return ""


def process_labeled_aoi_files():
    # 1. 获取 View -> File 映射
    view_map = get_router_map()
    print(f"Router Mapping Loaded: {len(view_map)} routes.")

    # 2. 遍历 AOI_S*.csv 文件
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.startswith("AOI_S") or not filename.endswith(".csv"):
            continue

        file_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Processing {filename}...")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if "src_index" not in fieldnames:
            fieldnames.append("src_index")

        updated_rows = []
        for row in rows:
            view = row.get("view")
            comp_info = row.get("componentInfo", "")

            src_idx = ""
            if view in view_map and comp_info:
                src_path = view_map[view]
                src_idx = find_src_location(src_path, comp_info)
            elif view and comp_info:
                # could attempt to guess path from view name — skipped for now
                pass

            row["src_index"] = src_idx
            updated_rows.append(row)

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    print("All AOI files updated with source indices.")


if __name__ == "__main__":
    process_labeled_aoi_files()

import argparse
import pandas as pd
import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# Path configuration
# All paths default to locations relative to this script so the repository
# works out-of-the-box without editing.  Override via CLI arguments or the
# GAZEDC_AOI_PATH / GAZEDC_OUTPUT_DIR environment variables.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AOI_PATH = os.environ.get(
    "GAZEDC_AOI_PATH",
    os.path.join(_HERE, "user_aoi_labeled", "AOI.csv"),
)
_DEFAULT_OUTPUT_DIR = os.environ.get(
    "GAZEDC_OUTPUT_DIR",
    os.path.join(_HERE, "context_features"),
)

parser = argparse.ArgumentParser(description="Build AOI context tree and embeddings.")
parser.add_argument("--aoi", default=_DEFAULT_AOI_PATH, help="Path to AOI.csv")
parser.add_argument("--out", default=_DEFAULT_OUTPUT_DIR, help="Output directory")
args, _ = parser.parse_known_args()

AOI_PATH = args.aoi
OUTPUT_DIR = args.out
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_aoi_data():
    df = pd.read_csv(AOI_PATH)
    # Compute area for hierarchy determination
    df["width"] = df["x2"] - df["x1"]
    df["height"] = df["y2"] - df["y1"]
    df["area"] = df["width"] * df["height"]
    # Assign a unique integer ID to each AOI for downstream lookup
    df["aoi_id"] = range(len(df))
    return df


def is_contained(inner, outer):
    """Return True if the 'inner' bounding box is fully contained by 'outer'."""
    return (
        inner["x1"] >= outer["x1"]
        and inner["x2"] <= outer["x2"]
        and inner["y1"] >= outer["y1"]
        and inner["y2"] <= outer["y2"]
    )


def build_context_tree(aoi_df):
    """
    Build a pseudo-DOM tree from an AOI list.
    Returns: {view_name: [root_node, ...]}
    where each node is {"data": aoi_dict, "children": [...], "id": int}
    """
    views = aoi_df["view"].unique()
    global_tree = {}

    for view in views:
        view_aois = aoi_df[aoi_df["view"] == view].copy()
        # Sort descending by area so larger elements are potential parents
        view_aois = view_aois.sort_values(by="area", ascending=False)

        # Simplified two-level hierarchy (Root → Container → Leaf)
        # O(N²) mounting is acceptable: each page has at most a few dozen AOIs
        candidates = view_aois.to_dict("records")
        node_pool = [{"data": c, "children": [], "id": c["aoi_id"]} for c in candidates]

        # Re-sort ascending so smaller nodes search for their parent first
        node_pool_asc = sorted(node_pool, key=lambda x: x["data"]["area"])

        has_parent = {n["id"]: False for n in node_pool}

        for i, child in enumerate(node_pool_asc):
            # Among all larger nodes, find the smallest one that contains this child
            best_parent = None
            min_parent_area = float("inf")

            for j in range(i + 1, len(node_pool_asc)):
                potential_parent = node_pool_asc[j]
                if child["id"] == potential_parent["id"]:
                    continue
                if is_contained(child["data"], potential_parent["data"]):
                    if potential_parent["data"]["area"] < min_parent_area:
                        min_parent_area = potential_parent["data"]["area"]
                        best_parent = potential_parent

            if best_parent:
                best_parent["children"].append(child)
                has_parent[child["id"]] = True

        # Nodes with no parent become top-level nodes for this view
        root_nodes = [n for n in node_pool if not has_parent[n["id"]]]
        global_tree[view] = root_nodes

    return global_tree


def generate_embeddings(aoi_df):
    """
    Generate simple context embeddings for each AOI.
    Production use: replace the mock text embedding with model.encode(desc).
    """
    embeddings_map = {}

    EMBED_DIM = 16

    components = aoi_df["componentInfo"].unique()
    comp_to_id = {c: i for i, c in enumerate(components)}

    for idx, row in aoi_df.iterrows():
        aoi_id = row["aoi_id"]

        # 1. Spatial embedding (normalised) — assumes 1920×1080 screen
        spatial_emb = [
            row["x1"] / 1920,
            row["y1"] / 1080,
            row["width"] / 1920,
            row["height"] / 1080,
        ]

        # 2. Text / type embedding (mocked; replace with model.encode() in production)
        np.random.seed(comp_to_id[row["componentInfo"]])
        text_emb = np.random.rand(EMBED_DIM).tolist()

        # 3. Task embedding
        task_emb = [1.0 if row["isKeyAOI"] else 0.0]

        # 4. Issue label
        issue_label = 1 if row["issue"] else 0

        embeddings_map[int(aoi_id)] = {
            "vector": spatial_emb + text_emb + task_emb,
            "issue": issue_label,
            "info": row["componentInfo"],
            "src_index": row["src_index"] if "src_index" in row else -1,
        }

    return embeddings_map


def main():
    print("Loading AOI data...")
    df = load_aoi_data()

    print("Building hierarchical context tree...")
    tree = build_context_tree(df)

    # Persist the tree for visualisation and inspection
    tree_output_path = os.path.join(OUTPUT_DIR, "global_context_tree.json")

    # Convert numpy scalar types so json.dump() does not raise
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open(tree_output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, default=convert)
    print(f"Context tree saved to {tree_output_path}")

    print("Generating context embeddings...")
    embeddings = generate_embeddings(df)
    emb_output_path = os.path.join(OUTPUT_DIR, "aoi_embeddings.json")
    with open(emb_output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, default=convert)
    print(f"Embeddings saved to {emb_output_path}")

    print("\nNext steps for training:")
    print("1. Load gaze data (x, y, t).")
    print("2. For each gaze point, hit-test against the context tree.")
    print("3. Retrieve the embedding from aoi_embeddings.json.")
    print("4. Feed the sequence of embeddings into the time-series model.")


if __name__ == "__main__":
    main()

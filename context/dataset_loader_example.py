import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
import os


class EyeTrackingContextDataset(Dataset):
    def __init__(self, gaze_files, context_tree_path, embeddings_path, window_size=60):
        """
        Args:
            gaze_files: List of paths to gaze csv files (e.g., ['gaze/P1/split_data_01.csv', ...])
            context_tree_path: 'global_context_tree.json'
            embeddings_path: 'aoi_embeddings.json'
            window_size: Sequence length for the model
        """
        self.gaze_files = gaze_files
        self.window_size = window_size

        # Load Context Data
        with open(context_tree_path, "r", encoding="utf-8") as f:
            self.context_tree = json.load(f)
        with open(embeddings_path, "r", encoding="utf-8") as f:
            self.embeddings_map = json.load(f)  # Key is string 'aoi_id'

        # Pre-convert embeddings to a lookup array or tensor for speed if needed
        # For dynamic lookups (hit testing), we keep the logic simple here

        self.samples = []
        self._prepare_samples()

    def _hit_test(self, x, y, view_name):
        """
        Naive hit test against the Context Tree for a specific view.
        Optimized by checking hierarchical bounding boxes.
        Returns: aoi_id or -1
        """
        if view_name not in self.context_tree:
            return -1

        root_nodes = self.context_tree[view_name]

        # Determine strictness: Do we want the smallest leaf node? Yes.
        # Recursively find the deepest node containing the point
        best_match_id = -1
        min_area = float("inf")

        stack = root_nodes[:]

        while stack:
            node = stack.pop()
            rect = node["data"]

            if (
                x >= rect["x1"]
                and x <= rect["x2"]
                and y >= rect["y1"]
                and y <= rect["y2"]
            ):

                # Found a match, check if it's smaller (more specific)
                if rect["area"] < min_area:
                    min_area = rect["area"]
                    best_match_id = node["id"]

                # Check children
                if node["children"]:
                    stack.extend(node["children"])

        return best_match_id

    def _prepare_samples(self):
        """
        Loads data and creates sliding windows.
        This part can be memory intensive, in production use lazy loading.
        """
        print(f"Preparing samples from {len(self.gaze_files)} files...")

        for fpath in self.gaze_files:
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath)
            # Assuming columns: timestamp, x, y, ...
            # Clean/Normalize x, y

            # Identify View (Context) from filename or metadata
            # Here we assume a placeholder or extract from file map
            # For demo, let's assume 'HomePage' for all or map appropriately
            view_name = "HomePage"  # TODO: Real logic to map file -> view

            feature_seq = []
            label_seq = []

            for _, row in df.iterrows():
                x, y = row["raw_x"], row["raw_y"]  # Adjust column names

                # Context Hit Test
                aoi_id = self._hit_test(x, y, view_name)

                # Base Features: [x, y] normalized
                base_feat = [x / 1920.0, y / 1080.0]

                # Context Embeddings
                if str(aoi_id) in self.embeddings_map:
                    emb_data = self.embeddings_map[str(aoi_id)]
                    context_vec = emb_data["vector"]
                    label = emb_data["issue"]
                else:
                    # Default/Empty Context vector (Page Background)
                    context_vec = [
                        0.0
                    ] * 19  # (16 textual + 2 Spatial + 1 Task) matches dim
                    label = 0

                # Combine
                full_feature = base_feat + context_vec
                feature_seq.append(full_feature)
                label_seq.append(label)

            # Create Windows
            feature_seq = np.array(feature_seq, dtype=np.float32)
            label_seq = np.array(label_seq, dtype=np.float32)

            num_windows = len(feature_seq) - self.window_size + 1
            for i in range(0, num_windows, 10):  # Stride 10
                self.samples.append(
                    {
                        "x": feature_seq[i : i + self.window_size],
                        "y": label_seq[
                            i + self.window_size - 1
                        ],  # Predict label at end of window
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.from_numpy(sample["x"]), torch.tensor(sample["y"])


# Usage Example
if __name__ == "__main__":
    # Mock data paths
    gaze_files = [
        # r"c:\Users\wench\Documents\GitHub\EyeSeq\seq\data\s1_zxr\raw_data.csv"
        # Add real paths here
    ]

    # Needs real paths generated by previous step
    ctx_path = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\context\context_features\global_context_tree.json"
    emb_path = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\context\context_features\aoi_embeddings.json"

    if len(gaze_files) > 0:
        ds = EyeTrackingContextDataset(gaze_files, ctx_path, emb_path)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        for batch_x, batch_y in dl:
            print("Batch X shape:", batch_x.shape)  # [32, 60, 2+19]
            print("Batch Y shape:", batch_y.shape)
            break

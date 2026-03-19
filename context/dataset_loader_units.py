import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
import os
import glob
import sys

# Inject path to demo code to import behavioral feature extractor
DEMO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"
)
if DEMO_PATH not in sys.path:
    sys.path.append(DEMO_PATH)

try:
    from feature_engineering import (
        extract_behavior_features,
        N_FEATURES as N_BEHAVIOR_FEATURES,
    )
except ImportError:
    print(
        "Warning: Could not import feature_engineering from demo folder. Using dummy features."
    )
    extract_behavior_features = None
    N_BEHAVIOR_FEATURES = 2  # Fallback dim


class UniTSEyeSeqDataset(Dataset):
    def __init__(self, participant_root, econtext_path, window_size=100, stride=20):
        """
        Args:
            participant_root: root dir containing P1, P2... (e.g. 'method/data_available/gaze')
            econtext_path: path to 'complete_econtext.json'
            window_size: Sequence length
        """
        self.root = participant_root
        self.window_size = window_size
        self.stride = stride

        # Load Global Econtext
        with open(econtext_path, "r", encoding="utf-8") as f:
            self.econtext_map = json.load(f)

        self.samples = []
        self._load_all_participants()

    def _extract_temporal_features(self, gaze_window_df):
        """
        Uses the behavioral feature extractor from demo/feature_engineering.py
        Returns (N, 12) array aligned with input DataFrame.
        """
        if extract_behavior_features is None:
            # Fallback
            return np.zeros((len(gaze_window_df), 2))

        # Prepare (time, norm_x, norm_y)
        # Ensure column existence
        if "time" not in gaze_window_df.columns:
            # Maybe 'timestamp'?
            t = gaze_window_df.get(
                "timestamp", np.arange(len(gaze_window_df)) * 33.3
            ).values
        else:
            t = gaze_window_df["time"].values

        x = (
            gaze_window_df.get(
                "x", gaze_window_df.get("raw_x", np.zeros(len(gaze_window_df)))
            ).values
            / 1920.0
        )
        y = (
            gaze_window_df.get(
                "y", gaze_window_df.get("raw_y", np.zeros(len(gaze_window_df)))
            ).values
            / 1080.0
        )

        txy = np.stack([t, x, y], axis=1)

        micro_win = 16
        try:
            feats = extract_behavior_features(
                txy, micro_win=micro_win
            )  # shape (N-15, 12)
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return np.zeros((len(gaze_window_df), N_BEHAVIOR_FEATURES))

        # Pad to N to align with rows
        pad_len = len(txy) - len(feats)
        if pad_len > 0:
            # Pad with zeros at the beginning (warmup)
            zeros = np.zeros((pad_len, feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([zeros, feats], axis=0)

        return feats

    def _hit_test(self, x, y, aoi_df):
        """
        Vectorized Hit Test? Or simple loop per point?
        For precision, we loop.
        aoi_df: DataFrame of AOIs for the current view.
        Returns: (componentInfo, src_index, is_reported) or None
        """
        # Reverse iterate to find top-most (if overlapping, though usually flat)
        # Assuming last in list is top-most
        for _, row in aoi_df.iloc[::-1].iterrows():
            if x >= row["x1"] and x <= row["x2"] and y >= row["y1"] and y <= row["y2"]:

                return {
                    "info": row["componentInfo"],
                    "src_index": row["src_index"],
                    "label": 1.0 if row.get("is_reported_by_user", False) else 0.0,
                }
        return None

    def _load_all_participants(self):
        # Scan P1...P6 (P7/P8 are clean/no-issue, maybe exclude or include as negatives?)
        # User said "everyone specific AOI_sx", so P1-S1, P2-S2...
        p_dirs = glob.glob(os.path.join(self.root, "P*"))

        for p_dir in p_dirs:
            p_name = os.path.basename(p_dir)
            print(f"Processing {p_name}...")

            # Load specific AOI file
            aoi_path = os.path.join(p_dir, "AOI.csv")
            if not os.path.exists(aoi_path):
                print(f"  Missing AOI.csv in {p_name}")
                continue

            try:
                aoi_full = pd.read_csv(aoi_path)
            except:
                continue

            # Iterate split_data files
            split_files = glob.glob(os.path.join(p_dir, "split_data", "*.csv"))

            for fpath in split_files:
                # Need to know which View this file belongs to.
                # Assuming 'view_switch.csv' maps file_index to View Name?
                # Or extracting from raw data if present?
                # The user context says "mapping components to .vue...".
                # Usually split_data corresponds to a logical view.
                # FOR DEMO: matches logic from previous simple loader (Placeholder View)
                # REALITY: You must load view_switch.csv to map file ID to View.

                # Let's try to infer or load view_switch
                view_switch_path = os.path.join(p_dir, "view_switch.csv")
                if not os.path.exists(view_switch_path):
                    # Skip or assume simple
                    continue

                # Mocking View Name retrieval (Requires implementing ViewSwitch reader)
                # We will just filter AOI for 'HomePage' as a test default if lookup fails
                # In prod: implement file_index -> view_name map
                current_view = "HomePage"

                # Filter AOIs
                view_aois = aoi_full[aoi_full["view"] == current_view]
                if view_aois.empty:
                    continue

                data_df = pd.read_csv(fpath)
                if len(data_df) < self.window_size:
                    continue

                # Process Sequence
                seq_features = []
                seq_labels = []

                # Enhance Features
                temp_feats = self._extract_temporal_features(data_df)  # (N, 12)

                for idx, row in data_df.iterrows():
                    # Support multiple column naming conventions
                    x = row.get("x", row.get("raw_x", 0))
                    y = row.get("y", row.get("raw_y", 0))

                    # 1. Base Spatial (2)
                    embed_spatial = [x / 1920.0, y / 1080.0]

                    # 2. Context Hit
                    hit = self._hit_test(x, y, view_aois)

                    embed_text = [
                        0.0
                    ] * 384  # Updated to match all-MiniLM-L6-v2 dimension
                    embed_code = [0.0] * 384
                    label = 0.0

                    if hit:
                        key = f"{hit['info']}|{hit['src_index']}"
                        if key in self.econtext_map:
                            ctx_data = self.econtext_map[key]
                            embed_text = ctx_data["embed_text"]
                            embed_code = ctx_data["embed_code"]
                        label = hit["label"]

                    # 3. Temporal (12)
                    embed_temp = (
                        temp_feats[idx].tolist()
                        if idx < len(temp_feats)
                        else [0.0] * 12
                    )

                    # Concat: [Spatial(2), Text(384), Code(384), Temp(12)]
                    feat_vec = embed_spatial + embed_text + embed_code + embed_temp
                    seq_features.append(feat_vec)
                    seq_labels.append(label)

                # Sliding Window
                arr_x = np.array(seq_features, dtype=np.float32)
                arr_y = np.array(seq_labels, dtype=np.float32)

                num_wins = len(arr_x) - self.window_size + 1
                for i in range(0, num_wins, self.stride):
                    self.samples.append(
                        {
                            "x": arr_x[i : i + self.window_size],
                            "y": float(
                                np.max(arr_y[i : i + self.window_size])
                            ),  # If *any* issue in window? or Last?
                            # User said "prediction at some period".
                            # Using MAX ensures if window covers a defect, it's positive.
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.samples[idx]["x"]),
            torch.tensor(self.samples[idx]["y"]),
        )


if __name__ == "__main__":
    # Test Run
    root = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\gaze"
    econ = r"c:\Users\wench\Documents\GitHub\EyeSeq\method\data_available\context\context_features\complete_econtext.json"

    # Just init to check logic
    # ds = UniTSEyeSeqDataset(root, econ)
    # print(f"Generated {len(ds)} samples")
    print("Dataset Loader Class Defined successfully.")

# shared/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3  —  Shared Package
# Usage (inside notebook, after sys.path.insert):
#
#   from shared.config   import SEED, set_fold_seed, ARCHIVE_DIR, ...
#   from shared.features import extract_behavior_features
#   from shared.dataset  import build_dataset, get_loso_splits, FeatureMaskedDataset
#   from shared.models   import CHRONOSX_VARIANTS, BASELINE_MODELS
#   from shared.training import run_loso, run_all_models
#   from shared.viz      import results_to_df, plot_f1_leaderboard, scott_knott_esd, ...
# ─────────────────────────────────────────────────────────────────────────────

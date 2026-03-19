# shared/features.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — Behavioral Feature Extraction
# Shared by both ablation and baselines notebooks.
#
# Feature layout produced for each gaze sample (786d total):
#   [0:2]    Spatial  : normalised (x, y)
#   [2:770]  Semantic : Text(384) + Code(384) embeddings
#   [770:778] Layer1  : 8d per-timestep micro-window stats (extract_layer1_features)
#   [778:786] Layer2  : 8d window-level macro stats broadcast to all timesteps
#                       (extract_layer2_features)
#
# Layer1 (8d, per-timestep micro-window, ~0.5s sliding window):
#   0  fixation_ratio   – fraction of low-velocity steps
#   1  saccade_amp      – mean saccade magnitude
#   2  saccade_std      – saccade amplitude std
#   3  velocity         – mean velocity (norm-units/s)
#   4  dispersion_x     – x-axis std
#   5  dispersion_y     – y-axis std
#   6  direction_change – mean absolute angle change
#   7  acceleration     – mean absolute acceleration
#
# Layer2 (8d, whole-window macro statistics, broadcast):
#   0  total_path_length – sum of all step distances
#   1  mean_velocity     – mean velocity across whole window
#   2  velocity_std      – velocity std across whole window
#   3  x_range           – max - min of normalised x
#   4  y_range           – max - min of normalised y
#   5  direction_entropy – 8-bin angular entropy over whole window
#   6  revisit_density   – fraction of second-half steps in first-half grid cells
#   7  centroid_shift    – first-half vs second-half centroid distance
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# ── Layer1: 8d per-timestep micro-window stats ────────────────────────────────
LAYER1_NAMES = [
    "fixation_ratio",  # 0
    "saccade_amp",  # 1
    "saccade_std",  # 2
    "velocity",  # 3
    "dispersion_x",  # 4
    "dispersion_y",  # 5
    "direction_change",  # 6
    "acceleration",  # 7
]
N_LAYER1 = len(LAYER1_NAMES)  # 8

# ── Layer2: 8d whole-window macro stats ───────────────────────────────────────
LAYER2_NAMES = [
    "total_path_length",  # 0
    "mean_velocity",  # 1
    "velocity_std",  # 2
    "x_range",  # 3
    "y_range",  # 4
    "direction_entropy",  # 5
    "revisit_density",  # 6
    "centroid_shift",  # 7
]
N_LAYER2 = len(LAYER2_NAMES)  # 8

N_BEHAV = N_LAYER1 + N_LAYER2  # 16  (used only for legacy compatibility)

# ── Legacy alias (12d combined) kept for dataset.py cache compat ─────────────
BEHAVIOR_FEATURE_NAMES = LAYER1_NAMES + LAYER2_NAMES  # 16 names total

FIX_VEL_THR = 0.8  # velocity threshold for fixation classification
N_DIR_BINS = 8  # angular bins for direction entropy


def extract_layer1_features(txy: np.ndarray, micro_win: int = 16) -> np.ndarray:
    """
    Slide a micro-window over raw gaze → compute N_LAYER1=8 per-timestep stats.

    Parameters
    ----------
    txy       : ndarray [N, 3]  — columns: (time_ms, norm_x, norm_y)
    micro_win : int             — micro-window size (16 samples @ ~33Hz ≈ 0.5 s)

    Returns
    -------
    ndarray [T, 8]  where T = N - micro_win + 1
    """
    if isinstance(txy, list):
        txy = np.array(txy, dtype=np.float32)

    N = len(txy)
    if N < micro_win:
        return np.zeros((0, N_LAYER1), dtype=np.float32)

    t, x, y = txy[:, 0], txy[:, 1], txy[:, 2]

    dx, dy = np.diff(x), np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    dt = np.diff(t)
    dt = np.where(dt < 1e-6, 30.0, dt)  # default ~30 ms
    vel = dist / dt * 1000.0  # normalised-units / s

    angles = np.arctan2(dy, dx)
    angle_change = np.abs(np.diff(angles))
    angle_change = np.minimum(angle_change, 2 * np.pi - angle_change)
    accel = np.abs(np.diff(vel))

    T_out = N - micro_win + 1
    out = np.zeros((T_out, N_LAYER1), dtype=np.float32)

    for i in range(T_out):
        sl = slice(i, i + micro_win)
        sl_d = slice(i, i + micro_win - 1)
        sl_a = slice(i, i + micro_win - 2)
        w_x, w_y = x[sl], y[sl]
        w_dist, w_vel = dist[sl_d], vel[sl_d]

        out[i, 0] = (w_vel < FIX_VEL_THR).mean()  # fixation_ratio
        out[i, 1] = w_dist.mean()  # saccade_amp
        out[i, 2] = w_dist.std()  # saccade_std
        out[i, 3] = w_vel.mean()  # velocity
        out[i, 4] = w_x.std()  # dispersion_x
        out[i, 5] = w_y.std()  # dispersion_y

        if micro_win > 2:
            w_ac = angle_change[sl_a]
            if len(w_ac):
                out[i, 6] = w_ac.mean()  # direction_change

        if micro_win > 2:
            w_accel = accel[sl_a]
            if len(w_accel):
                out[i, 7] = w_accel.mean()  # acceleration

    return out


def extract_layer2_features(txy: np.ndarray) -> np.ndarray:
    """
    Compute N_LAYER2=8 macro stats over the entire gaze window.

    Parameters
    ----------
    txy  : ndarray [N, 3]  — columns: (time_ms, norm_x, norm_y)

    Returns
    -------
    ndarray [8]   — one 8d vector for the whole window (to be broadcast)
    """
    if isinstance(txy, list):
        txy = np.array(txy, dtype=np.float32)

    N = len(txy)
    feat = np.zeros(N_LAYER2, dtype=np.float32)

    if N < 2:
        return feat

    t, x, y = txy[:, 0], txy[:, 1], txy[:, 2]
    dx, dy = np.diff(x), np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    dt = np.diff(t)
    dt = np.where(dt < 1e-6, 30.0, dt)
    vel = dist / dt * 1000.0

    feat[0] = dist.sum()  # total_path_length
    feat[1] = vel.mean()  # mean_velocity
    feat[2] = vel.std()  # velocity_std
    feat[3] = x.max() - x.min()  # x_range
    feat[4] = y.max() - y.min()  # y_range

    # direction_entropy over whole window
    angles = np.arctan2(dy, dx)
    bins = ((angles + np.pi) / (2 * np.pi) * N_DIR_BINS).astype(int)
    bins = np.clip(bins, 0, N_DIR_BINS - 1)
    cnts = np.bincount(bins, minlength=N_DIR_BINS).astype(float)
    cnts_nz = cnts[cnts > 0]
    if len(cnts_nz):
        probs = cnts_nz / cnts_nz.sum()
        feat[5] = -np.sum(probs * np.log2(probs + 1e-12))  # direction_entropy

    # revisit_density
    half = N // 2
    if half > 0:
        grid_first = {(int(x[j] * 4), int(y[j] * 4)) for j in range(half)}
        feat[6] = sum(
            (int(x[j] * 4), int(y[j] * 4)) in grid_first for j in range(half, N)
        ) / max(1, N - half)

    # centroid_shift
    cx1, cy1 = x[:half].mean(), y[:half].mean()
    cx2, cy2 = x[half:].mean(), y[half:].mean()
    feat[7] = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    return feat


def extract_behavior_features(txy: np.ndarray, micro_win: int = 16) -> np.ndarray:
    """
    Legacy wrapper: returns L1 features only (8d per timestep).
    Kept for backward compatibility with older dataset caches.
    New code should call extract_layer1_features() directly.
    """
    return extract_layer1_features(txy, micro_win=micro_win)

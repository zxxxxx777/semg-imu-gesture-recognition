import os
import numpy as np
import scipy.io as scio
from config import (
    DATA_ROOT,
    DROP_MIXED_WINDOW,
    EMG_CHANNELS,
    EPN_REPS_PER_CLASS,
    EPN_SAMPLING_RATE,
    IMU_CHANNELS,
    NUM_CLASSES,
    OUTER_TRAIN_GROUP,
    STEP_SIZE,
    WINDOW_SIZE,
)


def format_user_id(user_id):
    if isinstance(user_id, int):
        return f"user_{user_id:03d}"
    if isinstance(user_id, str) and user_id.isdigit():
        return f"user_{int(user_id):03d}"
    return str(user_id)


def load_user_data(user_id, outer_group=OUTER_TRAIN_GROUP, data_root=DATA_ROOT):
    user_name = format_user_id(user_id)
    mat_path = os.path.join(data_root, outer_group, user_name, "userData.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"EPN userData.mat not found: {mat_path}")
    return scio.loadmat(mat_path, simplify_cells=True)["userData"]


def list_available_200hz_users(data_root=DATA_ROOT, outer_group=OUTER_TRAIN_GROUP):
    group_dir = os.path.join(data_root, outer_group)
    users = []
    for name in sorted(os.listdir(group_dir)):
        if not name.startswith("user_"):
            continue
        user_data = load_user_data(name, outer_group=outer_group, data_root=data_root)
        sr = int(user_data["deviceInfo"]["emgSamplingRate"])
        if sr != EPN_SAMPLING_RATE:
            continue
        first_trial = user_data["training"][15]
        gyro = np.asarray(first_trial.get("gyro", []))
        if gyro.size == 0:
            continue
        users.append(name)
    return users


def _interp_to_length(x, target_len):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for interpolation, got shape {x.shape}")
    if len(x) == target_len:
        return x
    if len(x) == 0:
        raise ValueError("Cannot interpolate an empty IMU array")

    old_t = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.empty((target_len, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(new_t, old_t, x[:, ch]).astype(np.float32)
    return out


def _trial_imu_to_emg_timeline(trial, emg_len):
    gyro = np.asarray(trial["gyro"], dtype=np.float32)
    accel = np.asarray(trial["accel"], dtype=np.float32)
    quat = np.asarray(trial["quaternions"], dtype=np.float32)

    if gyro.size == 0 or accel.size == 0 or quat.size == 0:
        raise ValueError("Encountered empty IMU field in a requested 200 Hz user")

    imu = np.concatenate([gyro, accel, quat], axis=1)
    if imu.shape[1] != IMU_CHANNELS:
        raise ValueError(f"Expected {IMU_CHANNELS} IMU channels, got {imu.shape[1]}")
    return _interp_to_length(imu, emg_len)


def _gesture_bounds(trial):
    ground_truth = np.asarray(trial["groundTruth"]).reshape(-1)
    active = np.where(ground_truth > 0)[0]
    if len(active) == 0:
        return None
    return int(active[0]), int(active[-1]) + 1


def load_epn_user_multimodal_with_rep(
    user_id,
    sections=("training", "testing"),
    outer_group=OUTER_TRAIN_GROUP,
    require_200hz=True,
):
    if isinstance(sections, str):
        sections = (sections,)

    user_data = load_user_data(user_id, outer_group=outer_group)
    sr = int(user_data["deviceInfo"]["emgSamplingRate"])
    if require_200hz and sr != EPN_SAMPLING_RATE:
        raise ValueError(f"{format_user_id(user_id)} has {sr} Hz data, expected {EPN_SAMPLING_RATE} Hz")

    emg_list, imu_list, label_list, rep_list = [], [], [], []

    for section in sections:
        trials = user_data[section]
        for trial_idx, trial in enumerate(trials):
            if "groundTruth" not in trial:
                continue

            label = trial_idx // EPN_REPS_PER_CLASS - 1
            rep = trial_idx % EPN_REPS_PER_CLASS + 1
            if label < 0 or label >= NUM_CLASSES:
                continue

            emg = np.asarray(trial["emg"], dtype=np.float32)
            if emg.ndim != 2 or emg.shape[1] != EMG_CHANNELS:
                raise ValueError(f"Expected EMG shape (T, {EMG_CHANNELS}), got {emg.shape}")

            bounds = _gesture_bounds(trial)
            if bounds is None:
                continue
            start, end = bounds
            imu = _trial_imu_to_emg_timeline(trial, len(emg))

            emg_seg = emg[start:end]
            imu_seg = imu[start:end]
            if len(emg_seg) == 0:
                continue

            emg_list.append(emg_seg)
            imu_list.append(imu_seg)
            label_list.append(np.full(len(emg_seg), label, dtype=np.int64))
            rep_list.append(np.full(len(emg_seg), rep, dtype=np.int64))

    if not emg_list:
        raise ValueError(f"No labeled multimodal trials loaded for {format_user_id(user_id)}")

    return (
        np.vstack(emg_list),
        np.vstack(imu_list),
        np.concatenate(label_list),
        np.concatenate(rep_list),
    )


def sliding_window_multimodal(X_emg, X_imu, y, step_size=STEP_SIZE):
    X_emg_w, X_imu_w, y_w = [], [], []
    if len(X_emg) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, X_emg.shape[1]), dtype=X_emg.dtype),
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    total = (len(X_emg) - WINDOW_SIZE) // step_size + 1
    for i in range(total):
        start = i * step_size
        end = start + WINDOW_SIZE
        wy = y[start:end]
        if DROP_MIXED_WINDOW and not np.all(wy == wy[0]):
            continue
        label = wy[0] if DROP_MIXED_WINDOW else np.bincount(wy).argmax()
        X_emg_w.append(X_emg[start:end])
        X_imu_w.append(X_imu[start:end])
        y_w.append(label)

    return np.asarray(X_emg_w), np.asarray(X_imu_w), np.asarray(y_w)


def sliding_window_multimodal_by_segment(X_emg, X_imu, y, rep, step_size=STEP_SIZE):
    X_emg_w, X_imu_w, y_w = [], [], []
    if len(y) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, X_emg.shape[1]), dtype=X_emg.dtype),
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    change_points = np.where((y[1:] != y[:-1]) | (rep[1:] != rep[:-1]))[0] + 1
    boundaries = np.concatenate(([0], change_points, [len(y)]))

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < WINDOW_SIZE:
            continue
        Xw_emg, Xw_imu, Yw = sliding_window_multimodal(
            X_emg[start:end],
            X_imu[start:end],
            y[start:end],
            step_size=step_size,
        )
        if len(Yw) == 0:
            continue
        X_emg_w.append(Xw_emg)
        X_imu_w.append(Xw_imu)
        y_w.append(Yw)

    if not y_w:
        return (
            np.empty((0, WINDOW_SIZE, X_emg.shape[1]), dtype=X_emg.dtype),
            np.empty((0, WINDOW_SIZE, X_imu.shape[1]), dtype=X_imu.dtype),
            np.empty((0,), dtype=y.dtype),
        )

    return np.concatenate(X_emg_w, axis=0), np.concatenate(X_imu_w, axis=0), np.concatenate(y_w, axis=0)

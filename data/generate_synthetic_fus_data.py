"""
generate_synthetic_fus_data.py
==============================
Generate synthetic functional ultrasound (fUS) data that faithfully mimics
the Caltech fUS-BMI dataset structure (Griggs et al., Nature Neuroscience 2023).

The real dataset contains Power Doppler images acquired at 2 Hz from the
posterior parietal cortex of rhesus macaques performing a delayed-reach task
with up to 8 movement directions.  Each .mat file stores variables such as
`dop`, `behavior`, `timestamps`, `actual_labels`, `state_num`, etc.

Because the original archive is 34.3 GB, we synthesise a lightweight replica
(~50 MB total) that preserves:
  - Spatial structure  : 128 x 100 pixel Power Doppler frames (width x depth)
  - Temporal structure : 2 Hz sampling over ~200 s per run
  - Trial structure    : 8 target directions, ~40 trials per run
  - Hemodynamic-like   : slow drift + direction-dependent activation + noise
  - State machine      : trialstart → fixation → cue → memory → go → reward → iti

Output
------
data/
  ├── project_record.json
  ├── rt_fUS_data_S1_R1.mat
  ├── rt_fUS_data_S1_R2.mat
  ├── rt_fUS_data_S2_R1.mat
  └── rt_fUS_data_S2_R2.mat
"""

import json
import os
import numpy as np
from scipy.io import savemat

# ── reproducibility ──────────────────────────────────────────────────
np.random.seed(42)

# ── global imaging parameters ────────────────────────────────────────
N_PIXELS_X = 128          # lateral pixels  (matches 128-element probe)
N_PIXELS_Z = 100          # axial pixels
FS = 2.0                  # Power Doppler frame rate (Hz)
DIRECTIONS = np.arange(8) # 0-7 → 0°, 45°, …, 315°

# Trial state machine (name → nominal duration in seconds)
STATE_DURATIONS = {
    "trialstart":       0.5,
    "initialfixation":  1.0,
    "fixationhold":     0.5,
    "cue":              1.0,
    "memory":           1.5,
    "target_acquire":   1.0,
    "target_hold":      0.5,
    "reward":           0.5,
    "iti":              2.0,
}
STATE_NAMES = list(STATE_DURATIONS.keys())
TRIAL_DUR = sum(STATE_DURATIONS.values())  # ~8.5 s

# ── Region-of-interest masks (simplified cortical areas) ─────────────
def _make_roi_masks(nx=N_PIXELS_X, nz=N_PIXELS_Z):
    """Return dict of boolean masks for five cortical regions."""
    masks = {}
    # Rough spatial layout (left→right, top→bottom)
    masks["LIP"] = _ellipse_mask(nx, nz, cx=30, cz=40, rx=12, rz=15)
    masks["MIP"] = _ellipse_mask(nx, nz, cx=64, cz=35, rx=14, rz=12)
    masks["VIP"] = _ellipse_mask(nx, nz, cx=98, cz=40, rx=12, rz=15)
    masks["Area5"] = _ellipse_mask(nx, nz, cx=45, cz=70, rx=18, rz=10)
    masks["Area7"] = _ellipse_mask(nx, nz, cx=85, cz=70, rx=18, rz=10)
    return masks


def _ellipse_mask(nx, nz, cx, cz, rx, rz):
    xx, zz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    return ((xx - cx) ** 2 / rx ** 2 + (zz - cz) ** 2 / rz ** 2) <= 1.0


# ── Direction-dependent activation profiles ──────────────────────────
def _direction_activation(direction, roi_masks):
    """
    Each direction preferentially activates a subset of ROIs,
    mimicking the tuning properties observed in PPC.
    """
    # Activation weights: rows = directions 0-7, cols = LIP MIP VIP A5 A7
    tuning = np.array([
        [1.0, 0.3, 0.2, 0.8, 0.1],  # 0°   (right)
        [0.8, 0.6, 0.1, 0.7, 0.3],  # 45°
        [0.3, 1.0, 0.2, 0.4, 0.6],  # 90°  (up)
        [0.1, 0.8, 0.5, 0.2, 0.9],  # 135°
        [0.2, 0.3, 1.0, 0.1, 0.8],  # 180° (left)
        [0.4, 0.2, 0.9, 0.3, 0.7],  # 225°
        [0.6, 0.1, 0.5, 0.9, 0.4],  # 270° (down)
        [0.9, 0.4, 0.3, 0.6, 0.2],  # 315°
    ])
    weights = tuning[direction]
    activation_map = np.zeros((N_PIXELS_X, N_PIXELS_Z))
    for w, (name, mask) in zip(weights, roi_masks.items()):
        activation_map += w * mask.astype(float)
    return activation_map


# ── Hemodynamic response function (simplified) ──────────────────────
def _hrf(t, peak=3.0, undershoot=6.0):
    """Simplified HRF adapted for fUS (faster than fMRI)."""
    from scipy.stats import gamma as gamma_dist
    h = gamma_dist.pdf(t, a=peak) - 0.2 * gamma_dist.pdf(t, a=undershoot)
    return h / h.max()


# ── Generate one session-run ─────────────────────────────────────────
def generate_run(n_trials=40, noise_level=0.45):
    """
    Returns
    -------
    dop          : (n_pixels_x, n_pixels_z, n_frames) float32
    timestamps   : (n_frames,) float64   – seconds from run start
    actual_labels: (n_frames,) int        – target direction per frame (-1 = ITI)
    state_num    : (n_frames,) int        – state index per frame
    behavior     : dict with trial-level info
    """
    roi_masks = _make_roi_masks()

    # Assign random directions to trials
    trial_dirs = np.random.choice(DIRECTIONS, size=n_trials)

    # Build frame-level timeline
    dt = 1.0 / FS
    total_time = n_trials * TRIAL_DUR + 5.0  # small buffer
    n_frames = int(total_time * FS)
    timestamps = np.arange(n_frames) * dt

    # Frame-level labels
    actual_labels = -np.ones(n_frames, dtype=int)
    state_num = np.zeros(n_frames, dtype=int)

    # Pre-compute HRF kernel
    hrf_t = np.arange(0, 8, dt)
    hrf_kernel = _hrf(hrf_t)

    # Neural drive time-course per pixel (accumulated)
    neural_drive = np.zeros((N_PIXELS_X, N_PIXELS_Z, n_frames))

    trial_records = []
    t_cursor = 2.0  # start after 2 s baseline

    for trial_idx in range(n_trials):
        direction = trial_dirs[trial_idx]
        activation = _direction_activation(direction, roi_masks)

        trial_start_time = t_cursor
        trial_rec = {"trial": trial_idx + 1, "direction": int(direction),
                      "direction_deg": int(direction) * 45,
                      "start_time": trial_start_time, "states": {}}

        for si, (sname, sdur) in enumerate(STATE_DURATIONS.items()):
            # Add jitter
            jittered_dur = sdur + np.random.uniform(-0.1, 0.1)
            jittered_dur = max(jittered_dur, dt)

            f_start = int(t_cursor * FS)
            f_end = int((t_cursor + jittered_dur) * FS)
            f_end = min(f_end, n_frames)

            for f in range(f_start, f_end):
                state_num[f] = si
                actual_labels[f] = int(direction)

            # Activation during cue / memory / target_acquire
            if sname in ("cue", "memory", "target_acquire", "target_hold"):
                for f in range(f_start, f_end):
                    if f < n_frames:
                        neural_drive[:, :, f] += activation * 0.5

            trial_rec["states"][sname] = {
                "start": round(t_cursor, 3),
                "end": round(t_cursor + jittered_dur, 3),
            }
            t_cursor += jittered_dur

        trial_records.append(trial_rec)

    # Convolve neural drive with HRF (per pixel, along time axis)
    # For efficiency, convolve the spatial-mean per ROI then broadcast
    dop = np.zeros((N_PIXELS_X, N_PIXELS_Z, n_frames), dtype=np.float32)

    # Baseline Power Doppler level
    baseline = 1000.0 + 200.0 * np.random.rand(N_PIXELS_X, N_PIXELS_Z).astype(np.float32)

    # Slow physiological drift (larger to mimic real physiology)
    drift = 120.0 * np.sin(2 * np.pi * 0.01 * timestamps)[np.newaxis, np.newaxis, :]
    drift += 60.0 * np.sin(2 * np.pi * 0.03 * timestamps + 1.5)[np.newaxis, np.newaxis, :]

    # Convolve neural drive with HRF for each pixel (vectorised over spatial dims)
    nd_flat = neural_drive.reshape(-1, n_frames)
    hrf_len = len(hrf_kernel)
    conv_flat = np.zeros_like(nd_flat)
    for i in range(nd_flat.shape[0]):
        c = np.convolve(nd_flat[i], hrf_kernel, mode="full")[:n_frames]
        conv_flat[i] = c
    activation_signal = conv_flat.reshape(N_PIXELS_X, N_PIXELS_Z, n_frames).astype(np.float32)

    # Scale activation (moderate to create realistic decoding challenge)
    activation_signal *= 80.0

    # Combine
    noise = noise_level * baseline[:, :, np.newaxis] * np.random.randn(
        N_PIXELS_X, N_PIXELS_Z, n_frames
    ).astype(np.float32)

    dop = baseline[:, :, np.newaxis] + drift.astype(np.float32) + activation_signal + noise

    # Ensure positive
    dop = np.clip(dop, 0, None)

    behavior = {
        "trials": trial_records,
        "n_trials": n_trials,
        "directions_used": sorted(set(int(d) for d in trial_dirs)),
        "n_directions": len(set(int(d) for d in trial_dirs)),
    }

    return dop, timestamps, actual_labels, state_num, behavior


# ── Main ─────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    sessions = [
        {"session": 1, "run": 1, "monkey": "K", "n_targets": 8, "slot": 1,
         "task": "hand", "pretraining": True, "retraining": True, "n_trials": 40},
        {"session": 1, "run": 2, "monkey": "K", "n_targets": 8, "slot": 1,
         "task": "hand", "pretraining": True, "retraining": True, "n_trials": 40},
        {"session": 2, "run": 1, "monkey": "K", "n_targets": 4, "slot": 2,
         "task": "eye", "pretraining": False, "retraining": True, "n_trials": 48},
        {"session": 2, "run": 2, "monkey": "K", "n_targets": 4, "slot": 2,
         "task": "eye", "pretraining": False, "retraining": True, "n_trials": 48},
    ]

    project_record = []

    for sess_info in sessions:
        s = sess_info["session"]
        r = sess_info["run"]
        fname = f"rt_fUS_data_S{s}_R{r}.mat"
        fpath = os.path.join(out_dir, fname)

        print(f"Generating {fname}  ({sess_info['n_trials']} trials) ...")
        dop, ts, labels, states, behavior = generate_run(
            n_trials=sess_info["n_trials"]
        )

        # Build neurovascular map (mean activation)
        neurovascular_map = np.mean(dop, axis=2)

        mat_data = {
            "dop": dop,
            "timestamps": ts,
            "actual_labels": labels,
            "state_num": states,
            "neurovascular_map": neurovascular_map,
            "session": np.array([s]),
            "run": np.array([r]),
            "frame_num": np.arange(len(ts)),
            "time_acq": ts,
            "behavior_json": json.dumps(behavior),
        }

        savemat(fpath, mat_data, do_compression=True)
        print(f"  → saved {fpath}  ({os.path.getsize(fpath) / 1e6:.1f} MB)")

        project_record.append({
            "Session": s,
            "Run": r,
            "Monkey": sess_info["monkey"],
            "nTargets": sess_info["n_targets"],
            "Slot": sess_info["slot"],
            "Task": sess_info["task"],
            "pretraining": sess_info["pretraining"],
            "retraining": sess_info["retraining"],
            "filename": fname,
        })

    # Save project record
    pr_path = os.path.join(out_dir, "project_record.json")
    with open(pr_path, "w") as f:
        json.dump(project_record, f, indent=2)
    print(f"\nProject record → {pr_path}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
data_loader.py
==============
Stage 1 – Data Engineering & Preprocessing

This module provides utilities for:
  1. Loading Caltech-format .mat fUS data files (scipy.io / h5py)
  2. Extracting ROI time-series from Power Doppler volumes
  3. Wrapping data into MNE-Python RawArray objects for filtering & artefact rejection
  4. Exploratory visualisation of blood-flow signals across movement intentions

References
----------
Griggs et al., "Decoding Motor Plans Using a Closed-Loop Ultrasonic
Brain-Machine Interface", Nature Neuroscience (2023).
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── ROI definitions (must match generate_synthetic_fus_data.py) ──────
ROI_DEFS = {
    "LIP":   {"cx": 30, "cz": 40, "rx": 12, "rz": 15},
    "MIP":   {"cx": 64, "cz": 35, "rx": 14, "rz": 12},
    "VIP":   {"cx": 98, "cz": 40, "rx": 12, "rz": 15},
    "Area5": {"cx": 45, "cz": 70, "rx": 18, "rz": 10},
    "Area7": {"cx": 85, "cz": 70, "rx": 18, "rz": 10},
}

STATE_NAMES = [
    "trialstart", "initialfixation", "fixationhold", "cue",
    "memory", "target_acquire", "target_hold", "reward", "iti",
]


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def load_fus_mat(filepath):
    """
    Load a single .mat fUS data file.

    Parameters
    ----------
    filepath : str
        Path to rt_fUS_data_S*_R*.mat

    Returns
    -------
    dict with keys: dop, timestamps, actual_labels, state_num,
                    neurovascular_map, behavior, session, run
    """
    print(f"[load] Reading {os.path.basename(filepath)} ...")
    mat = loadmat(filepath, squeeze_me=True)

    data = {
        "dop": mat["dop"],                          # (nx, nz, n_frames)
        "timestamps": mat["timestamps"].flatten(),   # (n_frames,)
        "actual_labels": mat["actual_labels"].flatten().astype(int),
        "state_num": mat["state_num"].flatten().astype(int),
        "neurovascular_map": mat["neurovascular_map"],
        "session": int(mat["session"]),
        "run": int(mat["run"]),
    }

    # Parse embedded JSON behaviour record
    if "behavior_json" in mat:
        beh_str = str(mat["behavior_json"])
        data["behavior"] = json.loads(beh_str)
    else:
        data["behavior"] = {}

    nx, nz, nf = data["dop"].shape
    print(f"  → Doppler volume : {nx} x {nz} pixels, {nf} frames")
    print(f"  → Duration       : {data['timestamps'][-1]:.1f} s  ({nf / 2.0:.1f} s at 2 Hz)")
    print(f"  → Unique labels  : {sorted(set(data['actual_labels']))}")
    return data


def load_all_sessions(data_dir=DATA_DIR):
    """Load all .mat files listed in project_record.json."""
    pr_path = os.path.join(data_dir, "project_record.json")
    with open(pr_path) as f:
        project_record = json.load(f)

    datasets = []
    for rec in project_record:
        fpath = os.path.join(data_dir, rec["filename"])
        d = load_fus_mat(fpath)
        d["meta"] = rec
        datasets.append(d)
    return datasets, project_record


# =====================================================================
# 2. ROI TIME-SERIES EXTRACTION
# =====================================================================
def _ellipse_mask(nx, nz, cx, cz, rx, rz):
    xx, zz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    return ((xx - cx) ** 2 / rx ** 2 + (zz - cz) ** 2 / rz ** 2) <= 1.0


def extract_roi_timeseries(dop, roi_defs=ROI_DEFS):
    """
    Compute mean Power Doppler within each ROI at every frame.

    Parameters
    ----------
    dop : ndarray (nx, nz, n_frames)

    Returns
    -------
    roi_ts : dict  {roi_name: 1-D array of length n_frames}
    """
    nx, nz, nf = dop.shape
    roi_ts = {}
    for name, params in roi_defs.items():
        mask = _ellipse_mask(nx, nz, **params)
        roi_ts[name] = dop[mask, :].mean(axis=0)
    return roi_ts


# =====================================================================
# 3. MNE-PYTHON INTEGRATION
# =====================================================================
def build_mne_raw(roi_ts, timestamps, sfreq=2.0):
    """
    Wrap ROI time-series into an MNE RawArray.

    Parameters
    ----------
    roi_ts   : dict {name: 1-D array}
    timestamps : 1-D array (seconds)
    sfreq    : float, sampling frequency

    Returns
    -------
    raw : mne.io.RawArray
    """
    ch_names = list(roi_ts.keys())
    n_channels = len(ch_names)
    n_times = len(timestamps)

    # Stack into (n_channels, n_times) – MNE convention
    data = np.stack([roi_ts[ch] for ch in ch_names], axis=0)

    # Normalise to micro-volts scale for MNE compatibility
    # (Power Doppler is arbitrary units; we z-score then scale)
    for i in range(n_channels):
        mu, sigma = data[i].mean(), data[i].std()
        if sigma > 0:
            data[i] = (data[i] - mu) / sigma * 1e-6  # µV-like scale

    # Use 'eeg' type so MNE treats these as data channels for filtering
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def apply_mne_filtering(raw, l_freq=0.01, h_freq=0.5):
    """
    Band-pass filter the MNE Raw object.

    For fUS at 2 Hz, the Nyquist is 1 Hz.  We keep 0.01–0.5 Hz to
    preserve hemodynamic fluctuations while removing drift and noise.
    """
    print(f"[MNE] Filtering {l_freq}–{h_freq} Hz ...")
    raw_filtered = raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq, method="fir",
        fir_window="hamming", verbose=False,
    )
    return raw_filtered


# =====================================================================
# 4. FEATURE EXPLORATION & VISUALISATION
# =====================================================================
def build_exploration_dataframe(roi_ts, actual_labels, timestamps, state_num):
    """
    Create a tidy Pandas DataFrame for exploratory analysis.
    """
    records = []
    for t_idx in range(len(timestamps)):
        for roi_name, ts_arr in roi_ts.items():
            records.append({
                "time": timestamps[t_idx],
                "roi": roi_name,
                "power_doppler": ts_arr[t_idx],
                "direction": actual_labels[t_idx],
                "state": STATE_NAMES[state_num[t_idx]] if state_num[t_idx] < len(STATE_NAMES) else "unknown",
            })
    df = pd.DataFrame(records)
    return df


def plot_neurovascular_map(nv_map, session, run, save=True):
    """Plot the mean Power Doppler image (neurovascular map)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(nv_map.T, aspect="auto", cmap="hot", origin="lower")
    ax.set_xlabel("Lateral position (pixels)")
    ax.set_ylabel("Axial depth (pixels)")
    ax.set_title(f"Neurovascular Map — Session {session}, Run {run}")
    plt.colorbar(im, ax=ax, label="Mean Power Doppler (a.u.)")

    # Overlay ROI contours
    for name, p in ROI_DEFS.items():
        theta = np.linspace(0, 2 * np.pi, 100)
        x = p["cx"] + p["rx"] * np.cos(theta)
        z = p["cz"] + p["rz"] * np.sin(theta)
        ax.plot(x, z, "c--", linewidth=1.2, alpha=0.8)
        ax.text(p["cx"], p["cz"] - p["rz"] - 2, name,
                ha="center", va="bottom", color="cyan", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"neurovascular_map_S{session}_R{run}.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_roi_timeseries(roi_ts, timestamps, actual_labels, session, run, save=True):
    """Plot ROI time-series with direction colour-coding."""
    fig, axes = plt.subplots(len(roi_ts), 1, figsize=(14, 2.5 * len(roi_ts)), sharex=True)
    if len(roi_ts) == 1:
        axes = [axes]

    cmap = plt.cm.Set1
    direction_colors = {d: cmap(d / 8.0) for d in range(8)}
    direction_colors[-1] = (0.8, 0.8, 0.8, 0.3)

    for ax, (name, ts) in zip(axes, roi_ts.items()):
        # Plot the time-series
        ax.plot(timestamps, ts, "k-", linewidth=0.5, alpha=0.7)

        # Shade by direction
        for d in range(8):
            mask = actual_labels == d
            if mask.any():
                ax.fill_between(timestamps, ts.min(), ts.max(),
                                where=mask, alpha=0.15, color=direction_colors[d],
                                label=f"Dir {d} ({d*45}°)" if ax == axes[0] else "")
        ax.set_ylabel(f"{name}\n(a.u.)")
        ax.set_xlim(timestamps[0], min(timestamps[-1], 100))  # first 100 s

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"ROI Blood-Flow Time-Series — Session {session}, Run {run}")
    axes[0].legend(loc="upper right", fontsize=7, ncol=4)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"roi_timeseries_S{session}_R{run}.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_direction_boxplot(df, session, run, save=True):
    """Box-plot of Power Doppler by direction for each ROI."""
    # Filter to active states only
    active = df[df["state"].isin(["cue", "memory", "target_acquire", "target_hold"])]
    if active.empty:
        print("  [warn] No active-state data for boxplot.")
        return

    rois = active["roi"].unique()
    fig, axes = plt.subplots(1, len(rois), figsize=(4 * len(rois), 5), sharey=False)
    if len(rois) == 1:
        axes = [axes]

    for ax, roi in zip(axes, rois):
        subset = active[active["roi"] == roi]
        directions = sorted(subset["direction"].unique())
        data_by_dir = [subset[subset["direction"] == d]["power_doppler"].values for d in directions]
        bp = ax.boxplot(data_by_dir, labels=[f"{d*45}°" for d in directions], patch_artist=True)
        colors = plt.cm.Set1(np.linspace(0, 1, len(directions)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(roi, fontweight="bold")
        ax.set_xlabel("Movement Direction")
        if ax == axes[0]:
            ax.set_ylabel("Power Doppler (a.u.)")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Blood-Flow Distribution by Direction (Active States) — S{session} R{run}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"direction_boxplot_S{session}_R{run}.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_mne_psd(raw, session, run, save=True):
    """Plot power spectral density from MNE Raw object."""
    fig = raw.compute_psd(fmin=0.01, fmax=0.9, verbose=False).plot(show=False)
    fig.suptitle(f"Power Spectral Density — Session {session}, Run {run}", fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"psd_S{session}_R{run}.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_filtered_comparison(raw_orig, raw_filt, timestamps, session, run, save=True):
    """Compare original vs filtered signals for one channel."""
    ch = raw_orig.ch_names[0]
    orig_data = raw_orig.get_data(picks=[ch])[0]
    filt_data = raw_filt.get_data(picks=[ch])[0]
    t = np.arange(len(orig_data)) / raw_orig.info["sfreq"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax1.plot(t, orig_data * 1e6, "b-", linewidth=0.6, alpha=0.8)
    ax1.set_ylabel(f"{ch} (z-scored)")
    ax1.set_title(f"Original Signal — Session {session}, Run {run}")

    ax2.plot(t, filt_data * 1e6, "r-", linewidth=0.6, alpha=0.8)
    ax2.set_ylabel(f"{ch} (z-scored)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("After Band-Pass Filter (0.01–0.5 Hz)")
    ax2.set_xlim(0, min(t[-1], 100))

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, f"filter_comparison_S{session}_R{run}.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_stage1():
    """Execute the full Stage 1 pipeline."""
    print("=" * 70)
    print("  STAGE 1: Data Engineering & Preprocessing")
    print("=" * 70)

    # ── 1. Load all data ─────────────────────────────────────────────
    print("\n[Step 1] Loading fUS data files ...\n")
    datasets, project_record = load_all_sessions()

    # ── 2. Process first session in detail ───────────────────────────
    d = datasets[0]
    s, r = d["session"], d["run"]

    print(f"\n[Step 2] Extracting ROI time-series for S{s}_R{r} ...")
    roi_ts = extract_roi_timeseries(d["dop"])
    for name, ts in roi_ts.items():
        print(f"  {name:>6s} : mean={ts.mean():.1f}, std={ts.std():.1f}, "
              f"min={ts.min():.1f}, max={ts.max():.1f}")

    # ── 3. MNE integration ───────────────────────────────────────────
    print(f"\n[Step 3] Building MNE RawArray ...")
    raw = build_mne_raw(roi_ts, d["timestamps"])
    print(f"  → {raw}")

    print(f"\n[Step 4] Applying MNE band-pass filter ...")
    raw_filt = apply_mne_filtering(raw, l_freq=0.01, h_freq=0.5)

    # ── 4. Exploratory analysis ──────────────────────────────────────
    print(f"\n[Step 5] Building exploration DataFrame ...")
    df = build_exploration_dataframe(roi_ts, d["actual_labels"], d["timestamps"], d["state_num"])
    print(f"  → DataFrame shape: {df.shape}")
    print(f"  → Columns: {list(df.columns)}")
    print(f"\n  Summary statistics (active states):")
    active_df = df[df["state"].isin(["cue", "memory", "target_acquire"])]
    if not active_df.empty:
        summary = active_df.groupby(["roi", "direction"])["power_doppler"].agg(["mean", "std"]).round(1)
        print(summary.head(20).to_string())

    # ── 5. Visualisations ────────────────────────────────────────────
    print(f"\n[Step 6] Generating visualisations ...")
    plot_neurovascular_map(d["neurovascular_map"], s, r)
    plot_roi_timeseries(roi_ts, d["timestamps"], d["actual_labels"], s, r)
    plot_direction_boxplot(df, s, r)
    plot_mne_psd(raw, s, r)
    plot_filtered_comparison(raw, raw_filt, d["timestamps"], s, r)

    # ── Process remaining sessions (lighter) ─────────────────────────
    for d2 in datasets[1:]:
        s2, r2 = d2["session"], d2["run"]
        print(f"\n[Processing S{s2}_R{r2}] ...")
        roi_ts2 = extract_roi_timeseries(d2["dop"])
        plot_neurovascular_map(d2["neurovascular_map"], s2, r2)
        plot_roi_timeseries(roi_ts2, d2["timestamps"], d2["actual_labels"], s2, r2)

    # ── 6. Save processed data for Stage 2 ───────────────────────────
    print(f"\n[Step 7] Saving processed data for downstream stages ...")
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed_roi_data.npz")
    all_roi_ts = {}
    all_labels = {}
    all_states = {}
    all_timestamps = {}
    for d in datasets:
        key = f"S{d['session']}_R{d['run']}"
        roi = extract_roi_timeseries(d["dop"])
        all_roi_ts[key] = np.stack([roi[ch] for ch in ROI_DEFS.keys()], axis=0)
        all_labels[key] = d["actual_labels"]
        all_states[key] = d["state_num"]
        all_timestamps[key] = d["timestamps"]

    np.savez_compressed(
        processed_path,
        roi_names=list(ROI_DEFS.keys()),
        **{f"roi_{k}": v for k, v in all_roi_ts.items()},
        **{f"labels_{k}": v for k, v in all_labels.items()},
        **{f"states_{k}": v for k, v in all_states.items()},
        **{f"time_{k}": v for k, v in all_timestamps.items()},
    )
    print(f"  → Saved {processed_path}")

    print("\n" + "=" * 70)
    print("  STAGE 1 COMPLETE")
    print("=" * 70)
    print(f"  Figures saved to: {FIG_DIR}")
    print(f"  Processed data  : {processed_path}")

    return datasets, project_record


if __name__ == "__main__":
    run_stage1()

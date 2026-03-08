"""
neural_decoder.py
=================
Stage 2 – Neural Decoding Model Construction

This module implements:
  1. Epoch segmentation using MNE-Python (continuous fUS → per-trial epochs)
  2. Time-domain feature extraction from each epoch
  3. SVM and Random Forest classifiers for movement-intention decoding
  4. Cross-validated evaluation with confusion matrices and performance curves

The pipeline mirrors the approach in Griggs et al. (2023), where fUS signals
from the posterior parietal cortex are used to decode planned movement
directions in a delayed-reach task.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
from scipy.io import loadmat

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────
ROI_NAMES = ["LIP", "MIP", "VIP", "Area5", "Area7"]
STATE_NAMES = [
    "trialstart", "initialfixation", "fixationhold", "cue",
    "memory", "target_acquire", "target_hold", "reward", "iti",
]
SFREQ = 2.0   # Hz
DIRECTION_LABELS = {i: f"{i*45}°" for i in range(8)}


# =====================================================================
# 1. DATA LOADING (reuse processed data from Stage 1)
# =====================================================================
def load_processed_data():
    """Load the processed ROI data saved by Stage 1."""
    npz_path = os.path.join(DATA_DIR, "processed_roi_data.npz")
    data = np.load(npz_path, allow_pickle=True)
    roi_names = list(data["roi_names"])

    sessions = {}
    for key in data.files:
        if key.startswith("roi_") and key != "roi_names":
            sess_key = key.replace("roi_", "")
            sessions[sess_key] = {
                "roi": data[key],                           # (n_ch, n_times)
                "labels": data[f"labels_{sess_key}"],       # (n_times,)
                "states": data[f"states_{sess_key}"],       # (n_times,)
                "time": data[f"time_{sess_key}"],           # (n_times,)
            }
    return sessions, roi_names


# =====================================================================
# 2. MNE EPOCH SEGMENTATION
# =====================================================================
def build_mne_epochs(roi_data, labels, states, sfreq=SFREQ):
    """
    Segment continuous fUS data into per-trial MNE Epochs.

    Strategy
    --------
    We detect trial boundaries by finding transitions where `state_num`
    goes to 0 (trialstart).  For each trial we extract a window covering
    the cue → target_hold period (states 3–6), which is the task-relevant
    interval where direction-specific activation is expected.

    Returns
    -------
    epochs : mne.EpochsArray
    epoch_labels : list of int (direction per epoch)
    epoch_features_df : DataFrame with per-epoch metadata
    """
    n_ch, n_times = roi_data.shape

    # ── Detect trial onsets (state transitions to trialstart=0) ──────
    trial_starts = []
    for i in range(1, n_times):
        if states[i] == 0 and states[i - 1] != 0:
            trial_starts.append(i)
    # Also check if first frame is a trial start
    if states[0] == 0:
        trial_starts.insert(0, 0)

    # ── For each trial, find the cue onset (state 3) and target_hold end (state 6→7)
    epoch_list = []
    epoch_labels = []
    epoch_meta = []

    # We want a fixed-length epoch: from cue onset to ~4 seconds later
    # At 2 Hz, 4 s = 8 samples
    EPOCH_SAMPLES = 8

    for t_start in trial_starts:
        # Find cue onset within this trial (search forward up to 20 samples)
        cue_onset = None
        for j in range(t_start, min(t_start + 20, n_times)):
            if states[j] == 3:  # cue state
                cue_onset = j
                break

        if cue_onset is None:
            continue

        # Check we have enough samples
        if cue_onset + EPOCH_SAMPLES > n_times:
            continue

        # Extract epoch data
        epoch_data = roi_data[:, cue_onset:cue_onset + EPOCH_SAMPLES]  # (n_ch, epoch_len)

        # Get direction label (majority vote in epoch window)
        epoch_lbl = labels[cue_onset:cue_onset + EPOCH_SAMPLES]
        valid = epoch_lbl[epoch_lbl >= 0]
        if len(valid) == 0:
            continue
        direction = int(np.bincount(valid).argmax())

        epoch_list.append(epoch_data)
        epoch_labels.append(direction)
        epoch_meta.append({
            "trial_start_sample": t_start,
            "cue_onset_sample": cue_onset,
            "cue_onset_time": cue_onset / sfreq,
            "direction": direction,
            "direction_deg": direction * 45,
        })

    if len(epoch_list) == 0:
        raise ValueError("No valid epochs found!")

    # Stack into (n_epochs, n_ch, n_times) array
    epochs_data = np.stack(epoch_list, axis=0)
    epoch_labels = np.array(epoch_labels)

    # Build MNE EpochsArray
    info = mne.create_info(ch_names=ROI_NAMES[:n_ch], sfreq=sfreq, ch_types="eeg")
    # Normalise to µV scale
    epochs_scaled = epochs_data.copy()
    for i in range(n_ch):
        mu = epochs_scaled[:, i, :].mean()
        sigma = epochs_scaled[:, i, :].std()
        if sigma > 0:
            epochs_scaled[:, i, :] = (epochs_scaled[:, i, :] - mu) / sigma * 1e-6

    events = np.column_stack([
        np.arange(len(epoch_labels)),
        np.zeros(len(epoch_labels), dtype=int),
        epoch_labels,
    ])
    event_id = {DIRECTION_LABELS[d]: d for d in sorted(set(epoch_labels))}

    epochs = mne.EpochsArray(
        epochs_scaled, info, events=events, event_id=event_id,
        tmin=0, verbose=False,
    )

    meta_df = pd.DataFrame(epoch_meta)
    print(f"  → {len(epoch_labels)} epochs extracted "
          f"({len(set(epoch_labels))} directions, {EPOCH_SAMPLES} samples each)")

    return epochs, epoch_labels, epochs_data, meta_df


# =====================================================================
# 3. FEATURE EXTRACTION
# =====================================================================
def extract_features(epochs_data, roi_names=ROI_NAMES):
    """
    Extract time-domain features from each epoch.

    Features per ROI per epoch:
      - mean, std, max, min, range
      - slope (linear trend)
      - peak latency (sample index of max)
      - energy (sum of squares)

    Returns
    -------
    X : ndarray (n_epochs, n_features)
    feature_names : list of str
    """
    n_epochs, n_ch, n_times = epochs_data.shape
    features = []
    feature_names = []

    for ch_idx, roi in enumerate(roi_names[:n_ch]):
        for ep_idx in range(n_epochs):
            if ch_idx == 0:
                features.append([])
            sig = epochs_data[ep_idx, ch_idx, :]

            feats = {
                f"{roi}_mean": np.mean(sig),
                f"{roi}_std": np.std(sig),
                f"{roi}_max": np.max(sig),
                f"{roi}_min": np.min(sig),
                f"{roi}_range": np.ptp(sig),
                f"{roi}_slope": np.polyfit(np.arange(n_times), sig, 1)[0],
                f"{roi}_peak_lat": np.argmax(sig),
                f"{roi}_energy": np.sum(sig ** 2),
            }

            if ep_idx == 0 and ch_idx == 0:
                feature_names = list(feats.keys())
            elif ch_idx > 0 and ep_idx == 0:
                feature_names.extend(feats.keys())

            features[ep_idx].extend(feats.values())

    X = np.array(features)
    print(f"  → Feature matrix: {X.shape[0]} epochs × {X.shape[1]} features")
    return X, feature_names


# =====================================================================
# 4. CLASSIFICATION
# =====================================================================
def train_and_evaluate(X, y, feature_names):
    """
    Train SVM and Random Forest classifiers with stratified 5-fold CV.

    Returns
    -------
    results : dict with models, scores, predictions
    """
    results = {}

    # ── Define classifiers ───────────────────────────────────────────
    classifiers = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=2,
                random_state=42, n_jobs=-1)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        print(f"\n  [{name}]")

        # Cross-validated accuracy
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"    CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"    Per-fold   : {[f'{s:.3f}' for s in scores]}")

        # Cross-validated predictions for confusion matrix
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        f1 = f1_score(y, y_pred, average="weighted")
        print(f"    Weighted F1: {f1:.3f}")

        # Train final model on all data
        clf.fit(X, y)

        # Feature importance (Random Forest only)
        importances = None
        if "rf" in clf.named_steps:
            importances = clf.named_steps["rf"].feature_importances_

        results[name] = {
            "model": clf,
            "cv_scores": scores,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "y_pred": y_pred,
            "f1_weighted": f1,
            "importances": importances,
        }

    return results


# =====================================================================
# 5. VISUALISATION
# =====================================================================
def plot_confusion_matrices(results, y_true, save=True):
    """Plot confusion matrices for all classifiers."""
    n_clf = len(results)
    fig, axes = plt.subplots(1, n_clf, figsize=(7 * n_clf, 6))
    if n_clf == 1:
        axes = [axes]

    direction_names = [f"{d*45}°" for d in sorted(set(y_true))]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_true, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=direction_names)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nAcc={res['cv_mean']:.1%} ± {res['cv_std']:.1%}",
                     fontweight="bold")
        ax.set_xlabel("Predicted Direction")
        ax.set_ylabel("True Direction")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "confusion_matrices.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_feature_importance(importances, feature_names, top_n=20, save=True):
    """Plot top-N feature importances from Random Forest."""
    if importances is None:
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx][::-1], color="steelblue", edgecolor="navy")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top-20 Feature Importances (Random Forest)", fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_learning_curves(X, y, save=True):
    """Plot learning curves for SVM and RF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classifiers = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ]),
    }

    for ax, (name, clf) in zip(axes, classifiers.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            clf, X, y, cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.2, 1.0, 8),
            random_state=42, n_jobs=-1,
        )
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.15, color="blue")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.15, color="red")
        ax.plot(train_sizes, train_scores.mean(axis=1), "b-o", label="Training")
        ax.plot(train_sizes, val_scores.mean(axis=1), "r-o", label="Validation")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve — {name}", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "learning_curves.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_epoch_erp(epochs, save=True):
    """Plot average evoked response per direction (ERP-like)."""
    event_ids = epochs.event_id
    n_dirs = len(event_ids)
    fig, axes = plt.subplots(1, min(n_dirs, 4), figsize=(4 * min(n_dirs, 4), 4), sharey=True)
    if min(n_dirs, 4) == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, len(epochs.ch_names)))
    selected_dirs = list(event_ids.keys())[:4]

    for ax, dir_name in zip(axes, selected_dirs):
        ep_sub = epochs[dir_name]
        avg = ep_sub.average()
        times = avg.times
        data = avg.data * 1e6  # back to z-score units

        for ch_idx, ch_name in enumerate(avg.ch_names):
            ax.plot(times, data[ch_idx], color=colors[ch_idx], label=ch_name, linewidth=1.5)

        ax.set_title(f"Direction {dir_name}", fontweight="bold")
        ax.set_xlabel("Time from cue (s)")
        if ax == axes[0]:
            ax.set_ylabel("Amplitude (z-scored)")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Average Evoked Response by Direction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "epoch_erp.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


def plot_cv_comparison(results, save=True):
    """Bar chart comparing CV accuracy across classifiers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    means = [results[n]["cv_mean"] for n in names]
    stds = [results[n]["cv_std"] for n in names]
    f1s = [results[n]["f1_weighted"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, means, width, yerr=stds, label="CV Accuracy",
                   color="steelblue", edgecolor="navy", capsize=5)
    bars2 = ax.bar(x + width/2, f1s, width, label="Weighted F1",
                   color="coral", edgecolor="darkred")

    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Comparison (5-Fold CV)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "classifier_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"  → Saved {path}")
    plt.close(fig)


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_stage2():
    """Execute the full Stage 2 pipeline."""
    print("=" * 70)
    print("  STAGE 2: Neural Decoding Model Construction")
    print("=" * 70)

    # ── 1. Load processed data ───────────────────────────────────────
    print("\n[Step 1] Loading processed ROI data ...")
    sessions, roi_names = load_processed_data()
    print(f"  → {len(sessions)} session-runs loaded: {list(sessions.keys())}")

    # ── 2. Epoch segmentation ────────────────────────────────────────
    print("\n[Step 2] Segmenting into MNE Epochs ...")
    all_epochs_data = []
    all_labels = []

    for sess_key, sess in sessions.items():
        print(f"\n  Processing {sess_key} ...")
        epochs, labels, raw_epochs, meta = build_mne_epochs(
            sess["roi"], sess["labels"], sess["states"]
        )
        all_epochs_data.append(raw_epochs)
        all_labels.append(labels)

        # Plot ERP for first session only
        if sess_key == list(sessions.keys())[0]:
            plot_epoch_erp(epochs)

    # Concatenate all sessions
    X_epochs = np.concatenate(all_epochs_data, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    print(f"\n  → Total: {len(y_all)} epochs across all sessions")
    print(f"  → Direction distribution:")
    for d in sorted(set(y_all)):
        print(f"      {d*45:>3}° : {np.sum(y_all == d)} epochs")

    # ── 3. Feature extraction ────────────────────────────────────────
    print("\n[Step 3] Extracting time-domain features ...")
    X, feature_names = extract_features(X_epochs, roi_names)

    # ── 4. Classification ────────────────────────────────────────────
    print("\n[Step 4] Training classifiers (5-fold stratified CV) ...")
    results = train_and_evaluate(X, y_all, feature_names)

    # ── 5. Visualisations ────────────────────────────────────────────
    print("\n[Step 5] Generating visualisations ...")
    plot_confusion_matrices(results, y_all)
    plot_cv_comparison(results)
    plot_learning_curves(X, y_all)

    # Feature importance (RF)
    rf_res = results.get("Random Forest")
    if rf_res and rf_res["importances"] is not None:
        plot_feature_importance(rf_res["importances"], feature_names)

    # ── 6. Save models ───────────────────────────────────────────────
    print("\n[Step 6] Saving trained models ...")
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}_model.pkl")
        joblib.dump(res["model"], model_path)
        print(f"  → {model_path}")

    # Save feature names and metadata
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    meta = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_epochs": int(len(y_all)),
        "n_directions": int(len(set(y_all))),
        "directions": sorted([int(d) for d in set(y_all)]),
        "roi_names": roi_names,
        "sfreq": SFREQ,
        "epoch_samples": 8,
        "classifiers": {},
    }
    for name, res in results.items():
        meta["classifiers"][name] = {
            "cv_accuracy_mean": float(res["cv_mean"]),
            "cv_accuracy_std": float(res["cv_std"]),
            "f1_weighted": float(res["f1_weighted"]),
        }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → {meta_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STAGE 2 COMPLETE")
    print("=" * 70)
    print(f"\n  Classification Results:")
    for name, res in results.items():
        print(f"    {name:20s} : Acc = {res['cv_mean']:.1%} ± {res['cv_std']:.1%}  |  F1 = {res['f1_weighted']:.3f}")
    print(f"\n  Models saved to : {MODEL_DIR}")
    print(f"  Figures saved to: {FIG_DIR}")

    return results


if __name__ == "__main__":
    run_stage2()

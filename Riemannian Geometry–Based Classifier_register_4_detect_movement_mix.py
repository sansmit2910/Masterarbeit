# -*- coding: utf-8 -*-
"""
Riemannian Geometry (Covariance) Classifier for EEG
- Train on four registration recordings (A,B,C,D) using sliding windows
- Cross-validate with GroupKFold (leave-one-registration-out)
- Detect 4-class sequence in the movement-mix target with trigger-locked + jitter scan
- Save confusion matrix & shift histogram figures; save detected sequence to .txt and .csv

Requires: mne, numpy, scikit-learn, pyriemann, matplotlib
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold

# --- Riemannian tools (pyriemann) ---
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
except Exception as e:
    raise ImportError(
        "This script requires 'pyriemann'. Install it with:\n\n"
        "    pip install pyriemann\n\n"
        f"Import error was: {e}"
    )

# --------------------------
# === USER FILE PATHS ===
# --------------------------
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Output directory
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_rg_abcd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_STATE = 42

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)  # mu + beta band, good for motor-related activity
NOTCH_HZ = None         # e.g., 50 or 60 if you need a notch; set to None to skip

# Windowing for training from registration recordings
WIN_SEC = 1.0           # window length (seconds)
STEP_SEC = 0.25         # step between windows (seconds)

# Event-based detection on target (movement-mix)
EVENT_DESC_PREF = "Stimulus/S  1"     # preferred annotation label
TMIN_SEC = 0.0                         # start from trigger (seconds)
SEG_SEC = 1.0                          # segment length for detection (seconds)
JITTER_SAMP = 30                        # +/- samples jitter sweep around trigger
UNK_THRESH = 0.45                       # if max class prob < this -> UNK

# Channels: try Laplacian first, then standard motor sites, else all EEG
PREFERRED_CHANSETS = [
    ["C3L", "CzL", "C4L"],
    ["C3", "Cz", "C4"],
]


# --------------------------
# === HELPERS ===
# --------------------------
def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw = raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw


def _pick_channels(raw: mne.io.BaseRaw) -> List[int]:
    # Try preferred sets in order; fall back to all EEG
    ch_names = [ch.upper() for ch in raw.ch_names]
    picks = None
    for wanted in PREFERRED_CHANSETS:
        idx = []
        for w in wanted:
            try:
                idx.append(ch_names.index(w.upper()))
            except ValueError:
                idx = []
                break
        if idx:
            picks = idx
            break
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True)
    return picks


def _sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    if win <= 0 or step <= 0 or win > T:
        raise ValueError("Bad window/step config.")
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float64)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s+win]
    return X  # (N, C, win)


def _build_dataset(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]:
    """
    Returns:
        X  : (N, C, T) windows from all registrations
        y  : (N,) labels as ints [0..K-1]
        g  : (N,) groups (one group per source file)
        labs: list of class labels in order
        sf : sampling frequency (after resample)
    """
    all_X = []
    all_y = []
    all_g = []
    labs = sorted(list(paths.keys()))
    print(f"[Training] Building covariance windows from registrations {tuple(labs)} ...")

    sf = None
    for lab_idx, lab in enumerate(labs):
        raw = _load_raw(paths[lab])
        if sf is None:
            sf = raw.info["sfreq"]
        picks = _pick_channels(raw)
        ch_used = [raw.ch_names[p] for p in picks]
        print(f"  [{lab}] raw={os.path.basename(paths[lab])} | picks={ch_used} | sf={sf:.1f} Hz")

        X_win = _sliding_windows(raw, picks, WIN_SEC, STEP_SEC)  # (N, C, T)
        y_win = np.full((X_win.shape[0],), lab_idx, dtype=int)
        g_win = np.full((X_win.shape[0],), lab_idx, dtype=int)   # group by registration label/file

        all_X.append(X_win)
        all_y.append(y_win)
        all_g.append(g_win)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    g = np.concatenate(all_g, axis=0)
    print(f"[Train set] X={X.shape} (N,C,T), classes={labs} | per-class counts: "
          f"{ {labs[i]: int((y==i).sum()) for i in range(len(labs))} }")
    return X, y, g, labs, sf


def _riemann_pipeline() -> Pipeline:
    clf = Pipeline(steps=[
        ("cov", Covariances(estimator="oas")),          # (N,C,T) -> (N,C,C)
        ("ts", TangentSpace(metric="riemann")),         # (N,C,C) -> (N, F)
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto", random_state=RANDOM_STATE)),
    ])
    return clf


def _cv_report(X: np.ndarray, y: np.ndarray, groups: np.ndarray, class_names: List[str], n_splits: int = 4):
    kf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    y_true_all, y_pred_all = [], []
    for fold, (tr, te) in enumerate(kf.split(X, y, groups)):
        clf = _riemann_pipeline()
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(yp)
        acc = accuracy_score(y[te], yp)
        print(f"[CV] Fold {fold+1} accuracy: {acc:.3f}")
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = accuracy_score(y_true, y_pred)
    print("\n[CV] Classification report (leave-one-registration-out):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    print(f"[CV] Riemannian(TS+LR) accuracy: {acc:.3f}")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized rows)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), v in np.ndenumerate(cm / cm.sum(axis=1, keepdims=True)):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    cm_path = os.path.join(OUT_DIR, "cv_confusion_matrix_rg_abcd.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] CV confusion matrix → {cm_path}")

    return acc, cm


def _extract_events(raw: mne.io.BaseRaw, prefer_desc: str = EVENT_DESC_PREF):
    """Return (events, chosen_id, desc_used)."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) == 0 or events.size == 0:
        # fallback try to find stim channel detections
        events = mne.find_events(raw, verbose="ERROR")
        if events.size == 0:
            raise RuntimeError("No events/annotations found in target file.")
        # choose the most frequent code
        unique, counts = np.unique(events[:, 2], return_counts=True)
        chosen = unique[np.argmax(counts)]
        return events, chosen, f"code {int(chosen)}"
    # choose preferred, else first key
    if prefer_desc in event_id:
        chosen = event_id[prefer_desc]
        return events, chosen, prefer_desc
    # else pick the most frequent event code
    unique, counts = np.unique(events[:, 2], return_counts=True)
    chosen = unique[np.argmax(counts)]
    # reverse lookup key
    rev = {v: k for k, v in event_id.items()}
    desc = rev.get(chosen, f"code {int(chosen)}")
    return events, chosen, desc


def _detect_sequence(
    clf: Pipeline,
    target_path: str,
    labs: List[str],
    tmin_sec: float,
    seg_sec: float,
    jitter_samp: int,
    unk_thresh: float
):
    raw = _load_raw(target_path)
    picks = _pick_channels(raw)
    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_offset = int(round(tmin_sec * sf))

    events, chosen_code, desc = _extract_events(raw, EVENT_DESC_PREF)
    ev = events[events[:, 2] == chosen_code]
    n_events = len(ev)
    print(f"\n[Target] Using event '{desc}'; N={n_events}")

    # Pre-extract data for speed
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape

    labels_out: List[str] = []
    probs_out: List[np.ndarray] = []
    shifts: List[int] = []

    # Iterate events
    for idx in range(n_events):
        center = ev[idx, 0] + base_offset
        best_label = None
        best_prob = -np.inf
        best_shift = 0
        best_proba_vec = None

        # Collect candidate segments for all shifts then predict in batch
        # Prepare a list of valid windows
        segs = []
        shift_vals = []
        for s in range(-jitter_samp, jitter_samp + 1):
            start = center + s
            end = start + seg_samp
            if start < 0 or end > T:
                continue
            segs.append(data[:, start:end])
            shift_vals.append(s)
        if len(segs) == 0:
            labels_out.append("UNK")
            probs_out.append(np.full((len(labs),), np.nan))
            shifts.append(0)
            continue

        X_evt = np.stack(segs, axis=0)  # (M, C, seg)
        # run through pipeline
        proba = clf.predict_proba(X_evt)  # (M, K)
        # pick shift with max class confidence
        m_idx, m_cls = np.unravel_index(np.nanargmax(proba), proba.shape)
        max_prob = proba[m_idx, m_cls]
        pred_lab = labs[m_cls]
        # unknown thresholding
        if max_prob < unk_thresh:
            labels_out.append("UNK")
        else:
            labels_out.append(pred_lab)
        probs_out.append(proba[m_idx])
        shifts.append(shift_vals[m_idx])

    # Display summary
    counts = {lab: labels_out.count(lab) for lab in labs}
    counts["UNK"] = labels_out.count("UNK")

    print("\n=== Detected Sequence (Riemannian TS+LR, 4-class, trigger-locked+jitter) ===")
    # pretty print in wrapped lines of 80 tokens
    wrapped = []
    line = []
    for token in labels_out:
        line.append(token)
        if len(line) >= 80:
            wrapped.append(" ".join(line))
            line = []
    if line:
        wrapped.append(" ".join(line))
    print("\n".join(wrapped))

    print("\n[Counts]")
    for k in labs + ["UNK"]:
        print(f"  {k}: {counts.get(k, 0)}")

    # Save sequence to disk
    seq_txt = os.path.join(OUT_DIR, "rg_abcd_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    print(f"[Saved sequence] {seq_txt}")

    # Save CSV with per-event probs (best-shift) if available
    csv_path = os.path.join(OUT_DIR, "rg_abcd_movement_mix_sequence.csv")
    header = "event_idx,label,best_shift," + ",".join([f"prob_{lab}" for lab in labs])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, (lab, sft, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if pv is None or np.any(np.isnan(pv)):
                row = [str(i), lab, str(sft)] + [""] * len(labs)
            else:
                row = [str(i), lab, str(sft)] + [f"{p:.6f}" for p in pv]
            f.write(",".join(row) + "\n")
    print(f"[Saved table]    {csv_path}")

    # Jitter shift histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(shifts, bins=np.arange(-jitter_samp - 0.5, jitter_samp + 1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best shift per event)")
    ax.set_xlabel("Shift (samples)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "target_shift_hist_rg_abcd.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Target jitter shift histogram → {hist_path}")

    return labels_out, counts


# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Build dataset from registrations A,B,C,D
    X, y, groups, class_names, sf = _build_dataset(TEMPLATE_SOURCES)

    # 2) Cross-validate (leave-one-registration-out)
    _ = _cv_report(X, y, groups, class_names, n_splits=4)

    # 3) Fit final model on all registration windows
    clf = _riemann_pipeline()
    clf.fit(X, y)

    # 4) Detect sequence in movement-mix (trigger-locked + jitter)
    _ = _detect_sequence(
        clf=clf,
        target_path=TARGET_FILE,
        labs=class_names,
        tmin_sec=TMIN_SEC,
        seg_sec=SEG_SEC,
        jitter_samp=JITTER_SAMP,
        unk_thresh=UNK_THRESH
    )

    print("\n[Done] All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()

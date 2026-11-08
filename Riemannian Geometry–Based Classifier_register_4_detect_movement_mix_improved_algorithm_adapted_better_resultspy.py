# -*- coding: utf-8 -*-
"""
Riemannian Geometry (Covariance) Classifier — A/B only
- Train on two registration recordings (A,B) using sliding windows
- Cross-validate with *time-binned* folds (keeps both classes in every fold)
- Detect A/B sequence in the movement-mix target with annotation-locked + jitter scan
- Save confusion matrix, ROC, and jitter-shift hist; save detected sequence to .txt and .csv

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)

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
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Output directory
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_rg_ab_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_STATE = 42
DISPLAY_PLOTS = True   # set False to only save figures

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)  # mu + beta band
NOTCH_HZ = None         # e.g., 50 or 60; None to skip

# Sliding windows for training (registrations)
WIN_SEC = 1.0           # window length (sec)
STEP_SEC = 0.25         # step between windows (sec)

# Cross-validation (paired time-bin splits that keep both A & B in train/test)
CV_BINS = 5             # 4–10 is fine; 5 is a good default

# Detection on target (movement-mix)
EVENT_DESC_PREF = "Stimulus/S  1"  # annotation label to use if present
TMIN_SEC = 0.0                      # start at event onset
SEG_SEC = 1.0                       # segment length (sec)
JITTER_SAMP = 30                    # ± samples around event
UNK_THRESH = 0.45                   # prob threshold below which -> UNK

# Channels: prefer standard motor set if available in BOTH A & B (and target)
PREFERRED_SET_CANDIDATES = [
    ["C3L", "CzL", "C4L"],
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"],
]

# Euclidean Alignment (EA) to reduce inter-session covariance shifts
USE_EUCLIDEAN_ALIGNMENT = True
EA_EPS = 1e-10  # numerical floor for eigvals


# --------------------------
# === UTILS / LOGGING ===
# --------------------------
def log(msg: str):
    print(msg, flush=True)


# --------------------------
# === IO & PREPROCESS ===
# --------------------------
def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw


def _find_common_center_channels(raw_a: mne.io.BaseRaw, raw_b: mne.io.BaseRaw) -> List[str]:
    names_a = [n.upper() for n in raw_a.ch_names]
    names_b = [n.upper() for n in raw_b.ch_names]

    # try preferred sets (must be present in BOTH)
    for cand in PREFERRED_SET_CANDIDATES:
        cand_up = [c.upper() for c in cand]
        if all(c in names_a for c in cand_up) and all(c in names_b for c in cand_up):
            return cand_up

    # fallback: intersection of EEG channels, keep motor-ish if possible
    inter = sorted(set(names_a).intersection(names_b))
    if not inter:
        raise RuntimeError("No common EEG channels between A and B.")
    # Prefer a small center subset if available
    preferred_pool = ["C3", "CZ", "C4", "FCZ", "FZ", "PZ"]
    ordered = [c for c in preferred_pool if c in inter]
    if len(ordered) >= 3:
        return ordered[:4]
    # else just take first 4 from intersection
    return inter[:4]


def _picks_for(raw: mne.io.BaseRaw, want_names_up: List[str]) -> List[int]:
    # Return indices in raw that match the given UPPERCASE names, in that order
    name_to_idx = {n.upper(): i for i, n in enumerate(raw.ch_names)}
    missing = [n for n in want_names_up if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing required channels in file: {missing}")
    return [name_to_idx[n] for n in want_names_up]


# --------------------------
# === WINDOWING ===
# --------------------------
def _sliding_windows_centered(data: np.ndarray, sfreq: float, win_sec: float, step_sec: float) -> np.ndarray:
    """
    data: (C, T) already picked
    returns: X (N, C, win)
    """
    C, T = data.shape
    win = int(round(win_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if win <= 0 or step <= 0 or win > T:
        raise ValueError("Bad window/step config.")
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float64)
    for i, s in enumerate(starts):
        seg = data[:, s:s + win]
        seg = seg - seg.mean(axis=1, keepdims=True)  # per-window de-mean (robust for covariance)
        X[i] = seg
    return X


def _assign_bins(n_windows: int, n_bins: int) -> np.ndarray:
    """Return length-n_windows array with values in {0..n_bins-1}."""
    if n_windows == 0:
        return np.empty((0,), dtype=int)
    edges = np.linspace(0, n_windows, n_bins + 1, dtype=int)
    bins = np.empty(n_windows, dtype=int)
    for b in range(n_bins):
        bins[edges[b]:edges[b + 1]] = b
    return bins


# --------------------------
# === RG / EA HELPERS ===
# --------------------------
def cov_euclidean_mean(Cs: np.ndarray) -> np.ndarray:
    """Simple Euclidean mean of covariance matrices."""
    return np.mean(Cs, axis=0)


def invsqrtm(C: np.ndarray, eps: float = EA_EPS) -> np.ndarray:
    """Inverse square root of SPD matrix C, with eigenvalue flooring."""
    from numpy.linalg import eigh
    vals, vecs = eigh(C)
    vals = np.clip(vals, eps, None)
    D_inv_sqrt = np.diag(vals ** -0.5)
    return vecs @ D_inv_sqrt @ vecs.T


def apply_ea(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply Euclidean alignment: X' = A @ X; X: (N,C,T), A: (C,C)."""
    return A @ X  # numpy matmul broadcasts over the leading N dimension


def _apply_global_ea_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a global EA transform A on training windows X, and return (X_ea, A).
    X: (N,C,T)
    """
    cov_est = Covariances(estimator="oas")
    covs = cov_est.fit_transform(X)        # (N,C,C)
    R = cov_euclidean_mean(covs)           # (C,C)
    A = invsqrtm(R)                        # (C,C)
    X_ea = apply_ea(X, A)                  # (N,C,T)
    return X_ea, A


# --------------------------
# === PIPELINES ===
# --------------------------
def _riemann_pipeline_binary() -> Pipeline:
    return Pipeline(steps=[
        ("cov", Covariances(estimator="oas")),              # (N,C,T) -> (N,C,C)
        ("ts", TangentSpace(metric="riemann")),             # (N,C,C) -> (N,F)
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, solver="lbfgs", multi_class="auto",
            class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


# --------------------------
# === DATASET (A & B) ===
# --------------------------
def _build_dataset_ab(paths: Dict[str, str]):
    """
    Build (X,y,groups,class_names,sfreq,pick_names,cv_bins_idx) using only A and B.
    y maps {A:0, B:1}
    groups kept per-registration (0/1) for reference (not used by CV now).
    cv_bins_idx in {0..CV_BINS-1} assigns a temporal bin to each window.
    """
    labs = ["A", "B"]
    log("[TRAIN] Building windows from registrations A & B ...")

    # Pre-load both to choose a common channel set
    raw_A = _load_raw(paths["A"])
    raw_B = _load_raw(paths["B"])
    common_names_up = _find_common_center_channels(raw_A, raw_B)
    picks_A = _picks_for(raw_A, common_names_up)
    picks_B = _picks_for(raw_B, common_names_up)
    pick_names = [raw_A.ch_names[i].upper() for i in picks_A]  # store canonical order

    sf = raw_A.info["sfreq"]
    ch_used = [raw_A.ch_names[p] for p in picks_A]
    log(f"  [A] {os.path.basename(paths['A'])} | picks={ch_used} | sf={sf:.1f} Hz")
    XA = _sliding_windows_centered(raw_A.get_data(picks=picks_A), sf, WIN_SEC, STEP_SEC)
    yA = np.zeros((XA.shape[0],), dtype=int)
    gA = np.zeros((XA.shape[0],), dtype=int)
    bA = _assign_bins(XA.shape[0], CV_BINS)

    ch_used = [raw_B.ch_names[p] for p in picks_B]
    log(f"  [B] {os.path.basename(paths['B'])} | picks={ch_used} | sf={sf:.1f} Hz")
    XB = _sliding_windows_centered(raw_B.get_data(picks=picks_B), sf, WIN_SEC, STEP_SEC)
    yB = np.ones((XB.shape[0],), dtype=int)
    gB = np.ones((XB.shape[0],), dtype=int)
    bB = _assign_bins(XB.shape[0], CV_BINS)

    X_all = np.concatenate([XA, XB], axis=0)
    y_all = np.concatenate([yA, yB], axis=0)
    g_all = np.concatenate([gA, gB], axis=0)
    cv_bins_idx = np.concatenate([bA, bB], axis=0)

    counts = {"A": int((y_all == 0).sum()), "B": int((y_all == 1).sum())}
    log(f"[TRAIN] Total windows: {X_all.shape}, per-class: {counts}")
    return X_all, y_all, g_all, labs, sf, pick_names, cv_bins_idx


# --------------------------
# === CROSS-VALIDATION ===
# --------------------------
def _cv_report_binary(X: np.ndarray, y: np.ndarray, bin_idx: np.ndarray, class_names: List[str]):
    """
    K-fold CV by time bins: for fold b, test all windows with bin_idx==b
    (from BOTH A and B). Train on the rest. This guarantees 2-class train/test.
    """
    y_true_all, y_pred_all, y_proba_all = [], [], []
    valid_folds = 0

    for b in range(CV_BINS):
        te = np.where(bin_idx == b)[0]
        tr = np.where(bin_idx != b)[0]

        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        # ensure both classes present
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue

        # EA: fit on training only, apply to both
        if USE_EUCLIDEAN_ALIGNMENT:
            Xtr_ea, A = _apply_global_ea_train(Xtr)
            Xte_ea = apply_ea(Xte, A)
        else:
            Xtr_ea, Xte_ea = Xtr, Xte

        clf = _riemann_pipeline_binary()
        clf.fit(Xtr_ea, ytr)
        yp = clf.predict(Xte_ea)
        yproba = clf.predict_proba(Xte_ea)[:, 1]

        acc = accuracy_score(yte, yp)
        log(f"[CV] Bin {b+1}/{CV_BINS} accuracy: {acc:.3f}")

        y_true_all.append(yte)
        y_pred_all.append(yp)
        y_proba_all.append(yproba)
        valid_folds += 1

    if valid_folds == 0:
        raise RuntimeError("CV produced no folds with both classes. Increase CV_BINS or check data.")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)
    acc = accuracy_score(y_true, y_pred)

    log("\n[CV] Classification report (time-binned CV):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    log(f"[CV] Riemannian(TS+LR, A/B) accuracy: {acc:.3f}")

    # Confusion matrix fig
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, origin="upper", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("A/B Confusion Matrix (normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), v in np.ndenumerate(cm_norm):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    cm_path = os.path.join(OUT_DIR, "cv_confusion_matrix_rg_ab.png")
    fig.tight_layout(); fig.savefig(cm_path, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)
    log(f"[Saved] CV confusion matrix → {cm_path}")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("A/B ROC")
    ax.legend(loc="lower right")
    roc_path = os.path.join(OUT_DIR, "cv_roc_rg_ab.png")
    fig.tight_layout(); fig.savefig(roc_path, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)
    log(f"[Saved] CV ROC → {roc_path}")

    return acc, cm


# --------------------------
# === TRAIN FINAL MODEL ===
# --------------------------
def train_final_model(X: np.ndarray, y: np.ndarray):
    """
    Train final model on all windows with EA (if enabled).
    Returns a dict with classifier, EA matrix, and metadata.
    """
    if USE_EUCLIDEAN_ALIGNMENT:
        X_ea, A = _apply_global_ea_train(X)
    else:
        X_ea, A = X, None

    clf = _riemann_pipeline_binary()
    clf.fit(X_ea, y)

    return {"clf": clf, "A": A}


# --------------------------
# === EVENTS / DETECTION ===
# --------------------------
def _extract_events(raw: mne.io.BaseRaw, prefer_desc: str = EVENT_DESC_PREF):
    """Return (events, chosen_id, desc_used). Prefer annotations; fallback to stim-find."""
    # First: annotations
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) > 0 and events.size > 0:
        if prefer_desc in event_id:
            chosen = event_id[prefer_desc]
            return events, chosen, prefer_desc
        # else: choose most frequent
        unique, counts = np.unique(events[:, 2], return_counts=True)
        chosen = unique[np.argmax(counts)]
        rev = {v: k for k, v in event_id.items()}
        desc = rev.get(chosen, f"code {int(chosen)}")
        return events, chosen, desc

    # Fallback: stim channel
    try:
        events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    except Exception:
        events = np.empty((0, 3), dtype=int)
    if events.size == 0:
        raise RuntimeError("No events/annotations found in target file.")
    unique, counts = np.unique(events[:, 2], return_counts=True)
    chosen = unique[np.argmax(counts)]
    return events, chosen, f"code {int(chosen)}"


def detect_sequence_AB(
    model: dict,
    target_path: str,
    pick_names_up: List[str],
    labs: List[str],
    tmin_sec: float,
    seg_sec: float,
    jitter_samp: int,
    unk_thresh: float,
):
    raw = _load_raw(target_path)
    sf = raw.info["sfreq"]

    # Ensure required channels exist in target, in same order
    picks = _picks_for(raw, pick_names_up)
    data = raw.get_data(picks=picks)  # (C, T)

    seg_samp = int(round(seg_sec * sf))
    base_offset = int(round(tmin_sec * sf))

    events, chosen_code, desc = _extract_events(raw, EVENT_DESC_PREF)
    ev = events[events[:, 2] == chosen_code]
    n_events = len(ev)
    log(f"\n[Target] Using event '{desc}'; N={n_events}")

    clf = model["clf"]
    A = model["A"]

    labels_out: List[str] = []
    probs_out: List[np.ndarray] = []
    shifts: List[int] = []

    C, T = data.shape
    for idx in range(n_events):
        center = ev[idx, 0] + base_offset

        # Collect valid jittered segments
        segs = []
        shift_vals = []
        for s in range(-jitter_samp, jitter_samp + 1):
            start = center + s
            end = start + seg_samp
            if start < 0 or end > T:
                continue
            seg = data[:, start:end]
            seg = seg - seg.mean(axis=1, keepdims=True)  # de-mean
            segs.append(seg)
            shift_vals.append(s)

        if not segs:
            labels_out.append("UNK")
            probs_out.append(np.array([np.nan, np.nan]))
            shifts.append(0)
            continue

        X_evt = np.stack(segs, axis=0)  # (M,C,T)
        if A is not None:
            X_evt = apply_ea(X_evt, A)

        proba = clf.predict_proba(X_evt)  # (M,2) -> cols correspond to [A(0), B(1)]
        m_idx, m_cls = np.unravel_index(np.nanargmax(proba), proba.shape)
        max_prob = proba[m_idx, m_cls]
        pred_lab = labs[m_cls]
        labels_out.append(pred_lab if max_prob >= unk_thresh else "UNK")
        probs_out.append(proba[m_idx])
        shifts.append(shift_vals[m_idx])

    # Summary
    counts = {lab: labels_out.count(lab) for lab in labs}
    counts["UNK"] = labels_out.count("UNK")

    print("\n=== Detected Sequence (RG A/B, annotation-locked + jitter) ===")
    # pretty print 80 tokens / line
    line, wrapped = [], []
    for tok in labels_out:
        line.append(tok)
        if len(line) >= 80:
            wrapped.append(" ".join(line)); line = []
    if line:
        wrapped.append(" ".join(line))
    print("\n".join(wrapped))

    print("\n[Counts]")
    for k in labs + ["UNK"]:
        print(f"  {k}: {counts.get(k, 0)}")

    # Save sequence
    seq_txt = os.path.join(OUT_DIR, "rg_ab_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    csv_path = os.path.join(OUT_DIR, "rg_ab_movement_mix_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("event_idx,label,best_shift,prob_A,prob_B\n")
        for i, (lab, sft, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if pv is None or np.any(np.isnan(pv)):
                f.write(f"{i},{lab},{sft},,\n")
            else:
                f.write(f"{i},{lab},{sft},{pv[0]:.6f},{pv[1]:.6f}\n")
    log(f"[Saved sequence] {seq_txt}")
    log(f"[Saved table]    {csv_path}")

    # Jitter shift histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(shifts, bins=np.arange(-jitter_samp - 0.5, jitter_samp + 1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best shift per event)")
    ax.set_xlabel("Shift (samples)")
    ax.set_ylabel("Count")
    hist_path = os.path.join(OUT_DIR, "target_shift_hist_rg_ab.png")
    fig.tight_layout(); fig.savefig(hist_path, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)
    log(f"[Saved] Target jitter shift histogram → {hist_path}")

    return labels_out, counts


# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Build dataset (A & B only) with common channels
    X, y, groups, class_names, sf, pick_names_up, cv_bins_idx = _build_dataset_ab(TEMPLATE_SOURCES)

    # 2) Cross-validate with time-binned folds (keeps both classes in each fold)
    _ = _cv_report_binary(X, y, cv_bins_idx, class_names)

    # 3) Train final model on all windows (with EA)
    model = train_final_model(X, y)

    # 4) Detect sequence in movement-mix (annotation-locked + jitter)
    _ = detect_sequence_AB(
        model=model,
        target_path=TARGET_FILE,
        pick_names_up=pick_names_up,
        labs=class_names,                 # ["A","B"]
        tmin_sec=TMIN_SEC,
        seg_sec=SEG_SEC,
        jitter_samp=JITTER_SAMP,
        unk_thresh=UNK_THRESH,
    )

    print("\n[Done] All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()

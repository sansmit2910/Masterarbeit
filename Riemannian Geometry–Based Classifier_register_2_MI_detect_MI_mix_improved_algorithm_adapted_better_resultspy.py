# -*- coding: utf-8 -*-
"""
Riemannian Geometry (Covariance) Classifier for EEG (C/D only — MI)
- Train on two MI recordings (C, D) using sliding windows (per-window mean-centering)
- Cross-validate with StratifiedKFold (both classes in every fold)
- Detect C/D sequence in the MI-mix target with annotation-locked + jitter scan
- Print detected sequence; also saves TXT/CSV and a couple of small figures

Requires: mne, numpy, scikit-learn, pyriemann, matplotlib
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold

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
SOURCE_PATHS_CD: Dict[str, str] = {
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

# Output directory
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_rg_CD_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_STATE = 42

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)  # mu+beta
NOTCH_HZ = None         # 50 or 60 if needed, else None

# Windowing (training)
WIN_SEC = 1.0
STEP_SEC = 0.25

# Target detection (annotation-locked)
EVENT_DESC_PREFS = ["Stimulus/S  1", "S  1", "S1"]  # try these in order; fallback to most frequent
TMIN_SEC = 0.0
SEG_SEC = 1.0
JITTER_SAMP = 30
UNK_THRESH = 0.45  # if max class prob < this -> UNK
SMOOTH_KERNEL = 5  # odd integer (median filter over label indices), or None/0 to disable

# Channel preference: try Laplacian first, then standard motor sites; else fall back to all EEG
PREFERRED_CHANSETS = [
    ["C3L", "CzL", "C4L"],
    ["C3", "Cz", "C4", "FCz"],
]

DISPLAY_PLOTS = True  # set False to suppress plt.show()


# --------------------------
# === HELPERS ===
# --------------------------
def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    # Modern picking (no legacy warnings)
    raw.pick(['eeg'])  # keep only EEG, excludes 'bads' by default
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw


def _pick_channels(raw: mne.io.BaseRaw) -> List[int]:
    ch_upper = {name.upper(): i for i, name in enumerate(raw.ch_names)}
    for wanted in PREFERRED_CHANSETS:
        idx = []
        ok = True
        for w in wanted:
            w_up = w.upper()
            if w_up in ch_upper:
                idx.append(ch_upper[w_up])
            else:
                ok = False
                break
        if ok and len(idx) > 0:
            return idx
    # fallback: all EEG indices
    return list(range(len(raw.ch_names)))


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
        seg = data[:, s:s + win]
        seg = seg - seg.mean(axis=1, keepdims=True)  # per-window mean-centering
        X[i] = seg
    return X  # (N, C, win)


def _build_dataset_cd(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str], float, List[str]]:
    """
    Returns:
        X  : (N, C, T) windows from C & D (mean-centered)
        y  : (N,) labels as ints [0,1] for C,D
        labs: ['C','D']
        sf : sampling frequency after resample
        pick_names: channel names used
    """
    labs = ["C", "D"]
    all_X, all_y = [], []
    sf = None
    print("| [TRAIN] Building windows from MI registrations C & D ...")
    pick_names = None
    for li, lab in enumerate(labs):
        path = paths[lab]
        raw = _load_raw(path)
        if sf is None:
            sf = raw.info["sfreq"]
        picks = _pick_channels(raw)
        pnames = [raw.ch_names[p] for p in picks]
        if pick_names is None:
            pick_names = pnames
        print(f"|   [{lab}] {os.path.basename(path)} | picks={pnames} | sf={sf:.1f} Hz")
        Xw = _sliding_windows(raw, picks, WIN_SEC, STEP_SEC)
        yw = np.full((Xw.shape[0],), li, dtype=int)
        all_X.append(Xw)
        all_y.append(yw)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"| [TRAIN] Total windows: {X.shape}, per-class: {{'C': {(y==0).sum()}, 'D': {(y==1).sum()}}}")
    return X, y, labs, sf, pick_names


def _riemann_pipeline_binary() -> Pipeline:
    # Balanced LR; we need probabilities for UNK thresholding
    return Pipeline(steps=[
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def _cv_report_binary(X: np.ndarray, y: np.ndarray, class_names: List[str], n_splits: int = 5) -> float:
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        print("| [CV] Not enough samples per class for CV; skipping.")
        return np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []
    print("| [CV] StratifiedKFold cross-validation:")
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = _riemann_pipeline_binary()
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        acc = accuracy_score(y[te], yp)
        print(f"|   Fold {i}: acc={acc:.3f} (n={len(te)})")
        y_true_all.append(y[te])
        y_pred_all.append(yp)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = accuracy_score(y_true, y_pred)
    print("\n| [CV] Classification report (binary C/D):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    print(f"| [CV] Riemannian(TS+LR, C/D) accuracy: {acc:.3f}")

    # Confusion matrix figure (normalized by rows)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    cm_n = cm / row_sums

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_n, origin="upper", interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized rows)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), v in np.ndenumerate(cm_n):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    cm_path = os.path.join(OUT_DIR, "cv_confusion_matrix_rg_CD.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    if DISPLAY_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"| [Saved] CV confusion matrix → {cm_path}")

    return acc


def _extract_events_annotations(raw: mne.io.BaseRaw, prefer_descs: List[str]) -> Tuple[np.ndarray, int, str]:
    """Return (events, chosen_code, desc_used) using annotations primarily, fallback to stim events."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if events is not None and len(events) > 0 and len(event_id) > 0:
        for key in prefer_descs:
            if key in event_id:
                return events, int(event_id[key]), key
        # else pick the most frequent annotation code
        unique, counts = np.unique(events[:, 2], return_counts=True)
        chosen = int(unique[np.argmax(counts)])
        rev = {v: k for k, v in event_id.items()}
        desc = rev.get(chosen, f"code {chosen}")
        return events, chosen, desc

    # Fallback: stim channel based
    events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    if len(events) == 0:
        raise RuntimeError("No events/annotations found in target file.")
    unique, counts = np.unique(events[:, 2], return_counts=True)
    chosen = int(unique[np.argmax(counts)])
    return events, chosen, f"code {chosen}"


def _median_filter_labels(idx_seq: np.ndarray, kernel: int) -> np.ndarray:
    """Median filter over integer label indices; -1 is kept as is (UNK)."""
    if kernel and kernel >= 3 and (kernel % 2 == 1):
        try:
            from scipy.signal import medfilt
            sm = idx_seq.copy()
            mask = sm >= 0
            if np.any(mask):
                sm[mask] = medfilt(sm[mask], kernel_size=kernel)
            return sm
        except Exception:
            return idx_seq
    return idx_seq


def _detect_sequence_cd(
    clf: Pipeline,
    target_path: str,
    labs: List[str],
    tmin_sec: float,
    seg_sec: float,
    jitter_samp: int,
    unk_thresh: float,
    picks_by_name: Optional[List[str]] = None,
):
    raw = _load_raw(target_path)

    # Try to keep the same picks used for training (by name)
    if picks_by_name:
        picks = [raw.ch_names.index(n) for n in picks_by_name if n in raw.ch_names]
        if len(picks) == 0:
            picks = _pick_channels(raw)
    else:
        picks = _pick_channels(raw)

    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_offset = int(round(tmin_sec * sf))

    events, chosen_code, desc = _extract_events_annotations(raw, EVENT_DESC_PREFS)
    ev = events[events[:, 2] == chosen_code]
    n_events = len(ev)
    print(f"\n| [Target] Using event '{desc}'; N={n_events}")

    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape

    labels_out: List[str] = []
    probs_out: List[np.ndarray] = []
    shifts: List[int] = []

    # Iterate all events with jitter scan
    for idx in range(n_events):
        center = ev[idx, 0] + base_offset
        segs = []
        s_vals = []
        for s in range(-jitter_samp, jitter_samp + 1):
            start = center + s
            end = start + seg_samp
            if start < 0 or end > T:
                continue
            seg = data[:, start:end]
            seg = seg - seg.mean(axis=1, keepdims=True)  # match training centering
            segs.append(seg)
            s_vals.append(s)
        if len(segs) == 0:
            labels_out.append("UNK")
            probs_out.append(np.array([np.nan, np.nan]))
            shifts.append(0)
            continue

        X_evt = np.stack(segs, axis=0)  # (M, C, seg)
        proba = clf.predict_proba(X_evt)  # (M, 2)
        m_idx, m_cls = np.unravel_index(np.nanargmax(proba), proba.shape)
        max_prob = proba[m_idx, m_cls]
        label = labs[int(m_cls)] if max_prob >= unk_thresh else "UNK"

        labels_out.append(label)
        probs_out.append(proba[m_idx])
        shifts.append(s_vals[m_idx])

    # Optional smoothing over label indices (C=0, D=1, UNK=-1)
    idx_map = {lab: i for i, lab in enumerate(labs)}
    idx_seq = np.array([idx_map.get(lbl, -1) for lbl in labels_out], dtype=int)
    if SMOOTH_KERNEL and SMOOTH_KERNEL >= 3 and SMOOTH_KERNEL % 2 == 1:
        idx_seq = _median_filter_labels(idx_seq, SMOOTH_KERNEL)
        labels_out = [labs[i] if i >= 0 else "UNK" for i in idx_seq]

    # Summary + printing
    counts = {lab: labels_out.count(lab) for lab in labs}
    counts["UNK"] = labels_out.count("UNK")

    print("\n=== Detected Sequence (RG C/D, annotation-locked + jitter) ===")
    # pretty print chunked
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

    # Also save (optional)
    seq_txt = os.path.join(OUT_DIR, "rg_CD_MI_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    csv_path = os.path.join(OUT_DIR, "rg_CD_MI_mix_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("event_idx,label,best_shift,prob_C,prob_D\n")
        for i, (lab, sft, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if pv is None or np.any(np.isnan(pv)):
                row = [str(i), lab, str(sft), "", ""]
            else:
                row = [str(i), lab, str(sft), f"{pv[0]:.6f}", f"{pv[1]:.6f}"]
            f.write(",".join(row) + "\n")
    print(f"[Saved sequence] {seq_txt}")
    print(f"[Saved table]    {csv_path}")

    # Plots (optional)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(shifts, bins=np.arange(-jitter_samp - 0.5, jitter_samp + 1.5), edgecolor="black")
    ax.set_title("Target jitter shift (best shift per event)")
    ax.set_xlabel("Shift (samples)"); ax.set_ylabel("Count")
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "target_shift_hist_rg_CD.png")
    fig.savefig(hist_path, dpi=150)
    if DISPLAY_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"[Saved] Target jitter shift histogram → {hist_path}")

    max_probs = [np.nan if (pv is None or np.any(np.isnan(pv))) else float(np.max(pv)) for pv in probs_out]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(max_probs, lw=1, marker="o", ms=2)
    ax.axhline(UNK_THRESH, ls="--")
    ax.set_title("Per-event maximum posterior probability")
    ax.set_xlabel("Event #"); ax.set_ylabel("Max p")
    fig.tight_layout()
    prob_path = os.path.join(OUT_DIR, "target_maxprob_rg_CD.png")
    fig.savefig(prob_path, dpi=150)
    if DISPLAY_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"[Saved] Target per-event max posterior plot → {prob_path}")

    return labels_out


# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Build dataset from C & D (MI)
    X, y, class_names, sf, pick_names = _build_dataset_cd(SOURCE_PATHS_CD)

    # 2) Cross-validate (StratifiedKFold to avoid single-class folds)
    _ = _cv_report_binary(X, y, class_names, n_splits=5)

    # 3) Fit final model on all C/D windows
    clf = _riemann_pipeline_binary()
    clf.fit(X, y)

    # 4) Detect sequence on MI-mix (annotation-locked + jitter), and PRINT it
    _ = _detect_sequence_cd(
        clf=clf,
        target_path=TARGET_FILE,
        labs=class_names,
        tmin_sec=TMIN_SEC,
        seg_sec=SEG_SEC,
        jitter_samp=JITTER_SAMP,
        unk_thresh=UNK_THRESH,
        picks_by_name=pick_names,
    )

    print("\n[Done] All outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()

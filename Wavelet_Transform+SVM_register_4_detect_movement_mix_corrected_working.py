# -*- coding: utf-8 -*-
"""
Welch Bandpower + SVM with explicit filepaths for A/B/C/D training and a movement-mix target.

What this script does
---------------------
1) Loads four .fif files (A,B,C,D) as training data using your explicit paths below.
2) Filters EEG (1–40 Hz), picks ['C3L','CzL','C4L'] else ['C3','Cz','C4'] else first 3 EEGs.
3) Segments 1.0 s windows (250 samples) locked to 'Stimulus/S  1' events.
4) Extracts per-channel relative bandpowers (mu 8–12, beta1 13–20, beta2 20–30) via Welch.
5) Trains an SVM (RBF, class_weight='balanced'); runs stratified CV and saves a diagnostics figure:
   - Confusion matrix heatmap (row-normalized)
   - Classification report table (precision/recall/F1/support)
   - Per-class metric bars
   - Per-fold accuracies
6) Fits on full training set, then predicts the movement-mix target with ±30-sample jitter voting.
   Low-confidence votes become "UNK".
7) Compares predicted sequence to your fixed ground sequence (strict + ignore-UNK accuracies).
8) Saves outputs (plots + sequence .txt/.csv) next to the target file in `eeg_welch_svm_output/`.

Dependencies
------------
Python 3.9+; numpy, scipy, scikit-learn, mne, matplotlib, pandas.

Edit only the TEMPLATE_SOURCES and TARGET_FILE below if paths change.
"""

import os
import re
import json
import math
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

import matplotlib.pyplot as plt

# -----------------------------
# CONFIG — EXPLICIT FILE PATHS
# -----------------------------

TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Output directory
OUT_DIR = os.path.join(os.path.dirname(TARGET_FILE), "eeg_welch_svm_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Event description to use (annotations)
EVENT_DESC = "Stimulus/S  1"

# -----------------------------
# CONFIG — SIGNAL / FEATURES
# -----------------------------
SEED = 42
np.random.seed(SEED)

PREFERRED_CHS = ["C3L", "CzL", "C4L"]
FALLBACK_CHS = ["C3", "Cz", "C4"]

HP, LP = 1.0, 40.0  # Hz bandpass

SEG_LEN_SAMPS = 250  # with sf=250 Hz -> 1.0 s segments
WELCH_FMIN = 4.0
WELCH_FMAX = 30.0
NPERSEG = 128
NOVERLAP = 64

BANDS = {
    "theta": (4.0, 7.0),
    "mu": (8.0, 12.0),
    "beta1": (13.0, 20.0),
    "beta2": (20.0, 30.0),
}
USE_FEATURES = ["mu", "beta1", "beta2"]  # per channel

SVM_C = 1.0
SVM_GAMMA = "scale"
CLASS_WEIGHT = "balanced"
N_SPLITS = 5

JITTER = 30  # samples
UNK_MIN_TOP_FRAC = 0.40  # if the winning class gets <40% of votes -> UNK
UNK_MIN_MARGIN = 0.10    # if the winning margin <10% -> UNK

ALL_LABELS = ["A", "B", "C", "D"]
EPS = 1e-12

# Your fixed ground sequence
GROUND_SEQ_STR = """
B A A A B B B A A A A A B B B B B B B B B A A A A A A B B B A A B B B B A A A A A A B B B B B B B B B A A A A A A A A A A A B B B B B B B B B B B B B B A A A A A A B B B A A A A A A A A A A A B B B B B B A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B A A A A A A A A A A B B B B B B B B B B A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B
""".strip()

# Fallback for numpy<2.0 where np.trapezoid may not exist
def _trap_area(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

# -----------------------------
# UTILITIES
# -----------------------------

def _check_paths(paths: Dict[str, str], target_path: str):
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing training files for labels: " + ", ".join(missing) +
            "\nPlease check TEMPLATE_SOURCES paths."
        )
    if not os.path.isfile(target_path):
        raise FileNotFoundError(
            f"Missing target file: {target_path}\nPlease check TARGET_FILE."
        )

def read_raw_with_filter(path_fif: str, hp: float, lp: float) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_fif(path_fif, preload=True, verbose="ERROR")
    raw.filter(hp, lp, fir_design="firwin", verbose="ERROR")
    return raw

def pick_channels_any(raw: mne.io.BaseRaw, prefer: List[str], fallback: List[str]) -> List[str]:
    picks = [ch for ch in prefer if ch in raw.ch_names]
    if len(picks) == len(prefer):
        return picks
    picks = [ch for ch in fallback if ch in raw.ch_names]
    if len(picks) == len(fallback):
        return picks
    eegs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    return [raw.ch_names[i] for i in eegs[:3]]

def get_event_samples(raw: mne.io.BaseRaw, desc: str) -> np.ndarray:
    if raw.annotations is not None and len(raw.annotations) > 0:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if desc in event_id:
            code = event_id[desc]
            events = events[events[:, 2] == code]
            return events[:, 0]
        # fallback: use all annotated events
        return events[:, 0]
    events = mne.find_events(raw, consecutive=True, verbose=False)
    if events.size == 0:
        raise RuntimeError("No events found in raw.")
    return events[:, 0]

def extract_epochs_array(raw: mne.io.BaseRaw, picks: List[str], event_samps: np.ndarray,
                         seg_len: int) -> np.ndarray:
    X = []
    data = raw.get_data(picks=picks)
    T = data.shape[1]
    for s in event_samps:
        s0, s1 = int(s), int(s) + seg_len
        if s0 < 0 or s1 > T:
            continue
        X.append(data[:, s0:s1])
    if not X:
        return np.zeros((0, len(picks), seg_len), dtype=np.float64)
    return np.stack(X, axis=0)

def welch_bandpowers(epochs: np.ndarray, sf: float) -> np.ndarray:
    N, C, T = epochs.shape
    feats = []
    for i in range(N):
        row = []
        for c in range(C):
            f, Pxx = welch(epochs[i, c, :], fs=sf, nperseg=NPERSEG, noverlap=NOVERLAP)
            mask_total = (f >= WELCH_FMIN) & (f <= WELCH_FMAX)
            ft = f[mask_total]
            px = Pxx[mask_total]
            total = _trap_area(px, ft) + EPS
            # band areas
            masks = {k: (f >= lo) & (f <= hi) for k, (lo, hi) in BANDS.items()}
            bp = {}
            for k in ["mu", "beta1", "beta2"]:
                m = masks[k]
                bp[k] = _trap_area(Pxx[m], f[m]) if np.any(m) else 0.0
            rel = {k: (bp[k] / total) for k in bp}
            row.extend([rel[k] for k in USE_FEATURES])
        feats.append(row)
    return np.asarray(feats, dtype=np.float64)

def make_clf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=SVM_C, gamma=SVM_GAMMA, kernel="rbf",
                    class_weight=CLASS_WEIGHT, probability=False, random_state=SEED))
    ])

def cv_diagnostics_figure(X, y, labels, save_path=None, title="Cross-validation diagnostics"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = make_clf()
    fold_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + EPS)
    report_dict = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
    class_rows = []
    for lbl in labels:
        d = report_dict.get(lbl, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        class_rows.append([lbl, d["precision"], d["recall"], d["f1-score"], int(d["support"])])
    overall_acc = accuracy_score(y, y_pred)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.2, 1.0], hspace=0.25, wspace=0.25)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm_norm, vmin=0, vmax=1)
    ax1.set_title("Confusion matrix (row-normalized)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels)
    ax1.set_yticks(range(len(labels))); ax1.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax1.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Recall")

    # Classification report table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]
    cell_text = [[c[0], f"{c[1]:.3f}", f"{c[2]:.3f}", f"{c[3]:.3f}", f"{c[4]}"] for c in class_rows]
    table = ax2.table(cellText=cell_text, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    ax2.set_title(f"Classification report (overall acc = {overall_acc:.3f})")

    # Per-class bars
    ax3 = fig.add_subplot(gs[1, 0])
    idx = np.arange(len(labels))
    width = 0.25
    prec = [c[1] for c in class_rows]
    rec  = [c[2] for c in class_rows]
    f1   = [c[3] for c in class_rows]
    ax3.bar(idx - width, prec, width, label="Precision")
    ax3.bar(idx,         rec,  width, label="Recall")
    ax3.bar(idx + width, f1,   width, label="F1")
    ax3.set_xticks(idx); ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Per-class metrics")
    ax3.legend()

    # Per-fold accuracies
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(1, len(fold_scores)+1), fold_scores)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Fold"); ax4.set_ylabel("Accuracy")
    ax4.set_title(f"Per-fold accuracy (mean={fold_scores.mean():.3f} ± {fold_scores.std():.3f})")

    fig.suptitle(title, fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return overall_acc, fold_scores, cm_norm, report_dict

def build_train_matrix(reg_paths_labels: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    all_X, all_y = [], []
    used_chs = None
    sf_used = None
    for pth, lbl in reg_paths_labels:
        raw = read_raw_with_filter(pth, HP, LP)
        if sf_used is None:
            sf_used = float(raw.info["sfreq"])
        chs = pick_channels_any(raw, PREFERRED_CHS, FALLBACK_CHS)
        if used_chs is None:
            used_chs = chs
        ev_samps = get_event_samples(raw, EVENT_DESC)
        epochs = extract_epochs_array(raw, used_chs, ev_samps, SEG_LEN_SAMPS)
        feats = welch_bandpowers(epochs, sf=sf_used)
        all_X.append(feats)
        all_y.extend([lbl]*feats.shape[0])

    X = np.vstack(all_X) if all_X else np.zeros((0, len(used_chs)*len(USE_FEATURES)))
    y = np.array(all_y)
    return X, y, sf_used, used_chs

def jittered_predict_on_target(clf: Pipeline, raw: mne.io.BaseRaw, picks: List[str],
                               event_samps: np.ndarray, seg_len: int, sf: float,
                               jitter: int) -> Tuple[List[str], dict]:
    counts = Counter()
    preds = []
    shifts_all = []
    data = raw.get_data(picks=picks)
    T = data.shape[1]
    for s in event_samps:
        votes = []
        s = int(s)
        for k in range(-jitter, jitter+1):
            s0, s1 = s + k, s + k + seg_len
            if s0 < 0 or s1 > T:
                continue
            seg = data[:, s0:s1][np.newaxis, ...]
            feat = welch_bandpowers(seg, sf)
            label = clf.predict(feat)[0]
            votes.append(label)
            shifts_all.append(k)
        if not votes:
            preds.append("UNK")
            continue
        c = Counter(votes)
        top_lbl, top_cnt = c.most_common(1)[0]
        sec_cnt = c.most_common(2)[1][1] if len(c) > 1 else 0
        total = sum(c.values())
        top_frac = top_cnt / total
        margin = (top_cnt - sec_cnt) / total if total > 0 else 0.0
        if (top_frac < UNK_MIN_TOP_FRAC) or (margin < UNK_MIN_MARGIN):
            preds.append("UNK")
        else:
            preds.append(top_lbl)
        counts.update([preds[-1]])

    shift_stats = {
        "mean": float(np.mean(shifts_all)) if shifts_all else 0.0,
        "std": float(np.std(shifts_all)) if shifts_all else 0.0,
        "min": int(np.min(shifts_all)) if shifts_all else 0,
        "max": int(np.max(shifts_all)) if shifts_all else 0,
    }
    return preds, {"counts": dict(counts), "shift_stats": shift_stats}

def parse_ground_sequence(seq_str: str) -> List[str]:
    toks = re.findall(r"[A-D]", seq_str.upper())
    return toks

def sequence_accuracy(pred: List[str], truth: List[str]) -> dict:
    n = min(len(pred), len(truth))
    pred = pred[:n]
    truth = truth[:n]
    strict_correct = sum(1 for p, t in zip(pred, truth) if p == t)
    strict_acc = strict_correct / n if n else 0.0
    known_idx = [i for i,(p,t) in enumerate(zip(pred, truth)) if p != "UNK"]
    known_n = len(known_idx)
    known_correct = sum(1 for i in known_idx if pred[i] == truth[i])
    known_acc = known_correct / known_n if known_n else 0.0
    return {
        "N": n,
        "strict_acc": strict_acc,
        "known_only_acc": known_acc,
        "known_support": known_n,
        "strict_correct": strict_correct,
        "known_correct": known_correct
    }

# -----------------------------
# MAIN
# -----------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mne.set_log_level("ERROR")

    # Verify paths exist
    _check_paths(TEMPLATE_SOURCES, TARGET_FILE)
    for lbl in ["A","B","C","D"]:
        print(f"[Resolved] {lbl} -> {TEMPLATE_SOURCES[lbl]}")
    print(f"[Target]   -> {TARGET_FILE}")

    print("\n[Training] Building features from registrations (A, B, C, D) ...")
    reg_paths_labels = [
        (TEMPLATE_SOURCES["A"], "A"),
        (TEMPLATE_SOURCES["B"], "B"),
        (TEMPLATE_SOURCES["C"], "C"),
        (TEMPLATE_SOURCES["D"], "D"),
    ]
    X, y, sf, used_chs = build_train_matrix(reg_paths_labels)

    cls_counts = {lbl: int((y == lbl).sum()) for lbl in ALL_LABELS}
    print(f"[Train set] X={X.shape} (N,F), picks={used_chs}, sf={sf:.1f} Hz  | per-class: {cls_counts}")

    # CV diagnostics figure
    cv_fig_path = os.path.join(OUT_DIR, "cv_diagnostics.png")
    title = "Welch bandpower + SVM (4-class) — Cross-validation diagnostics"
    acc, fold_scores, cm_norm, report_dict = cv_diagnostics_figure(
        X, y, labels=ALL_LABELS, save_path=cv_fig_path, title=title
    )
    print(f"\n[CV] Accuracy: {acc:.3f} (± {fold_scores.std():.3f})")
    print("[CV] Confusion matrix (row-normalized):")
    with np.printoptions(precision=3, suppress=True):
        print(cm_norm)
    print("[CV] Classification report:")
    print(classification_report(
        y,
        cross_val_predict(make_clf(), X, y, cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)),
        labels=ALL_LABELS, zero_division=0
    ))
    print(f"[CV] Saved diagnostics figure -> {cv_fig_path}")

    # Fit on full train
    clf = make_clf()
    clf.fit(X, y)

    # Predict on target with jittered voting
    print("\n[Target] Loading movement-mix target ...")
    raw_t = read_raw_with_filter(TARGET_FILE, HP, LP)
    picks_t = pick_channels_any(raw_t, PREFERRED_CHS, FALLBACK_CHS)
    ev_samps_t = get_event_samples(raw_t, EVENT_DESC)
    print(f"[Target] Using event '{EVENT_DESC}'; N={len(ev_samps_t)}")
    print(f"[Predict] Jitter scan: ±{JITTER} samples (~±{JITTER*1000.0/sf:.0f} ms)  | seg_len={SEG_LEN_SAMPS} | file={os.path.basename(TARGET_FILE)}")

    preds, meta = jittered_predict_on_target(clf, raw_t, picks_t, ev_samps_t, SEG_LEN_SAMPS, sf, JITTER)
    print(f"[Shift stats] mean={meta['shift_stats']['mean']:.1f} samp, std={meta['shift_stats']['std']:.1f}, "
          f"min={meta['shift_stats']['min']}, max={meta['shift_stats']['max']}")

    print("\n=== Detected Sequence (Welch+SVM, 4-class, trigger-locked+jitter) ===")
    line = []
    for i, p in enumerate(preds, 1):
        line.append(p)
        if (i % 80) == 0:
            print(" ".join(line)); line = []
    if line: print(" ".join(line))

    print("\n[Counts]")
    for k, v in Counter(preds).most_common():
        print(f"  {k}: {v}")

    # Save sequence
    seq_txt = os.path.join(OUT_DIR, "welch_svm_movement_mix_sequence.txt")
    seq_csv = os.path.join(OUT_DIR, "welch_svm_movement_mix_sequence.csv")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(preds) + "\n")
    pd.DataFrame({"idx": np.arange(1, len(preds)+1), "pred": preds}).to_csv(seq_csv, index=False)
    print(f"\n[Saved sequence] {seq_txt}\n[Saved table]    {seq_csv}")

    # Compare to ground truth sequence
    truth = parse_ground_sequence(GROUND_SEQ_STR)
    accs = sequence_accuracy(preds, truth)
    print("\n[Ground sequence evaluation]")
    print(f"  N aligned: {accs['N']}")
    print(f"  Strict accuracy (UNK wrong): {accs['strict_acc']:.3f}  | correct={accs['strict_correct']}/{accs['N']}")
    print(f"  Known-only accuracy (ignore UNK): {accs['known_only_acc']:.3f}  | support={accs['known_support']}  | correct={accs['known_correct']}/{accs['known_support'] if accs['known_support'] else 1}")

    # Simple histogram for target predictions
    hist_path = os.path.join(OUT_DIR, "target_label_hist.png")
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    labs = ["A","B","C","D","UNK"]
    vals = [Counter(preds).get(k,0) for k in labs]
    ax.bar(labs, vals)
    ax.set_title("Predicted labels on movement-mix")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    print(f"[Saved] target label histogram -> {hist_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()

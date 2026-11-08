# -*- coding: utf-8 -*-
"""
MI (C/D) → Welch+SVM → predict on MI-mix target (trigger-locked + jitter scan)
Saves: CV confusion matrix figure, classification report figure, jitter histogram,
       predicted sequence (txt + csv), and counts bar chart.

Requirements: mne, numpy, scipy, scikit-learn, matplotlib
"""

import os
import sys
import json
import math
import time
import textwrap
from collections import Counter, defaultdict

import numpy as np
import mne
from scipy.signal import welch
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# -----------------------------
# ======= PATHS: EDIT ME =======
# -----------------------------
PATH_C = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif"
PATH_D = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif"

# Set your MI-mix target file here (e.g., "...imagine-mix_raw.fif")
TARGET_MI_MIX = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

# Output directory
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_welch_svm_CD_output_MI"

# -----------------------------
# ======= PARAMETERS ==========
# -----------------------------
# Channels to use (first try Laplacian, then fallback)
DESIRED_CHS_MAIN = ["C3L", "CzL", "C4L"]
DESIRED_CHS_FALLBACK = ["C3", "Cz", "C4"]

# Epoching relative to trigger: start sample = event_sample + T0_OFFSET
# Segment length in samples is computed from SEG_LEN_SEC * sfreq
T0_OFFSET_SEC = 0.0       # start exactly at trigger
SEG_LEN_SEC   = 1.0       # 1.0 s windows (like before)

# Welch PSD parameters
WELCH_NPERSEG = 128
WELCH_NOVERLP = 64
WELCH_NFFT    = 256

# Frequency bands (Hz)
BANDS = {
    "theta": (4., 7.),
    "mu":    (8., 12.),
    "beta1": (13., 20.),
    "beta2": (21., 30.),
}
FMAX = 40.0  # limit PSD to this max frequency (Hz)

# Classifier / CV
N_SPLITS = 5
SVM_C = 4.0
SVM_GAMMA = "scale"
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42

# Jitter scan around trigger (± samples)
JITTER_MS = 120          # e.g., ±120 ms (with 250 Hz ≈ ±30 samples)
JITTER_STEP_SAMPLES = 1  # step of 1 sample (same as before)
UNK_PROBA_MIN = 0.55     # if max class prob < this → UNK

# Plots
DPI = 140

# -----------------------------
# ======= UTILITIES ===========
# -----------------------------

def _check_path(p, name):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing {name} file: '{p}'")
    return p

def pick_channels_or_fallback(raw, desired=DESIRED_CHS_MAIN, fallback=DESIRED_CHS_FALLBACK):
    chs = [ch for ch in desired if ch in raw.ch_names]
    if not chs:
        chs = [ch for ch in fallback if ch in raw.ch_names]
    if not chs:
        eeg_like = [ch for ch in raw.ch_names if ch.upper().startswith("C")]
        if eeg_like:
            chs = eeg_like[:3]
    if not chs:
        raise RuntimeError(f"No desired EEG channels found. Raw has: {raw.ch_names[:10]} ...")
    return chs

def get_event_samples(raw):
    """Prefer 'Stimulus/S  1' from annotations; else fall back to stim channel events."""
    # Try annotations first
    ev_anno, ev_id = mne.events_from_annotations(raw, verbose=False)
    if len(ev_anno):
        # prefer an annotation containing 'S  1' or code == 1
        target_keys = sorted(ev_id.keys(), key=lambda k: (("S  1" not in k), k))
        for k in target_keys:
            if "S  1" in k or k.strip().endswith("1"):
                code = ev_id[k]
                idx = (ev_anno[:, 2] == code)
                if idx.any():
                    return ev_anno[idx, 0], k
        # else pick the most frequent
        # count by code
        counts = Counter(ev_anno[:, 2])
        common_code = counts.most_common(1)[0][0]
        idx = (ev_anno[:, 2] == common_code)
        # find key for reporting
        key = [k for k, v in ev_id.items() if v == common_code]
        key = key[0] if key else f"code={common_code}"
        return ev_anno[idx, 0], key

    # Fallback: stim channel events
    ev_stim = mne.find_events(raw, shortest_event=1, initial_event=True, verbose=False)
    if len(ev_stim):
        # choose most frequent code
        counts = Counter(ev_stim[:, 2])
        common_code = counts.most_common(1)[0][0]
        idx = (ev_stim[:, 2] == common_code)
        return ev_stim[idx, 0], f"stim_code={common_code}"

    raise RuntimeError("No events found from annotations or stim channels.")

def segment_from_events(raw, chs, event_samp, seg_len_samp, t0_offset_samp=0):
    """Return segments (n_events, n_ch, seg_len) using fixed start at (event + offset)."""
    data = raw.get_data(picks=chs)
    n_ch, n_samp = data.shape
    segments = []
    kept_idx = []

    for i, s in enumerate(event_samp):
        start = s + t0_offset_samp
        end = start + seg_len_samp
        if start >= 0 and end <= n_samp:
            segments.append(data[:, start:end])
            kept_idx.append(i)
    if not segments:
        raise RuntimeError("No segments within bounds. Check SEG_LEN_SEC and offsets.")
    X = np.stack(segments, axis=0)  # (N, C, T)
    return X, np.array(kept_idx, dtype=int)

def welch_features(X, sf, bands=BANDS, fmax=FMAX):
    """
    X: (N, C, T)
    Returns features (N, C * n_bands + C extra ratios).
    Features per channel:
      - relative power in each band (theta, mu, beta1, beta2)
      - mu / (beta1 + beta2) ratio
    """
    N, C, T = X.shape
    # shape for welch: (C, T)
    feats = []
    for i in range(N):
        ch_feats = []
        for c in range(C):
            f, Pxx = welch(X[i, c, :], fs=sf, nperseg=WELCH_NPERSEG,
                           noverlap=WELCH_NOVERLP, nfft=WELCH_NFFT)
            mask_f = (f <= fmax)
            f = f[mask_f]
            Pxx = Pxx[mask_f]

            # Total power for relative bands
            total = np.trapezoid(Pxx, f) + 1e-12

            # band powers
            rels = []
            band_vals = {}
            for bn, (f1, f2) in bands.items():
                mask_b = (f >= f1) & (f <= f2)
                bp = np.trapezoid(Pxx[mask_b], f[mask_b]) if np.any(mask_b) else 0.0
                band_vals[bn] = bp
                rels.append(bp / total)

            # ratio mu/(beta1+beta2)
            denom = (band_vals["beta1"] + band_vals["beta2"]) + 1e-12
            ratio_mu_b = band_vals["mu"] / denom

            ch_feats.extend(rels + [ratio_mu_b])
        feats.append(ch_feats)
    feats = np.asarray(feats, dtype=float)
    # log(1+x) for stability
    feats = np.log1p(feats)
    return feats  # (N, C*(len(bands)+1))

def make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=SVM_C, gamma=SVM_GAMMA,
                    class_weight=CLASS_WEIGHT,
                    probability=True, random_state=RANDOM_STATE))
    ])

def plot_confmat_and_report(y_true, y_pred, labels, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, digits=3)

    fig = plt.figure(figsize=(10, 4), dpi=DPI)
    # Confusion matrix
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(cm, interpolation='nearest')
    ax1.set_title(f"CV Confusion Matrix (acc={acc:.3f})")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.5 else "black")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # Classification report text
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_title("Classification Report")
    ax2.text(0.01, 0.99, report, va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return acc, cm, report

def save_sequence_txt_csv(seq, out_txt, out_csv):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(seq) + "\n")
    # CSV with index+label
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "label"])
        for i, lab in enumerate(seq):
            w.writerow([i, lab])

def bar_counts_plot(seq, out_png, title="Predicted label counts"):
    cnt = Counter(seq)
    labs = list(cnt.keys())
    vals = [cnt[k] for k in labs]

    fig = plt.figure(figsize=(6, 4), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(labs, vals)
    ax.set_title(title)
    ax.set_ylabel("Count")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def jitter_scan_predict(raw, chs, clf, sf, seg_len_samp, jitter_ms=120, step=1, unk_pmin=0.55):
    """
    For each event, scan shifts in [-J, +J] samples, collect probabilities, and:
      - choose label with highest mean probability across shifts
      - if its mean prob < unk_pmin -> 'UNK'
    Returns sequence (list[str]) and chosen shift statistics.
    """
    ev_samp, ev_key = get_event_samples(raw)
    J = int(round(jitter_ms * 1e-3 * sf))
    t0_off = int(round(T0_OFFSET_SEC * sf))
    chosen_shifts = []  # shift with max avg prob per event (for histogram)
    seq = []

    # For per-shift probabilities, we need class order from clf
    classes = clf.named_steps["svc"].classes_

    # Preload data for speed
    data = raw.get_data(picks=chs)
    n_ch, n_samp = data.shape

    for i, s0 in enumerate(ev_samp):
        # collect mean probability across all valid shifts
        sum_proba = np.zeros(len(classes), dtype=float)
        n_valid = 0
        best_shift = 0
        best_pmax = -1.0

        for sh in range(-J, J + 1, step):
            s = s0 + t0_off + sh
            e = s + seg_len_samp
            if s < 0 or e > n_samp:
                continue
            seg = data[:, s:e][None, :, :]  # (1, C, T)
            feats = welch_features(seg, sf)  # (1, F)
            proba = clf.predict_proba(feats)[0]  # (n_classes,)
            sum_proba += proba
            n_valid += 1
            # track the best single-shift confidence peak
            pmax = np.max(proba)
            if pmax > best_pmax:
                best_pmax = pmax
                best_shift = sh

        if n_valid == 0:
            seq.append("UNK")
            chosen_shifts.append(0)
            continue

        mean_proba = sum_proba / n_valid
        k = int(np.argmax(mean_proba))
        label = classes[k]
        if mean_proba[k] < unk_pmin:
            label = "UNK"
        seq.append(label)
        chosen_shifts.append(best_shift)

    return seq, np.array(chosen_shifts), ev_key

def shift_histogram_plot(shift_samps, sf, out_png, title="Target jitter shift histogram"):
    fig = plt.figure(figsize=(6, 4), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(shift_samps, bins=31)
    ax.set_title(title)
    ax.set_xlabel("Chosen shift (samples)")
    ax.set_ylabel("Count")
    mean = np.mean(shift_samps) if len(shift_samps) else float('nan')
    std = np.std(shift_samps) if len(shift_samps) else float('nan')
    ax.text(0.02, 0.95, f"mean={mean:.1f} samp, std={std:.1f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# -----------------------------
# ========== MAIN =============
# -----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load C & D (MI registrations), build train set
    path_c = _check_path(PATH_C, "MI registration C")
    path_d = _check_path(PATH_D, "MI registration D")
    print("[Training] Loading MI registrations C & D ...")
    raw_c = mne.io.read_raw_fif(path_c, preload=True, verbose=False)
    raw_d = mne.io.read_raw_fif(path_d, preload=True, verbose=False)

    # Infer sampling frequency; check consistency
    sf_c = raw_c.info["sfreq"]
    sf_d = raw_d.info["sfreq"]
    if abs(sf_c - sf_d) > 1e-6:
        raise RuntimeError(f"Sampling rates differ: C={sf_c}, D={sf_d}")
    sf = float(sf_c)
    seg_len_samp = int(round(SEG_LEN_SEC * sf))
    t0_offset_samp = int(round(T0_OFFSET_SEC * sf))

    # Pick channels (same for both)
    chs_c = pick_channels_or_fallback(raw_c)
    chs_d = pick_channels_or_fallback(raw_d)
    # unify channel order
    common_chs = [ch for ch in chs_c if ch in chs_d]
    if not common_chs:
        raise RuntimeError("No common channels between C and D.")
    chs = common_chs
    print(f"[Train set] channels={chs}, sf={sf:.1f} Hz")

    # Events & segments
    ev_c, key_c = get_event_samples(raw_c)
    ev_d, key_d = get_event_samples(raw_d)
    print(f"[Events] C → '{key_c}', N={len(ev_c)} | D → '{key_d}', N={len(ev_d)}")

    Xc, keep_c = segment_from_events(raw_c, chs, ev_c, seg_len_samp, t0_offset_samp)
    Xd, keep_d = segment_from_events(raw_d, chs, ev_d, seg_len_samp, t0_offset_samp)

    yc = np.array(["C"] * Xc.shape[0])
    yd = np.array(["D"] * Xd.shape[0])

    X_train = np.concatenate([Xc, Xd], axis=0)  # (N, C, T)
    y_train = np.concatenate([yc, yd], axis=0)

    # Features
    Feat = welch_features(X_train, sf)  # (N, F)
    print(f"[Train set] X={Feat.shape} (N,F)")

    # 2) CV & train classifier
    pipeline = make_clf()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(pipeline, Feat, y_train, cv=cv, n_jobs=-1, method="predict")
    # Fit proba-aware for later
    pipeline.fit(Feat, y_train)

    # Save CV visuals
    cm_png = os.path.join(OUT_DIR, "cv_confmat_CD.png")
    acc, cm, rep = plot_confmat_and_report(y_train, y_pred_cv, labels=["C", "D"], out_png=cm_png)
    print("\n[CV] Classification report (C/D):")
    print(rep)
    print(f"[CV] Welch+SVM (C/D) accuracy: {acc:.3f}")
    print(f"[Saved] CV confusion matrix → {cm_png}")

    # 3) Predict on MI-mix target with jitter scan
    target_path = _check_path(TARGET_MI_MIX, "MI-mix target")
    print("\n[Target] Loading MI-mix target ...")
    raw_t = mne.io.read_raw_fif(target_path, preload=True, verbose=False)
    # Ensure channel order matches
    chs_t = [ch for ch in chs if ch in raw_t.ch_names]
    if len(chs_t) != len(chs):
        raise RuntimeError(f"Target missing some channels from train set. Needed {chs}, found {chs_t}.")
    seq, shift_samps, ev_key = jitter_scan_predict(
        raw_t, chs_t, pipeline, sf, seg_len_samp,
        jitter_ms=JITTER_MS, step=JITTER_STEP_SAMPLES, unk_pmin=UNK_PROBA_MIN
    )
    print(f"[Predict] Jitter scan: ±{JITTER_MS} ms | seg_len={seg_len_samp} samples | events key='{ev_key}'")
    print(f"[Shift stats] mean={np.mean(shift_samps):.1f} samp, std={np.std(shift_samps):.1f}, "
          f"min={np.min(shift_samps)}, max={np.max(shift_samps)}")

    # 4) Print & save sequence
    # compact pretty print (80 cols)
    print("\n=== Detected Sequence (Welch+SVM C/D, trigger-locked+jitter) ===")
    line = []
    line_len = 0
    for lab in seq:
        tok = lab
        if line_len + len(tok) + 1 > 80:
            print(" ".join(line))
            line = [tok]
            line_len = len(tok)
        else:
            line.append(tok)
            line_len += len(tok) + 1
    if line:
        print(" ".join(line))

    counts = Counter(seq)
    print("\n[Counts]")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    # Save outputs
    out_seq_txt = os.path.join(OUT_DIR, "welch_svm_CD_mi_mix_sequence.txt")
    out_seq_csv = os.path.join(OUT_DIR, "welch_svm_CD_mi_mix_sequence.csv")
    save_sequence_txt_csv(seq, out_seq_txt, out_seq_csv)
    print(f"[Saved sequence] {out_seq_txt}")
    print(f"[Saved table]    {out_seq_csv}")

    # Plots: shift histogram & counts
    shift_png = os.path.join(OUT_DIR, "target_shift_hist_CD.png")
    shift_histogram_plot(shift_samps, sf, shift_png,
                         title="Target jitter shift histogram (C/D)")
    print(f"[Saved] Target jitter shift histogram → {shift_png}")

    counts_png = os.path.join(OUT_DIR, "predicted_counts_CD.png")
    bar_counts_plot(seq, counts_png, title="Predicted label counts (C/D)")
    print(f"[Saved] Predicted label counts → {counts_png}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)

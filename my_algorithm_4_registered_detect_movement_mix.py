# -*- coding: utf-8 -*-
"""
Mask-based motif discovery & sequence detection for BCI (A/B/C/D → MI-mix)

Algorithm (as requested):
1) Automatic motif discovery per class:
   - For each trigger-aligned event, compute channel-wise z-score vs a pre-trigger baseline.
   - Mark post-trigger samples "significant" if |z| >= Z_THR (optionally time-smoothed).
   - Overlay all event significance maps and keep positions that are common to at least
     CONSENSUS_FRAC of events → this is the class "mask".

2) Registration & detection:
   - For any new event, compute its significance map the same way.
   - Score against each class mask using:
        * coverage: fraction of mask-True positions covered by event-True
        * jaccard:  |event∧mask| / |event∨mask|
     Combined score = ALPHA*coverage + (1-ALPHA)*jaccard.
   - Optionally scan ±JITTER_SAMP samples to maximize the score for that event.
   - Assign the label with the best score, unless both coverage < COV_MIN or score < SCORE_MIN → UNK.

Outputs:
- Learned masks (PNG heatmaps)
- Detected sequence (stdout, .txt, .csv)
- Shift histogram & per-event max score plots
- Optional accuracy vs ground-truth string (if provided)

Requires: mne, numpy, scipy, scikit-learn, matplotlib
"""

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from scipy.ndimage import binary_closing, binary_opening
from sklearn.metrics import accuracy_score

# ----------------------------- USER PATHS -----------------------------
# Registration files (per-class)
REG_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
}
# Target MI mix for detection
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Optional: paste ground-truth (space-separated labels like "B A A B ...") to compute accuracy
GROUND_TRUTH_STR = ""  # leave empty "" to skip accuracy

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_mask_motif_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- CONFIG -----------------------------
RANDOM_SEED = 1337

# EEG preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)  # mu+beta
NOTCH_HZ = None         # 50 or 60 if needed, else None

# Channels (by name) in preference order. Will fall back to common EEG intersection.
PREFERRED_CHANS = ["C3", "Cz", "C4", "FCz"]

# Events / epoching
ANNOT_KEY = "Stimulus/S  1"  # prefer this annotation code; otherwise take most frequent
BASELINE_SEC = 0.2           # pre-trigger baseline length (s)
TMIN_SEC = 0.0               # segment start relative to trigger (s)
SEG_SEC = 1.0                # post-trigger segment duration (s)
JITTER_SAMP = 30             # ± samples to scan per event during detection

# Motif discovery & masks
Z_THR = 2.5                  # z-score threshold for significance
TIME_SMOOTH = 9              # odd int median filter kernel over time (per channel); 1 disables
CONSENSUS_FRAC = 0.6         # fraction of events that must be significant at (ch,t) to include in mask
MORPH_CLOSE = 3              # binary closing (time axis) size; 0 disables
MORPH_OPEN  = 0              # binary opening size; 0 disables

# Classification thresholds
ALPHA = 0.7                  # weight for coverage in score: score = ALPHA*coverage + (1-ALPHA)*jaccard
COV_MIN = 0.25               # minimum required mask coverage to accept a class
SCORE_MIN = 0.20             # minimum combined score to accept; else UNK
SMOOTH_LABEL_MED = 5         # median filter over label indices (odd int); 1 disables

# Printing width
PRINT_WRAP = 90

# ----------------------------- UTILS -----------------------------
def log(msg: str):
    print(msg, flush=True)

def _read_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    # keep EEG only
    raw.pick(picks=mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"))
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def _find_events(raw: mne.io.BaseRaw, prefer: str = ANNOT_KEY) -> Tuple[np.ndarray, int, str]:
    # Prefer annotations
    ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
    if ev is not None and len(ev) and len(eid):
        if prefer in eid:
            return ev, int(eid[prefer]), prefer
        # fallback: most frequent
        vals, counts = np.unique(ev[:, 2], return_counts=True)
        chosen = int(vals[np.argmax(counts)])
        rev = {v: k for k, v in eid.items()}
        return ev, chosen, rev.get(chosen, f"code {chosen}")
    # Fallback stim
    ev = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    if len(ev) == 0:
        raise RuntimeError("No events/annotations found.")
    vals, counts = np.unique(ev[:, 2], return_counts=True)
    chosen = int(vals[np.argmax(counts)])
    return ev, chosen, f"code {chosen}"

def _pick_common_channels(raws: List[mne.io.BaseRaw]) -> List[int]:
    # Prefer PREFERRED_CHANS if present in all raws; else use intersection of all EEG channels
    sets = [set(ch for ch in r.ch_names) for r in raws]
    inter = set.intersection(*sets) if sets else set()
    preferred_present = [c for c in PREFERRED_CHANS if c in inter]
    if len(preferred_present) >= 3:  # at least main motor channels
        names = preferred_present
    else:
        # fall back to sorted intersection (stable across files)
        names = sorted(inter)
    if not names:
        raise RuntimeError("No common EEG channels across files.")
    # Use names order from the first raw
    idx = [raws[0].ch_names.index(n) for n in names]
    return idx

def _epoch_arrays(raw: mne.io.BaseRaw,
                  picks: List[int],
                  events: np.ndarray,
                  code: int,
                  tmin: float,
                  seg_sec: float,
                  base_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_len = int(round(base_sec * sf))
    tmin_samp = int(round(tmin * sf))
    data = raw.get_data(picks=picks)
    T = data.shape[1]

    ev = events[events[:, 2] == code]
    X = []  # segments (C,Tseg)
    B = []  # baselines (C,Tbase)
    for e in ev:
        start = e[0] + tmin_samp
        end = start + seg_samp
        bstart = max(0, start - base_len)
        bend = start
        if start < 0 or end > T or bend <= bstart:
            continue
        X.append(data[:, start:end])
        B.append(data[:, bstart:bend])
    if len(X) == 0:
        return np.zeros((0, len(picks), seg_samp)), np.zeros((0, len(picks), base_len))
    return np.stack(X, axis=0), np.stack(B, axis=0)

def _event_significance(seg: np.ndarray,
                        base: np.ndarray,
                        z_thr: float,
                        time_med: int,
                        morph_close: int,
                        morph_open: int) -> np.ndarray:
    """
    seg: (C,T), base: (C,Tb)
    returns bool mask (C,T) where |z|>=z_thr after simple smoothing & morphology
    """
    mu = base.mean(axis=1, keepdims=True)
    sd = base.std(axis=1, keepdims=True) + 1e-6
    z = np.abs((seg - mu) / sd)

    sig = (z >= z_thr)

    # optional temporal median filter per channel
    if time_med and time_med >= 3 and (time_med % 2 == 1):
        s2 = sig.copy()
        for c in range(sig.shape[0]):
            s2[c] = medfilt(s2[c].astype(np.uint8), kernel_size=time_med).astype(bool)
        sig = s2

    # optional morphology in time dimension (treat per-channel as 1D)
    if morph_close and morph_close >= 2:
        s2 = sig.copy()
        for c in range(sig.shape[0]):
            s2[c] = binary_closing(s2[c], structure=np.ones(morph_close, dtype=bool))
        sig = s2
    if morph_open and morph_open >= 2:
        s2 = sig.copy()
        for c in range(sig.shape[0]):
            s2[c] = binary_opening(s2[c], structure=np.ones(morph_open, dtype=bool))
        sig = s2

    return sig

def _build_class_mask(raw: mne.io.BaseRaw,
                      picks: List[int],
                      prefer_code: str,
                      label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mask   : (C,T) bool, consensus significant positions
      frac   : (C,T) float in [0,1], fraction of events significant at (c,t)
    """
    events, code, desc = _find_events(raw, prefer_code)
    X, B = _epoch_arrays(raw, picks, events, code, TMIN_SEC, SEG_SEC, BASELINE_SEC)
    if X.shape[0] == 0:
        raise RuntimeError(f"No valid epochs for label {label}.")
    C, T = X.shape[1], X.shape[2]

    sigs = []
    for i in range(X.shape[0]):
        sig = _event_significance(
            seg=X[i], base=B[i],
            z_thr=Z_THR, time_med=TIME_SMOOTH,
            morph_close=MORPH_CLOSE, morph_open=MORPH_OPEN
        )  # (C,T)
        sigs.append(sig.astype(np.uint8))
    S = np.stack(sigs, axis=0)  # (N,C,T)
    frac = S.mean(axis=0)       # (C,T) fraction significant
    mask = frac >= CONSENSUS_FRAC

    # Save visualization
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.8), sharex=True)
    im0 = axes[0].imshow(frac, aspect='auto', origin='lower', vmin=0, vmax=1)
    axes[0].set_title(f"Significance fraction (label {label})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].imshow(mask, aspect='auto', origin='lower')
    axes[1].set_title(f"Consensus mask ≥ {int(CONSENSUS_FRAC*100)}%")
    axes[1].set_xlabel("Time (samples)"); axes[0].set_ylabel("Channels"); axes[1].set_ylabel("Channels")
    fig.tight_layout()
    pth = os.path.join(OUT_DIR, f"mask_{label}.png")
    fig.savefig(pth, dpi=150); plt.close(fig)
    log(f"[Saved] Mask for {label} → {pth} (events={X.shape[0]})")
    return mask, frac

def _score_event_to_masks(event_sig: np.ndarray,
                          masks: Dict[str, np.ndarray]) -> Tuple[str, float, dict]:
    """
    Returns (best_label, best_score, per_label_scores_dict)
    score = ALPHA*coverage + (1-ALPHA)*jaccard
    coverage = |E ∧ M| / |M| ; jaccard = |E ∧ M| / |E ∨ M|
    """
    e = event_sig.astype(bool).ravel()
    scores = {}
    best_label, best_score = "UNK", -1.0
    for lab, M in masks.items():
        m = M.ravel()
        inter = np.count_nonzero(e & m)
        uni = np.count_nonzero(e | m)
        mpos = max(1, np.count_nonzero(m))
        cov = inter / mpos
        jac = (inter / max(1, uni)) if uni > 0 else 0.0
        sc = ALPHA * cov + (1.0 - ALPHA) * jac
        scores[lab] = (cov, jac, sc)
        if sc > best_score:
            best_score = sc
            best_label = lab
    # apply thresholds
    cov_best, jac_best, sc_best = scores.get(best_label, (0.0, 0.0, 0.0))
    if (cov_best < COV_MIN) or (sc_best < SCORE_MIN):
        return "UNK", sc_best, scores
    return best_label, sc_best, scores

def _detect_on_target(target_path: str,
                      picks: List[int],
                      masks: Dict[str, np.ndarray],
                      prefer_code: str):
    raw = _read_raw(target_path)
    events, code, desc = _find_events(raw, prefer_code)
    data = raw.get_data(picks=picks)
    sf = raw.info["sfreq"]
    seg_samp = int(round(SEG_SEC * sf))
    base_len = int(round(BASELINE_SEC * sf))
    tmin_samp = int(round(TMIN_SEC * sf))
    T = data.shape[1]

    ev = events[events[:, 2] == code]
    labels: List[str] = []
    max_scores: List[float] = []
    best_shifts: List[int] = []

    for e in ev:
        center = e[0] + tmin_samp
        best = ("UNK", -1.0, 0)  # (lab, score, shift)
        # Scan jitter
        for s in range(-JITTER_SAMP, JITTER_SAMP + 1):
            start = center + s
            end = start + seg_samp
            bstart = start - base_len
            bend = start
            if start < 0 or end > T or bstart < 0:
                continue
            seg = data[:, start:end]
            base = data[:, bstart:bend]
            sig = _event_significance(seg, base, Z_THR, TIME_SMOOTH, MORPH_CLOSE, MORPH_OPEN)
            lab, sc, _scores = _score_event_to_masks(sig, masks)
            if sc > best[1]:
                best = (lab, sc, s)
        labels.append(best[0])
        max_scores.append(best[1] if best[1] >= 0 else 0.0)
        best_shifts.append(best[2])

    # Optional median filter over labels (A=0,B=1,C=2,D=3, UNK=-1)
    if SMOOTH_LABEL_MED and SMOOTH_LABEL_MED >= 3 and (SMOOTH_LABEL_MED % 2 == 1):
        order = {k: i for i, k in enumerate(sorted(masks.keys()))}  # deterministic index
        enc = np.array([order.get(l, -1) for l in labels], dtype=int)
        mask = enc >= 0
        if np.any(mask):
            enc_s = enc.copy()
            enc_s[mask] = medfilt(enc[mask], kernel_size=SMOOTH_LABEL_MED)
            inv = {v: k for k, v in order.items()}
            labels = [inv.get(v, "UNK") for v in enc_s]

    # Pretty print
    log("\n=== Detected Sequence (Mask-motif, annotation-locked + jitter) ===")
    line = []
    for i, tok in enumerate(labels, 1):
        line.append(tok)
        if len(line) >= PRINT_WRAP:
            log(" ".join(line)); line = []
    if line: log(" ".join(line))

    # Counts
    labs_sorted = sorted(masks.keys())
    counts = {lab: labels.count(lab) for lab in labs_sorted}
    counts["UNK"] = labels.count("UNK")
    log("\n[Counts]")
    for k in labs_sorted + ["UNK"]:
        log(f"  {k}: {counts.get(k, 0)}")

    # Save sequence
    seq_txt = os.path.join(OUT_DIR, "mask_motif_mi_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels) + "\n")
    seq_csv = os.path.join(OUT_DIR, "mask_motif_mi_mix_sequence.csv")
    with open(seq_csv, "w", encoding="utf-8") as f:
        f.write("event_idx,label,shift,max_score\n")
        for i, (l, s, sc) in enumerate(zip(labels, best_shifts, max_scores), 1):
            f.write(f"{i},{l},{s},{sc:.6f}\n")
    log(f"[Saved] Sequence TXT → {seq_txt}")
    log(f"[Saved] Sequence CSV → {seq_csv}")

    # Plots
    # Shift histogram
    bins = np.arange(-JITTER_SAMP - 0.5, JITTER_SAMP + 1.5, 1)
    plt.figure(figsize=(6, 3))
    plt.hist(best_shifts, bins=bins, edgecolor="black")
    plt.xlabel("Best shift (samples)"); plt.ylabel("Count")
    plt.title("Target best-shift histogram")
    plt.tight_layout()
    hpth = os.path.join(OUT_DIR, "target_shift_hist_mask_motif.png")
    plt.savefig(hpth, dpi=150); plt.close()
    log(f"[Saved] Shift histogram → {hpth}")

    # Score trace
    plt.figure(figsize=(8, 3))
    plt.plot(max_scores, lw=1)
    plt.axhline(SCORE_MIN, ls="--", alpha=0.5)
    plt.xlabel("Event #"); plt.ylabel("Max score")
    plt.title("Per-event max similarity score")
    plt.tight_layout()
    spth = os.path.join(OUT_DIR, "target_maxscore_mask_motif.png")
    plt.savefig(spth, dpi=150); plt.close()
    log(f"[Saved] Score trace → {spth}")

    return labels

# ----------------------------- MAIN -----------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 0) Load raws once (to choose common channels)
    raws = {lab: _read_raw(p) for lab, p in REG_SOURCES.items()}
    # Also load target to include in channel intersection
    raw_target = _read_raw(TARGET_FILE)
    picks = _pick_common_channels(list(raws.values()) + [raw_target])
    ch_names = [list(raws.values())[0].ch_names[i] for i in picks]
    log(f"| [Channels] Using common set: {ch_names}")

    # 1) Build masks per class
    masks = {}
    for lab, raw in raws.items():
        mask, frac = _build_class_mask(raw, picks, ANNOT_KEY, lab)
        masks[lab] = mask

    # 2) Detect sequence on target
    labels = _detect_on_target(TARGET_FILE, picks, masks, ANNOT_KEY)

    # 3) Optional accuracy vs ground truth
    gt_str = GROUND_TRUTH_STR.strip()
    if gt_str:
        gt_tokens = [t.strip() for t in gt_str.split() if t.strip()]
        n = min(len(gt_tokens), len(labels))
        if n > 0:
            acc = accuracy_score(gt_tokens[:n], labels[:n])
            log(f"[Accuracy vs GT] {acc*100:.2f}% (truncated to common length)")
        else:
            log("[Accuracy vs GT] (no overlap in lengths)")
    else:
        log("[Accuracy vs GT] (skipped — no GT pasted)")

    log("\n[Done] All outputs written to: " + OUT_DIR)

if __name__ == "__main__":
    main()

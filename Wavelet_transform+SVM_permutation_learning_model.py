# -*- coding: utf-8 -*-
"""
Train C vs D on registrations → Predict per-event labels on MIX
Wavelet (db4) relative-band energies + SVM.
Prints the predicted sequence and (optionally) accuracy if GT is provided.

Deps: mne, pywt, numpy, scikit-learn, matplotlib (only if you later add plots)
"""

import os
import sys
import json
from collections import Counter

import numpy as np
import mne
import pywt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# ======= PATHS: EDIT ME ======
# -----------------------------
PATH_C   = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif"
PATH_D   = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif"
PATH_MIX =  r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

# Optional ground-truth labels (text file with C/D tokens)
GT_PATH  = r""  # e.g., r"C:\Users\User\Desktop\masters_thesis_directory\mix_ground_truth_CD.txt"
OUT_DIR  = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_wavelet_svm_CD_predict_mix"

# -----------------------------
# ======= PARAMETERS ==========
# -----------------------------
DESIRED_CHS_MAIN     = ["C3L", "CzL", "C4L"]
DESIRED_CHS_FALLBACK = ["C3", "Cz", "C4"]

T0_OFFSET_SEC = 0.0
SEG_LEN_SEC   = 1.0  # 1-second windows locked to triggers

WAVELET   = "db4"
LEVEL     = None      # None → use max level allowed by segment length
USE_LOG1P = True

SVM_C         = 4.0
SVM_GAMMA     = "scale"
CLASS_WEIGHT  = "balanced"
RANDOM_STATE  = 42
CV_SPLITS     = 5

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
        raise RuntimeError(f"No desired EEG channels found. Example channels: {raw.ch_names[:10]}")
    return chs

def get_event_samples(raw):
    ev_anno, ev_id = mne.events_from_annotations(raw, verbose=False)
    if len(ev_anno):
        # Prefer 'S  1' or any code == 1; else most frequent code
        target_keys = sorted(ev_id.keys(), key=lambda k: (("S  1" not in k), k))
        for k in target_keys:
            if "S  1" in k or k.strip().endswith("1"):
                code = ev_id[k]
                idx = (ev_anno[:, 2] == code)
                if idx.any():
                    return ev_anno[idx, 0], k
        counts = Counter(ev_anno[:, 2])
        common_code = counts.most_common(1)[0][0]
        idx = (ev_anno[:, 2] == common_code)
        key = [k for k, v in ev_id.items() if v == common_code]
        key = key[0] if key else f"code={common_code}"
        return ev_anno[idx, 0], key

    ev_stim = mne.find_events(raw, shortest_event=1, initial_event=True, verbose=False)
    if len(ev_stim):
        counts = Counter(ev_stim[:, 2])
        common_code = counts.most_common(1)[0][0]
        idx = (ev_stim[:, 2] == common_code)
        return ev_stim[idx, 0], f"stim_code={common_code}"

    raise RuntimeError("No events found in annotations or stim channel.")

def segment_from_events(raw, chs, event_samp, seg_len_samp, t0_offset_samp=0):
    data = raw.get_data(picks=chs)
    n_ch, n_samp = data.shape
    segments, kept_idx = [], []
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

def wavelet_features(X, wavelet=WAVELET, level=LEVEL, use_log1p=USE_LOG1P):
    """
    X: (N, C, T) → features: relative energy per subband for each channel.
    """
    N, C, T = X.shape
    feats = []
    filt_len  = pywt.Wavelet(wavelet).dec_len
    max_level = pywt.dwt_max_level(T, filt_len)
    L = level if (level is not None and level > 0) else max(1, max_level)

    for i in range(N):
        row = []
        for c in range(C):
            coeffs = pywt.wavedec(X[i, c, :], wavelet=wavelet, level=L, mode="symmetric")
            energies = [float(np.sum(np.square(cc))) for cc in coeffs]  # [A_L, D_L, ..., D1]
            total = sum(energies) + 1e-12
            rel = [e / total for e in energies]
            row.extend(rel)
        feats.append(row)

    feats = np.asarray(feats, dtype=float)
    if use_log1p:
        feats = np.log1p(feats)
    return feats  # (N, C*(L+1))

def make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=SVM_C, gamma=SVM_GAMMA,
                    class_weight=CLASS_WEIGHT,
                    probability=False,
                    random_state=RANDOM_STATE))
    ])

def load_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            tokens.extend(line.strip().split())
    # Normalize tokens (accept 'A/B' or 'C/D'—map A→C, B→D if needed)
    norm = []
    for t in tokens:
        u = t.strip().upper()
        if u in ("C", "D"):
            norm.append(u)
        elif u == "A":
            norm.append("C")
        elif u == "B":
            norm.append("D")
        else:
            # ignore unknown tokens
            pass
    return np.array(norm, dtype="<U1")

# -----------------------------
# ========== MAIN =============
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load training (C & D) ----
    print("[Train] Loading C and D registrations ...")
    raw_c = mne.io.read_raw_fif(_check_path(PATH_C, "C"), preload=True, verbose=False)
    raw_d = mne.io.read_raw_fif(_check_path(PATH_D, "D"), preload=True, verbose=False)

    sf_c = raw_c.info["sfreq"]
    sf_d = raw_d.info["sfreq"]
    if abs(sf_c - sf_d) > 1e-6:
        raise RuntimeError(f"Sampling rates differ: C={sf_c}, D={sf_d}")
    sf = float(sf_c)
    seg_len_samp   = int(round(SEG_LEN_SEC * sf))
    t0_offset_samp = int(round(T0_OFFSET_SEC * sf))

    chs_c = pick_channels_or_fallback(raw_c)
    chs_d = pick_channels_or_fallback(raw_d)
    chs   = [ch for ch in chs_c if ch in chs_d]
    if not chs:
        raise RuntimeError("No common channels between C and D.")
    print(f"[Train] channels={chs}, sf={sf:.1f} Hz")

    ev_c, key_c = get_event_samples(raw_c)
    ev_d, key_d = get_event_samples(raw_d)
    print(f"[Train] Events: C→'{key_c}' N={len(ev_c)} | D→'{key_d}' N={len(ev_d)}")

    Xc, _ = segment_from_events(raw_c, chs, ev_c, seg_len_samp, t0_offset_samp)
    Xd, _ = segment_from_events(raw_d, chs, ev_d, seg_len_samp, t0_offset_samp)
    yc = np.array(["C"] * Xc.shape[0])
    yd = np.array(["D"] * Xd.shape[0])

    X_train = np.concatenate([Xc, Xd], axis=0)
    y_train = np.concatenate([yc, yd], axis=0)
    Feat_tr = wavelet_features(X_train)

    print(f"[Train] Features: {Feat_tr.shape} (N,F)")

    # ---- Fit classifier (optionally print CV score just for sanity) ----
    clf = make_clf()

    # (Optional sanity CV; comment out if you want max speed)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # You could do cross_val_score here if needed

    clf.fit(Feat_tr, y_train)
    print("[Train] Classifier fitted.")

    # ---- Load MIX and predict ----
    print("[MIX] Loading mix file ...")
    raw_mix = mne.io.read_raw_fif(_check_path(PATH_MIX, "MIX"), preload=True, verbose=False)

    # Ensure same channels exist in mix
    chs_mix = pick_channels_or_fallback(raw_mix, desired=chs, fallback=chs)
    if chs_mix != chs:
        # reorder mix to match training order
        chs_mix = chs
    ev_mix, key_mix = get_event_samples(raw_mix)
    print(f"[MIX] Events: '{key_mix}' N={len(ev_mix)}")

    X_mix, kept_idx = segment_from_events(raw_mix, chs_mix, ev_mix, seg_len_samp, t0_offset_samp)
    Feat_mix = wavelet_features(X_mix)

    y_pred = clf.predict(Feat_mix)  # array of "C"/"D"
    seq_str = " ".join(y_pred.tolist())

    # Save and print sequence
    out_seq = os.path.join(OUT_DIR, "mix_predicted_sequence_CD.txt")
    with open(out_seq, "w", encoding="utf-8") as f:
        f.write(seq_str + "\n")
    print("\n===== Predicted sequence (C/D) on MIX =====")
    print(seq_str)
    print(f"\n[Saved] Predicted sequence → {out_seq}")

    # ---- Optional: accuracy against ground truth ----
    if GT_PATH and os.path.isfile(GT_PATH):
        y_true = load_ground_truth(GT_PATH)
        if len(y_true) != len(y_pred):
            n = min(len(y_true), len(y_pred))
            print(f"[WARN] GT length ({len(y_true)}) != predictions ({len(y_pred)}). "
                  f"Comparing first {n} events.")
            y_true = y_true[:n]
            y_pred = y_pred[:n]

        acc = accuracy_score(y_true, y_pred)
        print(f"\n[MIX] Overall accuracy vs GT: {acc:.4f}")
        print("\n[MIX] Classification report:")
        print(classification_report(y_true, y_pred, labels=["C","D"], digits=3))

        cm = confusion_matrix(y_true, y_pred, labels=["C","D"])
        cm_json = {
            "labels": ["C", "D"],
            "matrix": cm.tolist(),
            "accuracy": float(acc),
            "n_events": int(len(y_true))
        }
        out_json = os.path.join(OUT_DIR, "mix_prediction_metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cm_json, f, indent=2)
        print(f"[Saved] Metrics JSON → {out_json}")
    else:
        print("\n[Info] No ground-truth file provided (GT_PATH empty or missing). "
              "Only sequence has been saved/printed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)

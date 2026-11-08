# -*- coding: utf-8 -*-
"""
Wavelet features + SVM (A/B/C/D) → detect sequence on movement-mix target
------------------------------------------------------------------------
- Train on four registrations (A,B,C,D).
- Derive virtual Laplacians: C3L, CzL, C4L (fallback to base if neighbors missing).
- Pre-first-event baseline z-scoring on Laplacians (shared per file).
- Epoch window [TMIN, TMAX] sec for both train & predict (same length).
- Features: per-channel Discrete Wavelet Transform (DWT) energies across levels
  + their log-energies and relative energies (stable-length scalar features).
- Classifier: StandardScaler → SVM (RBF, class_weight='balanced', probability=True).
- Prediction: trigger-locked with ± jitter search; UNK if low confidence / margin.
- Saves sequence (.txt, .csv), prints counts, CV accuracy & confusion matrix.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from mne.decoding import CSP  # not used; left imported if you want to compare later
import pywt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# ======= CONFIG ==========
# =========================

# --- EDIT THESE PATHS (use your real files) ---
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"
TARGET_EVENT = None  # None → pick the most frequent event code automatically
OUTPUT_DIR  = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_wavelet_svm_output"

# Epoch window (seconds)
TMIN = 0.00
TMAX = 1.00

# Resampling (set a common Hz to force identical fs across files; None keeps native)
RESAMPLE_HZ: Optional[float] = None  # e.g., 250.0

# Jitter at prediction (seconds, scanned ±)
JITTER_SEC = 0.12   # ~±120 ms if window=1s
JITTER_STEP_SAMP = 1

# Baseline on Laplacians from pre-first-event tail
BASELINE_USE = True
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
BASELINE_MODE = "zscore"  # 'zscore' or 'demean'

# Wavelet feature params
WAVELET = "db4"
WAVELET_LEVELS = 5  # D1..D5 + A5 → 6 energies per channel
USE_LOG_ENERGY = True
USE_REL_ENERGY = True  # energies / sum(energies)

# SVM
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_PROBABILITY = True
RANDOM_STATE = 42

# UNK gating
CONF_THRESH   = 0.55  # min p_top
MARGIN_THRESH = 0.05  # min (p_top - p_second)

# Optional ground truth string to print accuracy against (paste A/B/C/D labels separated by spaces)
GROUND_TRUTH_STR: Optional[str] = None

# =========================
# ====== UTILITIES ========
# =========================

EPS = 1e-12

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_raw(path: str, resample_hz: Optional[float]) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-6:
        raw.resample(resample_hz, npad="auto")
    return raw

def get_events_any(raw: mne.io.BaseRaw):
    """Prefer annotations; fallback to stim channel."""
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(events) > 0:
            return events, event_id
    except Exception:
        pass
    events = mne.find_events(raw, verbose="ERROR")
    codes = np.unique(events[:, 2]) if len(events) else np.array([])
    event_id = {f"EVT/{int(c)}": int(c) for c in codes}
    return events, event_id

def resolve_event_auto(events, event_id_map, requested):
    """Return (event_code, label)."""
    if requested is not None:
        if isinstance(requested, int):
            return requested, f"_code:{requested}"
        if requested in event_id_map:
            return event_id_map[requested], requested
        for k, v in event_id_map.items():
            if requested.lower() in k.lower():
                return v, k
        raise ValueError(f"Could not resolve event '{requested}'.")
    if len(event_id_map) == 0:
        raise RuntimeError("No events/annotations found.")
    from collections import Counter
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for lbl, c in event_id_map.items():
        if c == code:
            return c, lbl
    return int(code), f"_code:{int(code)}"

# ---------- Virtual Laplacian ----------
CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}

def derive_laplacian_from_arrays(data_map: Dict[str, np.ndarray], name: str) -> np.ndarray:
    base, neigh = CENTRAL_BASES[name]
    if base not in data_map:
        raise KeyError(f"Base channel {base} missing for {name}.")
    base_arr = data_map[base]
    neigh_avail = [data_map[k] for k in neigh if k in data_map]
    if len(neigh_avail) == 2:
        return base_arr - 0.5 * (neigh_avail[0] + neigh_avail[1])
    else:
        return base_arr.copy()

def derive_laplacian_continuous(raw: mne.io.BaseRaw,
                                start: Optional[int] = None,
                                stop: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Return (C', T) for [C3L, CzL, C4L] from continuous Raw (with fallbacks)."""
    if start is None:
        start = 0
    if stop is None:
        stop = raw.n_times

    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)

    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError("None of the required central/neighbor channels are present.")

    full = raw.get_data(picks=picks, start=start, stop=stop)  # (K, T)
    picked_names = [raw.info["ch_names"][i] for i in picks]
    data_map = {n: full[idx] for idx, n in enumerate(picked_names)}

    out, out_names = [], []
    for lap_name in ["C3L", "CzL", "C4L"]:
        try:
            sig = derive_laplacian_from_arrays(data_map, lap_name)
            out.append(sig); out_names.append(lap_name)
        except KeyError:
            base, _ = CENTRAL_BASES[lap_name]
            if base in data_map:
                out.append(data_map[base]); out_names.append(base)
    if len(out) == 0:
        raise RuntimeError("No central/Laplacian signals could be derived.")
    return np.vstack(out), out_names

# ---------- Baseline ----------
def pre_annotation_baseline_window(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    sf = raw.info["sfreq"]
    events, _ = get_events_any(raw)
    if len(events) > 0:
        first_samp = int(np.min(events[:, 0]))
        end = max(0, first_samp - int(round(margin_sec * sf)))
        start = max(0, end - int(round(min_sec * sf)))
    else:
        start = 0
        end = min(int(round(min_sec * sf)), raw.n_times)
    if end <= start:
        end = min(int(round(min_sec * sf)), raw.n_times)
        start = 0
    return start, end

def baseline_on_laplacian(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    start, end = pre_annotation_baseline_window(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
    lap, names = derive_laplacian_continuous(raw, start, end)  # (C', T0)
    mu = lap.mean(axis=1)
    sd = lap.std(axis=1) + EPS
    return mu, sd, names

def apply_baseline_norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str):
    if mode == "zscore":
        return (arr - mean[None, :, None]) / std[None, :, None] if arr.ndim == 3 else (arr - mean[:, None]) / std[:, None]
    else:
        return (arr - mean[None, :, None]) if arr.ndim == 3 else (arr - mean[:, None])

# ---------- Epoching ----------
def epochs_from_file(fpath: str, tmin: float, tmax: float,
                     resample_hz: Optional[float]) -> Tuple[np.ndarray, List[str], float, int, str]:
    """
    Returns:
      lap_epochs: (N, C', T), lap_names, sfreq, N, chosen_event_label
    """
    raw = load_raw(fpath, resample_hz)
    events, eid = get_events_any(raw)
    if len(events) == 0:
        raise RuntimeError(f"No events found in {os.path.basename(fpath)}.")
    code, chosen_lbl = resolve_event_auto(events, eid, requested=None)

    # Pick base + neighbor channels needed for Laplacians
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError(f"Required channels not found in {os.path.basename(fpath)}.")

    # Compute Lap baseline
    if BASELINE_USE:
        bmu, bsd, lap_bnames = baseline_on_laplacian(raw)

    # Epoch
    ep = mne.Epochs(raw, events[events[:, 2] == code], event_id={chosen_lbl: code},
                    tmin=tmin, tmax=tmax, baseline=None, picks=picks,
                    preload=True, detrend=0, verbose="ERROR", reject_by_annotation=True)
    if len(ep) == 0:
        raise RuntimeError(f"No epochs after epoching in {os.path.basename(fpath)}.")

    # Extract base + neighbors epochs -> derive Laplacians per epoch
    data = ep.get_data()  # (N, K, T)
    ch_names = ep.ch_names
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    out, out_names = [], []
    for lap_name in ["C3L", "CzL", "C4L"]:
        base, neigh = CENTRAL_BASES[lap_name]
        if base not in name_to_idx:
            continue
        base_idx = name_to_idx[base]
        neigh_idxs = [name_to_idx[k] for k in neigh if k in name_to_idx]
        if len(neigh_idxs) == 2:
            lap = data[:, base_idx, :] - 0.5 * (data[:, neigh_idxs[0], :] + data[:, neigh_idxs[1], :])
            out.append(lap); out_names.append(lap_name)
        else:
            out.append(data[:, base_idx, :]); out_names.append(base)
    if len(out) == 0:
        raise RuntimeError("No Laplacian channels derived in epochs.")
    laps = np.stack(out, axis=1)  # (N, C', T)
    sf = ep.info["sfreq"]

    # Align baseline order if needed and normalize
    if BASELINE_USE:
        if lap_bnames != out_names:
            idx_map = {n: i for i, n in enumerate(lap_bnames)}
            idx = [idx_map[n] for n in out_names if n in idx_map]
            bmu = bmu[idx]; bsd = bsd[idx]
        laps = apply_baseline_norm(laps, bmu, bsd, BASELINE_MODE)

    return laps, out_names, float(sf), laps.shape[0], chosen_lbl

def align_to_ref_order(X: np.ndarray, names: List[str], ref_order: List[str]) -> np.ndarray:
    """X: (N,C,T) -> reorder to ref_order subset."""
    if names == ref_order:
        return X
    idx = [names.index(n) for n in ref_order if n in names]
    if not idx:
        raise RuntimeError("No overlapping Laplacian channels to align.")
    return X[:, idx, :]

# ---------- Features: Wavelet energies ----------
def wavelet_energy_features(sig: np.ndarray,
                            wavelet: str = WAVELET,
                            levels: int = WAVELET_LEVELS,
                            use_log: bool = True,
                            use_rel: bool = True) -> np.ndarray:
    """
    sig: (T,) time-series
    Returns feature vector:
      [E_A(L), E_D(L), ..., E_D(1), (optional logs), (optional relatives)]
    Energies are sum of squares of coeffs at each level.
    """
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=levels, mode="symmetric")
    # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    energies = np.array([np.sum(c * c) for c in coeffs], dtype=float)
    feats = [energies]
    if use_log:
        feats.append(np.log10(energies + EPS))
    if use_rel:
        total = np.sum(energies) + EPS
        feats.append(energies / total)
    return np.concatenate(feats, axis=0)

def make_features(lap_epochs: np.ndarray) -> np.ndarray:
    """
    lap_epochs: (N, C, T)
    Returns: (N, F) wavelet features stacked over channels.
    """
    N, C, _ = lap_epochs.shape
    feats = []
    for n in range(N):
        f_parts = []
        for c in range(C):
            f_parts.append(wavelet_energy_features(lap_epochs[n, c]))
        feats.append(np.concatenate(f_parts, axis=0))
    return np.vstack(feats)

# ---------- Build training ----------
def build_train_set(files: Dict[str, str], tmin: float, tmax: float,
                    resample_hz: Optional[float]):
    X_list, y_list, name_list, sf_list = [], [], [], []
    chosen_events = {}
    print("[Training] Building features from registrations (A,B, C, D) ...")
    for lbl in ["A", "B", "C", "D"]:
        fpath = files[lbl]
        laps, lap_names, sf, n, chosen_lbl = epochs_from_file(fpath, tmin, tmax, resample_hz)
        X_list.append(laps)
        y_list.append(np.full(n, lbl, dtype=object))
        name_list.append(lap_names)
        sf_list.append(sf)
        chosen_events[lbl] = chosen_lbl

    # enforce same sfreq
    if len(set(sf_list)) != 1:
        raise RuntimeError(f"Sampling rates differ: {sf_list}. Set RESAMPLE_HZ to a common value.")
    sf = sf_list[0]

    # choose reference lap order
    ref_order = name_list[0]
    X_aligned = []
    for X, names in zip(X_list, name_list):
        X_aligned.append(align_to_ref_order(X, names, ref_order))
    X_lap = np.concatenate(X_aligned, axis=0)  # (N, C, T)
    y = np.concatenate(y_list, axis=0)

    # wavelet features
    X = make_features(X_lap)

    print(f"[Train set] X={X.shape} (N,F), Lap={ref_order}, sf={sf:.1f} Hz  | per-class:",
          {k: int(np.sum(y == k)) for k in ["A","B","C","D"]})
    return X, y, ref_order, sf

# ---------- Classifier ----------
def make_wavelet_svm_pipeline() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(C=SVM_C, kernel="rbf", gamma=SVM_GAMMA,
                    class_weight=SVM_CLASS_WEIGHT, probability=SVM_PROBABILITY,
                    random_state=RANDOM_STATE))
    ])
    return pipe

def cv_score(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std())

# ---------- Target prediction ----------
def predict_sequence_with_jitter(raw: mne.io.BaseRaw,
                                 ref_order: List[str],
                                 tmin: float, tmax: float,
                                 resample_hz: Optional[float],
                                 pipe: Pipeline,
                                 event_req=None):
    """Return labels, per-event prob vectors, and chosen shifts."""
    sf = raw.info["sfreq"]
    events, eid = get_events_any(raw)
    if len(events) == 0:
        raise RuntimeError("No events in target file.")
    code, lbl = resolve_event_auto(events, eid, requested=event_req)
    evs = events[events[:, 2] == code]
    print(f"[Target] Using event '{lbl}' (code {code}); N={len(evs)}")

    # Build continuous Laplacians & baseline-normalize them
    lap_all, lap_names = derive_laplacian_continuous(raw)
    if lap_names != ref_order:
        m = {n: i for i, n in enumerate(lap_names)}
        idx = [m[n] for n in ref_order if n in m]
        lap_all = lap_all[idx]
        lap_names = [ref_order[i] for i in range(len(idx))]
    if BASELINE_USE:
        bmu, bsd, lap_bnames = baseline_on_laplacian(raw)
        if lap_bnames != lap_names:
            m = {n: i for i, n in enumerate(lap_bnames)}
            idx = [m[n] for n in lap_names if n in m]
            bmu, bsd = bmu[idx], bsd[idx]
        lap_all = (lap_all - bmu[:, None]) / bsd[:, None] if BASELINE_MODE == "zscore" else (lap_all - bmu[:, None])

    # Jitter config
    L = int(round((tmax - tmin) * sf))
    jitter = int(round(JITTER_SEC * sf))
    N = lap_all.shape[1]

    labels, prob_list, shifts = [], [], []
    print(f"[Predict] Jitter scan: ±{jitter} samples (~±{1000.0 * jitter / sf:.0f} ms)  | seg_len={L} | T_ext={N}")

    for ev in tqdm(evs[:, 0], desc="[Predict] Events", unit="ev"):
        start0 = int(ev + tmin * sf)
        seg_start = max(0, start0 - jitter)
        seg_end   = min(N, start0 + L + jitter)
        if seg_end - seg_start < L + 1:
            labels.append("UNK"); prob_list.append(np.zeros(4)); shifts.append(0); continue

        best_p, best_lbl, best_shift = -1.0, "UNK", 0
        best_vec = None

        for s in range(0, (seg_end - seg_start) - L + 1, max(1, JITTER_STEP_SAMP)):
            seg = lap_all[:, seg_start + s: seg_start + s + L]   # (C, L)
            # → wavelet features
            f_parts = []
            for c in range(seg.shape[0]):
                f_parts.append(wavelet_energy_features(seg[c]))
            feat = np.concatenate(f_parts, axis=0)[None, :]  # (1, F)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_vec = pipe.predict_proba(feat)[0]  # order pipe.classes_

            # Map to A,B,C,D prob vector (fixed order)
            classes = list(pipe.classes_)
            p_full = np.zeros(4)
            for i, cl in enumerate(['A','B','C','D']):
                p_full[i] = p_vec[classes.index(cl)] if cl in classes else 0.0

            idx_top = int(np.argmax(p_full))
            p_top   = float(p_full[idx_top])
            p_second = float(np.partition(p_full, -2)[-2])
            cand_lbl = ['A','B','C','D'][idx_top]

            if p_top > best_p:
                best_p = p_top
                best_lbl = cand_lbl if (p_top >= CONF_THRESH and (p_top - p_second) >= MARGIN_THRESH) else "UNK"
                best_shift = s - jitter
                best_vec = p_full

        labels.append(best_lbl)
        prob_list.append(best_vec if best_vec is not None else np.zeros(4))
        shifts.append(best_shift)

    return labels, np.vstack(prob_list), np.array(shifts, dtype=int)

# ---------- Saving & reporting ----------
def save_sequence(labels: List[str], probs: np.ndarray, outdir: str, stem: str):
    ensure_outdir(outdir)
    txt_path = os.path.join(outdir, f"{stem}.txt")
    csv_path = os.path.join(outdir, f"{stem}.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(labels) + "\n")
    df = pd.DataFrame({
        "idx": np.arange(len(labels)),
        "label": labels,
        "pA": probs[:, 0] if probs.size else np.nan,
        "pB": probs[:, 1] if probs.size else np.nan,
        "pC": probs[:, 2] if probs.size else np.nan,
        "pD": probs[:, 3] if probs.size else np.nan,
    })
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved sequence] {txt_path}\n[Saved table]    {csv_path}")

def print_counts(labels: List[str]):
    from collections import Counter
    c = Counter(labels)
    print("\n[Counts]")
    for k in ['A','B','C','D','UNK']:
        if c.get(k, 0) > 0:
            print(f"  {k}: {c[k]}")

def confusion_report(pipe: Pipeline, X: np.ndarray, y: np.ndarray, outdir: str):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    labels = ["A","B","C","D"]
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
    print("\n[CV] Confusion matrix (normalized rows):")
    with np.printoptions(precision=3, suppress=True):
        print(cm)
    print("\n[CV] Classification report:")
    print(classification_report(y, y_pred, labels=labels, zero_division=0, digits=3))
    # Optional: save as CSV
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(outdir, "cv_confusion_matrix.csv"), index=True)

def accuracy_vs_gt(pred_labels: List[str], gt_str: str):
    gt = [t for t in gt_str.strip().split() if t]
    n = min(len(pred_labels), len(gt))
    if n == 0:
        return
    acc = 100.0 * sum(1 for i in range(n) if pred_labels[i] == gt[i]) / n
    best_acc, best_shift = acc, 0
    for s in range(-3, 3+1):
        if s == 0: continue
        # compare pred shifted circularly against gt on n
        acc_s = 100.0 * sum(1 for i in range(n) if pred_labels[(i - s) % n] == gt[i]) / n
        if acc_s > best_acc:
            best_acc, best_shift = acc_s, s
    print("\n[Accuracy vs your ground truth]")
    print(f"  Lengths -> pred={len(pred_labels)}, gt={len(gt)}  (compared on {n})")
    print(f"  No shift: {acc:.2f}%")
    print(f"  Best shift in [-3..+3]: shift={best_shift:+d}, acc={best_acc:.2f}%")

# =========================
# ========= MAIN ==========
# =========================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_outdir(OUTPUT_DIR)

    # sanity: paths
    for k, p in TEMPLATE_SOURCES.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing template {k}: {p}")
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")

    # -------- Build train set (wavelet features)
    X, y, lap_order, sf = build_train_set(TEMPLATE_SOURCES, TMIN, TMAX, RESAMPLE_HZ)

    # -------- Pipeline + CV
    pipe = make_wavelet_svm_pipeline()
    mean_cv, std_cv = cv_score(pipe, X, y)
    print(f"\n[CV] Wavelet+SVM accuracy: {mean_cv:.3f} (± {std_cv:.3f})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    confusion_report(pipe, X, y, OUTPUT_DIR)

    # -------- Target
    print("\n[Target] Loading movement-mix target ...")
    raw_t = load_raw(TARGET_FILE, RESAMPLE_HZ)
    labels, probs, shifts = predict_sequence_with_jitter(raw_t, lap_order, TMIN, TMAX, RESAMPLE_HZ, pipe, event_req=TARGET_EVENT)
    print(f"[Shift stats] mean={float(np.mean(shifts)):.1f} samp, std={float(np.std(shifts)):.1f}, "
          f"min={int(np.min(shifts))}, max={int(np.max(shifts))}")

    # -------- Report & save
    print("\n=== Detected Sequence (Wavelet+SVM, 4-class, trigger-locked+jitter) ===")
    wrap = 100
    for i in range(0, len(labels), wrap):
        print(" ".join(labels[i:i+wrap]))

    print_counts(labels)
    save_sequence(labels, probs, OUTPUT_DIR, stem="wavelet_svm_movement_mix_sequence")

    if GROUND_TRUTH_STR:
        accuracy_vs_gt(labels, GROUND_TRUTH_STR)

    print("\nDone.")

if __name__ == "__main__":
    main()

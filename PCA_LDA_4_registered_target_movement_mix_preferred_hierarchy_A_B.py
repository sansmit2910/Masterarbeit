# -*- coding: utf-8 -*-
"""
Hierarchical PCA+LDA EEG Motif Classifier (A,B,C,D → movement-mix target)
------------------------------------------------------------------------
Level-1: LDA (AB vs CD) with configurable priors (favor AB by default).
Level-2: LDA (A vs B) or (C vs D) inside the chosen group.

Features:
  • Virtual Laplacians (C3L, CzL, C4L) with fallback if neighbors missing
  • Per-file shared baseline on Laplacians (pre-first-event segment)
  • Time-domain (per-epoch z-scored, flattened)
  • Mu/Beta envelope (log-variance per band & channel)
  • PCA → LDA pipelines for each classifier
  • Confidence gating + UNK + temporal smoothing
  • Safe epoching (drop edge-clipping events)
  • Progress bars, sequence save (TXT/CSV), per-label counts

Deps: mne, numpy, scipy, scikit-learn, tqdm, matplotlib (only for optional CM fig)
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

import numpy as np
import mne
from scipy.signal import hilbert
from mne.filter import filter_data
from tqdm import tqdm

# ML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

# -----------------------
# ----- CONFIG ----------
# -----------------------

DATA_DIR = r"C:\Users\User\Desktop\masters_thesis_directory"

REG_FILES: Dict[str, str] = {
    "A": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-1-reg.fif"),
    "B": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-2-reg.fif"),
    "C": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-imagine-move-1_raw.fif"),
    "D": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-imagine-move-2_raw.fif"),
}

TARGET_FILE = os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-mix_raw.fif")
TARGET_EVENT: Optional[Union[int, str]] = None  # None -> auto choose most frequent

# Epoch window (s)
TMIN = 0.0
TMAX = 0.6

# Resample to unify sampling rate (None to keep native)
RESAMPLE_HZ: Optional[float] = 200.0

# Laplacians
CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}

# Baseline on Laplacians (pre-first-event)
BASELINE_USE_PRE_ANNOT = True
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
BASELINE_MODE = "zscore"
EPS = 1e-12

# Envelope bands and smoothing
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]
ENV_SMOOTH_SEC = 0.05

# PCA+LDA dims
PCA_MAX_COMPONENTS_GATE = 30   # for AB vs CD
PCA_MAX_COMPONENTS_AB   = 30   # for A vs B
PCA_MAX_COMPONENTS_CD   = 30   # for C vs D
N_FOLDS_CV = 5

# Gate priors & thresholds
GATE_PRIOR_AB = 0.8             # favor AB in movement-mix
GATE_PROBA_MIN = 0.55           # else -> UNK
SUB_PROBA_MIN  = 0.55           # inside AB or CD; else -> UNK

# Temporal smoothing (windowed majority vote)
SMOOTH_WIN = 5                  # odd number recommended; 1 disables smoothing
SMOOTH_INCLUDE_UNK = False      # if False, UNK does not vote (only breaks ties)

# Allow CD at all? If you know target is physical movement, you can de-prioritize CD via GATE_PRIOR_AB
ALLOW_CD = True                 # keep True (you asked not to hard-code A/B only)

# Output
OUTPUT_DIR = os.path.join(DATA_DIR, "eeg_pca_lda_hier_output")
SAVE_TXT_CSV = True

# -----------------------
# ----- UTILS -----------
# -----------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_raw(path: str, resample_hz: Optional[float]) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-6:
        raw.resample(resample_hz, npad="auto")
    return raw

def get_events_any(raw: mne.io.BaseRaw):
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(events) > 0:
            return events, event_id
    except Exception:
        pass
    events = mne.find_events(raw, verbose="ERROR")
    codes = np.unique(events[:, 2])
    event_id = {f"EVT/{int(c)}": int(c) for c in codes}
    return events, event_id

def resolve_event_auto(events, event_id_map, requested):
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
    if len(event_id_map) == 1:
        k, v = next(iter(event_id_map.items()))
        return v, k
    from collections import Counter
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for lbl, c in event_id_map.items():
        if c == code:
            return c, lbl
    return int(code), f"_code:{int(code)}"

def valid_events_for_window(raw: mne.io.BaseRaw, events: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    """Drop events that would clip the epoch window at the data boundary."""
    sf = raw.info["sfreq"]
    tmin_s = int(np.floor(tmin * sf))
    tmax_s = int(np.ceil(tmax * sf))
    n_times = raw.n_times
    valid_mask = []
    for s, _, _ in events:
        ok = (s + tmin_s >= 0) and (s + tmax_s < n_times)
        valid_mask.append(ok)
    valid_mask = np.array(valid_mask, dtype=bool)
    dropped = np.sum(~valid_mask)
    if dropped > 0:
        print(f"  [Safe epoching] Dropping {dropped} events that clip the window (kept {np.sum(valid_mask)}).")
    return events[valid_mask]

def derive_laplacian_from_arrays(data_map: Dict[str, np.ndarray], name: str) -> np.ndarray:
    base, neigh = CENTRAL_BASES[name]
    if base not in data_map:
        raise KeyError(f"Base {base} missing for {name}.")
    base_arr = data_map[base]
    neigh_avail = [data_map[k] for k in neigh if k in data_map]
    if len(neigh_avail) == 2:
        return base_arr - 0.5 * (neigh_avail[0] + neigh_avail[1])
    return base_arr.copy()

def derive_laplacian_continuous(raw: mne.io.BaseRaw, start: Optional[int] = None, stop: Optional[int] = None):
    if start is None: start = 0
    if stop  is None: stop  = raw.n_times
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError("Central/neighbor channels missing.")
    full = raw.get_data(picks=picks, start=start, stop=stop)
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
    if not out:
        raise RuntimeError("No Laplacian signals derived.")
    return np.vstack(out), out_names

def derive_laplacian_epochs(epochs_data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    out, out_names = [], []
    for lap_name in ["C3L", "CzL", "C4L"]:
        base, neigh = CENTRAL_BASES[lap_name]
        if base not in name_to_idx:
            continue
        base_idx = name_to_idx[base]
        neigh_idxs = [name_to_idx[k] for k in neigh if k in name_to_idx]
        if len(neigh_idxs) == 2:
            lap = epochs_data[:, base_idx, :] - 0.5 * (epochs_data[:, neigh_idxs[0], :] + epochs_data[:, neigh_idxs[1], :])
            out.append(lap); out_names.append(lap_name)
        else:
            out.append(epochs_data[:, base_idx, :]); out_names.append(base)
    if not out:
        raise RuntimeError("No Laplacian signals in epochs.")
    return np.stack(out, axis=1), out_names  # (N, C', T)

def pre_annotation_baseline_window(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    sf = raw.info["sfreq"]
    events, _ = get_events_any(raw)
    if len(events) > 0:
        first_samp = int(np.min(events[:, 0]))
        end = max(0, first_samp - int(round(margin_sec * sf)))
        start = 0
        if end - start < int(round(min_sec * sf)):
            end = min(int(round(min_sec * sf)), raw.n_times); start = 0
    else:
        start = 0; end = min(int(round(min_sec * sf)), raw.n_times)
    if end <= start:
        end = min(int(round(min_sec * sf)), raw.n_times)
    return start, end

def compute_baseline_stats_on_laplacian(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    start, end = pre_annotation_baseline_window(raw, min_sec, margin_sec)
    lap, lap_names = derive_laplacian_continuous(raw, start=start, stop=end)
    mean = lap.mean(axis=1)
    std = lap.std(axis=1) + EPS
    seconds = (end - start) / raw.info["sfreq"]
    return mean, std, seconds, lap_names

def apply_baseline_norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str):
    if arr.ndim == 2:
        return (arr - mean[:, None]) / std[:, None] if mode == "zscore" else (arr - mean[:, None])
    elif arr.ndim == 3:
        return (arr - mean[None, :, None]) / std[None, :, None] if mode == "zscore" else (arr - mean[None, :, None])
    raise ValueError("Unsupported array ndim.")

# ----- Envelope features -----

def _safe_filter_data(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    nyq = sfreq / 2.0
    h = min(h_freq, nyq - 0.5)
    if l_freq >= h:
        return data.copy()
    return filter_data(data.copy(), sfreq, l_freq=l_freq, h_freq=h, fir_design='firwin', verbose="ERROR")

def smooth_envelope(arr: np.ndarray, sfreq: float, smooth_sec: float) -> np.ndarray:
    k = max(1, int(round(smooth_sec * sfreq)))
    if k <= 1:
        return arr
    w = np.ones(k) / k
    return np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, arr)

def band_envelopes_logvar(epoch: np.ndarray, sfreq: float,
                          bands: List[Tuple[float, float]], smooth_sec: float) -> np.ndarray:
    feats = []
    for (l, h) in bands:
        X = _safe_filter_data(epoch, sfreq, l, h)
        env = np.abs(hilbert(X, axis=-1))
        env = smooth_envelope(env, sfreq, smooth_sec)
        v = np.var(env, axis=-1) + 1e-12
        feats.append(np.log(v))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0,), dtype=epoch.dtype)

def make_features_from_epochs(lap_epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """lap_epochs: (N, C, T) -> features per epoch: [TD(flat z) + EN(logvar by band&ch)]"""
    N, C, T = lap_epochs.shape
    X_list = []
    for i in range(N):
        ep = lap_epochs[i]
        mu = ep.mean(axis=1, keepdims=True)
        sd = ep.std(axis=1, keepdims=True) + 1e-8
        ep_z = (ep - mu) / sd
        td = ep_z.reshape(C * T)
        en = band_envelopes_logvar(ep, sfreq, ENV_BANDS, ENV_SMOOTH_SEC)
        X_list.append(np.concatenate([td, en], axis=0))
    return np.vstack(X_list)

# ----- ML helpers -----

def build_pipe(pca_max: int, n_features: int, n_samples: int, n_classes: int) -> Pipeline:
    pca_n = int(min(pca_max, n_features, max(2, n_samples - n_classes)))
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=pca_n, svd_solver="auto", random_state=0)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])

def set_lda_priors(pipe: Pipeline, priors: List[float]):
    # swap priors into the already-constructed pipeline's LDA
    lda = pipe.named_steps["lda"]
    pipe.named_steps["lda"] = LinearDiscriminantAnalysis(solver=lda.solver, priors=np.array(priors, dtype=float))

# ----- Smoothing -----

def majority_smooth(labels: List[str], win: int, include_unk: bool) -> List[str]:
    if win <= 1 or win % 2 == 0:
        return labels
    k = win // 2
    out = labels[:]
    for i in range(len(labels)):
        s = max(0, i - k)
        e = min(len(labels), i + k + 1)
        window = labels[s:e]
        if not include_unk:
            window = [x for x in window if x != "UNK"]
            if not window:
                continue
        counts = Counter(window)
        # if ties, keep original
        best = max(counts.items(), key=lambda kv: (kv[1], -abs((s+e)//2 - i)))[0]
        out[i] = best
    return out

# ----- Save -----

def save_sequence(labels: List[str], probs: List[float], outdir: str, prefix: str):
    ensure_outdir(outdir)
    txt = " ".join(labels)
    txt_path = os.path.join(outdir, f"{prefix}_sequence.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    csv_path = os.path.join(outdir, f"{prefix}_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,label,max_proba\n")
        for i, (l, p) in enumerate(zip(labels, probs)):
            f.write(f"{i},{l},{p:.6f}\n")
    print("\n=== Detected Sequence (Hier-PCA+LDA, trigger-locked) ===")
    print(txt)
    print(f"\n[Saved sequence] {txt_path}")
    print(f"[Saved table]    {csv_path}")

# -----------------------
# -------- MAIN ---------
# -----------------------

def main():
    ensure_outdir(OUTPUT_DIR)

    # -------- Build training features --------
    X_all, y_all = [], []
    sfreq_ref = None
    ref_order = None

    print("[Training] Building features from registrations (A,B,C,D) ...")
    for lbl in tqdm(["A", "B", "C", "D"], total=4, desc="[Build Train]"):
        fpath = REG_FILES[lbl]
        raw = load_raw(fpath, RESAMPLE_HZ)
        sf = raw.info["sfreq"] if RESAMPLE_HZ is None else RESAMPLE_HZ
        if sfreq_ref is None:
            sfreq_ref = sf
        else:
            assert abs(sfreq_ref - sf) < 1e-6, "All files must share the same sampling rate (set RESAMPLE_HZ)."

        events, eid = get_events_any(raw)
        code, chosen = resolve_event_auto(events, eid, requested=None)
        events = valid_events_for_window(raw, events[events[:, 2] == code], TMIN, TMAX)

        # Picks for Laplacians
        names_needed = set()
        for base, neigh in CENTRAL_BASES.values():
            names_needed.add(base); names_needed.update(neigh)
        picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
        if len(picks) == 0:
            raise RuntimeError(f"[{lbl}] Central/neighbor channels missing in {os.path.basename(fpath)}.")

        epochs = mne.Epochs(raw, events, event_id={chosen: code}, tmin=TMIN, tmax=TMAX,
                            baseline=None, picks=picks, preload=True, detrend=0,
                            reject_by_annotation=False, verbose="ERROR")
        if len(epochs) == 0:
            raise RuntimeError(f"No epochs found in {os.path.basename(fpath)} for '{chosen}'.")

        data = epochs.get_data()
        ch_names = [epochs.ch_names[i] for i in range(len(epochs.ch_names))]
        lap_ep, lap_names = derive_laplacian_epochs(data, ch_names)

        # Baseline on Laplacians
        if BASELINE_USE_PRE_ANNOT:
            bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
            if lap_names_b != lap_names:
                idx_map = {n: i for i, n in enumerate(lap_names_b)}
                idx = [idx_map[n] for n in lap_names if n in idx_map]
                bmean, bstd = bmean[idx], bstd[idx]
            lap_ep = apply_baseline_norm(lap_ep, bmean, bstd, BASELINE_MODE)

        # Reference Lap order
        if ref_order is None:
            ref_order = lap_names
        else:
            idx_map = {n: i for i, n in enumerate(lap_names)}
            idx = [idx_map[n] for n in ref_order if n in idx_map]
            lap_ep = lap_ep[:, idx, :]

        X = make_features_from_epochs(lap_ep, sf)
        y = np.array([lbl] * X.shape[0])
        X_all.append(X); y_all.append(y)

    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    n_samples, n_features = X_train.shape

    # Split training by groups
    idx_AB = np.isin(y_train, ["A", "B"])
    idx_CD = np.isin(y_train, ["C", "D"])

    # Pipelines
    pipe_gate = build_pipe(PCA_MAX_COMPONENTS_GATE, n_features, n_samples, 2)
    set_lda_priors(pipe_gate, [GATE_PRIOR_AB, 1.0 - GATE_PRIOR_AB])  # ['AB','CD'] order we define below

    pipe_AB = build_pipe(PCA_MAX_COMPONENTS_AB, n_features, np.sum(idx_AB), 2)
    pipe_CD = build_pipe(PCA_MAX_COMPONENTS_CD, n_features, np.sum(idx_CD), 2)

    # Build labels for gate
    y_gate = np.where(np.isin(y_train, ["A", "B"]), "AB", "CD")

    # CV summaries
    print(f"\n[CV] Gate ABvsCD (priors AB={GATE_PRIOR_AB:.2f}, CD={1-GATE_PRIOR_AB:.2f}) ...")
    cv = StratifiedKFold(n_splits=min(N_FOLDS_CV, np.bincount([0 if z=='AB' else 1 for z in y_gate]).min()), shuffle=True, random_state=0)
    gate_scores = cross_val_score(pipe_gate, X_train, y_gate, cv=cv)
    print(f"  • Gate CV acc: {gate_scores.mean():.3f} (± {gate_scores.std():.3f})")

    print("[CV] A vs B ...")
    if np.sum(idx_AB) >= 4:
        cv_ab = StratifiedKFold(n_splits=min(N_FOLDS_CV, min(np.bincount([0 if z=='A' else 1 for z in y_train[idx_AB]]))), shuffle=True, random_state=0)
        ab_scores = cross_val_score(pipe_AB, X_train[idx_AB], y_train[idx_AB], cv=cv_ab)
        print(f"  • AB CV acc: {ab_scores.mean():.3f} (± {ab_scores.std():.3f})")
    else:
        print("  • Not enough AB samples for CV.")

    print("[CV] C vs D ...")
    if np.sum(idx_CD) >= 4:
        cv_cd = StratifiedKFold(n_splits=min(N_FOLDS_CV, min(np.bincount([0 if z=='C' else 1 for z in y_train[idx_CD]]))), shuffle=True, random_state=0)
        cd_scores = cross_val_score(pipe_CD, X_train[idx_CD], y_train[idx_CD], cv=cv_cd)
        print(f"  • CD CV acc: {cd_scores.mean():.3f} (± {cd_scores.std():.3f})")
    else:
        print("  • Not enough CD samples for CV.")

    # Fit all on full data
    pipe_gate.fit(X_train, y_gate)
    pipe_AB.fit(X_train[idx_AB], y_train[idx_AB])
    if ALLOW_CD and np.sum(idx_CD) >= 2:
        pipe_CD.fit(X_train[idx_CD], y_train[idx_CD])

    # -------- Target classification --------
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw_t = load_raw(TARGET_FILE, RESAMPLE_HZ)
    sf = raw_t.info["sfreq"]
    events_all, eid_all = get_events_any(raw_t)
    code_t, lbl_t = resolve_event_auto(events_all, eid_all, requested=TARGET_EVENT)
    events_t = valid_events_for_window(raw_t, events_all[events_all[:, 2] == code_t], TMIN, TMAX)
    print(f"[Target] Using event '{lbl_t}' (code {code_t}); N={len(events_t)}")

    # Build Lap epochs and baseline
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks_t = mne.pick_channels(raw_t.info["ch_names"], include=list(names_needed))
    if len(picks_t) == 0:
        raise RuntimeError("Target: central/neighbor channels missing.")

    epochs_t = mne.Epochs(raw_t, events_t, event_id={lbl_t: code_t}, tmin=TMIN, tmax=TMAX,
                          baseline=None, picks=picks_t, preload=True, detrend=0,
                          reject_by_annotation=False, verbose="ERROR")
    data_t = epochs_t.get_data()
    ch_names_t = [epochs_t.ch_names[i] for i in range(len(epochs_t.ch_names))]
    lap_t, lap_names_t = derive_laplacian_epochs(data_t, ch_names_t)

    # Baseline on target Laplacians
    if BASELINE_USE_PRE_ANNOT:
        bmean_t, bstd_t, secs_t, lap_names_b_t = compute_baseline_stats_on_laplacian(raw_t, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_b_t != lap_names_t:
            idx_map = {n: i for i, n in enumerate(lap_names_b_t)}
            idx = [idx_map[n] for n in lap_names_t if n in idx_map]
            bmean_t, bstd_t = bmean_t[idx], bstd_t[idx]
        lap_t = apply_baseline_norm(lap_t, bmean_t, bstd_t, BASELINE_MODE)

    # Reorder target Lap order to training ref
    if lap_names_t != ref_order:
        idx_map = {n: i for i, n in enumerate(lap_names_t)}
        idx = [idx_map[n] for n in ref_order if n in idx_map]
        lap_t = lap_t[:, idx, :]

    # Extract features for target
    print("\n[Predict] Extracting features & running hierarchical classification ...")
    X_target = []
    for i in tqdm(range(lap_t.shape[0]), total=lap_t.shape[0], desc="[Features]"):
        X_target.append(make_features_from_epochs(lap_t[i:i+1], sf)[0])
    X_target = np.vstack(X_target)

    # Gate AB vs CD
    proba_gate = pipe_gate.predict_proba(X_target)  # columns correspond to pipe_gate.classes_
    # We define order to map probabilities: find indices
    classes_gate = pipe_gate.classes_.tolist()  # e.g., ['AB','CD'] (order guaranteed by training labels)
    idx_ABc = classes_gate.index("AB")
    idx_CDc = classes_gate.index("CD")

    pred_final = []
    max_probs  = []

    for i in tqdm(range(X_target.shape[0]), total=X_target.shape[0], desc="[Classify]"):
        p_ab = proba_gate[i, idx_ABc]
        p_cd = proba_gate[i, idx_CDc]
        group = "UNK"
        p_group = max(p_ab, p_cd)

        if p_ab >= GATE_PROBA_MIN:
            group = "AB"
        elif ALLOW_CD and (p_cd >= GATE_PROBA_MIN):
            group = "CD"
        else:
            pred_final.append("UNK"); max_probs.append(p_group); continue

        # Within-group decision
        if group == "AB":
            proba_sub = pipe_AB.predict_proba(X_target[i:i+1])[0]
            # align class indices
            classes_sub = pipe_AB.classes_.tolist()  # ['A','B'] order
            jmax = int(np.argmax(proba_sub))
            if proba_sub[jmax] >= SUB_PROBA_MIN:
                pred_final.append(classes_sub[jmax]); max_probs.append(proba_sub[jmax])
            else:
                pred_final.append("UNK"); max_probs.append(proba_sub[jmax])
        elif group == "CD":
            if not ALLOW_CD:
                pred_final.append("UNK"); max_probs.append(p_group); continue
            proba_sub = pipe_CD.predict_proba(X_target[i:i+1])[0]
            classes_sub = pipe_CD.classes_.tolist()  # ['C','D'] order
            jmax = int(np.argmax(proba_sub))
            if proba_sub[jmax] >= SUB_PROBA_MIN:
                pred_final.append(classes_sub[jmax]); max_probs.append(proba_sub[jmax])
            else:
                pred_final.append("UNK"); max_probs.append(proba_sub[jmax])

    # Temporal smoothing
    if SMOOTH_WIN > 1:
        pred_sm = majority_smooth(pred_final, SMOOTH_WIN, include_unk=SMOOTH_INCLUDE_UNK)
    else:
        pred_sm = pred_final

    # Report & save
    counts = Counter(pred_sm)
    print("\n[Counts after smoothing]")
    for k in ["A", "B", "C", "D", "UNK"]:
        if counts.get(k, 0) > 0:
            print(f"  {k}: {counts[k]}")
    if SAVE_TXT_CSV:
        save_sequence(pred_sm, max_probs, OUTPUT_DIR, prefix="hier_pca_lda_movement_mix")

    print("\nDone.")

if __name__ == "__main__":
    main()

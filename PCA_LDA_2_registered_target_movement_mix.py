# -*- coding: utf-8 -*-
"""
PCA+LDA (A/B only) with ERD/ERS features + wide jitter scan

- Templates from two registrations: A (fist close) and B (pinch).
- Features: per-channel mu/beta Hilbert-envelope log-variance (ERD/ERS-like).
- Classifier: StandardScaler -> PCA (auto) -> LDA (lsqr, shrinkage='auto').
- Trigger-locked detection on target, but with wide +/- jitter search.
- Prints detected sequence and accuracy vs your provided ground-truth string.

Requires: mne, numpy, scipy, scikit-learn, tqdm, matplotlib (optional)
"""

import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import mne
from scipy.signal import hilbert
from mne.filter import filter_data
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import csv

# =======================
# ====== CONFIG =========
# =======================

# -- YOUR FILES HERE --
REG_A = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif"
REG_B = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif"
TARGET = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Epoch window relative to trigger (seconds)
TMIN = 0.0
TMAX = 0.6

# Wide earlier search for best-discriminative segment (seconds)
SEARCH_JITTER_SEC = 0.60  # try 0.60–0.80 if still pegged early

# Envelope (ERD/ERS) bands and smoothing
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]  # mu, beta
ENV_SMOOTH_SEC = 0.050  # ~50 ms moving-average smoothing

# Optional: mark low-confidence as UNK (set to None to disable)
MIN_CONF: Optional[float] = None  # e.g., 0.10

# Fixed resample rate (set None to keep native; ERD features are length-invariant)
RESAMPLE_HZ: Optional[float] = None  # e.g., 250.0

# Output dir
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_pca_lda_ab_output"

# Ground-truth sequence (your A/B truth for movement-mix)
GROUND_TRUTH_STR = """
B A A A B B B A A A A A B B B B B B B B B A A A A A A B B B A A B B B B A A A A A A B B B B B B B B B A A A A A A A A A A A B B B B B B B B B B B B B B A A A A A A B B B A A A A A A A A A A A B B B B B B A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B A A A A A A A A A A B B B B B B B B B B A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B
"""

# =======================
# ====== UTILS ==========
# =======================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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
    codes = np.unique(events[:, 2])
    event_id = {f"EVT/{int(c)}": int(c) for c in codes}
    return events, event_id

def resolve_most_frequent_event(events, event_id_map):
    """Return (event_code, label) of most frequent code."""
    if len(event_id_map) == 0:
        raise RuntimeError("No events/annotations found.")
    from collections import Counter
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for lbl, c in event_id_map.items():
        if c == code:
            return c, lbl
    return int(code), f"_code:{int(code)}"

# ---------- Laplacian (C3L, CzL, C4L) ----------

CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}

def _derive_laplacian_from_arrays(data_map: Dict[str, np.ndarray], name: str) -> np.ndarray:
    base, neigh = CENTRAL_BASES[name]
    if base not in data_map:
        raise KeyError(f"Base channel {base} missing for {name}.")
    base_arr = data_map[base]
    neigh_avail = [data_map[k] for k in neigh if k in data_map]
    if len(neigh_avail) == 2:
        return base_arr - 0.5 * (neigh_avail[0] + neigh_avail[1])
    return base_arr.copy()  # fallback

def derive_laplacian_epochs(epochs_data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """epochs_data: (N,C,T) -> (N,C',T) for C3L, CzL, C4L (fallback if neighbors missing)."""
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
    if len(out) == 0:
        raise RuntimeError("No C3/Cz/C4(-neighbors) present.")
    return np.stack(out, axis=1), out_names  # (N, C', T)

# ---------- Baseline window helper (optional; not used in features) ----------

def safe_epoch(raw: mne.io.BaseRaw, events: np.ndarray, code: int,
               tmin: float, tmax: float, picks) -> mne.Epochs:
    """Epochs that drop clipped events automatically."""
    ep = mne.Epochs(raw, events[events[:, 2] == code], event_id={"X": code},
                    tmin=tmin, tmax=tmax, baseline=None, picks=picks,
                    preload=True, detrend=0, verbose="ERROR", on_missing='ignore')
    # Epochs already drops clipped events; we can log how many remain:
    return ep

# ---------- ERD/ERS feature builder ----------

def _safe_bandpass(x: np.ndarray, sf: float, l: float, h: float) -> np.ndarray:
    nyq = sf / 2.0
    h = min(h, nyq - 0.5)
    if l >= h:
        return x.copy()
    return filter_data(x.copy(), sfreq=sf, l_freq=l, h_freq=h, fir_design='firwin', verbose='ERROR')

def _smooth_env(env: np.ndarray, sf: float, sec: float) -> np.ndarray:
    k = max(1, int(round(sec * sf)))
    if k <= 1:
        return env
    w = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, w, mode='same'), -1, env)

def make_features(lap_epochs: np.ndarray, sf: float) -> np.ndarray:
    """
    ERD/ERS features per epoch:
      for each band (mu, beta) and each Laplacian channel -> log-variance of the smoothed Hilbert envelope.
    lap_epochs: (N, C, T)
    returns: (N, C * len(ENV_BANDS))
    """
    N, C, T = lap_epochs.shape
    feats = []
    for (l, h) in ENV_BANDS:
        X = np.empty_like(lap_epochs)
        for n in range(N):
            for c in range(C):
                X[n, c] = _safe_bandpass(lap_epochs[n, c], sf, l, h)
        env = np.abs(hilbert(X, axis=-1))
        env = _smooth_env(env, sf, ENV_SMOOTH_SEC)
        v = np.log(np.var(env, axis=-1) + 1e-12)  # (N, C)
        feats.append(v)
    F = np.hstack(feats)  # (N, C*bands)
    return F

# ---------- Dataset builders ----------

def build_train_AB(reg_a_path: str, reg_b_path: str, resample_hz: Optional[float]):
    raws = {}
    for k, p in {"A": reg_a_path, "B": reg_b_path}.items():
        raws[k] = load_raw(p, resample_hz)

    # Pick C3/Cz/C4 + neighbors
    need = set()
    for base, neigh in CENTRAL_BASES.values():
        need.add(base); need.update(neigh)
    picks_a = mne.pick_channels(raws["A"].info["ch_names"], include=list(need))
    picks_b = mne.pick_channels(raws["B"].info["ch_names"], include=list(need))
    if len(picks_a) == 0 or len(picks_b) == 0:
        raise RuntimeError("Missing central/neighbor channels in registrations.")

    # Events / choose most frequent code in each registration (usually the same)
    ev_a, eid_a = get_events_any(raws["A"])
    ev_b, eid_b = get_events_any(raws["B"])
    code_a, lbl_a = resolve_most_frequent_event(ev_a, eid_a)
    code_b, lbl_b = resolve_most_frequent_event(ev_b, eid_b)

    ep_a = safe_epoch(raws["A"], ev_a, code_a, TMIN, TMAX, picks=picks_a)
    ep_b = safe_epoch(raws["B"], ev_b, code_b, TMIN, TMAX, picks=picks_b)
    if len(ep_a) == 0 or len(ep_b) == 0:
        raise RuntimeError("No epochs in A or B.")

    # Derive Laplacian per epoch
    X_a_raw = ep_a.get_data()  # (Na, K, T)
    X_b_raw = ep_b.get_data()

    ch_a = [ep_a.ch_names[i] for i in range(len(ep_a.ch_names))]
    ch_b = [ep_b.ch_names[i] for i in range(len(ep_b.ch_names))]

    lap_a, lap_names_a = derive_laplacian_epochs(X_a_raw, ch_a)  # (Na,C',T)
    lap_b, lap_names_b = derive_laplacian_epochs(X_b_raw, ch_b)

    sf_a = ep_a.info["sfreq"]; sf_b = ep_b.info["sfreq"]
    if abs(sf_a - sf_b) > 1e-6:
        raise RuntimeError(f"Sampling rates differ A={sf_a} vs B={sf_b}. Resample both to RESAMPLE_HZ.")

    # If Lap channel sets differ (fallback bases), keep common subset & reorder to C3L,CzL,C4L preference
    def reorder_to_common(lap_a, names_a, lap_b, names_b):
        pref = ["C3L", "CzL", "C4L", "C3", "Cz", "C4"]
        common = [n for n in pref if n in names_a and n in names_b]
        if len(common) == 0:
            raise RuntimeError("No common central/Lap channels between A and B.")
        idx_a = [names_a.index(n) for n in common]
        idx_b = [names_b.index(n) for n in common]
        return lap_a[:, idx_a, :], lap_b[:, idx_b, :], common
    lap_a, lap_b, lap_names = reorder_to_common(lap_a, lap_names_a, lap_b, lap_names_b)

    # Build ERD/ERS features
    sf = float(sf_a)
    Feat_A = make_features(lap_a, sf=sf)
    Feat_B = make_features(lap_b, sf=sf)

    # Balance classes (downsample bigger one)
    nA, nB = Feat_A.shape[0], Feat_B.shape[0]
    if nA != nB:
        if nA > nB:
            idx = np.random.RandomState(7).choice(nA, size=nB, replace=False)
            Feat_A = Feat_A[idx]
            nA = nB
        else:
            idx = np.random.RandomState(7).choice(nB, size=nA, replace=False)
            Feat_B = Feat_B[idx]
            nB = nA

    X = np.vstack([Feat_A, Feat_B])
    y = np.array(["A"] * nA + ["B"] * nB)

    return X, y, sf, lap_names, int(round((TMAX - TMIN) * sf))

def build_target_epochs(raw_path: str, sf_train: float, lap_names_ref: List[str],
                        resample_hz: Optional[float]):
    raw = load_raw(raw_path, resample_hz)
    if abs(raw.info["sfreq"] - sf_train) > 1e-6:
        raise RuntimeError(f"Target sf={raw.info['sfreq']} differs from train sf={sf_train}. "
                           f"Set RESAMPLE_HZ to a fixed value for all files.")
    events, eid = get_events_any(raw)
    code, lbl = resolve_most_frequent_event(events, eid)

    # Picks for lap
    need = set()
    for base, neigh in CENTRAL_BASES.values():
        need.add(base); need.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(need))
    if len(picks) == 0:
        raise RuntimeError("Missing central/neighbor channels in target.")

    # Safe epoching (drops clipped automatically)
    ep = safe_epoch(raw, events, code, TMIN - SEARCH_JITTER_SEC, TMAX + SEARCH_JITTER_SEC, picks=picks)
    X_raw = ep.get_data()  # (N, K, Text)
    ch = [ep.ch_names[i] for i in range(len(ep.ch_names))]
    lap, lap_names_t = derive_laplacian_epochs(X_raw, ch)  # (N, C', Text)

    # Reorder/Subset to ref Lap names
    name_to_idx = {n: i for i, n in enumerate(lap_names_t)}
    idx = [name_to_idx[n] for n in lap_names_ref if n in name_to_idx]
    lap = lap[:, idx, :]
    lap_names = [lap_names_ref[i] for i in range(len(idx))]

    return ep.events[:, 0], lap, lap_names, raw.info["sfreq"]

# ---------- Classifier ----------

def auto_pca_dim(X: np.ndarray, min_k: int = 2, max_k: Optional[int] = None) -> int:
    # For ERD features, dims are small (C*bands). Use min(#feats, 6) unless feats<2.
    d = X.shape[1]
    if max_k is None:
        max_k = d
    return max(min_k, min(d, 6))

def make_pipeline(n_pca: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_pca, svd_solver="auto", random_state=7)),
        ("lda", LDA(solver="lsqr", shrinkage="auto"))
    ])

# ---------- Prediction with jitter ----------

def predict_sequence_with_jitter(lap_ext: np.ndarray, sf: float, seg_len: int,
                                 pipe: Pipeline) -> Tuple[List[str], List[float], List[int]]:
    """
    lap_ext: (N, C, T_ext) epochs already extracted on [TMIN-τ, TMAX+τ]
    seg_len: length corresponding to (TMAX-TMIN) in samples
    Returns: labels, max_proba, chosen_shift_samples
    """
    N, C, T_ext = lap_ext.shape
    jitter_samp = int(round(SEARCH_JITTER_SEC * sf))
    nominal_start = jitter_samp  # segment nominally starts after τ seconds
    labels, probs, shifts = [], [], []

    for n in tqdm(range(N), desc="[Predict] Classifying (jitter)", leave=False):
        best_lab, best_p, best_shift = "UNK", -1.0, 0
        # sweep start s from 0..(2*jitter)
        for s in range(0, 2 * nominal_start + 1):
            start = s
            end = start + seg_len
            if end > T_ext:
                break
            seg = lap_ext[n, :, start:end][np.newaxis, :, :]  # (1,C,seg_len)
            X = make_features(seg, sf=sf)  # (1, C*bands)
            proba = pipe.predict_proba(X)[0]  # [p(A), p(B)]
            idx = int(np.argmax(proba))
            lab = "A" if idx == 0 else "B"
            pmax = float(proba[idx])

            if pmax > best_p:
                best_p = pmax
                best_lab = lab
                best_shift = start - nominal_start  # negative = earlier than nominal

        if MIN_CONF is not None and best_p < MIN_CONF:
            best_lab = "UNK"

        labels.append(best_lab)
        probs.append(best_p)
        shifts.append(best_shift)

    return labels, probs, shifts

# ---------- Accuracy vs GT ----------

def parse_labels_string(s: str) -> List[str]:
    return [ch for ch in s.replace("\n", " ").split() if ch in ("A", "B")]

def accuracy_report(pred: List[str], gt_str: str, shift_window: int = 5):
    gt = parse_labels_string(gt_str)
    n_pred, n_gt = len(pred), len(gt)
    n_cmp = min(n_pred, n_gt)
    if n_cmp == 0:
        print("[Accuracy] No comparable length (zero).")
        return

    def acc_at_shift(sh: int) -> float:
        # compare pred[t] vs gt[t+sh], only when both indices valid
        if sh >= 0:
            L = min(n_cmp, n_gt - sh, n_pred)
            if L <= 0: return float("nan")
            ok = sum(1 for i in range(L) if pred[i] == gt[i + sh])
            return 100.0 * ok / L
        else:
            sh = -sh
            L = min(n_cmp, n_pred - sh, n_gt)
            if L <= 0: return float("nan")
            ok = sum(1 for i in range(L) if pred[i + sh] == gt[i])
            return 100.0 * ok / L

    acc0 = acc_at_shift(0)
    best_sh, best_acc = 0, acc0
    for sh in range(-shift_window, shift_window + 1):
        a = acc_at_shift(sh)
        if a == a and a > best_acc:  # a==a to check not NaN
            best_sh, best_acc = sh, a

    print("\n[Accuracy vs your fixed ground truth]")
    print(f"  Lengths -> pred={n_pred}, gt={n_gt}  (compared on ~{min(n_pred, n_gt)} aligned samples)")
    print(f"  No shift: {acc0:.2f}%")
    print(f"  Best shift in [-{shift_window}..+{shift_window}]: shift={best_sh}, acc={best_acc:.2f}%")

# =======================
# ====== MAIN ===========
# =======================

def main():
    ensure_dir(OUT_DIR)

    # ------ Train on registrations ------
    print("[Training] Building A/B features from registrations ...")
    X, y, sf, lap_names, seg_len = build_train_AB(REG_A, REG_B, RESAMPLE_HZ)
    print(f"[Train set] X={X.shape}, A={np.sum(y=='A')}, B={np.sum(y=='B')}")
    print(f"[Info] Lap channels used: {lap_names} | seg_len={seg_len} samples @ sf={sf:.2f} Hz")
    # Auto PCA dim for small ERD feature space
    n_pca = auto_pca_dim(X)
    print(f"[Auto PCA] n_components={n_pca}")
    pipe = make_pipeline(n_pca)

    # CV (stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"[CV] PCA({n_pca})+LDA(shrinkage) accuracy: {np.mean(scores):.3f} (± {np.std(scores):.3f})")
    pipe.fit(X, y)

    # ------ Target epochs with extended window ------
    print("\n[Target] Loading target & building extended epochs ...")
    ev_samps, lap_ext, lap_names_t, sf_t = build_target_epochs(TARGET, sf, lap_names, RESAMPLE_HZ)
    print(f"[Target] Events kept: N={lap_ext.shape[0]} | ext_len={lap_ext.shape[-1]} | sf={sf_t:.2f} Hz")
    print(f"[Predict] Jitter scan: ±{int(round(SEARCH_JITTER_SEC*sf))} samples "
          f"(±{SEARCH_JITTER_SEC:.2f}s) | seg_len={seg_len} | T_ext={lap_ext.shape[-1]}")

    # ------ Predict with jitter ------
    labels, probs, shifts = predict_sequence_with_jitter(lap_ext, sf, seg_len, pipe)

    # ------ Save + report ------
    txt_path = os.path.join(OUT_DIR, "pca_lda_ab_sequence.txt")
    csv_path = os.path.join(OUT_DIR, "pca_lda_ab_sequence.csv")

    with open(txt_path, "w") as f:
        f.write(" ".join(labels) + "\n")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pmax", "shift_samples"])
        for i, (lab, p, sh) in enumerate(zip(labels, probs, shifts)):
            w.writerow([i, lab, f"{p:.3f}", sh])

    # counts
    from collections import Counter
    cnt = Counter(labels)
    print("\n[Counts]")
    for k in ["A", "B", "UNK"]:
        if cnt.get(k, 0) > 0:
            print(f"  {k}: {cnt[k]}")

    print("\n=== Detected Sequence (PCA+LDA, A/B only, trigger-locked) ===")
    print(" ".join(labels))

    print(f"\n[Saved sequence] {txt_path}")
    print(f"[Saved table]    {csv_path}")

    # shifts
    if len(shifts):
        print(f"[Shift stats] mean={np.mean(shifts):.1f} samp, std={np.std(shifts):.1f}, "
              f"min={np.min(shifts)}, max={np.max(shifts)}")

    # Accuracy vs ground truth
    accuracy_report(labels, GROUND_TRUTH_STR, shift_window=5)


if __name__ == "__main__":
    main()

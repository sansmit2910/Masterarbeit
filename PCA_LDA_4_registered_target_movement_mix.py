# -*- coding: utf-8 -*-
"""
PCA + LDA EEG Motif Classifier (A,B,C,D → movement-mix target)
--------------------------------------------------------------
- Registration datasets: 4 .fif files (A,B,C,D), single trigger type.
- Features from virtual Laplacians (C3L, CzL, C4L):
    • Time-domain (flattened epoch)
    • Mu/Beta Hilbert envelopes (log-variance per band & channel)
- Shared baseline (pre-first-event) computed on Laplacians of each file.
- Pipeline: StandardScaler → PCA → LDA (multi-class)
- Cross-validated training accuracy + confusion matrix
- Predicts per-trigger labels on the target movement-mix file.
- Saves sequence as TXT + CSV.

Deps: mne, numpy, scipy, matplotlib, scikit-learn, tqdm
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from mne.filter import filter_data

from tqdm import tqdm

# ---- ML ----
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import confusion_matrix, accuracy_score
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. Install with:\n"
        "    pip install scikit-learn\n"
        f"Original import error: {e}"
    )

# =======================
# ====== CONFIG =========
# =======================

DATA_DIR = r"C:\Users\User\Desktop\masters_thesis_directory"

TEMPLATE_SOURCES: Dict[str, str] = {
    "A": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-1-reg.fif"),
    "B": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-2-reg.fif"),
    "C": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-imagine-move-1_raw.fif"),
    "D": os.path.join(DATA_DIR, "preprocessed_raw_sub-001-imagine-move-2_raw.fif"),
}

TARGET_FILE = os.path.join(DATA_DIR, "preprocessed_raw_sub-001-movement-mix_raw.fif")
TARGET_EVENT: Optional[Union[int, str]] = None  # None → auto-pick most frequent (should be S1)

# Epoch window (sec)
TMIN = 0.0
TMAX = 0.6

# Optional resampling to unify T across files (None = keep native)
RESAMPLE_HZ: Optional[float] = 200.0

# Baseline (on derived Laplacians) from BEFORE first event
BASELINE_USE_PRE_ANNOT = True
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
BASELINE_MODE = "zscore"      # 'zscore' or 'demean'
EPS = 1e-12

# Laplacians to derive (with graceful fallback if neighbors missing)
CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}

# Envelope bands (mu, beta)
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]
ENV_SMOOTH_SEC = 0.05

# PCA + LDA
PCA_MAX_COMPONENTS = 40          # cap; actual will be min(this, n_features, n_samples-classes)
N_FOLDS_CV = 5                   # cross-val folds on training epochs

# Decision threshold (optional UNK)
USE_PROBA_THRESHOLD = False      # set True → low-confidence → UNK
PROBA_THRESHOLD = 0.40

# Visualization / output
OUTPUT_DIR = os.path.join(DATA_DIR, "eeg_pca_lda_output")
SAVE_PLOTS = True
SHOW_PLOTS = True

# =======================
# ====== UTILS ==========
# =======================

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
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for lbl, c in event_id_map.items():
        if c == code:
            return c, lbl
    return int(code), f"_code:{int(code)}"

def derive_laplacian_from_arrays(data_map: Dict[str, np.ndarray], name: str) -> np.ndarray:
    base, neigh = CENTRAL_BASES[name]
    if base not in data_map:
        raise KeyError(f"Base {base} missing for {name}.")
    base_arr = data_map[base]
    neigh_avail = [data_map[k] for k in neigh if k in data_map]
    if len(neigh_avail) == 2:
        return base_arr - 0.5 * (neigh_avail[0] + neigh_avail[1])
    else:
        return base_arr.copy()

def derive_laplacian_continuous(raw: mne.io.BaseRaw,
                                start: Optional[int] = None,
                                stop: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    if start is None: start = 0
    if stop  is None: stop  = raw.n_times

    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError("Required central/neighbor channels are missing.")

    full = raw.get_data(picks=picks, start=start, stop=stop)  # (K,T)
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
        raise RuntimeError("No Laplacian signals could be derived.")
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
        raise RuntimeError("No Laplacian signals available in epochs.")
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

def compute_baseline_stats_on_laplacian(raw: mne.io.BaseRaw,
                                        min_sec: float, margin_sec: float):
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
    else:
        raise ValueError("Unsupported array ndim")

# ---------- Envelope features ----------

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
    """
    epoch: (C, T) Laplacian data
    returns: (C * len(bands),) log-variance of smoothed Hilbert envelope per band/channel
    """
    feats = []
    for (l, h) in bands:
        X = _safe_filter_data(epoch, sfreq, l, h)  # (C,T)
        env = np.abs(hilbert(X, axis=-1))
        env = smooth_envelope(env, sfreq, smooth_sec)
        v = np.var(env, axis=-1) + 1e-12
        feats.append(np.log(v))
    return np.concatenate(feats, axis=0) if feats else np.zeros((0,), dtype=epoch.dtype)

# ---------- Feature extraction ----------

def make_features_from_epochs(lap_epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """
    lap_epochs: (N, C, T)
    Returns X: (N, F) combining:
      - flattened time-domain (C*T) after per-epoch, per-channel z-score
      - envelope log-var features per band and channel
    """
    N, C, T = lap_epochs.shape
    X_list = []
    for i in range(N):
        ep = lap_epochs[i]  # (C,T)
        # per-channel z-score over time
        mu = ep.mean(axis=1, keepdims=True)
        sd = ep.std(axis=1, keepdims=True) + 1e-8
        ep_z = (ep - mu) / sd
        td = ep_z.reshape(C * T)  # time-domain vector
        en = band_envelopes_logvar(ep, sfreq, ENV_BANDS, ENV_SMOOTH_SEC)
        X_list.append(np.concatenate([td, en], axis=0))
    return np.vstack(X_list)

# ---------- Confusion matrix plot ----------

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, outpath: str, show: bool):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    if show: plt.show()
    else: plt.close(fig)

# ---------- Save sequence ----------

def save_sequence(labels: List[str], probs_max: List[float], outdir: str, prefix: str):
    ensure_outdir(outdir)
    txt = " ".join(labels)
    txt_path = os.path.join(outdir, f"{prefix}_sequence.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    csv_path = os.path.join(outdir, f"{prefix}_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,label,max_proba\n")
        for i, (l, p) in enumerate(zip(labels, probs_max)):
            f.write(f"{i},{l},{p:.6f}\n")
    print("\n=== Detected Sequence (PCA+LDA, trigger-locked) ===")
    print(txt)
    print(f"\n[Saved sequence] {txt_path}")
    print(f"[Saved table]    {csv_path}")

# =======================
# ====== MAIN ===========
# =======================

def main():
    ensure_outdir(OUTPUT_DIR)

    # ---------- Build training set ----------
    all_X = []
    all_y = []
    sfreq_ref = None
    ref_order = None

    print("[Training] Building features from registrations (A,B,C,D) ...")
    for lbl in tqdm(["A", "B", "C", "D"], total=4, desc="[Build Train]"):
        fpath = TEMPLATE_SOURCES[lbl]
        raw = load_raw(fpath, RESAMPLE_HZ)
        sf = raw.info["sfreq"] if RESAMPLE_HZ is None else RESAMPLE_HZ
        if sfreq_ref is None:
            sfreq_ref = sf
        else:
            assert abs(sfreq_ref - sf) < 1e-6, "All files must share the same sampling rate (set RESAMPLE_HZ)."

        events, eid = get_events_any(raw)
        code, chosen = resolve_event_auto(events, eid, requested=None)

        # Build epochs for central + neighbors
        names_needed = set()
        for base, neigh in CENTRAL_BASES.values():
            names_needed.add(base); names_needed.update(neigh)
        picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
        if len(picks) == 0:
            raise RuntimeError(f"[{lbl}] Required central/neighbor channels missing in {os.path.basename(fpath)}.")

        epochs = mne.Epochs(raw, events[events[:, 2] == code],
                            event_id={chosen: code}, tmin=TMIN, tmax=TMAX,
                            baseline=None, picks=picks, preload=True, detrend=0, verbose="ERROR")
        if len(epochs) == 0:
            raise RuntimeError(f"No epochs found in {os.path.basename(fpath)} for event '{chosen}'.")

        data = epochs.get_data()  # (N,K,T)
        ch_names = [epochs.ch_names[i] for i in range(len(epochs.ch_names))]
        lap_epochs, lap_names = derive_laplacian_epochs(data, ch_names)  # (N, C', T)

        # Shared baseline per file on Laplacians
        if BASELINE_USE_PRE_ANNOT:
            bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
            if lap_names_b != lap_names:
                idx_map = {n: i for i, n in enumerate(lap_names_b)}
                idx = [idx_map[n] for n in lap_names if n in idx_map]
                bmean, bstd = bmean[idx], bstd[idx]
            lap_epochs = apply_baseline_norm(lap_epochs, bmean, bstd, BASELINE_MODE)

        # Fix a reference Lap order from first label
        if ref_order is None:
            ref_order = lap_names
        else:
            # reorder to match ref_order
            idx_map = {n: i for i, n in enumerate(lap_names)}
            idx = [idx_map[n] for n in ref_order if n in idx_map]
            lap_epochs = lap_epochs[:, idx, :]

        X = make_features_from_epochs(lap_epochs, sf)
        y = np.array([lbl] * X.shape[0])
        all_X.append(X); all_y.append(y)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    # ---------- Build pipeline ----------
    n_classes = len(np.unique(y_train))
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    pca_n = int(min(PCA_MAX_COMPONENTS, n_features, max(2, n_samples - n_classes)))
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=pca_n, svd_solver="auto", random_state=0)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])

    # ---------- Cross-validated accuracy ----------
    print(f"\n[Training] CV ({N_FOLDS_CV}-fold) with PCA({pca_n}) + LDA on {n_samples} epochs, {n_features} features ...")
    cv = StratifiedKFold(n_splits=min(N_FOLDS_CV, n_samples // n_classes), shuffle=True, random_state=0)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=None)
    print(f"  • Mean CV accuracy: {cv_scores.mean():.3f}  (± {cv_scores.std():.3f})")

    y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=None)
    cm = confusion_matrix(y_train, y_pred_cv, labels=["A", "B", "C", "D"])
    print("  • Confusion matrix (CV):")
    print(cm)

    ensure_outdir(OUTPUT_DIR)
    plot_confusion(cm, ["A", "B", "C", "D"],
                   title="PCA+LDA (CV) — Confusion Matrix",
                   outpath=os.path.join(OUTPUT_DIR, "confusion_matrix_cv.png"),
                   show=SHOW_PLOTS if SAVE_PLOTS else False)

    # ---------- Fit on full training ----------
    pipe.fit(X_train, y_train)

    # ---------- Target classification ----------
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw_t = load_raw(TARGET_FILE, RESAMPLE_HZ)
    sf = raw_t.info["sfreq"]
    events_all, eid_all = get_events_any(raw_t)
    code_t, lbl_t = resolve_event_auto(events_all, eid_all, requested=TARGET_EVENT)
    events_t = events_all[events_all[:, 2] == code_t]
    print(f"[Target] Using event '{lbl_t}' (code {code_t}); N={len(events_t)}")

    # Build epochs with the same Lap order & baseline
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks_t = mne.pick_channels(raw_t.info["ch_names"], include=list(names_needed))
    if len(picks_t) == 0:
        raise RuntimeError("Target: required central/neighbor channels missing.")

    epochs_t = mne.Epochs(raw_t, events_t, event_id={lbl_t: code_t}, tmin=TMIN, tmax=TMAX,
                          baseline=None, picks=picks_t, preload=True, detrend=0, verbose="ERROR")
    data_t = epochs_t.get_data()
    ch_names_t = [epochs_t.ch_names[i] for i in range(len(epochs_t.ch_names))]
    lap_t, lap_names_t = derive_laplacian_epochs(data_t, ch_names_t)

    # Baseline on target
    if BASELINE_USE_PRE_ANNOT:
        bmean_t, bstd_t, secs_t, lap_names_b_t = compute_baseline_stats_on_laplacian(raw_t, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_b_t != lap_names_t:
            idx_map = {n: i for i, n in enumerate(lap_names_b_t)}
            idx = [idx_map[n] for n in lap_names_t if n in idx_map]
            bmean_t, bstd_t = bmean_t[idx], bstd_t[idx]
        lap_t = apply_baseline_norm(lap_t, bmean_t, bstd_t, BASELINE_MODE)

    # Reorder to reference Lap order
    if lap_names_t != ref_order:
        idx_map = {n: i for i, n in enumerate(lap_names_t)}
        idx = [idx_map[n] for n in ref_order if n in idx_map]
        lap_t = lap_t[:, idx, :]

    print("\n[Target] Extracting features & predicting ...")
    X_target = []
    for i in tqdm(range(lap_t.shape[0]), total=lap_t.shape[0], desc="[Predict]"):
        X_target.append(make_features_from_epochs(lap_t[i:i+1], sf)[0])
    X_target = np.vstack(X_target)

    proba = pipe.predict_proba(X_target)  # (N, 4)
    labels_pred = pipe.classes_[np.argmax(proba, axis=1)].tolist()
    max_proba = np.max(proba, axis=1)

    if USE_PROBA_THRESHOLD:
        labels_pred = [l if p >= PROBA_THRESHOLD else "UNK" for l, p in zip(labels_pred, max_proba)]

    # Report / save sequence
    save_sequence(labels_pred, max_proba.tolist(), OUTPUT_DIR, prefix="pca_lda_movement_mix")

    # Summary counts
    counts = Counter(labels_pred)
    print("\n[Counts]")
    for k in ["A", "B", "C", "D", "UNK"]:
        if counts.get(k, 0) > 0:
            print(f"  {k}: {counts[k]}")
    if "UNK" in counts:
        print(f"UNK count: {counts['UNK']} / {len(labels_pred)}")

    # (Optional) quick training accuracy on full fit (not CV)
    y_pred_train = pipe.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f"\n[Training fit] Accuracy on training data (optimistic): {acc_train:.3f}")

    if SAVE_PLOTS:
        print(f"\nFigures & outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

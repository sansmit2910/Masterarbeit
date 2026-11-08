# -*- coding: utf-8 -*-
"""
A/B Registration Motifs + Continuous Full-Scan Detection
+ Optional Regularization (ideal 229 @ ~0.66 Hz) + Accuracy vs Ground Truth
---------------------------------------------------------------------------
- Learns motifs from A,B registration files.
- Displays ONE figure with raw C3/Cz/C4, Laplacians, and 4 ERD bands per motif.
- Scans the target continuously (no triggers). TD+EN fusion NCC, dynamic thresholds,
  peak picking, surgical dedup, global ISI.
- By default, prints:
    (1) the deduplicated sequence (variable length, no UNK)
    (2) the Best-229 contiguous detections (no UNK, no grid)
    (3) accuracy vs user-supplied ground truth (if provided)
- Optional: enable optimized regularization (period + offset) to force exactly 229
  outputs; may introduce UNK if a grid slot has no nearby detection.

Deps: mne, numpy, scipy, matplotlib
"""

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert, fftconvolve, correlate, find_peaks
from mne.filter import filter_data
from numpy.linalg import eigh
from collections import Counter

# =========================
# ======== CONFIG =========
# =========================

# Registration templates: A & B only
TEMPLATE_SOURCES: Dict[str, Union[str, Tuple[str, Union[int, str, None]]]] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

# Target (full-scan; no triggers used)
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

# Template epoch window (seconds)
TMIN = 0.0
TMAX = 0.6
MAX_JITTER_SEC = 0.08   # template alignment tolerance

# Baseline on pre-first-annotation if available; otherwise first X sec
BASELINE_USE_PRE_ANNOT = True
BASELINE_MODE = "zscore"         # 'zscore' or 'demean'
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
EPS = 1e-12

# ERD sub-bands (Hz) for display and envelope feature
ERD_BANDS = [
    (8.0, 12.0),   # mu
    (13.0, 20.0),  # beta1
    (20.0, 26.0),  # beta2
    (26.0, 30.0),  # beta3
]
ENV_SMOOTH_SEC = 0.05

# Fusion of TD & EN
USE_FUSION = True
FUSION_ALPHA = 0.65  # TD-heavier

# Continuous-scan: per-label dynamic thresholds (on each label's NCC series)
NULL_PERCENTILE = 97.5      # percentile of fused NCC used as hard threshold
SOFT_FACTOR = 0.95          # allow 95% of dyn threshold (soft pass)
Z_FLOOR = 0.50              # ≥ 0.5 SD above label’s null mean
ABS_FLOOR = 0.16            # absolute fused-score floor

# Peak picking / dedup in continuous scan
SCAN_MIN_ISI_SEC = 0.35     # minimal spacing for peak finder per label
GLOBAL_MIN_ISI_SEC = 0.30   # global refractory after dedup (across labels)
TARGET_MAX_COUNT = None     # e.g., 229 to cap detections before regularization, or None

# Whitening & per-label channel weighting (TD)
USE_WHITEN_TD = True
WEIGHT_POWER = 1.0   # 0=no weighting, 1=rms, 2=rms^2

# ---- Regularized "ideal" sequence (grid) ----
# Turn OFF to avoid UNKs; we still output Best-229 contiguous detections.
USE_IDEAL_GRID = False
IDEAL_COUNT = 229
IDEAL_FREQ_HZ = 0.66           # nominal (~0.66 Hz → period ≈ 1.515 s)
GRID_TOL_SEC = 0.50            # allowed ± window around each grid time
PERIOD_SEARCH_PCT = 0.10       # search period within ±10% around nominal
PERIOD_STEPS = 41
OFFSET_STEPS = 241
ALIGN_PENALTY_PER_S = 0.02     # slight tie-break penalty for |dt|

# --- Ground truth evaluation ---
# Paste your 229-long A/B sequence here (space or comma separated). Leave empty to skip.
GROUND_TRUTH_SEQ_STR = ""  # e.g. "B A A A B B B A A A ... B"
GROUND_TRUTH_TRIM_K = 0    # e.g., 7 to ignore first & last 7 positions in accuracy
COUNT_UNK_AS_CORRECT = False  # only affects grid-based evaluation (if enabled)

# Visualization / output
OUTPUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_template_match_output"
ONE_IMAGE_NAME = "registration_motifs_AB.png"
SEQ_TXT_NAME = "recognized_sequence_fullscan_AB.txt"
SEQ_TXT_NAME_IDEAL = "recognized_sequence_fullscan_AB_IDEAL229.txt"
SAVE_PLOTS = True
SHOW_PLOTS = True

# =========================
# ======== HELPERS ========
# =========================

CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}
SENSOR_BASES = ["C3", "Cz", "C4"]

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_raw(path: str) -> mne.io.BaseRaw:
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")

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

def pre_annotation_baseline_window(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    """Prefer pre-first-annotation; fallback to [0:min_sec] if not available."""
    sf = raw.info["sfreq"]
    try:
        events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception:
        events = np.empty((0, 3), dtype=int)
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

def compute_baseline_stats(raw: mne.io.BaseRaw, picks: List[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    start, end = pre_annotation_baseline_window(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
    X = raw.get_data(picks=picks, start=start, stop=end)  # (K, T0)
    mean = X.mean(axis=1)
    std = X.std(axis=1) + EPS
    secs = (end - start) / raw.info["sfreq"]
    return mean, std, secs

def apply_baseline(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str):
    if arr.ndim == 2:
        return (arr - mean[:, None]) / std[:, None] if mode == "zscore" else (arr - mean[:, None])
    elif arr.ndim == 3:
        return (arr - mean[None, :, None]) / std[None, :, None] if mode == "zscore" else (arr - mean[None, :, None])
    else:
        raise ValueError("arr must be 2D or 3D")

def derive_laplacian_from_arrays(data_map: Dict[str, np.ndarray], name: str) -> np.ndarray:
    base, neigh = CENTRAL_BASES[name]
    if base not in data_map:
        raise KeyError(f"Base {base} missing for {name}.")
    base_arr = data_map[base]
    neigh_avail = [data_map[k] for k in neigh if k in data_map]
    if len(neigh_avail) == 2:
        return base_arr - 0.5 * (neigh_avail[0] + neigh_avail[1])
    return base_arr.copy()

def derive_laplacian_epoch(epoch_data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    out, out_names = [], []
    for lap in ["C3L", "CzL", "C4L"]:
        base, neigh = CENTRAL_BASES[lap]
        if base not in name_to_idx:
            continue
        base_idx = name_to_idx[base]
        neigh_idx = [name_to_idx[k] for k in neigh if k in name_to_idx]
        if len(neigh_idx) == 2:
            sig = epoch_data[base_idx, :] - 0.5 * (epoch_data[neigh_idx[0], :] + epoch_data[neigh_idx[1], :])
            out.append(sig); out_names.append(lap)
        else:
            out.append(epoch_data[base_idx, :]); out_names.append(base)
    if not out:
        raise RuntimeError("No Laplacian channels could be derived for this epoch.")
    return np.vstack(out), out_names

def jitter_align_to_median(epochs: np.ndarray, sfreq: float, max_jitter_sec: float) -> np.ndarray:
    """
    Align each epoch (C,T) to the running median by brute-force lag search in ±max_jitter samples.
    epochs: (N, C, T)
    """
    aligned = epochs.copy()
    max_shift = int(round(max(0.0, max_jitter_sec) * sfreq))
    if max_shift <= 0:
        return aligned

    ref = np.nanmedian(aligned, axis=0)  # (C,T)
    for _ in range(2):  # two passes
        for i in range(aligned.shape[0]):
            best_lag, best_score = 0, -np.inf
            for lag in range(-max_shift, max_shift + 1):
                if lag < 0:
                    a = aligned[i, :, :lag]; b = ref[:, -lag:]
                elif lag > 0:
                    a = aligned[i, :, lag:]; b = ref[:, :-lag]
                else:
                    a = aligned[i];         b = ref
                a0 = a - a.mean(axis=-1, keepdims=True)
                b0 = b - b.mean(axis=-1, keepdims=True)
                s = float(np.sum(a0 * b0))
                if s > best_score:
                    best_score, best_lag = s, lag
            aligned[i] = np.roll(aligned[i], shift=int(best_lag), axis=-1)
        ref = np.nanmedian(aligned, axis=0)
    return aligned

def safe_band_envelope(x: np.ndarray, sfreq: float, l: float, h: float, smooth_sec: float) -> np.ndarray:
    nyq = sfreq / 2.0
    h_eff = min(h, nyq - 0.5)
    if l >= h_eff:
        xf = x.copy()
    else:
        xf = filter_data(x.copy(), sfreq, l_freq=l, h_freq=h_eff, fir_design='firwin', verbose="ERROR")
    env = np.abs(hilbert(xf, axis=-1))
    k = max(1, int(round(smooth_sec * sfreq)))
    if k > 1:
        w = np.ones(k) / k
        env = np.apply_along_axis(lambda s: np.convolve(s, w, mode="same"), -1, env)
    return env

def make_env_feature(data_lap: np.ndarray, sfreq: float,
                     bands: List[Tuple[float, float]], smooth_sec: float) -> np.ndarray:
    if not bands:
        if data_lap.ndim == 2:
            return np.zeros((0, data_lap.shape[1]), dtype=data_lap.dtype)
        elif data_lap.ndim == 3:
            return np.zeros((data_lap.shape[0], 0, data_lap.shape[2]), dtype=data_lap.dtype)
        else:
            raise ValueError("data_lap must be 2D or 3D when bands=[]")
    if data_lap.ndim == 2:
        data = data_lap[np.newaxis, :, :]
        squeeze = True
    elif data_lap.ndim == 3:
        data = data_lap; squeeze = False
    else:
        raise ValueError("data_lap must be 2D or 3D")
    N, C, T = data.shape
    feats = []
    for (l, h) in bands:
        X = np.empty_like(data)
        for n in range(N):
            X[n] = filter_data(data[n].copy(), sfreq, l_freq=l,
                               h_freq=min(h, sfreq/2.0 - 0.5),
                               fir_design='firwin', verbose="ERROR")
        env = np.abs(hilbert(X, axis=-1))
        k = max(1, int(round(smooth_sec * sfreq)))
        if k > 1:
            w = np.ones(k) / k
            env = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, env)
        feats.append(env)
    feat = np.concatenate(feats, axis=1)
    return feat[0] if squeeze else feat

def ncc_multichannel(signal: np.ndarray, template: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    signal: (C, N); template: (C, L). Optional weights (C,).
    Returns NCC of length N-L+1
    """
    C, N = signal.shape
    Ct, L = template.shape
    assert C == Ct, f"Channel count mismatch {C}!={Ct}"
    if weights is None:
        weights = np.ones(C, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != C:
            raise ValueError("weights length must equal channel count")
    S_t_total = np.sum([weights[c] * np.sum(template[c] ** 2) for c in range(C)]) + EPS
    ones = np.ones(L)
    numerator = None
    denom_sum = None
    for c in range(C):
        w = weights[c]
        x = signal[c]
        t0 = template[c]
        r_num = correlate(x, t0, mode="valid", method="fft")
        s1 = fftconvolve(x, ones, mode="valid")
        s2 = fftconvolve(x * x, ones, mode="valid")
        lvar = s2 - (s1 * s1) / L
        lvar = np.maximum(lvar, EPS)
        numerator = r_num * w if numerator is None else (numerator + w * r_num)
        denom_sum = lvar * w if denom_sum is None else (denom_sum + w * lvar)
    denom = np.sqrt(S_t_total * np.maximum(denom_sum, EPS))
    return numerator / denom

# =========================
# ====== BUILD MOTIFS =====
# =========================

def build_motif_for_file(label: str, fif_path: str, event_key: Optional[Union[int, str]], tmin: float, tmax: float):
    raw = load_raw(fif_path)
    sf = raw.info["sfreq"]
    events, eid = get_events_any(raw)
    if len(eid) == 0:
        raise RuntimeError(f"[{label}] No annotations/events in {os.path.basename(fif_path)}.")
    if len(eid) == 1:
        code = list(eid.values())[0]; chosen_lbl = list(eid.keys())[0]
    else:
        counts = Counter(events[:, 2].tolist())
        code, _ = counts.most_common(1)[0]
        chosen_lbl = [k for k, v in eid.items() if v == code][0]

    names_needed = set(["C3","Cz","C4"])
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError(f"[{label}] Missing required central/neighbor channels in {os.path.basename(fif_path)}.")

    # Baseline on picked channels
    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, _ = compute_baseline_stats(raw, picks)
        X_all = apply_baseline(raw.get_data(picks=picks), bmean, bstd, BASELINE_MODE)
    else:
        X_all = raw.get_data(picks=picks)

    info = mne.pick_info(raw.info, sel=picks)
    raw2 = mne.io.RawArray(X_all, info, verbose="ERROR")
    sel_events = events[events[:, 2] == code]
    if len(sel_events) == 0:
        raise RuntimeError(f"No events '{chosen_lbl}' in {os.path.basename(fif_path)}.")
    epochs = mne.Epochs(raw2, sel_events, event_id={chosen_lbl: code},
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, detrend=0, verbose="ERROR")
    data = epochs.get_data()  # (N,K,T)
    ch_names = epochs.ch_names
    N, K, T = data.shape

    # Sensor-space motif: C3/Cz/C4
    sensor_idxs = [ch_names.index(n) for n in ["C3","Cz","C4"] if n in ch_names]
    if len(sensor_idxs) == 0:
        raise RuntimeError(f"[{label}] None of C3/Cz/C4 present in {os.path.basename(fif_path)}.")
    sensor_epochs = data[:, sensor_idxs, :]
    sensor_names = [ch_names[i] for i in sensor_idxs]

    # Laplacian per-epoch
    lap_epochs = []
    lap_names = None
    for i in range(N):
        lap_i, lap_names_i = derive_laplacian_epoch(data[i], ch_names)  # baseline already applied
        lap_epochs.append(lap_i)
        if lap_names is None:
            lap_names = lap_names_i
    lap_epochs = np.stack(lap_epochs, axis=0)

    # Jitter-align to median
    sensor_aligned = jitter_align_to_median(sensor_epochs, sf, MAX_JITTER_SEC)
    lap_aligned = jitter_align_to_median(lap_epochs, sf, MAX_JITTER_SEC)

    # Median motifs
    motif_sensor = np.median(sensor_aligned, axis=0)  # (Cs,T)
    motif_lap = np.median(lap_aligned, axis=0)        # (Cl,T)

    # ERD envelopes on Laplacian motif (for display)
    band_envs = []
    for (l, h) in ERD_BANDS:
        env = safe_band_envelope(motif_lap, sf, l, h, ENV_SMOOTH_SEC)
        band_envs.append(env)

    # TD+EN templates for detection
    templ_td = motif_lap - motif_lap.mean(axis=-1, keepdims=True)
    env_epochs = []
    for (l, h) in ERD_BANDS:
        env_e = np.abs(hilbert(filter_data(lap_aligned.copy(), sf, l_freq=l,
                                           h_freq=min(h, sf/2 - 0.5),
                                           fir_design='firwin', verbose="ERROR"), axis=-1))
        k = max(1, int(round(ENV_SMOOTH_SEC * sf)))
        if k > 1:
            w = np.ones(k) / k
            env_e = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, env_e)
        env_epochs.append(env_e)
    if env_epochs:
        env_stack = np.concatenate(env_epochs, axis=1)  # (N, C*B, T)
        templ_en = np.median(env_stack, axis=0)
        templ_en = templ_en - templ_en.mean(axis=-1, keepdims=True)
    else:
        templ_en = np.zeros((0, templ_td.shape[1]))

    return dict(
        label=label,
        sfreq=sf,
        t_axis=np.arange(T)/sf + TMIN,
        sensor_names=sensor_names,
        lap_names=lap_names,
        motif_sensor=motif_sensor,
        motif_lap=motif_lap,
        band_envs=band_envs,
        chosen_event="auto",
        n_epochs=N,
        templ_td=templ_td,
        templ_en=templ_en
    )

# =========================
# ====== PLOT (ONE) =======
# =========================

def plot_all_in_one(motifs: Dict[str, dict], outdir: str, fname: str,
                    save: bool = True, show: bool = True):
    ensure_outdir(outdir)
    labels = [k for k in ["A", "B"] if k in motifs]

    n_rows = 6  # raw, lap, + 4 ERD rows
    n_cols = len(labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.8*n_rows), sharex='col')
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    for j, lab in enumerate(labels):
        m = motifs[lab]
        axes[0, j].set_title(f"Motif {lab} (N={m['n_epochs']})")

    row_labels = [
        "Raw sensors: C3/Cz/C4",
        "Laplacians: C3L/CzL/C4L",
        f"ERD {ERD_BANDS[0][0]:.0f}-{ERD_BANDS[0][1]:.0f} Hz",
        f"ERD {ERD_BANDS[1][0]:.0f}-{ERD_BANDS[1][1]:.0f} Hz",
        f"ERD {ERD_BANDS[2][0]:.0f}-{ERD_BANDS[2][1]:.0f} Hz",
        f"ERD {ERD_BANDS[3][0]:.0f}-{ERD_BANDS[3][1]:.0f} Hz",
    ]
    for i in range(n_rows):
        axes[i, 0].set_ylabel(row_labels[i])

    for j, lab in enumerate(labels):
        m = motifs[lab]
        t = m["t_axis"]

        ax = axes[0, j]
        for k, ch in enumerate(m["sensor_names"]):
            ax.plot(t, m["motif_sensor"][k] + 4*k, label=ch)
        if j == n_cols-1:
            ax.legend(loc="upper right", fontsize=8, ncols=min(3, len(m["sensor_names"])))
        ax.grid(alpha=0.2)

        ax = axes[1, j]
        for k, ch in enumerate(m["lap_names"]):
            ax.plot(t, m["motif_lap"][k] + 4*k, label=ch)
        if j == n_cols-1:
            ax.legend(loc="upper right", fontsize=8, ncols=min(3, len(m["lap_names"])))
        ax.grid(alpha=0.2)

        for b_idx, env in enumerate(m["band_envs"]):
            ax = axes[2 + b_idx, j]
            env_disp = (env - env.mean(axis=1, keepdims=True)) / (env.std(axis=1, keepdims=True) + EPS)
            for k in range(env.shape[0]):
                ax.plot(t, env_disp[k] + 2.0*k, label=m["lap_names"][k] if k < len(m["lap_names"]) else f"Ch{k+1}")
            if j == n_cols-1:
                ax.legend(loc="upper right", fontsize=8, ncols=min(3, env.shape[0]))
            ax.grid(alpha=0.2)

        axes[-1, j].set_xlabel("Time (s)")

    fig.suptitle("Registration Motifs A/B (All in One Window)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    if save:
        out_path = os.path.join(outdir, fname)
        fig.savefig(out_path, dpi=160)
        print(f"[Saved figure] {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# =========================
# ==== CONTINUOUS SCAN ====
# =========================

def compute_whitener_from_baseline_lap(raw: mne.io.BaseRaw, ref_order: List[str]) -> np.ndarray:
    start, end = pre_annotation_baseline_window(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError("Required central/neighbor channels not present for baseline.")
    full = raw.get_data(picks=picks, start=start, stop=end)
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
        raise RuntimeError("No central/Laplacian signals could be derived for baseline.")
    lap = np.vstack(out); lap_names = out_names
    if lap_names != ref_order:
        idx = [lap_names.index(n) for n in ref_order if n in lap_names]
        lap = lap[idx, :]
    C = lap.shape[0]
    cov = (lap @ lap.T) / max(1, lap.shape[1])
    cov += 1e-8 * np.eye(C)
    d, V = eigh(cov); d = np.maximum(d, 1e-10)
    Dm12 = np.diag(1.0 / np.sqrt(d))
    W = V @ Dm12 @ V.T
    return W

def derive_laplacian_continuous(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, List[str]]:
    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError("Required central/neighbor channels not present.")
    full = raw.get_data(picks=picks)
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

def build_ref_and_indices(motifs: Dict[str, dict], labels_use: List[str], ERD_BANDS: List[Tuple[float,float]]):
    ref_order = motifs[labels_use[0]]['lap_names']
    templates = {}
    label_idx_td = {}
    label_idx_en = {}
    B_env = len(ERD_BANDS)
    for l in labels_use:
        td = motifs[l]['templ_td']
        en = motifs[l]['templ_en']
        names = motifs[l]['lap_names']
        idx_td_names = [names.index(n) for n in ref_order if n in names]
        td_re = td[idx_td_names, :]
        templates[l] = {"TD": td_re}
        label_idx_td[l] = [ref_order.index(n) for n in ref_order if n in names]
        if en.size and B_env > 0:
            en_blocks = np.split(en, B_env, axis=0)
            en_blocks_reordered = []
            for blk in en_blocks:
                blk_re = blk[idx_td_names, :]
                en_blocks_reordered.append(blk_re)
            en_re = np.concatenate(en_blocks_reordered, axis=0)
            templates[l]["EN"] = en_re
            idx_en = []
            for b in range(B_env):
                base = b * len(ref_order)
                idx_en.extend([base + i for i in label_idx_td[l]])
            label_idx_en[l] = idx_en
        else:
            templates[l]["EN"] = np.zeros((0, td.shape[1]))
            label_idx_en[l] = []
    return ref_order, templates, label_idx_td, label_idx_en

def dedup_overlaps(events: List[dict], L: int, strategy: str = "overlap_fused_ncc") -> List[dict]:
    """events: list of dicts with keys: 't0','t1','center','label','score'."""
    if not events:
        return []
    ev_sorted = sorted(events, key=lambda d: d['t0'])
    groups = [[ev_sorted[0]]]
    for ev in ev_sorted[1:]:
        if ev['t0'] <= groups[-1][-1]['t1']:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    kept = []
    for grp in groups:
        best = max(grp, key=lambda d: d['score'])
        kept.append(best)
    kept = sorted(kept, key=lambda d: d['center'])
    return kept

def enforce_global_min_isi(kept: List[dict], sf: float, min_isi_sec: float) -> List[dict]:
    if not kept:
        return kept
    kept_sorted = sorted(kept, key=lambda d: d['center'])
    out = [kept_sorted[0]]
    min_dist = int(round(min_isi_sec * sf))
    for ev in kept_sorted[1:]:
        if ev['center'] - out[-1]['center'] >= min_dist:
            out.append(ev)
        elif ev['score'] > out[-1]['score']:
            out[-1] = ev
    return out

# ---------- Optimized regularizer: search period + offset ----------
def regularize_to_grid_optim(
    events: List[dict],
    sf: float,
    approx_freq_hz: float,
    count: int,
    tol_sec: float,
    period_search_pct: float = 0.10,
    period_steps: int = 41,
    offset_steps: int = 241,
    align_penalty_per_s: float = 0.02,
    use_scores: bool = True,
) -> Tuple[List[str], float, float, Tuple[int, float, float]]:
    """
    Build exactly `count` labels on a uniform grid by optimizing period and offset
    using deduplicated detections.
    Returns: (seq, best_t0, best_period, metrics=(UNK, sum_score, total_abs_dt))
    """
    if not events:
        return ["UNK"] * count, 0.0, 1.0 / max(approx_freq_hz, 1e-9), (count, 0.0, 0.0)

    times = np.array([ev['center'] / sf for ev in events], dtype=float)
    scores = np.array([ev['score'] for ev in events], dtype=float)
    labels = [ev['label'] for ev in events]
    order = np.argsort(times)
    times = times[order]; scores = scores[order]
    labels = [labels[i] for i in order]

    # Robust period estimate from detections + nominal blend
    diffs = np.diff(times)
    if diffs.size:
        d_clip = diffs[(diffs > np.percentile(diffs, 5)) & (diffs < np.percentile(diffs, 95))]
        P_est = np.median(d_clip) if d_clip.size else 1.0 / approx_freq_hz
    else:
        P_est = 1.0 / approx_freq_hz
    P0 = 0.5 * (P_est + 1.0 / approx_freq_hz)

    Plow  = max(0.2, P0 * (1.0 - period_search_pct))
    Phigh = P0 * (1.0 + period_search_pct)

    tmin, tmax = times[0], times[-1]
    best = None

    for P in np.linspace(Plow, Phigh, max(2, int(period_steps))):
        span = (count - 1) * P
        if tmax - tmin >= span * 0.5:
            t0_min = tmin
            t0_max = max(t0_min, tmax - span)
        else:
            t0_min = tmin
            t0_max = tmin + P

        for alpha in np.linspace(t0_min, t0_max, max(2, int(offset_steps))):
            used = np.zeros(len(times), dtype=bool)
            seq = []
            unk = 0
            sum_score = 0.0
            sum_abs_dt = 0.0

            for k in range(count):
                t_target = alpha + k * P
                left, right = t_target - tol_sec, t_target + tol_sec
                i0 = np.searchsorted(times, left, side="left")
                i1 = np.searchsorted(times, right, side="right")

                best_i, best_val = -1, -np.inf
                for i in range(i0, i1):
                    if used[i]:
                        continue
                    dt = abs(times[i] - t_target)
                    val = (scores[i] if use_scores else 1.0) - align_penalty_per_s * dt
                    if val > best_val:
                        best_val = val
                        best_i = i

                if best_i >= 0:
                    used[best_i] = True
                    seq.append(labels[best_i])
                    sum_score += float(scores[best_i])
                    sum_abs_dt += float(abs(times[best_i] - t_target))
                else:
                    seq.append("UNK")
                    unk += 1

            key = (unk, -sum_score, sum_abs_dt)  # minimize UNK, maximize score, minimize |dt|
            if (best is None) or (key < best[0]):
                best = (key, alpha, P, seq)

    (unk, neg_sum_score, total_abs_dt), best_t0, best_P, best_seq = best
    return best_seq, float(best_t0), float(best_P), (unk, -neg_sum_score, total_abs_dt)

# =========================
# ===== Accuracy utils ====
# =========================

def parse_seq_string(s: str) -> List[str]:
    """Parse a string of labels into a list. Keeps tokens 'A','B','UNK' (case-insensitive)."""
    if not s:
        return []
    toks = [t.strip().upper() for t in s.replace(",", " ").split()]
    out = []
    for t in toks:
        if t in ("A", "B", "UNK"):
            out.append(t)
    return out

def compute_accuracy(pred: List[str], truth: List[str],
                     count_unk_as_correct: bool = False,
                     trim_k: int = 0) -> float:
    """Return accuracy (%) over aligned lists. Optionally trim K from both ends."""
    assert len(pred) == len(truth), "Length mismatch for accuracy comparison"
    n = len(pred)
    a = trim_k
    b = n - trim_k
    if a < 0 or b <= a:
        a, b = 0, n
    if b <= a:
        return 0.0
    matches = 0
    total = b - a
    for i in range(a, b):
        if pred[i] == truth[i]:
            matches += 1
        elif count_unk_as_correct and pred[i] == "UNK":
            matches += 1
    return 100.0 * matches / max(1, total)

def best_window_accuracy(pred: List[str], truth: List[str],
                         count_unk_as_correct: bool = False,
                         trim_k: int = 0) -> Tuple[float, int, List[str]]:
    """Slide truth-length window over pred to maximize accuracy. Returns (acc, start_idx, window_seq)."""
    m, n = len(pred), len(truth)
    if m < n or n == 0:
        return 0.0, -1, []
    best_acc, best_i = -1.0, -1
    best_seq = []
    for i in range(0, m - n + 1):
        window = pred[i:i+n]
        acc = compute_accuracy(window, truth, count_unk_as_correct=count_unk_as_correct, trim_k=trim_k)
        if acc > best_acc:
            best_acc, best_i, best_seq = acc, i, window
    return best_acc, best_i, best_seq

# =========================
# ========= MAIN ==========
# =========================

def main():
    ensure_outdir(OUTPUT_DIR)

    # ---- 1) Build motifs & templates for A,B ----
    print("[Registration] Building motifs and templates (A,B) ...")
    motifs: Dict[str, dict] = {}
    for label, entry in TEMPLATE_SOURCES.items():
        fpath = entry if isinstance(entry, str) else entry[0]
        print(f"  • {label}: {os.path.basename(fpath)}")
        mot = build_motif_for_file(label, fpath, None, TMIN, TMAX)
        motifs[label] = mot
        print(f"    -> epochs: {mot['n_epochs']}, sfreq: {mot['sfreq']:.2f} Hz")

    # One big figure with both motifs (and ERDs)
    print("\n[Plot] One-window registration motifs (A,B) ...")
    plot_all_in_one(motifs, OUTPUT_DIR, ONE_IMAGE_NAME, save=SAVE_PLOTS, show=SHOW_PLOTS)

    # ---- 2) Prepare templates & indices ----
    labels_use = list(motifs.keys())  # ['A', 'B']
    print(f"\n[Detect] Using labels: {labels_use}")

    sfreqs = [motifs[k]['sfreq'] for k in motifs]
    if not all(abs(s - sfreqs[0]) < 1e-6 for s in sfreqs):
        raise RuntimeError("Template files must share the same sampling rate.")
    sf = sfreqs[0]

    ref_order, templates, label_idx_td, label_idx_en = build_ref_and_indices(motifs, labels_use, ERD_BANDS)

    # Per-label TD weights
    ch_weights: Dict[str, np.ndarray] = {}
    for l in labels_use:
        rms = np.sqrt(np.mean(templates[l]["TD"] ** 2, axis=1))
        w = (rms ** WEIGHT_POWER) if WEIGHT_POWER != 0 else np.ones_like(rms)
        w = w + EPS
        w = w / np.mean(w)
        ch_weights[l] = w

    # ---- 3) Load target & build continuous Laplacians ----
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw_t = load_raw(TARGET_FILE)
    if abs(raw_t.info["sfreq"] - sf) > 1e-6:
        raw_t.resample(sf, npad="auto")

    lap_all, lap_names_target = derive_laplacian_continuous(raw_t)
    if lap_names_target != ref_order:
        name_to_idx = {n: i for i, n in enumerate(lap_names_target)}
        idx = [name_to_idx[n] for n in ref_order if n in name_to_idx]
        lap_all = lap_all[idx]
        lap_names_target = [ref_order[i] for i in range(len(idx))]
    C_ref, N = lap_all.shape

    # Baseline (same Lap channels)
    if BASELINE_USE_PRE_ANNOT:
        bstart, bend = pre_annotation_baseline_window(raw_t, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        lap_bl = lap_all[:, bstart:bend]
        bmean = lap_bl.mean(axis=1)
        bstd = lap_bl.std(axis=1) + EPS
        lap_all = (lap_all - bmean[:, None]) / bstd[:, None] if BASELINE_MODE == "zscore" else (lap_all - bmean[:, None])

    # Envelope features for target (continuous)
    env_all = make_env_feature(lap_all, sf, ERD_BANDS, ENV_SMOOTH_SEC)  # (C_ref*B, N) or (0,N)

    # Whitening (TD)
    if USE_WHITEN_TD:
        W = compute_whitener_from_baseline_lap(raw_t, ref_order)
        lap_all = W @ lap_all
        for l in templates:
            templates[l]["TD"] = W @ templates[l]["TD"]

    # ---- 4) Continuous NCC per label ----
    print("\n[Scan] Computing NCC time series (TD + EN) per label ...")
    L = next(iter(templates.values()))["TD"].shape[1]
    fused_series: Dict[str, np.ndarray] = {}
    td_series: Dict[str, np.ndarray] = {}
    en_series: Dict[str, np.ndarray] = {}

    for lab in labels_use:
        seg_td = lap_all[label_idx_td[lab], :]          # (C_lab, N)
        ncc_td = ncc_multichannel(seg_td, templates[lab]["TD"], weights=ch_weights[lab])  # (N-L+1,)
        td_series[lab] = ncc_td

        if env_all.size and templates[lab]["EN"].size and label_idx_en[lab]:
            seg_en = env_all[label_idx_en[lab], :]      # (C_lab*B, N)
            ncc_en = ncc_multichannel(seg_en, templates[lab]["EN"])
            en_series[lab] = ncc_en
            sc = FUSION_ALPHA * ncc_td + (1.0 - FUSION_ALPHA) * ncc_en if USE_FUSION else ncc_td
        else:
            en_series[lab] = np.zeros_like(ncc_td)
            sc = ncc_td
        fused_series[lab] = sc

    # ---- 5) Per-label dynamic thresholds + soft gates → peak picking ----
    print("\n[Calibrate] Dynamic thresholds from NCC distributions...")
    thr, mu, sd = {}, {}, {}
    for lab in labels_use:
        s = fused_series[lab]
        thr[lab] = float(np.percentile(s, NULL_PERCENTILE))
        mu[lab]  = float(np.mean(s))
        sd[lab]  = float(np.std(s) + 1e-9)
        print(f"  • {lab}: P{NULL_PERCENTILE:.1f}={thr[lab]:.3f}, μ={mu[lab]:.3f}, σ={sd[lab]:.3f}")

    min_dist = int(round(max(1, SCAN_MIN_ISI_SEC * sf)))
    candidates: List[dict] = []
    for lab in labels_use:
        s = fused_series[lab]
        soft_h = max(ABS_FLOOR, SOFT_FACTOR * thr[lab])
        pks, props = find_peaks(s, height=soft_h, distance=min_dist)
        for k, h in zip(pks, props["peak_heights"]):
            z = (h - mu[lab]) / (sd[lab] if sd[lab] > 1e-9 else 1.0)
            if z >= Z_FLOOR or h >= thr[lab]:
                t0 = k
                t1 = k + L
                center = k + (L // 2)
                candidates.append(dict(label=lab, k=int(k), center=int(center), t0=int(t0), t1=int(t1),
                                       score=float(h), z=float(z)))

    print(f"[Scan] Raw candidates found: {len(candidates)}")

    # ---- 6) Surgical dedup across labels + global ISI ----
    kept = dedup_overlaps(candidates, L=L, strategy="overlap_fused_ncc")
    kept = enforce_global_min_isi(kept, sf, GLOBAL_MIN_ISI_SEC)
    if TARGET_MAX_COUNT is not None and len(kept) > TARGET_MAX_COUNT:
        kept = sorted(kept, key=lambda d: d['score'], reverse=True)[:TARGET_MAX_COUNT]
        kept = sorted(kept, key=lambda d: d['center'])
    print(f"[Scan] After dedup + global ISI: {len(kept)}")

    # ---- 7) Output sequences ----
    seq_ref = [ev['label'] for ev in kept]

    print("\n=== Detected Motif Order (Continuous full-scan, A/B only; dedup view) ===")
    print(" ".join(seq_ref) if seq_ref else "[none]")
    print(f"Total detections (dedup): {len(seq_ref)}")

    if SAVE_PLOTS:
        out_seq = os.path.join(OUTPUT_DIR, SEQ_TXT_NAME)
        with open(out_seq, "w", encoding="utf-8") as f:
            f.write("Detected order (dedup):\n")
            f.write((" ".join(seq_ref) if seq_ref else "[none]") + "\n")
            f.write(f"\nTotal detections (dedup): {len(seq_ref)}\n")
        print(f"[Saved sequence] {out_seq}")

    # ---- 8) Accuracy vs Ground Truth (if provided) ----
    gt = parse_seq_string(GROUND_TRUTH_SEQ_STR)
    if gt:
        if len(gt) != IDEAL_COUNT:
            print(f"\n[Accuracy] Ground truth length={len(gt)} but IDEAL_COUNT={IDEAL_COUNT}.")
            print("           Accuracy will be computed only for Best-229 window if lengths match.")
        # Best-window accuracy for the dedup sequence (sliding)
        acc_win, win_start, win_seq = best_window_accuracy(seq_ref, gt,
                                                           count_unk_as_correct=False,
                                                           trim_k=GROUND_TRUTH_TRIM_K)
        if win_start >= 0:
            print(f"\n[Accuracy] Dedup vs GT (best sliding window, trim={GROUND_TRUTH_TRIM_K}): "
                  f"{acc_win:.2f}%  (start_idx={win_start})")
        else:
            print("\n[Accuracy] Dedup vs GT: not enough detections to compare with sliding window.")

    # ---- 9) Best-229 contiguous detections (no grid, no UNK) ----
    if len(kept) >= IDEAL_COUNT:
        scores = np.array([ev['score'] for ev in kept], dtype=float)
        Lw = IDEAL_COUNT
        csum = np.concatenate(([0.0], np.cumsum(scores)))
        best_i = int(np.argmax(csum[Lw:] - csum[:-Lw]))
        best_seq_229 = [ev['label'] for ev in kept[best_i: best_i+Lw]]

        print("\n=== Best-Window 229 Sequence (no grid, no UNK) ===")
        print(" ".join(best_seq_229))
        print(f"Window start index in dedup list: {best_i}  (len={Lw})")

        if SAVE_PLOTS:
            out_seq3 = os.path.join(OUTPUT_DIR, "recognized_sequence_fullscan_AB_BEST229.txt")
            with open(out_seq3, "w", encoding="utf-8") as f:
                f.write("Best-window 229 order (no grid):\n")
                f.write(" ".join(best_seq_229) + "\n")
                f.write(f"\nstart_idx={best_i}, length={Lw}\n")
            print(f"[Saved best-229 sequence] {out_seq3}")

        # Accuracy for Best-229 vs GT (if provided and lengths match)
        if gt and len(gt) == IDEAL_COUNT:
            acc_best = compute_accuracy(best_seq_229, gt,
                                        count_unk_as_correct=False,
                                        trim_k=GROUND_TRUTH_TRIM_K)
            print(f"[Accuracy] Best-229 vs GT (trim={GROUND_TRUTH_TRIM_K}): {acc_best:.2f}%")
    else:
        print("\n[Note] Fewer than 229 detections; skipping best-window output.")

    # ---- 10) Optional: grid-regularized 229 (may introduce UNK) ----
    if USE_IDEAL_GRID:
        print("\n[Regularize] Optimizing period & offset "
              f"(target={IDEAL_COUNT}, nominal≈{IDEAL_FREQ_HZ:.2f} Hz, tol±{GRID_TOL_SEC:.2f}s)...")
        seq_ideal, best_t0, best_P, metrics = regularize_to_grid_optim(
            events=kept,                      # use deduplicated hits
            sf=sf,
            approx_freq_hz=IDEAL_FREQ_HZ,
            count=IDEAL_COUNT,
            tol_sec=GRID_TOL_SEC,
            period_search_pct=PERIOD_SEARCH_PCT,
            period_steps=PERIOD_STEPS,
            offset_steps=OFFSET_STEPS,
            align_penalty_per_s=ALIGN_PENALTY_PER_S,
            use_scores=True,
        )
        unk, sum_score, total_abs_dt = metrics
        print("\n=== Regularized Ideal Sequence (exact 229, optimized grid) ===")
        print(" ".join(seq_ideal))
        print(f"UNK count: {seq_ideal.count('UNK')} / {IDEAL_COUNT}")
        print(f"  -> best_t0={best_t0:.3f}s, best_period={best_P:.3f}s "
              f"(freq≈{1.0/best_P:.3f} Hz), avg|dt|={total_abs_dt/IDEAL_COUNT:.3f}s")

        if SAVE_PLOTS:
            out_seq2 = os.path.join(OUTPUT_DIR, SEQ_TXT_NAME_IDEAL)
            with open(out_seq2, "w", encoding="utf-8") as f:
                f.write("Regularized ideal order (229, optimized grid):\n")
                f.write(" ".join(seq_ideal) + "\n")
                f.write(f"\nbest_t0={best_t0:.3f}s, best_period={best_P:.3f}s "
                        f"(freq≈{1.0/best_P:.3f} Hz), UNK={seq_ideal.count('UNK')}/{IDEAL_COUNT}, "
                        f"avg|dt|={total_abs_dt/IDEAL_COUNT:.3f}s\n")
            print(f"[Saved ideal sequence] {out_seq2}")

        # Accuracy for grid-regularized sequence (with optional UNK handling)
        if gt and len(gt) == IDEAL_COUNT:
            acc_grid = compute_accuracy(seq_ideal, gt,
                                        count_unk_as_correct=COUNT_UNK_AS_CORRECT,
                                        trim_k=GROUND_TRUTH_TRIM_K)
            mode = "UNK=correct" if COUNT_UNK_AS_CORRECT else "UNK=wrong"
            print(f"[Accuracy] Grid-regularized vs GT (trim={GROUND_TRUTH_TRIM_K}, {mode}): {acc_grid:.2f}%")

if __name__ == "__main__":
    main()

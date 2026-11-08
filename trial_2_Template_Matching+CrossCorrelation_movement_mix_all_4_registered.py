# -*- coding: utf-8 -*-
"""
EEG Template Matching (Trigger-locked) with surgical deduplication:
- Motor Laplacians (C3L, CzL, C4L)
- TD NCC + Mu/Beta Envelope NCC (fusion)
- Pre-whitening (baseline covariance)
- Per-label channel weighting
- Dynamic per-label thresholds from null calibration
- Overlap-aware deduplication using fused NCC (surgical method)

A = fist, B = pinch

Deps: mne, numpy, scipy, matplotlib
"""

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlate, hilbert, find_peaks
from numpy.linalg import eigh

# =========================
# ========= CONFIG ========
# =========================

TEMPLATE_SOURCES: Dict[str, Union[str, Tuple[str, Union[int, str, None]]]] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"
TARGET_EVENT: Optional[Union[int, str]] = None   # leave None to auto-pick most frequent

# Template epoch window (sec). If A still weak: TMIN=0.1, TMAX=0.7
TMIN = 0.0
TMAX = 0.6

# Keep native fs
RESAMPLE_HZ: Optional[float] = None

# Template build alignment & target search jitter (sec)
MAX_JITTER_SEC = 0.05
SEARCH_JITTER_SEC = 0.40

# Baseline (pre-first-annotation) on Laplacians
BASELINE_USE_PRE_ANNOT = True
BASELINE_MODE = "zscore"      # 'zscore' or 'demean'
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
EPS = 1e-12

# Focus classification on real movements first; set None to include C,D
CLASSIFY_LABELS: Optional[List[str]] = ["A", "B"]

# Envelope feature (set [] to disable)
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]
ENV_SMOOTH_SEC = 0.05

# Fusion of TD & EN
USE_FUSION = True
FUSION_ALPHA = 0.5  # 1=TD only, 0=EN only

# Dynamic thresholds from target null windows
NULL_SAMPLES = 400
NULL_MIN_GAP_SEC = 0.5
NULL_PERCENTILE = 97.5  # looser: 95; stricter: 99

# Competition rule
RATIO_ALPHA = 1.05
DELTA_MARGIN = 0.005

# Whitening & per-label channel weighting (TD)
USE_WHITEN_TD = True
USE_CH_WEIGHTS = True
WEIGHT_POWER = 1.0   # 0=no weighting, 1=rms, 2=rms^2

# Deduplication (surgical)
# Options: "none", "gap", "overlap_first", "overlap_energy", "overlap_fused_ncc"
DEDUP_STRATEGY = "overlap_fused_ncc"
MIN_ISI_SEC = 0.7    # used only for "gap" strategy

# Visualization
NCC_THRESHOLD_GLOBAL = 0.30
OUTPUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_template_match_output"
SAVE_PLOTS = True
SHOW_PLOTS = True

# =========================
# ========= UTILS =========
# =========================

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

# ---------- Laplacians ----------

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
        raise RuntimeError("Required central/neighbor channels not present.")

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
    if len(out) == 0:
        raise RuntimeError("No central/Laplacian signals could be derived.")
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
    if len(out) == 0:
        raise RuntimeError("No central/Laplacian signals available in epochs.")
    return np.stack(out, axis=1), out_names

# ---------- Baseline, whitening ----------

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
    std  = lap.std(axis=1) + EPS
    seconds = (end - start) / raw.info["sfreq"]
    return mean, std, seconds, lap_names

def apply_baseline_norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str):
    if arr.ndim == 2:
        return (arr - mean[:, None]) / std[:, None] if mode == "zscore" else (arr - mean[:, None])
    elif arr.ndim == 3:
        return (arr - mean[None, :, None]) / std[None, :, None] if mode == "zscore" else (arr - mean[None, :, None])
    raise ValueError("Unsupported array ndim")

def compute_whitener_from_baseline(raw: mne.io.BaseRaw, ref_order: List[str]) -> np.ndarray:
    start, end = pre_annotation_baseline_window(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
    lap, lap_names = derive_laplacian_continuous(raw, start=start, stop=end)
    if lap_names != ref_order:
        idx = [lap_names.index(n) for n in ref_order if n in lap_names]
        lap = lap[idx, :]
    C = lap.shape[0]
    cov = (lap @ lap.T) / max(1, lap.shape[1])
    cov = cov + 1e-8 * np.eye(C)
    d, V = eigh(cov)
    d = np.maximum(d, 1e-10)
    Dm12 = np.diag(1.0 / np.sqrt(d))
    W = V @ Dm12 @ V.T
    return W

def apply_whiten(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return W @ X
    elif X.ndim == 3:
        return np.einsum('ij,njt->nit', W, X)
    else:
        raise ValueError("X ndim must be 2 or 3")

# ---------- Envelope feature ----------

def _safe_filter_data(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    from mne.filter import filter_data
    nyq = sfreq / 2.0
    h = min(h_freq, nyq - 0.5)
    if l_freq >= h:
        return data.copy()
    return filter_data(data.copy(), sfreq, l_freq=l_freq, h_freq=h, fir_design='firwin', verbose="ERROR")

def hilbert_envelope(data: np.ndarray, axis: int = -1) -> np.ndarray:
    analytic = hilbert(data, axis=axis)
    return np.abs(analytic)

def make_env_feature(data_lap: np.ndarray, sfreq: float,
                     bands: List[Tuple[float, float]], smooth_sec: float) -> np.ndarray:
    if not bands:
        if data_lap.ndim == 1:
            return np.zeros((0, data_lap.shape[0]), dtype=data_lap.dtype)
        if data_lap.ndim == 2:
            return np.zeros((0, data_lap.shape[1]), dtype=data_lap.dtype)
        if data_lap.ndim == 3:
            return np.zeros((data_lap.shape[0], 0, data_lap.shape[2]), dtype=data_lap.dtype)
        raise ValueError(f"data_lap ndim must be 1, 2 or 3, got {data_lap.ndim}")
    if data_lap.ndim == 1:
        data = data_lap[np.newaxis, np.newaxis, :]
        squeeze = True
    elif data_lap.ndim == 2:
        data = data_lap[np.newaxis, :, :]
        squeeze = True
    elif data_lap.ndim == 3:
        data = data_lap
        squeeze = False
    else:
        raise ValueError
    N, C, T = data.shape
    feats = []
    for (l, h) in bands:
        X = np.empty_like(data)
        for n in range(N):
            X[n] = _safe_filter_data(data[n], sfreq, l, h)
        env = hilbert_envelope(X, axis=-1)
        k = max(1, int(round(smooth_sec * sfreq)))
        if k > 1:
            w = np.ones(k) / k
            env = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, env)
        feats.append(env)
    feat = np.concatenate(feats, axis=1)
    return feat[0] if squeeze else feat

# ---------- Weighted NCC ----------

def ncc_multichannel(signal: np.ndarray, template: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    signal: (C, N); template: (C, L). Optional per-channel weights (C,).
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
# ====== TEMPLATES ========
# =========================

def build_template_from_file(
    label: str,
    fif_path: str,
    event_key: Optional[Union[int, str]],
    tmin: float,
    tmax: float,
    resample_hz: Optional[float],
):
    raw = load_raw(fif_path, resample_hz)
    events, eid = get_events_any(raw)
    code, chosen_lbl = resolve_event_auto(events, eid, requested=event_key)

    names_needed = set()
    for base, neigh in CENTRAL_BASES.values():
        names_needed.add(base); names_needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
    if len(picks) == 0:
        raise RuntimeError(f"[{label}] Required central/neighbor channels not present in {os.path.basename(fif_path)}.")

    epochs = mne.Epochs(raw, events[events[:, 2] == code],
                        event_id={chosen_lbl: code}, tmin=tmin, tmax=tmax,
                        baseline=None, picks=picks, preload=True, detrend=0, verbose="ERROR")
    if len(epochs) == 0:
        raise RuntimeError(f"No epochs found in {os.path.basename(fif_path)} for event '{chosen_lbl}'.")

    data = epochs.get_data()  # (N,K,T)
    ch_names = [epochs.ch_names[i] for i in range(len(epochs.ch_names))]

    lap_epochs, lap_names = derive_laplacian_epochs(data, ch_names)
    sfreq = epochs.info["sfreq"]

    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_b != lap_names:
            idx_map = {n: i for i, n in enumerate(lap_names_b)}
            idx = [idx_map[n] for n in lap_names if n in idx_map]
            bmean, bstd = bmean[idx], bstd[idx]
        lap_epochs = apply_baseline_norm(lap_epochs, bmean, bstd, BASELINE_MODE)

    # Jitter-align & median template (TD)
    max_shift = int(round(MAX_JITTER_SEC * sfreq))
    aligned = lap_epochs.copy()
    ref = np.nanmedian(aligned, axis=0)
    if max_shift > 0:
        for _ in range(2):
            for i in range(aligned.shape[0]):
                best_lag, best_score = 0, -np.inf
                for lag in range(-max_shift, max_shift + 1):
                    if lag < 0:
                        a = aligned[i, :, :lag]; b = ref[:, -lag:]
                    elif lag > 0:
                        a = aligned[i, :, lag:]; b = ref[:, :-lag]
                    else:
                        a = aligned[i];         b = ref
                    s = np.sum((a - a.mean(axis=-1, keepdims=True)) * (b - b.mean(axis=-1, keepdims=True)))
                    if s > best_score:
                        best_score, best_lag = s, lag
                aligned[i] = np.roll(aligned[i], shift=best_lag, axis=-1)
            ref = np.nanmedian(aligned, axis=0)
    templ_td = np.median(aligned, axis=0)
    templ_td = templ_td - templ_td.mean(axis=-1, keepdims=True)

    # Envelope template
    env_epochs = make_env_feature(lap_epochs, sfreq, ENV_BANDS, ENV_SMOOTH_SEC)
    templ_en = np.median(env_epochs, axis=0) if env_epochs.size else np.zeros((0, templ_td.shape[1]))
    if templ_en.size:
        templ_en = templ_en - templ_en.mean(axis=-1, keepdims=True)

    return {"TD": templ_td, "EN": templ_en}, sfreq, lap_names, chosen_lbl

# =========================
# ====== PLOTTING =========
# =========================

def zscore_channels(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True) + EPS
    return (arr - mu) / sd

def plot_motif_summary(label, templ_td, lap_names, sfreq, ncc_td, templ_en, ncc_en, best_snippet_td, tmin, outdir, show=False):
    ensure_outdir(outdir)
    L = templ_td.shape[1]
    t_axis = np.arange(L) / sfreq + tmin

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(f"Motif {label} — Laplacian TD + Envelope", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("TD Template (z-scored per channel)")
    im1 = ax1.imshow(zscore_channels(templ_td), aspect="auto", origin="lower",
                     extent=[t_axis[0], t_axis[-1], 0, templ_td.shape[0]])
    ax1.set_ylabel("Lap Channels")
    if len(lap_names) >= 2:
        ax1.set_yticks([0, len(lap_names)-1]); ax1.set_yticklabels([lap_names[0], lap_names[-1]])
    cb1 = plt.colorbar(im1, ax=ax1); cb1.set_label("z")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Global NCC (TD and Envelope) over target")
    if ncc_td is not None and ncc_td.size:
        ax2.plot(ncc_td, label="TD")
    if ncc_en is not None and ncc_en.size:
        ax2.plot(ncc_en, label="EN", alpha=0.8)
    ax2.set_ylabel("NCC"); ax2.set_xlabel("Window start (samples)")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(3, 1, 3)
    if best_snippet_td is not None and best_snippet_td.size:
        im3 = ax3.imshow(zscore_channels(best_snippet_td), aspect="auto", origin="lower",
                         extent=[t_axis[0], t_axis[-1], 0, best_snippet_td.shape[0]])
        ax3.set_title("Best TD detection (z-scored per channel)")
        ax3.set_ylabel("Lap Channels"); ax3.set_xlabel("Time (s)")
        if len(lap_names) >= 2:
            ax3.set_yticks([0, len(lap_names)-1]); ax3.set_yticklabels([lap_names[0], lap_names[-1]])
        cb3 = plt.colorbar(im3, ax=ax3); cb3.set_label("z")
    else:
        ax3.set_title("No confident TD detection shown")
        ax3.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if SAVE_PLOTS:
        fig.savefig(os.path.join(outdir, f"motif_{label}_lap_fusion.png"), dpi=150)
    if show: plt.show()
    else: plt.close(fig)

# =========================
# ====== DEDUP LOGIC ======
# =========================

def dedup_events_by_gap(events, sfreq, min_isi_sec=0.7, keep="first"):
    """Collapse triggers closer than min_isi_sec apart."""
    if len(events) == 0:
        return events
    min_gap = int(round(min_isi_sec * sfreq))
    kept_idxs = [0]
    for i in range(1, len(events)):
        if events[i, 0] - events[kept_idxs[-1], 0] >= min_gap:
            kept_idxs.append(i)
        elif keep == "last":
            kept_idxs[-1] = i
    return events[kept_idxs]

def build_search_interval(ev_samp: int, sf: float, L: int, jitter_sec: float, tmin: float, tmax: float, N: int) -> Tuple[int, int]:
    """Return [seg_start, seg_end) window (samples) we search for this trigger."""
    start0 = int(ev_samp + tmin * sf)
    seg_start = max(0, start0 - int(round(jitter_sec * sf)))
    seg_end = min(N, start0 + L + int(round(jitter_sec * sf)))
    return seg_start, seg_end

def dedup_events_by_overlap(
    events: np.ndarray, sfreq: float, N: int, L: int, jitter_sec: float,
    strategy: str,
    tmin: float, tmax: float,
    lap_all: Optional[np.ndarray] = None,
    env_all: Optional[np.ndarray] = None,
    templates: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    label_idx_td: Optional[Dict[str, List[int]]] = None,
    label_idx_en: Optional[Dict[str, List[int]]] = None,
    fusion_alpha: float = 0.5, use_fusion: bool = True,
    ch_weights: Optional[Dict[str, np.ndarray]] = None,
    labels_use: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Group events whose search windows overlap and keep one per group.
    strategy:
      - 'overlap_first'  : keep first per overlapping group
      - 'energy'         : keep window with max Laplacian energy
      - 'overlap_fused_ncc' : keep window with max fused NCC across labels
    """
    if len(events) == 0:
        return events

    # Build intervals
    intervals = []
    for i in range(len(events)):
        a, b = build_search_interval(events[i, 0], sfreq, L, jitter_sec, tmin, tmax, N)
        intervals.append((a, b, i))
    # Sort by start
    intervals.sort(key=lambda x: x[0])

    groups = [[intervals[0]]]
    for itv in intervals[1:]:
        if itv[0] <= groups[-1][-1][1]:  # overlap
            groups[-1].append(itv)
        else:
            groups.append([itv])

    kept = []
    for grp in groups:
        if strategy == "overlap_first":
            kept.append(grp[0][2]); continue
        if strategy == "energy":
            best_i, best_val = grp[0][2], -np.inf
            for (a, b, idx) in grp:
                energy = float(np.sum(lap_all[:, a:b] ** 2))
                if energy > best_val:
                    best_val, best_i = energy, idx
            kept.append(best_i); continue
        if strategy == "overlap_fused_ncc":
            best_i, best_val = grp[0][2], -np.inf
            for (a, b, idx) in grp:
                # compute best fused score across labels in [a,b)
                best_sc = -np.inf
                for lab in labels_use:
                    td = lap_all[label_idx_td[lab], a:b]
                    sc_td = float(np.max(ncc_multichannel(td, templates[lab]["TD"],
                                                          weights=ch_weights[lab])))
                    if env_all.size and templates[lab]["EN"].size and label_idx_en[lab]:
                        en = env_all[label_idx_en[lab], a:b]
                        sc_en = float(np.max(ncc_multichannel(en, templates[lab]["EN"])))
                        sc = fusion_alpha * sc_td + (1.0 - fusion_alpha) * sc_en if use_fusion else sc_td
                    else:
                        sc = sc_td
                    if sc > best_sc:
                        best_sc = sc
                if best_sc > best_val:
                    best_val, best_i = best_sc, idx
            kept.append(best_i); continue
        raise ValueError("Unknown overlap strategy")
    kept = sorted(set(kept))
    return events[kept]

# =========================
# ========= MAIN ==========
# =========================

def main():
    ensure_outdir(OUTPUT_DIR)
    assert len(TEMPLATE_SOURCES) == 4, "Expecting A,B,C,D templates."

    # 1) Build templates
    templates: Dict[str, Dict[str, np.ndarray]] = {}
    lap_names_by_label: Dict[str, List[str]] = {}
    sfreq_ref: Optional[float] = None

    print("[Templates] Building TD+EN (Laplacian) ...")
    for label, entry in TEMPLATE_SOURCES.items():
        if isinstance(entry, (list, tuple)):
            fpath, ev = entry[0], (entry[1] if len(entry) > 1 else None)
        else:
            fpath, ev = entry, None
        print(f"  • {label}: {os.path.basename(fpath)}")
        templ_dict, sfreq, lap_names, chosen = build_template_from_file(label, fpath, ev, TMIN, TMAX, RESAMPLE_HZ)
        print(f"    -> Event: '{chosen}'. Lap channels: {lap_names}")
        templates[label] = templ_dict
        lap_names_by_label[label] = lap_names
        if sfreq_ref is None:
            sfreq_ref = sfreq
        else:
            assert abs(sfreq_ref - sfreq) < 1e-6, "All templates must share sampling rate."
    assert sfreq_ref is not None

    # 2) Labels for classification
    labels_all = list(templates.keys())
    labels_use = labels_all if CLASSIFY_LABELS is None else CLASSIFY_LABELS
    print(f"\n[Classify] Using labels: {labels_use}")

    # 3) Reference lap order & reorder per label
    ref_order = lap_names_by_label[labels_use[0]]

    def reorder(template: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
        idx = [names.index(n) for n in ref_order if n in names]
        return template[idx], [names[i] for i in idx]

    label_idx_td: Dict[str, List[int]] = {}
    label_idx_en: Dict[str, List[int]] = {}

    B_env = len(ENV_BANDS)
    for l in labels_use:
        names = lap_names_by_label[l]
        td = templates[l]["TD"]; en = templates[l]["EN"]

        td_re, names_re = reorder(td, names)
        templates[l]["TD"] = td_re
        lap_names_by_label[l] = names_re

        idx_td = [ref_order.index(n) for n in names_re]
        label_idx_td[l] = idx_td

        if en.size and B_env > 0:
            en_blocks = np.split(en, B_env, axis=0)
            en_blocks_reordered = []
            for blk in en_blocks:
                en_blk_re, _ = reorder(blk, names)
                en_blocks_reordered.append(en_blk_re)
            templates[l]["EN"] = np.concatenate(en_blocks_reordered, axis=0)
            idx_en = []
            for b in range(B_env):
                base = b * len(ref_order)
                idx_en.extend([base + i for i in idx_td])
            label_idx_en[l] = idx_en
        else:
            templates[l]["EN"] = en
            label_idx_en[l] = []

    # 4) Load target & events
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw = load_raw(TARGET_FILE, resample_hz=sfreq_ref)
    events_all, eid_all = get_events_any(raw)
    code, lbl = resolve_event_auto(events_all, eid_all, requested=TARGET_EVENT)
    events = events_all[events_all[:, 2] == code]
    print(f"[Target] Using event '{lbl}' (code {code}); N={len(events)}")
    if len(events) == 0:
        raise RuntimeError("No matching target events found.")

    # 5) Laplacians over target + baseline norm
    lap_all, lap_names_target = derive_laplacian_continuous(raw)
    if lap_names_target != ref_order:
        name_to_idx = {n: i for i, n in enumerate(lap_names_target)}
        idx = [name_to_idx[n] for n in ref_order if n in name_to_idx]
        lap_all = lap_all[idx]
        lap_names_target = [ref_order[i] for i in range(len(idx))]

    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_base = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_base != ref_order:
            map_idx = {n: i for i, n in enumerate(lap_names_base)}
            idx = [map_idx[n] for n in ref_order if n in map_idx]
            bmean, bstd = bmean[idx], bstd[idx]
        print(f"[Target] Shared Lap baseline: {secs:.2f}s ({BASELINE_MODE}) on {lap_all.shape[0]} ch.")
        lap_all = apply_baseline_norm(lap_all, bmean, bstd, BASELINE_MODE)

    # 6) Envelope feature (continuous)
    env_all = make_env_feature(lap_all, sfreq_ref, ENV_BANDS, ENV_SMOOTH_SEC)

    # 7) Whitening (TD) + channel weights
    W = None
    if USE_WHITEN_TD:
        W = compute_whitener_from_baseline(raw, ref_order)
        lap_all = apply_whiten(W, lap_all)
        for lab in templates:
            templates[lab]["TD"] = apply_whiten(W, templates[lab]["TD"])

    ch_weights: Dict[str, np.ndarray] = {}
    for lab in labels_use:
        td = templates[lab]["TD"]
        rms = np.sqrt(np.mean(td ** 2, axis=1))
        w = (rms ** WEIGHT_POWER) if WEIGHT_POWER != 0 else np.ones_like(rms)
        w = w + EPS
        w = w / np.mean(w)
        ch_weights[lab] = w

    # 8) Dynamic thresholds from null windows (calibrate BEFORE dedup is fine)
    print("\n[Calibrate] Estimating per-label null thresholds ...")
    sf = sfreq_ref
    L = next(iter(templates.values()))["TD"].shape[1]
    N = lap_all.shape[1]
    null_gap = int(round(NULL_MIN_GAP_SEC * sf))
    trigger_samps = events[:, 0]
    forbidden = np.zeros(N, dtype=bool)
    for s in trigger_samps:
        a = max(0, s - (L + null_gap)); b = min(N, s + (L + null_gap))
        forbidden[a:b] = True

    rng = np.random.default_rng(12345)
    starts = []
    attempts = 0
    while len(starts) < NULL_SAMPLES and attempts < NULL_SAMPLES * 20:
        s = rng.integers(0, max(1, N - L))
        if not forbidden[s:s+L].any():
            starts.append(int(s))
        attempts += 1
    if len(starts) < max(50, NULL_SAMPLES // 2):
        print(f"  (warn) only sampled {len(starts)} null windows; thresholds may be noisy")

    dyn_thresh: Dict[str, float] = {}
    for lab in labels_use:
        idx_td = label_idx_td[lab]
        idx_en = label_idx_en[lab]
        t_td = templates[lab]["TD"]
        t_en = templates[lab]["EN"]
        w_td = ch_weights[lab]

        scores = []
        for s in starts:
            seg_td = lap_all[idx_td, s:s+L]
            sc_td = float(np.max(ncc_multichannel(seg_td, t_td, weights=w_td)))
            if env_all.size and t_en.size and idx_en:
                seg_en = env_all[idx_en, s:s+L]
                sc_en = float(np.max(ncc_multichannel(seg_en, t_en)))
                sc = FUSION_ALPHA * sc_td + (1.0 - FUSION_ALPHA) * sc_en if USE_FUSION else sc_td
            else:
                sc = sc_td
            scores.append(sc)
        thr = float(np.percentile(scores, NULL_PERCENTILE)) if len(scores) else 0.0
        dyn_thresh[lab] = thr
        print(f"  • {lab}: null-threshold @ P{NULL_PERCENTILE:.1f} = {thr:.3f}")

    # 9) ---- SURGICAL DEDUP: overlap-aware grouping + fused NCC selection ----
    print("\n[Dedup] Strategy:", DEDUP_STRATEGY)
    if DEDUP_STRATEGY == "none":
        pass
    elif DEDUP_STRATEGY == "gap":
        events = dedup_events_by_gap(events, sfreq_ref, min_isi_sec=MIN_ISI_SEC, keep="first")
    elif DEDUP_STRATEGY in ("overlap_first", "overlap_energy", "overlap_fused_ncc"):
        events = dedup_events_by_overlap(
            events, sfreq_ref, N=N, L=L, jitter_sec=SEARCH_JITTER_SEC,
            strategy=("overlap_first" if DEDUP_STRATEGY=="overlap_first"
                      else "energy" if DEDUP_STRATEGY=="overlap_energy"
                      else "overlap_fused_ncc"),
            tmin=TMIN, tmax=TMAX,
            lap_all=lap_all, env_all=env_all, templates=templates,
            label_idx_td=label_idx_td, label_idx_en=label_idx_en,
            fusion_alpha=FUSION_ALPHA, use_fusion=USE_FUSION,
            ch_weights=ch_weights, labels_use=labels_use
        )
    else:
        raise ValueError("Unknown dedup strategy.")
    print(f"[Dedup] Triggers after dedup: {len(events)}")

    # 10) Trigger-locked classification (dynamic thresholds + ratio)
    print("\n[Classify] One label per trigger (dynamic thresholds + ratio) ...")
    win_center_samps = int(((TMIN + TMAX) / 2.0) * sf)
    jitter = int(round(SEARCH_JITTER_SEC * sf))

    labels_pred, times_pred = [], []
    for ev in events[:, 0]:
        best_label, best_score, second_best = "UNK", -1.0, -1.0
        best_time = (ev + win_center_samps) / sf

        start0 = int(ev + TMIN * sf)
        seg_start = max(0, start0 - jitter)
        seg_end   = min(N, start0 + L + jitter)
        if seg_end - seg_start < L + 1:
            labels_pred.append("UNK"); times_pred.append(best_time); continue

        seg_td_common = lap_all[:, seg_start:seg_end]
        seg_en_common = env_all[:, seg_start:seg_end] if env_all.size else np.zeros((0, seg_end - seg_start))

        per_label_best = {}
        for lab in labels_use:
            seg_td = seg_td_common[label_idx_td[lab], :]
            w_td = ch_weights[lab]
            ncc_td = ncc_multichannel(seg_td, templates[lab]["TD"], weights=w_td)
            k_td = int(np.argmax(ncc_td)); sc_td = float(ncc_td[k_td])

            if env_all.size and templates[lab]["EN"].size and label_idx_en[lab]:
                seg_en = seg_en_common[label_idx_en[lab], :]
                ncc_en = ncc_multichannel(seg_en, templates[lab]["EN"])
                k_en = int(np.argmax(ncc_en)); sc_en = float(ncc_en[k_en])
                sc = FUSION_ALPHA * sc_td + (1.0 - FUSION_ALPHA) * sc_en if USE_FUSION else sc_td
                k = k_td
            else:
                sc = sc_td
                k = k_td

            per_label_best[lab] = (sc, k)

        sorted_labs = sorted(per_label_best.items(), key=lambda kv: kv[1][0], reverse=True)
        (lab1, (sc1, k1)) = sorted_labs[0]
        (lab2, (sc2, k2)) = sorted_labs[1] if len(sorted_labs) > 1 else ("UNK", (-1.0, 0))

        passes = sc1 >= dyn_thresh.get(lab1, -1.0)
        better = (sc1 >= RATIO_ALPHA * max(sc2, -1.0)) or (sc1 - max(sc2, -1.0) >= DELTA_MARGIN)

        if passes and better:
            best_label = lab1
            best_time = (seg_start + k1 + (L // 2)) / sf
        else:
            best_label = "UNK"

        labels_pred.append(best_label)
        times_pred.append(best_time)

    print("\n=== Recognized Pattern Order (Trigger-locked, dynamic + surgical dedup) ===")
    print(" ".join(labels_pred))
    unk = sum(1 for x in labels_pred if x == "UNK")
    print(f"UNK count: {unk} / {len(labels_pred)}")

    # 11) Visualization over full target (TD & EN)
    print("\n[Visualization] Global NCC per motif ...")
    ensure_outdir(OUTPUT_DIR)
    for lab in templates.keys():
        td_global = lap_all[label_idx_td.get(lab, []), :] if lab in label_idx_td else lap_all
        w_td = ch_weights.get(lab, np.ones(td_global.shape[0]))
        ncc_td = ncc_multichannel(td_global, templates[lab]["TD"], weights=w_td)

        if (env_all.size and templates[lab]["EN"].size and label_idx_en.get(lab, [])):
            en_global = env_all[label_idx_en[lab], :]
            ncc_en = ncc_multichannel(en_global, templates[lab]["EN"])
        else:
            ncc_en = np.array([])

        peaks, _ = find_peaks(ncc_td, height=NCC_THRESHOLD_GLOBAL, distance=max(1, int(round(0.35 * sf))))
        best_snippet = None
        L = templates[lab]["TD"].shape[1]
        if len(peaks) > 0:
            best_idx = int(peaks[np.argmax(ncc_td[peaks])])
            start = best_idx; end = start + L
            start = max(0, start); end = min(lap_all.shape[1], end)
            best_snippet = td_global[:, start:end]
            if best_snippet.shape[1] < L:
                pad = np.zeros((best_snippet.shape[0], L - best_snippet.shape[1]))
                best_snippet = np.hstack([best_snippet, pad])

        plot_motif_summary(
            label=lab,
            templ_td=templates[lab]["TD"],
            lap_names=lap_names_by_label.get(lab, ref_order),
            sfreq=sf,
            ncc_td=ncc_td,
            templ_en=templates[lab]["EN"],
            ncc_en=ncc_en,
            best_snippet_td=best_snippet,
            tmin=TMIN,
            outdir=OUTPUT_DIR,
            show=SHOW_PLOTS
        )

    if SAVE_PLOTS:
        print(f"Figures saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

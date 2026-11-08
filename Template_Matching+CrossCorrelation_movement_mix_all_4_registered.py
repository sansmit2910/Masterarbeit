# -*- coding: utf-8 -*-
"""
EEG Template Matching: Trigger-locked + Laplacian + Mu/Beta Envelope Fusion
---------------------------------------------------------------------------
Goal: separate A (fist) vs B (pinch) more reliably.

• One decision per trigger (± jitter).
• Motor-focused virtual Laplacians: C3L, CzL, C4L (C* - mean(FC*, CP*)), with graceful fallback.
• Two features per motif:
    (1) Time-domain NCC on Laplacians (TD).
    (2) Mu/Beta Hilbert-envelope NCC (EN) on the same Laplacians.
  Final fused score = FUSION_ALPHA * TD + (1 - FUSION_ALPHA) * EN.  (Set USE_FUSION=False or ENV_BANDS=[] for TD-only.)
• Shared baseline (pre-first-annotation) computed on derived Laplacians; applied to both templates & target.
• Common reference order across labels (Laplacians) with per-label subsetting so channel counts always match.
• Figures shown + saved.

Deps: mne, numpy, scipy, matplotlib
"""

import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlate, hilbert, find_peaks

# =======================
# ====== CONFIG =========
# =======================

TEMPLATE_SOURCES: Dict[str, Union[str, Tuple[str, Union[int, str, None]]]] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"
TARGET_EVENT: Optional[Union[int, str]] = None  # leave None to auto-pick most frequent

# Epoch window for templates (sec)
TMIN = 0.0
TMAX = 0.6

RESAMPLE_HZ: Optional[float] = None       # keep native fs
MAX_JITTER_SEC = 0.05                     # align epochs when building templates
SEARCH_JITTER_SEC = 0.40                  # local search around trigger during classification

# Baseline on derived Laplacians from BEFORE first annotation
BASELINE_USE_PRE_ANNOT = True
BASELINE_MODE = "zscore"                  # 'zscore' or 'demean'
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
EPS = 1e-12

# Focus on real movements first
CLASSIFY_LABELS: Optional[List[str]] = ["A", "B"]  # set None to allow all four later

# Envelope feature (set [] to disable)
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]
ENV_SMOOTH_SEC = 0.05

# Fusion
USE_FUSION = True
FUSION_ALPHA = 0.5   # 1=TD only, 0=EN only

# Thresholds
PER_LABEL_FUSED_THRESH = {'A': 0.20, 'B': 0.20, 'C': 0.20, 'D': 0.20}
DELTA_MARGIN = 0.015

# Viz
NCC_THRESHOLD_GLOBAL = 0.30
OUTPUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_template_match_output"
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

# ---------- Virtual Laplacian helpers ----------

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
        return base_arr.copy()  # fallback to base only

def derive_laplacian_continuous(raw: mne.io.BaseRaw,
                                start: Optional[int] = None,
                                stop: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Return (C', T) of [C3L, CzL, C4L] derived from continuous raw."""
    # Coerce None -> ints (older MNE wants ints here)
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
        raise RuntimeError("No central/Laplacian signals could be derived. Check your channel names.")
    return np.vstack(out), out_names

def derive_laplacian_epochs(epochs_data: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """epochs_data: (N, C, T) → (N, C', T) for C3L, CzL, C4L (fallbacks if neighbors missing)."""
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
    return np.stack(out, axis=1), out_names  # (N, C', T)

# ---------- Baseline ----------

def pre_annotation_baseline_window(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    sf = raw.info["sfreq"]   # FIX: .info (no call)
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
    """Baseline mean/std computed on derived Laplacians from pre-first-event segment."""
    start, end = pre_annotation_baseline_window(raw, min_sec, margin_sec)
    lap, lap_names = derive_laplacian_continuous(raw, start=start, stop=end)  # (C', T0)
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

# ---------- Envelope feature (optional) ----------

def _safe_filter_data(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Band-pass arrays (not Raw) with a safe highcut below Nyquist. Returns a copy."""
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
    """
    Build band-limited Hilbert envelopes on Laplacian signals.

    Input:
      data_lap: (C, T) for continuous, or (N, C, T) for epochs
    Output:
      (C*B, T) for continuous, or (N, C*B, T) for epochs
    """
    # No bands → return empty feature with correct shape
    if not bands:
        if data_lap.ndim == 1:
            return np.zeros((0, data_lap.shape[0]), dtype=data_lap.dtype)
        if data_lap.ndim == 2:  # (C, T)
            return np.zeros((0, data_lap.shape[1]), dtype=data_lap.dtype)
        elif data_lap.ndim == 3:  # (N, C, T)
            return np.zeros((data_lap.shape[0], 0, data_lap.shape[2]), dtype=data_lap.dtype)
        else:
            raise ValueError(f"data_lap ndim must be 1, 2 or 3, got {data_lap.ndim}")

    # Ensure explicit epoch axis
    if data_lap.ndim == 1:            # (T) -> (1, 1, T)
        data = data_lap[np.newaxis, np.newaxis, :]
        squeeze = True
    elif data_lap.ndim == 2:          # (C, T) -> (1, C, T)
        data = data_lap[np.newaxis, :, :]
        squeeze = True
    elif data_lap.ndim == 3:          # (N, C, T)
        data = data_lap
        squeeze = False
    else:
        raise ValueError(f"data_lap ndim must be 1, 2 or 3, got {data_lap.ndim}")

    N, C, T = data.shape
    feats = []

    for (l, h) in bands:
        # Safe band-pass per epoch
        X = np.empty_like(data)
        for n in range(N):
            X[n] = _safe_filter_data(data[n], sfreq, l, h)  # (C, T)
        env = hilbert_envelope(X, axis=-1)  # (N, C, T)

        # Smooth
        k = max(1, int(round(smooth_sec * sfreq)))
        if k > 1:
            w = np.ones(k) / k
            env = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, env)

        feats.append(env)

    feat = np.concatenate(feats, axis=1)  # (N, C*B, T)
    return feat[0] if squeeze else feat

# ---------- NCC ----------

def ncc_multichannel(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    signal: (C, N); template: (C, L). Returns NCC of length N-L+1
    """
    C, N = signal.shape
    Ct, L = template.shape
    assert C == Ct, f"Channel count mismatch {C}!={Ct}"
    S_t_total = np.sum(np.sum(template ** 2, axis=1)) + EPS
    ones = np.ones(L)
    numerator = None
    denom_sum = None
    for c in range(C):
        x = signal[c]
        t0 = template[c]
        r_num = correlate(x, t0, mode="valid", method="fft")
        s1 = fftconvolve(x, ones, mode="valid")
        s2 = fftconvolve(x * x, ones, mode="valid")
        lvar = s2 - (s1 * s1) / L
        lvar = np.maximum(lvar, EPS)
        numerator = r_num if numerator is None else (numerator + r_num)
        denom_sum = lvar if denom_sum is None else (denom_sum + lvar)
    denom = np.sqrt(S_t_total * np.maximum(denom_sum, EPS))
    return numerator / denom

# ---------- Templates ----------

def build_template_from_file(
    label: str,
    fif_path: str,
    event_key: Optional[Union[int, str]],
    tmin: float,
    tmax: float,
    resample_hz: Optional[float],
):
    """
    Returns:
      template_dict: {'TD': (C',L), 'EN': (C_env,L)} both on Laplacian channels,
      sfreq, lap_channel_names, chosen_event_label
    """
    raw = load_raw(fif_path, resample_hz)
    events, eid = get_events_any(raw)
    code, chosen_lbl = resolve_event_auto(events, eid, requested=event_key)

    # Build epochs for base+neighbors needed by Laplacians
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

    # Derive Laplacians per epoch → (N, C', T)
    lap_epochs, lap_names = derive_laplacian_epochs(data, ch_names)
    sfreq = epochs.info["sfreq"]

    # Baseline on derived Laplacians using pre-first-annotation segment
    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        # align baseline order
        if lap_names_b != lap_names:
            idx_map = {n: i for i, n in enumerate(lap_names_b)}
            idx = [idx_map[n] for n in lap_names if n in idx_map]
            bmean, bstd = bmean[idx], bstd[idx]
        lap_epochs = apply_baseline_norm(lap_epochs, bmean, bstd, BASELINE_MODE)

    # Jitter align & median template (TD)
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

    # Envelope template on Laplacians (optional bands)
    env_epochs = make_env_feature(lap_epochs, sfreq, ENV_BANDS, ENV_SMOOTH_SEC)  # (N, C*B, T) or (N,0,T)
    templ_en = np.median(env_epochs, axis=0) if env_epochs.size else np.zeros((0, templ_td.shape[1]))
    if templ_en.size:
        templ_en = templ_en - templ_en.mean(axis=-1, keepdims=True)

    return {"TD": templ_td, "EN": templ_en}, sfreq, lap_names, chosen_lbl

# ---------- Plotting ----------

def zscore_channels(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True) + EPS
    return (arr - mu) / sd

def plot_motif_summary(
    label: str,
    templ_td: np.ndarray,
    lap_names: List[str],
    sfreq: float,
    ncc_td: Optional[np.ndarray],
    templ_en: np.ndarray,
    ncc_en: Optional[np.ndarray],
    best_snippet_td: Optional[np.ndarray],
    tmin: float,
    outdir: str,
    show: bool = False
):
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
        fname = f"motif_{label}_lap_fusion.png".replace(os.sep, "-")
        fig.savefig(os.path.join(outdir, fname), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

# =======================
# ====== MAIN ===========
# =======================

def main():
    ensure_outdir(OUTPUT_DIR)
    assert len(TEMPLATE_SOURCES) == 4, "Expecting A,B,C,D templates."

    # 1) Build templates (TD & EN) on Laplacians
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
        templ_dict, sfreq, lap_names, chosen = build_template_from_file(
            label, fpath, ev, TMIN, TMAX, RESAMPLE_HZ
        )
        print(f"    -> Event: '{chosen}'. Lap channels: {lap_names}")
        templates[label] = templ_dict
        lap_names_by_label[label] = lap_names
        if sfreq_ref is None:
            sfreq_ref = sfreq
        else:
            assert abs(sfreq_ref - sfreq) < 1e-6, "All templates must share sampling rate."
    assert sfreq_ref is not None

    # 2) Classification labels
    labels_all = list(templates.keys())
    labels_use = labels_all if CLASSIFY_LABELS is None else CLASSIFY_LABELS
    print(f"\n[Classify] Using labels: {labels_use}")

    # 3) Establish a reference Laplacian order & per-label indices into it
    ref_order = lap_names_by_label[labels_use[0]]

    def reorder(template: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
        idx = [names.index(n) for n in ref_order if n in names]
        return template[idx], [names[i] for i in idx]

    # Reorder templates to reference order; compute per-label channel indices
    label_idx_td: Dict[str, List[int]] = {}
    label_idx_en: Dict[str, List[int]] = {}

    B_env = len(ENV_BANDS)
    for l in labels_use:
        names = lap_names_by_label[l]
        td = templates[l]["TD"]; en = templates[l]["EN"]

        td_re, names_re = reorder(td, names)
        templates[l]["TD"] = td_re
        lap_names_by_label[l] = names_re

        # Indices into ref_order for this label's TD channels
        idx_td = [ref_order.index(n) for n in names_re]
        label_idx_td[l] = idx_td

        # Reorder EN blocks channel-wise to the same subset/order
        if en.size and B_env > 0:
            C_td = td_re.shape[0]
            en_blocks = np.split(en, B_env, axis=0)  # stacked per band
            en_blocks_reordered = []
            for blk in en_blocks:
                en_blk_re, _ = reorder(blk, names)  # reorder using original names -> ref subset
                en_blocks_reordered.append(en_blk_re)
            en_re = np.concatenate(en_blocks_reordered, axis=0)
            templates[l]["EN"] = en_re

            # Build indices for EN selection on lap-based env_all: replicate TD idx per band
            idx_en = []
            for b in range(B_env):
                base = b * len(ref_order)
                idx_en.extend([base + i for i in idx_td])
            label_idx_en[l] = idx_en
        else:
            templates[l]["EN"] = en  # (0, L) when disabled
            label_idx_en[l] = []

    # 4) Load target & pick event type
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw = load_raw(TARGET_FILE, resample_hz=sfreq_ref)
    events_all, eid_all = get_events_any(raw)
    code, lbl = resolve_event_auto(events_all, eid_all, requested=TARGET_EVENT)
    events = events_all[events_all[:, 2] == code]
    print(f"[Target] Using event '{lbl}' (code {code}); N={len(events)}")
    if len(events) == 0:
        raise RuntimeError("No matching target events found.")

    # 5) Build continuous Laplacians for the whole target; shared baseline on them
    lap_all, lap_names_target = derive_laplacian_continuous(raw)  # (C_ref, N)

    # Align lap_all order to ref_order
    if lap_names_target != ref_order:
        name_to_idx = {n: i for i, n in enumerate(lap_names_target)}
        idx = [name_to_idx[n] for n in ref_order if n in name_to_idx]
        lap_all = lap_all[idx]
        lap_names_target = [ref_order[i] for i in range(len(idx))]
    C_ref = lap_all.shape[0]

    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_base = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_base != ref_order:
            map_idx = {n: i for i, n in enumerate(lap_names_base)}
            idx = [map_idx[n] for n in ref_order if n in map_idx]
            bmean, bstd = bmean[idx], bstd[idx]
        print(f"[Target] Shared Lap baseline: {secs:.2f}s ({BASELINE_MODE}) on {C_ref} ch.")
        lap_all = apply_baseline_norm(lap_all, bmean, bstd, BASELINE_MODE)

    # 6) Envelope feature on the same Laplacians (continuous)
    env_all = make_env_feature(lap_all, sfreq_ref, ENV_BANDS, ENV_SMOOTH_SEC)  # (C_ref*B, N) or (0,N)

    # 7) Trigger-locked classification (fused TD+EN), with per-label subsetting
    print("\n[Trigger-locked] Classifying one label per trigger (TD+EN fusion) ...")
    sf = sfreq_ref
    L = next(iter(templates.values()))["TD"].shape[1]
    N = lap_all.shape[1]
    win_center_samps = int(((TMIN + TMAX) / 2.0) * sf)
    jitter = int(round(SEARCH_JITTER_SEC * sf))

    labels_pred, times_pred, scores_pred = [], [], []
    for ev in events[:, 0]:
        best_label, best_score, second_best = "UNK", -1.0, -1.0
        best_time = (ev + win_center_samps) / sf

        start0 = int(ev + TMIN * sf)
        seg_start = max(0, start0 - jitter)
        seg_end   = min(N, start0 + L + jitter)
        if seg_end - seg_start < L + 1:
            labels_pred.append("UNK"); times_pred.append(best_time); scores_pred.append(-1.0); continue

        # Common windows from continuous features
        seg_td_common = lap_all[:, seg_start:seg_end]            # (C_ref, Tloc)
        seg_en_common = env_all[:, seg_start:seg_end] if env_all.size else np.zeros((0, seg_end - seg_start))

        for lab in labels_use:
            # Subset per label
            seg_td = seg_td_common[label_idx_td[lab], :]
            ncc_td = ncc_multichannel(seg_td, templates[lab]["TD"])
            k_td = int(np.argmax(ncc_td)); sc_td = float(ncc_td[k_td])

            if env_all.size and templates[lab]["EN"].size and label_idx_en[lab]:
                seg_en = seg_en_common[label_idx_en[lab], :]
                ncc_en = ncc_multichannel(seg_en, templates[lab]["EN"])
                k_en = int(np.argmax(ncc_en)); sc_en = float(ncc_en[k_en])
                sc = FUSION_ALPHA * sc_td + (1.0 - FUSION_ALPHA) * sc_en if USE_FUSION else sc_td
                k  = k_td  # use TD center
            else:
                sc = sc_td
                k  = k_td

            if sc < PER_LABEL_FUSED_THRESH.get(lab, -1.0):
                continue
            if sc > best_score:
                second_best = best_score
                best_score = sc
                best_label = lab
                best_time = (seg_start + k + (L // 2)) / sf
            elif sc > second_best:
                second_best = sc

        if best_score >= 0 and best_score - max(second_best, -1.0) >= DELTA_MARGIN:
            labels_pred.append(best_label)
        else:
            labels_pred.append("UNK")
        times_pred.append(best_time)
        scores_pred.append(best_score)

    print("\n=== Recognized Pattern Order (Trigger-locked, TD+EN fusion) ===")
    print(" ".join(labels_pred))

    # Report counts
    used = labels_use
    counts = {l: 0 for l in used}
    unk = 0
    for l in labels_pred:
        if l in counts: counts[l] += 1
        else: unk += 1
    miss = [l for l, c in counts.items() if c == 0]
    if miss:
        print("\nMotifs NOT found (trigger-locked): " + ", ".join(miss))
    print(f"UNK count: {unk} / {len(labels_pred)}")

    # 8) Visualization per motif (global NCC for TD & EN) — subset channels per label
    print("\n[Visualization] Computing global NCC (TD & EN) per motif ...")
    ensure_outdir(OUTPUT_DIR)
    for lab in templates.keys():
        # TD global NCC
        td_global = lap_all[label_idx_td.get(lab, []), :] if lab in label_idx_td else lap_all
        ncc_td = ncc_multichannel(td_global, templates[lab]["TD"])

        # EN global NCC
        if (env_all.size and templates[lab]["EN"].size and label_idx_en.get(lab, [])):
            en_global = env_all[label_idx_en[lab], :]
            ncc_en = ncc_multichannel(en_global, templates[lab]["EN"])
        else:
            ncc_en = np.array([])

        # Best TD snippet (for plotting)
        peaks, _ = find_peaks(ncc_td, height=NCC_THRESHOLD_GLOBAL, distance=max(1, int(round(0.35 * sf))))
        best_snippet = None
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
            lap_names=lap_names_by_label[lab],
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
        print(f"\nFigures saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

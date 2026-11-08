# -*- coding: utf-8 -*-
"""
EEG Template Matching via DTW (Motor Imagery only: C & D)
---------------------------------------------------------
- Templates from MI .fif files (C, D) using virtual Laplacians (C3L, CzL, C4L; safe fallbacks).
- Hilbert envelopes in mu/beta bands with smoothing.
- Shared baseline (pre-first-annotation) on Laplacians.
- Jitter realign + optional template trimming.
- DTW with Sakoe–Chiba band; sign-invariant; local z per window.
- Fusion TD + EN; per-label z-calibration + auto-thresholds; argmax fallback.
- Trigger-locked classification with a single overall progress bar.
- Prints + saves sequence (TXT + CSV).
- Per-motif visuals: template + global DTW similarity + best snippet.

Deps: mne, numpy, scipy, matplotlib, tqdm
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter, defaultdict

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
from mne.filter import filter_data

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get('total', 0))

# =======================
# ====== CONFIG =========
# =======================

# --- MI templates only ---
TEMPLATE_SOURCES: Dict[str, Union[str, Tuple[str, Union[int, str, None]]]] = {
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

# --- MI target (edit if needed) ---
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"
TARGET_EVENT: Optional[Union[int, str]] = None  # None -> auto-pick most frequent

# Epoch window for templates (sec)
TMIN = 0.0
TMAX = 0.6

RESAMPLE_HZ: Optional[float] = None       # keep native fs
MAX_JITTER_SEC = 0.06                     # align epochs when building templates
SEARCH_JITTER_SEC = 0.25                  # local search around trigger during classification

# Baseline (on derived Laplacians) from BEFORE first annotation in each file
BASELINE_USE_PRE_ANNOT = True
BASELINE_MODE = "zscore"                  # 'zscore' or 'demean'
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
EPS = 1e-12

# Classification labels (None => all templates compete) -> only C,D exist
CLASSIFY_LABELS: Optional[List[str]] = None

# Envelope feature (MI-friendly)
ENV_BANDS = [(8.0, 13.0), (13.0, 30.0)]
ENV_SMOOTH_SEC = 0.06

# Fusion
USE_FUSION = True
FUSION_ALPHA = 0.35   # slightly EN-leaning for MI (TD weight)

# DTW parameters
DTW_BAND_SEC = 0.05         # Sakoe–Chiba half-band (sec)
DTW_SIM_SCALE = 1.0         # distance->similarity scaling
DTW_STRIDE_GLOBAL = 8       # stride for global curves (viz speed)

# Decision thresholds (base); auto-thresholds will be max(base, auto)
PER_LABEL_FUSED_THRESH = {'C': 0.18, 'D': 0.18}
DELTA_MARGIN_RANK = 0.20    # min z-score gap best vs second

# Auto-thresholding from fused-score distribution (per label)
AUTO_THRESH = True
AUTO_THRESH_Q = 0.60
ARGMAX_FALLBACK = True

# Template trimming (keep top-Q most self-similar epochs)
TRIM_TEMPLATES = True
TRIM_Q = 0.70
TRIM_DTW_BAND_SEC = 0.04

# Per-label z-calibration across events
USE_LABEL_ZCAL = True

# Group prior OFF (both are MI)
USE_GROUP_PRIOR = False

# Viz
SIM_THRESHOLD_GLOBAL = 0.30
OUTPUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_template_match_output"
SAVE_PLOTS = True
SHOW_PLOTS = True

# Diagnostics
PRINT_PER_LABEL_MEANS = True
PRINT_SCORE_RANGES = True

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
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for lbl, c in event_id_map.items():
        if c == code:
            return c, lbl
    return int(code), f"_code:{int(code)}"

# ---------- Virtual Laplacians ----------

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
        raise RuntimeError("No central/Laplacian signals could be derived. Check your channel names.")
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
    return np.stack(out, axis=1), out_names  # (N, C', T)

# ---------- Baseline ----------

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

# ---------- Envelope feature ----------

def _safe_filter_data(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
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
        if data_lap.ndim == 2:
            return np.zeros((0, data_lap.shape[1]), dtype=data_lap.dtype)
        elif data_lap.ndim == 3:
            return np.zeros((data_lap.shape[0], 0, data_lap.shape[2]), dtype=data_lap.dtype)
        else:
            raise ValueError(f"data_lap ndim must be 2 or 3, got {data_lap.ndim}")

    if data_lap.ndim == 2:          # (C, T) -> (1, C, T)
        data = data_lap[np.newaxis, :, :]
        squeeze = True
    elif data_lap.ndim == 3:        # (N, C, T)
        data = data_lap
        squeeze = False
    else:
        raise ValueError(f"data_lap ndim must be 2 or 3, got {data_lap.ndim}")

    N, C, T = data.shape
    feats = []
    for (l, h) in bands:
        X = np.empty_like(data)
        for n in range(N):
            X[n] = _safe_filter_data(data[n], sfreq, l, h)
        env = hilbert_envelope(X, axis=-1)  # (N, C, T)
        k = max(1, int(round(smooth_sec * sfreq)))
        if k > 1:
            w = np.ones(k) / k
            env = np.apply_along_axis(lambda x: np.convolve(x, w, mode="same"), -1, env)
        feats.append(env)
    feat = np.concatenate(feats, axis=1)  # (N, C*B, T)
    return feat[0] if squeeze else feat

# ---------- DTW ----------

def _local_z(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + eps
    return (x - mu) / sd

def dtw_distance_multichannel(seg: np.ndarray, templ: np.ndarray,
                              band_radius: int,
                              weights: Optional[np.ndarray] = None) -> float:
    C, Ls = seg.shape
    Ct, Lt = templ.shape
    assert C == Ct, f"Channel mismatch {C}!={Ct}"
    if weights is None:
        weights = np.ones(C, dtype=float)
    else:
        weights = np.asarray(weights, float)
        assert weights.shape[0] == C

    # cost matrix
    diff = (templ[:, :, None] - seg[:, None, :])  # (C, Lt, Ls)
    diff2 = (weights[:, None, None] * diff) ** 2
    cost = np.sqrt(np.sum(diff2, axis=0))  # (Lt, Ls)

    r = int(max(0, band_radius))
    INF = 1e18
    D = np.full((Lt+1, Ls+1), INF, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, Lt+1):
        j_start = max(1, i - r)
        j_end   = min(Ls, i + r)
        row = cost[i-1]
        Di_1 = D[i-1]
        for j in range(j_start, j_end+1):
            c = row[j-1]
            D[i, j] = c + min(Di_1[j], D[i, j-1], Di_1[j-1])
    dist = D[Lt, Ls]
    norm = max(1.0, Lt + Ls)
    return dist / norm

def dtw_similarity(seg: np.ndarray, templ: np.ndarray,
                   band_radius: int, weights: Optional[np.ndarray] = None,
                   scale: float = 1.0) -> float:
    d = dtw_distance_multichannel(seg, templ, band_radius, weights=weights)
    return 1.0 / (1.0 + (d / max(scale, 1e-9)))

def dtw_similarity_sign_invariant(seg: np.ndarray, templ: np.ndarray,
                                  band_radius: int, weights: Optional[np.ndarray] = None,
                                  scale: float = 1.0) -> float:
    s1 = dtw_similarity(seg,  templ, band_radius, weights=weights, scale=scale)
    s2 = dtw_similarity(-seg, templ, band_radius, weights=weights, scale=scale)
    return max(s1, s2)

# ---------- Template helpers ----------

def jitter_realign_median(lap_epochs: np.ndarray, sfreq: float, max_jitter_sec: float) -> np.ndarray:
    max_shift = int(round(max_jitter_sec * sfreq))
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
    return aligned

def trim_epochs_by_dtw(lap_epochs: np.ndarray, keep_q: float, sfreq: float, band_sec: float) -> np.ndarray:
    if lap_epochs.shape[0] <= 4 or keep_q >= 0.999:
        return lap_epochs
    provisional = np.median(lap_epochs, axis=0)  # (C,T)
    band = int(round(band_sec * sfreq))
    dists = []
    for i in range(lap_epochs.shape[0]):
        seg = _local_z(lap_epochs[i], axis=-1)
        d = 1.0 - dtw_similarity_sign_invariant(seg, provisional, band_radius=band, weights=None, scale=DTW_SIM_SCALE)
        dists.append(d)
    dists = np.array(dists, float)
    k = max(3, int(np.ceil(keep_q * lap_epochs.shape[0])))
    keep_idx = np.argsort(dists)[:k]
    return lap_epochs[keep_idx]

# ---------- Build template ----------

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

    # Build epochs for needed channels
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

    # Baseline
    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_b != lap_names:
            idx_map = {n: i for i, n in enumerate(lap_names_b)}
            idx = [idx_map[n] for n in lap_names if n in idx_map]
            bmean, bstd = bmean[idx], bstd[idx]
        lap_epochs = apply_baseline_norm(lap_epochs, bmean, bstd, BASELINE_MODE)

    lap_epochs = jitter_realign_median(lap_epochs, sfreq, MAX_JITTER_SEC)
    if TRIM_TEMPLATES:
        lap_epochs = trim_epochs_by_dtw(lap_epochs, TRIM_Q, sfreq, TRIM_DTW_BAND_SEC)

    templ_td = np.median(lap_epochs, axis=0)
    templ_td = templ_td - templ_td.mean(axis=-1, keepdims=True)

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
    sim_td: Optional[np.ndarray],
    templ_en: np.ndarray,
    sim_en: Optional[np.ndarray],
    best_snippet_td: Optional[np.ndarray],
    tmin: float,
    outdir: str,
    show: bool = False
):
    ensure_outdir(outdir)
    L = templ_td.shape[1]
    t_axis = np.arange(L) / sfreq + tmin

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(f"Motif {label} — Laplacian TD + Envelope (DTW, MI)", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title("TD Template (z-scored per channel)")
    im1 = ax1.imshow(zscore_channels(templ_td), aspect="auto", origin="lower",
                     extent=[t_axis[0], t_axis[-1], 0, templ_td.shape[0]])
    ax1.set_ylabel("Lap Channels")
    if len(lap_names) >= 2:
        ax1.set_yticks([0, len(lap_names)-1]); ax1.set_yticklabels([lap_names[0], lap_names[-1]])
    cb1 = plt.colorbar(im1, ax=ax1); cb1.set_label("z")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title("Global DTW similarity (TD and Envelope) over target")
    if sim_td is not None and sim_td.size:
        ax2.plot(sim_td, label="TD")
    if sim_en is not None and sim_en.size:
        ax2.plot(sim_en, label="EN", alpha=0.8)
    ax2.set_ylabel("Similarity"); ax2.set_xlabel("Window start (samples)")
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
        fname = f"motif_{label}_MI_lap_dtw.png".replace(os.sep, "-")
        fig.savefig(os.path.join(outdir, fname), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------- Save & show sequence ----------

def save_and_show_sequence(labels: List[str], times: List[float], scores: List[float],
                           outdir: str, prefix: str = "recognized_trigger_locked_DTW_MI"):
    ensure_outdir(outdir)
    seq_str = " ".join(labels)
    txt_path = os.path.join(outdir, f"{prefix}_sequence.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(seq_str + "\n")

    csv_path = os.path.join(outdir, f"{prefix}_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("idx,time_s,label,score\n")
        for i, (t, l, s) in enumerate(zip(times, labels, scores)):
            s_val = s if np.isfinite(s) else -1.0
            f.write(f"{i},{t:.6f},{l},{s_val:.6f}\n")

    print("\n=== Detected Sequence (Trigger-locked, DTW fusion • MI) ===")
    print(seq_str)
    print(f"\n[Saved sequence] {txt_path}")
    print(f"[Saved table]    {csv_path}")

# =======================
# ====== MAIN ===========
# =======================

def main():
    ensure_outdir(OUTPUT_DIR)
    assert set(TEMPLATE_SOURCES.keys()) == {"C", "D"}, "This MI script expects templates 'C' and 'D' only."

    # 1) Build templates
    templates: Dict[str, Dict[str, np.ndarray]] = {}
    lap_names_by_label: Dict[str, List[str]] = {}
    sfreq_ref: Optional[float] = None

    print("[Templates] Building TD+EN (Laplacian, MI only) ...")
    for label in tqdm(list(TEMPLATE_SOURCES.keys()), desc="[Templates] Build", total=len(TEMPLATE_SOURCES)):
        entry = TEMPLATE_SOURCES[label]
        if isinstance(entry, (list, tuple)):
            fpath, ev = entry[0], (entry[1] if len(entry) > 1 else None)
        else:
            fpath, ev = entry, None
        templ_dict, sfreq, lap_names, chosen = build_template_from_file(
            label, fpath, ev, TMIN, TMAX, RESAMPLE_HZ
        )
        print(f"  • {label}: {os.path.basename(fpath)}  -> Event: '{chosen}'. Lap: {lap_names}")
        templates[label] = templ_dict
        lap_names_by_label[label] = lap_names
        if sfreq_ref is None:
            sfreq_ref = sfreq
        else:
            assert abs(sfreq_ref - sfreq) < 1e-6, "All templates must share sampling rate."
    assert sfreq_ref is not None
    sf = sfreq_ref
    dtw_band = int(round(DTW_BAND_SEC * sf))

    # 2) Classification labels (C, D)
    labels_all = list(templates.keys())
    labels_use = labels_all if CLASSIFY_LABELS is None else CLASSIFY_LABELS
    print(f"\n[Classify] Using labels: {labels_use}")
    assert set(labels_use) <= {"C", "D"}

    # 3) Reference Lap order & per-label indices
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
            en_re = np.concatenate(en_blocks_reordered, axis=0)
            templates[l]["EN"] = en_re

            idx_en = []
            for b in range(B_env):
                base = b * len(ref_order)
                idx_en.extend([base + i for i in idx_td])
            label_idx_en[l] = idx_en
        else:
            templates[l]["EN"] = en
            label_idx_en[l] = []

    # 4) Target & events
    print(f"\n[Target] Loading {os.path.basename(TARGET_FILE)} ...")
    raw = load_raw(TARGET_FILE, resample_hz=sfreq_ref)
    events_all, eid_all = get_events_any(raw)
    code, lbl = resolve_event_auto(events_all, eid_all, requested=TARGET_EVENT)
    events = events_all[events_all[:, 2] == code]
    print(f"[Target] Using event '{lbl}' (code {code}); N={len(events)}")
    if len(events) == 0:
        raise RuntimeError("No matching target events found.")

    # 5) Laplacians for target; baseline
    lap_all, lap_names_target = derive_laplacian_continuous(raw)
    if lap_names_target != ref_order:
        name_to_idx = {n: i for i, n in enumerate(lap_names_target)}
        idx = [name_to_idx[n] for n in ref_order if n in name_to_idx]
        lap_all = lap_all[idx]
        lap_names_target = [ref_order[i] for i in range(len(idx))]
    C_ref, N = lap_all.shape

    if BASELINE_USE_PRE_ANNOT:
        bmean, bstd, secs, lap_names_base = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_base != ref_order:
            map_idx = {n: i for i, n in enumerate(lap_names_base)}
            idx = [map_idx[n] for n in ref_order if n in map_idx]
            bmean, bstd = bmean[idx], bstd[idx]
        print(f"[Target] Shared Lap baseline: {secs:.2f}s ({BASELINE_MODE}) on {C_ref} ch.")
        lap_all = apply_baseline_norm(lap_all, bmean, bstd, BASELINE_MODE)

    env_all = make_env_feature(lap_all, sf, ENV_BANDS, ENV_SMOOTH_SEC)

    # 6) Trigger-locked classification (DTW fusion) with ONE overall progress bar
    print("\n[Trigger-locked] Classifying one label per trigger (DTW fusion, MI) ...")
    L = next(iter(templates.values()))["TD"].shape[1]
    win_center_samps = int(((TMIN + TMAX) / 2.0) * sf)
    jitter = int(round(SEARCH_JITTER_SEC * sf))

    ch_weights: Dict[str, Optional[np.ndarray]] = {lab: None for lab in labels_use}

    labels_pred, times_pred, scores_pred = [], [], []
    fused_matrix: List[Dict[str, float]] = []

    for ev in tqdm(list(events[:, 0]), desc="[Events]", total=len(events)):
        best_time = (ev + win_center_samps) / sf
        start0 = int(ev + TMIN * sf)
        seg_start = max(0, start0 - jitter)
        seg_end   = min(N, start0 + L + jitter)
        if seg_end - seg_start < L + 1:
            labels_pred.append("UNK"); times_pred.append(best_time); scores_pred.append(-1.0); fused_matrix.append({}); continue

        seg_td_common = lap_all[:, seg_start:seg_end]
        seg_en_common = env_all[:, seg_start:seg_end] if env_all.size else np.zeros((0, seg_end - seg_start))

        fused_by_label_event: Dict[str, float] = {}
        # TD search
        for lab in labels_use:
            idx_td = label_idx_td[lab]
            templ_td = templates[lab]["TD"]
            td_best_sim = -1.0
            for k in range(0, max(0, seg_td_common.shape[1] - L + 1)):
                seg_td = seg_td_common[idx_td, k:k+L]
                seg_td = _local_z(seg_td, axis=-1)
                sim_td = dtw_similarity_sign_invariant(seg_td, templ_td,
                                                       band_radius=int(round(DTW_BAND_SEC * sf)),
                                                       weights=ch_weights[lab],
                                                       scale=DTW_SIM_SCALE)
                td_best_sim = max(td_best_sim, sim_td)

            # EN search
            if seg_en_common.size and templates[lab]["EN"].size and label_idx_en[lab]:
                idx_en = label_idx_en[lab]
                templ_en = templates[lab]["EN"]
                en_best_sim = -1.0
                for k in range(0, max(0, seg_en_common.shape[1] - L + 1)):
                    seg_en = seg_en_common[idx_en, k:k+L]
                    seg_en = _local_z(seg_en, axis=-1)
                    sim_en = dtw_similarity_sign_invariant(seg_en, templ_en,
                                                           band_radius=max(1, int(round(0.5 * DTW_BAND_SEC * sf))),
                                                           weights=None,
                                                           scale=DTW_SIM_SCALE)
                    en_best_sim = max(en_best_sim, sim_en)
                fused = FUSION_ALPHA * td_best_sim + (1.0 - FUSION_ALPHA) * en_best_sim if USE_FUSION else td_best_sim
            else:
                fused = td_best_sim

            fused_by_label_event[lab] = fused

        fused_matrix.append(fused_by_label_event)
        times_pred.append(best_time)
        labels_pred.append("UNK")           # temporary; will relabel below
        scores_pred.append(-1.0)

    # === Global calibration (per-label) ===
    per_label_all = {lab: [] for lab in labels_use}
    for ev_dict in fused_matrix:
        for lab in labels_use:
            v = ev_dict.get(lab, np.nan)
            if np.isfinite(v):
                per_label_all[lab].append(v)

    auto_thresh = {}
    for lab in labels_use:
        arr = np.array(per_label_all[lab], float)
        auto_thresh[lab] = float(np.quantile(arr, AUTO_THRESH_Q)) if (AUTO_THRESH and arr.size) else -np.inf

    z_mu, z_sd = {}, {}
    for lab in labels_use:
        arr = np.array(per_label_all[lab], float)
        if arr.size:
            z_mu[lab] = float(np.mean(arr)); z_sd[lab] = float(np.std(arr) + 1e-6)
        else:
            z_mu[lab], z_sd[lab] = 0.0, 1.0

    # === Second-pass relabel with calibrated ranking ===
    labels_second, times_second, scores_second = [], [], []
    for i, ev_dict in enumerate(fused_matrix):
        ranked = []
        for lab in labels_use:
            raw_f = ev_dict.get(lab, -np.inf)
            rank = ((raw_f - z_mu[lab]) / z_sd[lab]) if USE_LABEL_ZCAL else raw_f
            ranked.append((lab, raw_f, rank))
        ranked.sort(key=lambda t: t[2], reverse=True)

        best_lab, best_raw, best_rank = ranked[0]
        second_rank = ranked[1][2] if len(ranked) > 1 else -np.inf

        thr = max(PER_LABEL_FUSED_THRESH.get(best_lab, -1.0), auto_thresh.get(best_lab, -np.inf))
        ok_thresh = (np.isfinite(best_raw) and best_raw >= thr)
        ok_margin = (best_rank - second_rank) >= DELTA_MARGIN_RANK

        if ok_thresh and ok_margin:
            labels_second.append(best_lab); scores_second.append(best_raw)
        else:
            if ARGMAX_FALLBACK and np.isfinite(best_raw):
                labels_second.append(best_lab); scores_second.append(best_raw)
            else:
                labels_second.append("UNK"); scores_second.append(best_raw if np.isfinite(best_raw) else -1.0)
        times_second.append(times_pred[i])

    labels_pred, times_pred, scores_pred = labels_second, times_second, scores_second

    # Diagnostics
    if PRINT_SCORE_RANGES:
        print("\n[Score ranges (best per label at each event)]")
        for lab in labels_use:
            vals = [ev.get(lab, np.nan) for ev in fused_matrix]
            vals = np.array(vals, float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                print(f"{lab}: min={finite.min():.3f}  median={np.median(finite):.3f}  max={finite.max():.3f}")

    save_and_show_sequence(labels_pred, times_pred, scores_pred, OUTPUT_DIR,
                           prefix="recognized_trigger_locked_DTW_MI")

    counts = {l: 0 for l in labels_use}
    unk = 0
    for l in labels_pred:
        if l in counts: counts[l] += 1
        else: unk += 1
    miss = [l for l, c in counts.items() if c == 0]
    if miss:
        print("\nMotifs NOT found (trigger-locked): " + ", ".join(miss))
    print(f"UNK count: {unk} / {len(labels_pred)}")

    if PRINT_PER_LABEL_MEANS:
        per_label = defaultdict(list)
        for lab, sc in zip(labels_pred, scores_pred):
            if lab != "UNK" and np.isfinite(sc):
                per_label[lab].append(sc)
        print("\n[Per-label mean fused score]")
        for lab in labels_use:
            arr = per_label.get(lab, [])
            mean_sc = np.mean(arr) if len(arr) else float('nan')
            print(f"{lab}: n={len(arr):3d}  mean={mean_sc:.3f}")

    # 7) Visualization (global DTW similarity curves)
    print("\n[Visualization] Computing global DTW similarity (TD & EN) per motif ...")
    ensure_outdir(OUTPUT_DIR)
    stride = max(1, int(DTW_STRIDE_GLOBAL))
    for lab in tqdm(list(templates.keys()), desc="[Viz] Per motif"):
        td_global = lap_all[label_idx_td.get(lab, []), :] if lab in label_idx_td else lap_all
        L = templates[lab]["TD"].shape[1]
        sims_td = []
        for k in range(0, td_global.shape[1] - L + 1, stride):
            seg = _local_z(td_global[:, k:k+L], axis=-1)
            s = dtw_similarity_sign_invariant(seg, templates[lab]["TD"], band_radius=int(round(DTW_BAND_SEC * sf)),
                                              weights=None, scale=DTW_SIM_SCALE)
            sims_td.append(s)
        sims_td = np.array(sims_td, dtype=float)

        if (env_all.size and templates[lab]["EN"].size and label_idx_en.get(lab, [])):
            en_global = env_all[label_idx_en[lab], :]
            sims_en = []
            for k in range(0, en_global.shape[1] - L + 1, stride):
                seg = _local_z(en_global[:, k:k+L], axis=-1)
                s = dtw_similarity_sign_invariant(seg, templates[lab]["EN"], band_radius=max(1, int(round(0.5 * DTW_BAND_SEC * sf))),
                                                  weights=None, scale=DTW_SIM_SCALE)
                sims_en.append(s)
            sims_en = np.array(sims_en, dtype=float)
        else:
            sims_en = np.array([])

        peaks, _ = find_peaks(sims_td, height=SIM_THRESHOLD_GLOBAL, distance=max(1, int(round(0.35 * (sf/stride)))))
        best_snippet = None
        if len(peaks) > 0:
            best_idx_stride = int(peaks[np.argmax(sims_td[peaks])])
            start = best_idx_stride * stride
            end = start + L
            start = max(0, start); end = min(lap_all.shape[1], end)
            best_td_global = td_global[:, start:end]
            if best_td_global.shape[1] < L:
                pad = np.zeros((best_td_global.shape[0], L - best_td_global.shape[1]))
                best_td_global = np.hstack([best_td_global, pad])
            best_snippet = best_td_global

        plot_motif_summary(
            label=lab,
            templ_td=templates[lab]["TD"],
            lap_names=lap_names_by_label.get(lab, []),
            sfreq=sf,
            sim_td=sims_td,
            templ_en=templates[lab]["EN"],
            sim_en=sims_en,
            best_snippet_td=best_snippet,
            tmin=TMIN,
            outdir=OUTPUT_DIR,
            show=SHOW_PLOTS
        )

    if SAVE_PLOTS:
        print(f"\nFigures saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

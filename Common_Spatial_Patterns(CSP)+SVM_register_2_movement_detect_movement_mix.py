# -*- coding: utf-8 -*-
"""
CSP + SVM (A/B only) → Detect sequence in movement-mix target
-------------------------------------------------------------
- Train on registration datasets: A & B (movement-1, movement-2).
- Derive Laplacians (C3L, CzL, C4L).
- Pre-first-event baseline z-score on Laplacians (shared per file).
- Trigger-locked epoching with jitter search at prediction time.
- CSP (average_power, OAS regularization) -> Standardize -> Linear SVM (probability=True).
- Probability threshold to emit UNK for low confidence.
- Tiny median smoothing (optional).
- Saves sequence (.txt, .csv) and figures.

Requires: mne, numpy, scikit-learn, pandas, matplotlib, tqdm
"""

import os
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import mne
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe non-interactive backend for saving figures
import matplotlib.pyplot as plt

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =======================
# ====== CONFIG =========
# =======================

# --- Paths (EDIT ME) ---
FILE_A = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif"
FILE_B = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif"
FILE_TARGET = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# --- Epoch window around triggers (sec) ---
TMIN = 0.00
TMAX = 1.00  # keep identical for train & predict

# --- Resampling (Hz); None=keep native fs. If you see mismatched fs across files, set a fixed value (e.g., 250.0)
RESAMPLE_HZ: Optional[float] = None

# --- Baseline on Laplacians from pre-first-event data ---
BASELINE_USE = True
BASELINE_MIN_SEC = 5.0
BASELINE_MARGIN_SEC = 0.0
BASELINE_MODE = "zscore"  # 'zscore' or 'demean'

# --- CSP & SVM ---
N_CSP = 2                         # with 3 Laplacian chans, 2 is a good choice
SVM_C = 1.0
SVM_CLASS_WEIGHT = "balanced"     # to be safer across sessions
SVM_PROBA = True

# --- Jitter around window start during prediction ---
JITTER_SEC = 0.12                 # ± jitter (seconds)
JITTER_STEP_SAMP = 1              # slide step in samples

# --- Decision thresholding & smoothing ---
UNK_PROBA_THR = 0.60              # if max P(A,B) < this → UNK
SMOOTH_MEDIAN = 3                 # odd number; 1 disables

# --- Output ---
OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_csp_svm_ab_output_register_2_detect_movement"
SAVE_PLOTS = True                 # save figures to OUT_DIR

# --- Optional ground truth (paste your A/B sequence, separated by spaces). Leave "" to skip accuracy.
GROUND_TRUTH_STR = ""

# =======================
# ====== UTILS ==========
# =======================

EPS = 1e-12

CENTRAL_BASES = {
    "C3L": ("C3", ["FC3", "CP3"]),
    "CzL": ("Cz", ["FCz", "CPz"]),
    "C4L": ("C4", ["FC4", "CP4"]),
}

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_raw(path: str, resample_hz: Optional[float]) -> mne.io.BaseRaw:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: "{path}"')
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-6:
        raw.resample(resample_hz, npad="auto")
    return raw

def get_events_any(raw: mne.io.BaseRaw):
    """Use annotations if present; else stim channel."""
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(events):
            return events, event_id
    except Exception:
        pass
    events = mne.find_events(raw, verbose="ERROR")
    codes = np.unique(events[:, 2]) if len(events) else np.array([])
    event_id = {f"Stimulus/S {int(c):>2}": int(c) for c in codes}
    return events, event_id

def resolve_event_auto(events, event_id_map, requested=None):
    """Return (event_code, label). If requested is None → most frequent."""
    if requested is not None:
        if isinstance(requested, int):
            return requested, f"_code:{requested}"
        if requested in event_id_map:
            return event_id_map[requested], requested
        for k, v in event_id_map.items():
            if requested.lower() in k.lower():
                return v, k
        raise ValueError(f"Could not resolve event '{requested}'.")
    # pick most frequent
    from collections import Counter
    counts = Counter(events[:, 2].tolist())
    code, _ = counts.most_common(1)[0]
    for k, v in event_id_map.items():
        if v == code:
            return v, k
    return int(code), f"_code:{int(code)}"

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
    """Return (C', T) of [C3L, CzL, C4L] from continuous raw (fallback to base if needed)."""
    if start is None:
        start = 0
    if stop is None:
        stop = raw.n_times

    needed = set()
    for base, neigh in CENTRAL_BASES.values():
        needed.add(base); needed.update(neigh)
    picks = mne.pick_channels(raw.info["ch_names"], include=list(needed))
    if len(picks) == 0:
        raise RuntimeError("Central/neighbor channels not found.")

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
    if not out:
        raise RuntimeError("No laplacian signals derived.")
    return np.vstack(out), out_names

def pre_annotation_baseline_window(raw: mne.io.BaseRaw, min_sec: float, margin_sec: float):
    sf = raw.info["sfreq"]
    events, _ = get_events_any(raw)
    if len(events) > 0:
        first = int(np.min(events[:, 0]))
        end = max(0, first - int(round(margin_sec * sf)))
        start = 0
        if end - start < int(round(min_sec * sf)):
            end = min(int(round(min_sec * sf)), raw.n_times); start = 0
    else:
        start = 0; end = min(int(round(min_sec * sf)), raw.n_times)
    if end <= start:
        end = min(int(round(min_sec * sf)), raw.n_times)
    return start, end

def baseline_on_laplacian(raw: mne.io.BaseRaw):
    start, end = pre_annotation_baseline_window(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
    lap, names = derive_laplacian_continuous(raw, start=start, stop=end)
    mu = lap.mean(axis=1)
    sd = lap.std(axis=1) + EPS
    return mu, sd, names

def apply_baseline_norm(arr: np.ndarray, mu: np.ndarray, sd: np.ndarray, mode: str):
    """arr: (C,T) or (N,C,T)."""
    if mode == "zscore":
        if arr.ndim == 2:
            return (arr - mu[:, None]) / sd[:, None]
        elif arr.ndim == 3:
            return (arr - mu[None, :, None]) / sd[None, :, None]
    elif mode == "demean":
        if arr.ndim == 2:
            return (arr - mu[:, None])
        elif arr.ndim == 3:
            return (arr - mu[None, :, None])
    return arr

def epochs_from_file(fpath: str, tmin: float, tmax: float, resample_hz: Optional[float]):
    """Return Laplacian epochs array (N,C,T), lap_names, sfreq, N."""
    raw = load_raw(fpath, resample_hz)
    sf = raw.info["sfreq"]

    # baseline on laps
    if BASELINE_USE:
        bmu, bsd, bnames = baseline_on_laplacian(raw)

    events, eid = get_events_any(raw)
    if len(events) == 0:
        raise RuntimeError(f"No events in {os.path.basename(fpath)}")

    code, lbl = resolve_event_auto(events, eid, requested=None)
    picks = mne.pick_channels(raw.info["ch_names"], include=list({x for s in CENTRAL_BASES.values() for x in [s[0], *s[1]]}))
    epochs = mne.Epochs(raw, events[events[:, 2] == code], event_id={lbl: code},
                        tmin=tmin, tmax=tmax, baseline=None, picks=picks,
                        preload=True, detrend=None, verbose="ERROR", reject_by_annotation=True)
    data = epochs.get_data()  # (N,K,T)
    ch_names = epochs.ch_names

    # derive lap per epoch
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    laps = []
    lap_names = []
    for lap_name in ["C3L", "CzL", "C4L"]:
        base, neigh = CENTRAL_BASES[lap_name]
        if base not in name_to_idx:
            continue
        bidx = name_to_idx[base]
        nidx = [name_to_idx[k] for k in neigh if k in name_to_idx]
        if len(nidx) == 2:
            lap = data[:, bidx, :] - 0.5 * (data[:, nidx[0], :] + data[:, nidx[1], :])
        else:
            lap = data[:, bidx, :]
        laps.append(lap); lap_names.append(lap_name if len(nidx) == 2 else base)
    if not laps:
        raise RuntimeError("Could not build Laplacian epochs.")
    laps = np.stack(laps, axis=1)  # (N,C',T)

    if BASELINE_USE:
        # reorder baseline stats to current lap_names (if needed)
        if bnames != lap_names:
            idx_map = {n: i for i, n in enumerate(bnames)}
            idx = [idx_map[n] for n in lap_names if n in idx_map]
            bmu, bsd = bmu[idx], bsd[idx]
        laps = apply_baseline_norm(laps, bmu, bsd, BASELINE_MODE)

    return laps, lap_names, sf, laps.shape[0]

def align_to_ref_order(X: np.ndarray, names: List[str], ref_order: List[str]) -> np.ndarray:
    """X: (N,C,T) -> reorder to ref_order subset."""
    idx = [names.index(n) for n in ref_order if n in names]
    if not idx:
        raise RuntimeError("No overlapping Laplacian channels to align.")
    return X[:, idx, :]

def build_train_set(files: Dict[str, str], tmin: float, tmax: float, resample_hz: Optional[float]):
    """Return X (N,C,T), y (N,), ref lap order, sfreq."""
    X_list, y_list, name_list, sf_list = [], [], [], []
    for lbl, fpath in files.items():
        laps, lap_names, sf, n = epochs_from_file(fpath, tmin, tmax, resample_hz)
        X_list.append(laps); y_list.append(np.array([lbl] * n))
        name_list.append(lap_names); sf_list.append(sf)

    # enforce same sfreq
    if len(set(sf_list)) != 1:
        raise RuntimeError(f"Sampling rates differ: {sf_list}. Set RESAMPLE_HZ to a common value.")
    sf = sf_list[0]

    # ref order from first entry
    ref_order = name_list[0]
    X_aligned = []
    for X, names in zip(X_list, name_list):
        if names != ref_order:
            X = align_to_ref_order(X, names, ref_order)
        X_aligned.append(X)

    X = np.concatenate(X_aligned, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, ref_order, sf

# ---------- Pipeline ----------

def make_csp_svm_pipeline(n_csp: int, C_val: float, class_w):
    # Correct CSP config: OAS regularizer (reg="oas"), compute covariance per-epoch (cov_est="epoch")
    try:
        csp = CSP(
            n_components=n_csp,
            reg="oas",
            cov_est="epoch",
            transform_into="average_power",
            log=True,
            norm_trace=False,
        )
    except TypeError:
        # Fallback for older MNE versions without `transform_into`
        csp = CSP(
            n_components=n_csp,
            reg="oas",
            cov_est="epoch",
            log=True,
            norm_trace=False,
        )

    svm = SVC(kernel="linear", C=C_val, class_weight=class_w,
              probability=True, random_state=42)
    pipe = Pipeline([
        ("csp", csp),
        ("scaler", StandardScaler()),
        ("svm", svm),
    ])
    return pipe

# ---------- Helpers ----------

def median_smooth(labels: List[str], k: int) -> List[str]:
    if k <= 1 or len(labels) < 3:
        return labels
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    r = k // 2
    out = labels[:]
    map_lab = {"A": 0, "B": 1, "UNK": 2}
    inv_map = {v: k for k, v in map_lab.items()}
    arr = np.array([map_lab.get(l, 2) for l in labels])
    for i in range(len(labels)):
        a = max(0, i - r); b = min(len(labels), i + r + 1)
        med = int(np.median(arr[a:b]))
        out[i] = inv_map[med]
    return out

def parse_gt(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    toks = s.replace(",", " ").split()
    return [t.upper() for t in toks if t.upper() in {"A", "B"}]

def accuracy_vs_gt(pred: List[str], gt: List[str], shifts=range(-3, 4)) -> None:
    if not gt:
        print("\n[Accuracy] No ground truth provided.")
        return
    n = min(len(pred), len(gt))
    p = pred[:n]; g = gt[:n]
    acc0 = 100.0 * sum(int(p[i] == g[i]) for i in range(n)) / n
    best = acc0; best_shift = 0
    for s in shifts:
        if s == 0:
            continue
        if s > 0:
            m = n - s
            if m <= 0:
                continue
            acc = 100.0 * sum(int(pred[i + s] == gt[i]) for i in range(m)) / m
        else:
            m = n + s
            if m <= 0:
                continue
            acc = 100.0 * sum(int(pred[i] == gt[i - s]) for i in range(m)) / m
        if acc > best:
            best, best_shift = acc, s
    print("\n[Accuracy vs your ground truth]")
    print(f"  Lengths -> pred={len(pred)}, gt={len(gt)}  (compared on {n})")
    print(f"  No shift: {acc0:.2f}%")
    print(f"  Best shift in [-3..+3]: shift={best_shift:+d}, acc={best:.2f}%")

# =======================
# ====== PLOTTING =======
# =======================

def _safe_show_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_confusion_cv(pipe, X, y, outdir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=skf)
    labels = ["A", "B"]
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("CV Confusion Matrix (CSP+SVM)")
    _safe_show_save(fig, os.path.join(outdir, "fig_cv_confusion_matrix.png"))

def plot_csp_bars(pipe, lap_order, outdir):
    csp = pipe.named_steps["csp"]
    patt = getattr(csp, "patterns_", None)
    if patt is None:
        return
    # patterns_ shape: (n_components, n_channels) OR (n_channels, n_components) depending on version
    if patt.shape[0] == len(lap_order):
        patt = patt.T  # make it (n_components, n_channels)
    n_comp = min(patt.shape[0], 4)
    fig, axs = plt.subplots(n_comp, 1, figsize=(6, 1.7*n_comp), sharex=True)
    if n_comp == 1:
        axs = [axs]
    for i in range(n_comp):
        axs[i].bar(lap_order, patt[i])
        axs[i].set_ylabel(f"CSP{i+1}")
    axs[-1].set_xlabel("Laplacian channels")
    fig.suptitle("CSP Spatial Patterns (weights per Laplacian)")
    _safe_show_save(fig, os.path.join(outdir, "fig_csp_patterns.png"))

def plot_csp_scatter(pipe, X, y, outdir):
    csp = pipe.named_steps["csp"]
    Z = csp.transform(X)  # (N, n_csp) log-variance features
    if Z.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for lab, color in [("A", "tab:blue"), ("B", "tab:orange")]:
        idx = (y == lab)
        if np.sum(idx):
            ax.scatter(Z[idx, 0], Z[idx, 1], s=12, alpha=0.6, label=lab)
    ax.set_xlabel("CSP1")
    ax.set_ylabel("CSP2")
    ax.legend()
    ax.set_title("CSP feature scatter (train)")
    _safe_show_save(fig, os.path.join(outdir, "fig_csp_scatter.png"))

def plot_evoked_from_X(X, y, sf, lap_order, outdir):
    # X: (N,C,T) Laplacian epochs
    t = np.arange(X.shape[2]) / sf
    fig, axs = plt.subplots(len(lap_order), 1, figsize=(7, 1.8*len(lap_order)), sharex=True)
    if len(lap_order) == 1:
        axs = [axs]
    for ci, ch in enumerate(lap_order):
        for lab, color in [("A", "tab:blue"), ("B", "tab:orange")]:
            idx = (y == lab)
            if np.sum(idx) == 0:
                continue
            mu = X[idx, ci, :].mean(axis=0)
            se = X[idx, ci, :].std(axis=0) / max(1, np.sqrt(np.sum(idx)))
            axs[ci].plot(t, mu, label=lab, color=color, lw=1.8)
            axs[ci].fill_between(t, mu-se, mu+se, color=color, alpha=0.15, linewidth=0)
        axs[ci].axvline(0, color="k", lw=0.8, ls="--")
        axs[ci].set_ylabel(ch)
    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()
    fig.suptitle("Train ERPs (Laplacian mean ± SEM)")
    _safe_show_save(fig, os.path.join(outdir, "fig_train_evoked.png"))

def plot_probs_and_sequence(labels, probs, outdir):
    # probs: list of (pA, pB)
    if not probs:
        return
    pA = [p[0] for p in probs]
    pB = [p[1] for p in probs]
    n = len(labels)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, pA, label="p(A)")
    ax.plot(x, pB, label="p(B)")
    # background bands for labels
    for i, lab in enumerate(labels):
        if lab == "UNK":
            ax.axvspan(i-0.5, i+0.5, color="grey", alpha=0.07)
        elif lab == "A":
            ax.axvspan(i-0.5, i+0.5, color="tab:blue", alpha=0.05)
        else:
            ax.axvspan(i-0.5, i+0.5, color="tab:orange", alpha=0.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Event index")
    ax.set_ylabel("Probability")
    ax.set_title("Target probabilities & predicted labels")
    ax.legend()
    _safe_show_save(fig, os.path.join(outdir, "fig_target_probabilities.png"))

def plot_shift_hist(shifts, sf, outdir):
    if shifts is None or len(shifts) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(shifts, bins=21, edgecolor="k")
    ax.set_xlabel(f"Chosen jitter shift (samples)  |  ~{(1.0/sf):.3f}s/sample")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of chosen jitter shifts")
    _safe_show_save(fig, os.path.join(outdir, "fig_jitter_shift_hist.png"))

def save_all_figs(pipe, X, y, sf, lap_order, labels_sm, probs, shifts, outdir):
    os.makedirs(outdir, exist_ok=True)
    # CV confusion
    try:
        plot_confusion_cv(pipe, X, y, outdir)
    except Exception as e:
        print(f"[Plot] CV confusion skipped: {e}")
    # CSP visuals (require fitted CSP)
    try:
        plot_csp_bars(pipe, lap_order, outdir)
        plot_csp_scatter(pipe, X, y, outdir)
    except Exception as e:
        print(f"[Plot] CSP visuals skipped: {e}")
    # Evoked
    try:
        plot_evoked_from_X(X, y, sf, lap_order, outdir)
    except Exception as e:
        print(f"[Plot] Evoked skipped: {e}")
    # Target probabilities & jitter hist
    try:
        plot_probs_and_sequence(labels_sm, probs, outdir)
        plot_shift_hist(shifts, sf, outdir)
    except Exception as e:
        print(f"[Plot] Target plots skipped: {e}")

# =======================
# ====== PREDICT =========
# =======================

def predict_sequence_with_jitter(raw_target: mne.io.BaseRaw,
                                 pipe: Pipeline,
                                 ref_order: List[str],
                                 sf: float,
                                 tmin: float,
                                 tmax: float,
                                 proba_thr: float,
                                 jitter_sec: float,
                                 jitter_step_samp: int) -> Tuple[List[str], List[Tuple[float, float]], List[int]]:
    """Return labels, probs [(pA,pB)], chosen_shifts (samples)."""
    events, eid = get_events_any(raw_target)
    code, lbl = resolve_event_auto(events, eid, requested=None)
    evs = events[events[:, 2] == code]
    if len(evs) == 0:
        raise RuntimeError("No target events for selected code.")
    print(f"[Target] Using event '{lbl}' (code {code}); N events={len(evs)}")

    # build continuous lap + baseline
    lap_all, lap_names = derive_laplacian_continuous(raw_target)
    if lap_names != ref_order:
        # bring to ref order subset
        name_to_idx = {n: i for i, n in enumerate(lap_names)}
        idx = [name_to_idx[n] for n in ref_order if n in name_to_idx]
        if not idx:
            raise RuntimeError("Target and train Laplacian channels do not overlap.")
        lap_all = lap_all[idx]
    if BASELINE_USE:
        bmu, bsd, bnames = baseline_on_laplacian(raw_target)
        # reorder baseline to ref_order
        if bnames != ref_order:
            idx = [bnames.index(n) for n in ref_order if n in bnames]
            bmu, bsd = bmu[idx], bsd[idx]
        lap_all = apply_baseline_norm(lap_all, bmu, bsd, BASELINE_MODE)

    L = int(round((tmax - tmin) * sf))
    jitter = int(round(jitter_sec * sf))
    N = lap_all.shape[1]

    labels, probs, shifts = [], [], []
    start_offset = int(round(tmin * sf))
    print(f"[Predict] Jitter scan: ±{jitter} samples (~±{1000.0 * jitter / sf:.0f} ms)  | seg_len={L}")
    for ev in tqdm(evs, desc="[Predict] Events", unit="ev"):
        ev_samp = ev[0]
        # nominal window
        w0 = ev_samp + start_offset
        best_lab, best_p, best_shift = "UNK", -1.0, 0
        best_pA, best_pB = 0.0, 0.0

        for k in range(-jitter, jitter + 1, jitter_step_samp):
            s = w0 + k
            e = s + L
            if s < 0 or e > N:
                continue
            seg = lap_all[:, s:e][None, ...]  # (1,C,L)
            proba = pipe.predict_proba(seg)[0]   # proba order pipe.classes_
            classes = list(pipe.classes_)
            pA_cur = proba[classes.index("A")] if "A" in classes else 0.0
            pB_cur = proba[classes.index("B")] if "B" in classes else 0.0
            pmax = pA_cur if pA_cur >= pB_cur else pB_cur
            lab_cur = "A" if pA_cur >= pB_cur else "B"
            if pmax > best_p:
                best_p = pmax
                best_lab = lab_cur
                best_shift = k
                best_pA, best_pB = pA_cur, pB_cur

        final_lab = best_lab if best_p >= proba_thr else "UNK"
        labels.append(final_lab)
        probs.append((best_pA, best_pB))
        shifts.append(best_shift)
    return labels, probs, shifts

# =======================
# ====== MAIN ===========
# =======================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_outdir(OUT_DIR)

    # --- Sanity: paths exist
    for p in [FILE_A, FILE_B, FILE_TARGET]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Path does not exist: "{p}"')

    # -------- Build train set
    print("[Training] Building A/B features from registrations ...")
    files = {"A": FILE_A, "B": FILE_B}
    X, y, lap_order, sf = build_train_set(files, TMIN, TMAX, RESAMPLE_HZ)
    print(f"[Train set] X={X.shape}, A={(y=='A').sum()}, B={(y=='B').sum()}  | Lap={lap_order}  | sf={sf:.1f} Hz")

    # -------- Pipeline & CV
    pipe = make_csp_svm_pipeline(N_CSP, SVM_C, SVM_CLASS_WEIGHT)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"[CV] CSP+SVM accuracy: {scores.mean():.3f} (± {scores.std():.3f})")

    # Fit on full train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    # -------- Target
    print("\n[Target] Loading movement-mix target ...")
    raw_t = load_raw(FILE_TARGET, RESAMPLE_HZ)

    labels, probs, shifts = predict_sequence_with_jitter(
        raw_t, pipe, lap_order, sf, TMIN, TMAX, UNK_PROBA_THR, JITTER_SEC, JITTER_STEP_SAMP
    )

    # Tiny smoothing
    labels_sm = median_smooth(labels, SMOOTH_MEDIAN)

    # -------- Save sequence
    seq_txt = os.path.join(OUT_DIR, "csp_svm_ab_sequence.txt")
    seq_csv = os.path.join(OUT_DIR, "csp_svm_ab_sequence.csv")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_sm) + "\n")
    df = pd.DataFrame({
        "idx": np.arange(len(labels_sm)),
        "label": labels_sm,
        "pA": [p[0] for p in probs],
        "pB": [p[1] for p in probs],
        "shift_samp": shifts
    })
    df.to_csv(seq_csv, index=False)
    print(f"\n[Saved sequence] {seq_txt}")
    print(f"[Saved table]    {seq_csv}")

    # -------- Counts
    from collections import Counter
    cnt = Counter(labels_sm)
    print("\n[Counts]")
    for k in ["A", "B", "UNK"]:
        print(f"  {k}: {cnt.get(k,0)}")

    # -------- Print sequence (wrapped)
    print("\n=== Detected Sequence (CSP+SVM, A/B only, trigger-locked+jitter) ===")
    wrap = 110
    for i in range(0, len(labels_sm), wrap):
        print(" ".join(labels_sm[i:i+wrap]))

    # -------- Accuracy vs GT (optional)
    gt = parse_gt(GROUND_TRUTH_STR)
    if gt:
        accuracy_vs_gt(labels_sm, gt)

    # -------- Save figures
    if SAVE_PLOTS:
        save_all_figs(pipe, X, y, sf, lap_order, labels_sm, probs, shifts, OUT_DIR)
        print(f"\n[Figures] Saved to: {OUT_DIR}")

    print("\nDone.")

if __name__ == "__main__":
    main()

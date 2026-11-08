# -*- coding: utf-8 -*-
"""
CSP + SVM (A/B/C/D) -> movement-mix sequence detection + FIGURES
-----------------------------------------------------------------
- Builds training data from 4 registrations (A,B,C,D).
- Virtual Laplacians: C3L, CzL, C4L (C* - mean(FC*, CP*)), graceful fallback.
- Shared pre-first-event baseline on Laplacians (z-score).
- CSP (average_power, OAS regularization) -> SVM (RBF, class_weight='balanced'),
  probabilities via CalibratedClassifierCV (compatible with both new/old sklearn).
- Trigger-locked classification with ± jitter scan and confidence/margin gating (UNK).
- Saves sequence (.txt/.csv) and prints counts & optional accuracy against ground truth.
- NEW: Visualizations (CV confusion matrix, CSP patterns, CSP scatter, train ERPs,
       target probabilities, jitter shift histogram).

Requires: mne, numpy, scipy, scikit-learn, tqdm, matplotlib
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import mne
from mne.decoding import CSP
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# ============== CONFIG ==============

TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE  = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"
TARGET_EVENT = None  # e.g., "Stimulus/S  1"; None -> auto-pick most frequent
OUTPUT_DIR   = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_csp_svm_output"

# Epoch window (seconds) for both training and target
TMIN = 0.0
TMAX = 1.0

# Resampling (None = keep native fs)
RESAMPLE_HZ: Optional[float] = None

# Jitter scan during prediction (fraction of window length)
JITTER_FRAC = 0.12  # ±12% of window (~±120 ms if window is 1.0 s)

# Baseline (on Laplacians) from pre-first-event tail of each RAW
BASELINE_USE_PRE_ANNOT = True
BASELINE_MIN_SEC   = 5.0
BASELINE_MARGIN_SEC = 0.0  # gap before first event
BASELINE_MODE = "zscore"   # 'zscore' or 'demean'
EPS = 1e-12

# CSP config
N_CSP = 6               # number of spatial filters
CSP_REG = "oas"         # covariance regularization

# SVM config
SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

# Decision thresholds
CONF_THRESH   = 0.55   # min top-class probability; else UNK
MARGIN_THRESH = 0.05   # min (p_top - p_second); else UNK

# Optional ground-truth string (paste here to get accuracy in console)
GROUND_TRUTH_STR: Optional[str] = None
# Example:
# GROUND_TRUTH_STR = ("B A A A B B B ...")

# Colors for plots
COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red", "UNK": "grey"}

# ============== UTILS ==============

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
        return base_arr.copy()  # fallback

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

# ---------- Baseline (pre-first-event) ----------

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
    if BASELINE_USE_PRE_ANNOT:
        bmu, bsd, lap_bnames = baseline_on_laplacian(raw)

    # Epoch (safe: drop events that clip the window)
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
    if BASELINE_USE_PRE_ANNOT:
        if lap_bnames != out_names:
            idx_map = {n: i for i, n in enumerate(lap_bnames)}
            idx = [idx_map[n] for n in out_names if n in idx_map]
            bmu = bmu[idx]; bsd = bsd[idx]
        laps = apply_baseline_norm(laps, bmu, bsd, BASELINE_MODE)

    return laps, out_names, float(sf), laps.shape[0], chosen_lbl

# ---------- Build train set ----------

def build_train_set(files: Dict[str, str], tmin: float, tmax: float,
                    resample_hz: Optional[float]):
    X_list, y_list, group_list = [], [], []
    ref_order: Optional[List[str]] = None
    sf_ref: Optional[float] = None

    print("[Training] Building features from registrations (A,B,C,D) ...")
    for lbl in ["A", "B", "C", "D"]:
        fpath = files[lbl]
        laps, lap_names, sf, n, chosen = epochs_from_file(fpath, tmin, tmax, resample_hz)
        if ref_order is None:
            ref_order = lap_names
            sf_ref = sf
        else:
            # Reorder channels to ref_order
            idx = [lap_names.index(n) for n in ref_order if n in lap_names]
            laps = laps[:, idx, :]
        X_list.append(laps)
        y_list.append(np.full(n, lbl, dtype=object))
        group_list.append(np.full(n, lbl, dtype=object))

    X = np.concatenate(X_list, axis=0)  # (N, C, T)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(group_list, axis=0)

    print(f"[Train set] N={X.shape[0]}, C={X.shape[1]}, T={X.shape[2]}  | per-class: "
          f"{{'A': {(y=='A').sum()}, 'B': {(y=='B').sum()}, 'C': {(y=='C').sum()}, 'D': {(y=='D').sum()}}}")
    return X, y, groups, ref_order, sf_ref

# ---------- Train CSP+SVM pipeline ----------

def make_calibrated(estimator, cv=3, method="sigmoid"):
    """Create CalibratedClassifierCV compatible with both new/old sklearn."""
    try:
        # Newer scikit-learn
        return CalibratedClassifierCV(estimator=estimator, cv=cv, method=method)
    except TypeError:
        # Older scikit-learn
        return CalibratedClassifierCV(base_estimator=estimator, cv=cv, method=method)

def make_csp_svm_pipeline(n_csp: int) -> Pipeline:
    csp = CSP(n_components=n_csp, reg=CSP_REG, transform_into="average_power", log=True, cov_est="epoch")
    # Use SVC without internal probability; we'll calibrate externally (faster & consistent).
    svm = SVC(C=SVM_C, kernel="rbf", gamma=SVM_GAMMA, class_weight=SVM_CLASS_WEIGHT, probability=False, random_state=0)
    clf = make_calibrated(svm, cv=3, method="sigmoid")
    pipe = Pipeline([("csp", csp), ("svm", clf)])
    return pipe

def get_pipe_classes(pipe: Pipeline) -> Optional[List[str]]:
    # Try pipeline attribute first
    if hasattr(pipe, "classes_"):
        return list(pipe.classes_)
    # else look at last step
    last_step = list(pipe.named_steps.values())[-1]
    if hasattr(last_step, "classes_"):
        return list(last_step.classes_)
    return None

def cv_score(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
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

    if BASELINE_USE_PRE_ANNOT:
        bmu, bsd, lap_bnames = baseline_on_laplacian(raw)
        if lap_bnames != lap_names:
            m = {n: i for i, n in enumerate(lap_bnames)}
            idx = [m[n] for n in lap_names if n in m]
            bmu, bsd = bmu[idx], bsd[idx]
        lap_all = (lap_all - bmu[:, None]) / bsd[:, None] if BASELINE_MODE == "zscore" else (lap_all - bmu[:, None])

    # Jitter config
    L = int(round((tmax - tmin) * sf))
    jitter = int(round(JITTER_FRAC * L))
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

        # Scan small jitter window
        for s in range(0, (seg_end - seg_start) - L + 1):
            seg = lap_all[:, seg_start + s: seg_start + s + L]   # (C, L)
            X = seg[None, :, :]                                  # (1, C, L)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_vec = pipe.predict_proba(X)[0]

            classes = get_pipe_classes(pipe) or []
            p_full = np.zeros(4)  # A,B,C,D
            for i, cl in enumerate(['A','B','C','D']):
                if cl in classes:
                    p_full[i] = p_vec[classes.index(cl)]
                else:
                    p_full[i] = 0.0

            idx_top = int(np.argmax(p_full))
            p_top = float(p_full[idx_top])
            p_second = float(np.partition(p_full, -2)[-2])
            cand_lbl = ['A','B','C','D'][idx_top]

            # Keep the highest-confidence candidate; apply gating when saving final label
            if p_top > best_p:
                best_p = p_top
                best_shift = s - jitter
                best_vec = p_full
                best_lbl = cand_lbl if (p_top >= CONF_THRESH and (p_top - p_second) >= MARGIN_THRESH) else "UNK"

        labels.append(best_lbl)
        prob_list.append(best_vec if best_vec is not None else np.zeros(4))
        shifts.append(best_shift)

    return labels, np.vstack(prob_list), np.array(shifts, dtype=int)

# ---------- Saving & Accuracy ----------

def ensure_outdir_and_save(text: str, rows: List[List], outdir: str, stem: str):
    ensure_outdir(outdir)
    txt_path = os.path.join(outdir, f"{stem}.txt")
    csv_path = os.path.join(outdir, f"{stem}.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pA", "pB", "pC", "pD"])
        for r in rows:
            w.writerow(r)
    print(f"\n[Saved sequence] {txt_path}\n[Saved table]    {csv_path}")

def save_sequence(labels: List[str], probs: np.ndarray, outdir: str, stem: str):
    text = " ".join(labels) + "\n"
    rows = []
    for i, lab in enumerate(labels):
        row = [i, lab] + (probs[i].tolist() if probs is not None and probs.size else [np.nan]*4)
        rows.append(row)
    ensure_outdir_and_save(text, rows, outdir, stem)

def print_counts(labels: List[str]):
    from collections import Counter
    c = Counter(labels)
    print("\n[Counts]")
    for k in ['A','B','C','D','UNK']:
        if c.get(k, 0) > 0:
            print(f"  {k}: {c[k]}")

def accuracy_vs_gt(pred_labels: List[str], gt_str: str):
    gt = [t for t in gt_str.strip().split() if t]
    n = min(len(pred_labels), len(gt))
    if n == 0:
        return
    acc = 100.0 * sum(1 for i in range(n) if pred_labels[i] == gt[i]) / n
    best_acc, best_shift = acc, 0
    for s in range(-3, 3+1):
        if s == 0: continue
        acc_s = 100.0 * sum(1 for i in range(n) if pred_labels[(i - s) % n] == gt[i]) / n
        if acc_s > best_acc:
            best_acc, best_shift = acc_s, s
    print("\n[Accuracy vs your ground truth]")
    print(f"  Lengths -> pred={len(pred_labels)}, gt={len(gt)}  (compared on {n})")
    print(f"  No shift: {acc:.2f}%")
    print(f"  Best shift in [-3..+3]: shift={best_shift:+d}, acc={best_acc:.2f}%")

# ---------- FIGURES ----------

def _safe_show_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_confusion_cv(pipe, X, y, outdir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(pipe, X, y, cv=skf)
    labels = ["A", "B", "C", "D"]
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("CV Confusion Matrix (CSP+SVM)")
    _safe_show_save(fig, os.path.join(outdir, "fig_cv_confusion_matrix.png"))

def plot_csp_bars(pipe, lap_order, outdir):
    csp = pipe.named_steps["csp"]
    patt = getattr(csp, "patterns_", None)
    if patt is None:
        return
    # patterns_ can be (n_channels, n_components) or (n_components, n_channels)
    if patt.shape[0] == len(lap_order):
        patt = patt.T
    n_comp = min(patt.shape[0], 6)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = csp.transform(X)  # (N, n_csp) log-variance features
    if Z.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for lab in ["A", "B", "C", "D"]:
        idx = (y == lab)
        if np.any(idx):
            ax.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.6, label=lab, color=COLORS.get(lab, None))
    ax.set_xlabel("CSP1")
    ax.set_ylabel("CSP2")
    ax.legend()
    ax.set_title("CSP feature scatter (train)")
    _safe_show_save(fig, os.path.join(outdir, "fig_csp_scatter.png"))

def plot_evoked_from_X(X, y, sf, lap_order, outdir):
    # X: (N,C,T) Laplacian epochs
    t = np.arange(X.shape[2]) / sf
    fig, axs = plt.subplots(len(lap_order), 1, figsize=(7.5, 1.9*len(lap_order)), sharex=True)
    if len(lap_order) == 1:
        axs = [axs]
    for ci, ch in enumerate(lap_order):
        for lab in ["A", "B", "C", "D"]:
            idx = (y == lab)
            if np.sum(idx) == 0:
                continue
            mu = X[idx, ci, :].mean(axis=0)
            se = X[idx, ci, :].std(axis=0) / max(1, np.sqrt(np.sum(idx)))
            axs[ci].plot(t, mu, label=lab, color=COLORS.get(lab, None), lw=1.8)
            axs[ci].fill_between(t, mu-se, mu+se, color=COLORS.get(lab, None), alpha=0.12, linewidth=0)
        axs[ci].axvline(0, color="k", lw=0.8, ls="--")
        axs[ci].set_ylabel(ch)
    axs[-1].set_xlabel("Time (s)")
    axs[0].legend(ncol=4, fontsize=9)
    fig.suptitle("Train ERPs (Laplacian mean ± SEM)")
    _safe_show_save(fig, os.path.join(outdir, "fig_train_evoked.png"))

def plot_probs_and_sequence(labels, probs, outdir):
    # probs: Nx4 columns for A,B,C,D (as constructed in predict loop)
    if probs is None or probs.size == 0:
        return
    pA, pB, pC, pD = probs[:,0], probs[:,1], probs[:,2], probs[:,3]
    n = len(labels)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x, pA, label="p(A)", color=COLORS["A"])
    ax.plot(x, pB, label="p(B)", color=COLORS["B"])
    ax.plot(x, pC, label="p(C)", color=COLORS["C"])
    ax.plot(x, pD, label="p(D)", color=COLORS["D"])
    # light bands for predicted labels
    for i, lab in enumerate(labels):
        col = COLORS.get(lab, "grey")
        ax.axvspan(i-0.5, i+0.5, color=col, alpha=0.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Event index")
    ax.set_ylabel("Probability")
    ax.set_title("Target probabilities & predicted labels")
    ax.legend(ncol=4, fontsize=9)
    _safe_show_save(fig, os.path.join(outdir, "fig_target_probabilities.png"))

def plot_shift_hist(shifts, sf, outdir):
    if shifts is None or len(shifts) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(shifts, bins=21, edgecolor="k")
    ax.set_xlabel(f"Chosen jitter shift (samples)  |  ~{(1.0/sf):.3f}s/sample")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of chosen jitter shifts")
    _safe_show_save(fig, os.path.join(outdir, "fig_jitter_shift_hist.png"))

def save_all_figs(pipe, X, y, sf, lap_order, labels_sm, probs, shifts, outdir):
    ensure_outdir(outdir)
    try:
        plot_confusion_cv(pipe, X, y, outdir)
    except Exception as e:
        print(f"[Figures] Confusion matrix skipped: {e}")
    try:
        plot_csp_bars(pipe, lap_order, outdir)
    except Exception as e:
        print(f"[Figures] CSP bars skipped: {e}")
    try:
        plot_csp_scatter(pipe, X, y, outdir)
    except Exception as e:
        print(f"[Figures] CSP scatter skipped: {e}")
    try:
        plot_evoked_from_X(X, y, sf, lap_order, outdir)
    except Exception as e:
        print(f"[Figures] Train ERPs skipped: {e}")
    try:
        plot_probs_and_sequence(labels_sm, probs, outdir)
    except Exception as e:
        print(f"[Figures] Probabilities plot skipped: {e}")
    try:
        plot_shift_hist(shifts, sf, outdir)
    except Exception as e:
        print(f"[Figures] Shift hist skipped: {e}")

# ============== MAIN ==============

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_outdir(OUTPUT_DIR)

    # Sanity: paths exist
    for k, p in TEMPLATE_SOURCES.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing template {k}: {p}")
    if not os.path.exists(TARGET_FILE):
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")

    # -------- Build train set
    X, y, groups, lap_order, sf = build_train_set(TEMPLATE_SOURCES, TMIN, TMAX, RESAMPLE_HZ)

    # -------- Make pipeline and CV
    pipe = make_csp_svm_pipeline(N_CSP)
    mean_cv, std_cv = cv_score(pipe, X, y)
    print(f"\n[CV] CSP+SVM accuracy: {mean_cv:.3f} (± {std_cv:.3f})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    # -------- Target
    print("\n[Target] Loading movement-mix target ...")
    raw_t = load_raw(TARGET_FILE, RESAMPLE_HZ)
    events, eid = get_events_any(raw_t)
    code, lbl = resolve_event_auto(events, eid, requested=TARGET_EVENT)
    print(f"[Target] Using event '{lbl}' (code {code})")

    # Predict with jitter scan
    labels, probs, shifts = predict_sequence_with_jitter(
        raw_t, lap_order, TMIN, TMAX, RESAMPLE_HZ, pipe, event_req=TARGET_EVENT
    )
    print(f"[Shift stats] mean={float(np.mean(shifts)):.1f} samp, std={float(np.std(shifts)):.1f}, "
          f"min={int(np.min(shifts))}, max={int(np.max(shifts))}")

    # -------- Report & save
    print("\n=== Detected Sequence (CSP+SVM, 4-class, trigger-locked+jitter) ===")
    wrap = 100
    for i in range(0, len(labels), wrap):
        print(" ".join(labels[i:i+wrap]))

    print_counts(labels)
    save_sequence(labels, probs, OUTPUT_DIR, stem="csp_svm_movement_mix_sequence")

    if GROUND_TRUTH_STR:
        accuracy_vs_gt(labels, GROUND_TRUTH_STR)

    # -------- Figures
    save_all_figs(pipe, X, y, sf, lap_order, labels, probs, shifts, OUTPUT_DIR)
    print(f"\n[Figures] Saved to: {OUTPUT_DIR}")

    print("\nDone.")

if __name__ == "__main__":
    main()

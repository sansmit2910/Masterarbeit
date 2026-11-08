# welch_svm_AB_coral_baseline.py
# Train on registrations A & B, detect on movement-mix.
# Improvements:
#  - Baseline-relative Welch features (event vs. pre-stim baseline).
#  - Unsupervised CORAL alignment of training features to target domain.
#  - Group-aware CV (prevents event-jitter leakage).
#  - Visualizations for CV confusion, target jitter shifts, and timeline vs GT.

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import mne
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import trapezoid
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

# ----------------------------
# === User paths & settings ===
# ----------------------------

# Train ONLY on A & B:
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
}

# Target (movement mix):
TARGET_FILE  = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

# Ground-truth sequence (A/B only). Keep None to skip target accuracy calc.
GROUND_SEQUENCE_STR: Optional[str] = (
    "B A A A B B B A A A A A B B B B B B B B B A A A A A A B B B A A B B B B A A A A A A B B B B B B B B B A A A A A A A A A A A B B B B B B B B B B B B B B A A A A A A B B B A A A A A A A A A A A B B B B B B A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B A A A A A A A A A A B B B B B B B B B B A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B"
)

# Channels: prefer Laplacian labels if present, else fallback to standard
PRIMARY_PICKS = ['C3L', 'CzL', 'C4L']
FALLBACK_PICKS = ['C3', 'Cz', 'C4']

# Annotations: use events with code 1 (often 'Stimulus/S  1')
EVENT_CODE = 1

# Segmenting / features
SEG_LEN_S = 1.0          # seconds per event window (and baseline window)
BANDPASS = (4.0, 40.0)   # bandpass before Welch
WELCH_NPERSEG = 128
WELCH_NOOVERLAP = 64
BANDS = {
    "theta": (4, 7),
    "mu": (8, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
}
TOTAL_BAND = (4, 40)

# Training augmentation / target scan
TRAIN_JITTERS = 2        # additional jittered samples per event for training
MAX_JITTER_SAMP = 25     # max jitter size (samples) for augmentation
TARGET_SCAN_JITTER = 30  # scan ± this many samples on target

# UNK threshold (binary model → 0.60 is slightly stricter than 0.50)
UNK_THRESHOLD = 0.60

# Domain alignment
USE_CORAL = True
CORAL_REG = 1e-3

# CV
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3
RANDOM_STATE = 42

# Output
OUT_DIR = os.path.join(os.path.dirname(TARGET_FILE), "eeg_welch_svm_AB_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# === Utilities & filters  ===
# ----------------------------

def find_event_key_for_code(raw: mne.io.BaseRaw, target_code: int) -> Tuple[str, np.ndarray]:
    """Return the annotation key that maps to `target_code` and the events with that code."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    candidate_keys = [k for k, v in event_id.items() if v == target_code]
    if not candidate_keys:
        candidate_keys = [k for k in event_id.keys() if 'Stimulus/S' in k and k.strip().endswith('1')]
        if not candidate_keys:
            if len(event_id) > 0:
                chosen = sorted(event_id.keys())[0]
                mask = events[:, 2] == event_id[chosen]
                return chosen, events[mask]
            else:
                raise RuntimeError("No annotations found in raw file.")
    key = candidate_keys[0]
    mask = events[:, 2] == event_id[key]
    return key, events[mask]


def pick_eeg(raw: mne.io.BaseRaw, prefer: List[str], fallback: List[str]) -> List[int]:
    chs = [c for c in prefer if c in raw.ch_names]
    if len(chs) < 3:
        chs = [c for c in fallback if c in raw.ch_names]
    if len(chs) < 3:
        raise RuntimeError(f"Could not find channels {prefer} nor {fallback} in file {raw.filenames[0]}")
    return mne.pick_channels(raw.ch_names, include=chs)


def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return b, a


def apply_bandpass(data: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    """Zero-phase bandpass (filtfilt) channel-wise."""
    b, a = butter_bandpass(band[0], band[1], fs, order=4)
    return filtfilt(b, a, data, axis=-1)


def per_trial_zscore(seg: np.ndarray) -> np.ndarray:
    """Channel-wise z-score per trial window."""
    mu = seg.mean(axis=-1, keepdims=True)
    sd = seg.std(axis=-1, keepdims=True) + 1e-9
    return (seg - mu) / sd


def welch_bandpowers(seg: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    seg: (C, T)
    Returns per-channel arrays:
      bp_mu, bp_b1, bp_b2, rel_mu, rel_b1, rel_b2, ratio
    Shapes: each (C,)
    """
    C, T = seg.shape
    f, Pxx = welch(seg, fs=fs, nperseg=min(WELCH_NPERSEG, T),
                   noverlap=min(WELCH_NOOVERLAP, T//2), axis=-1)
    def band_mask(lo, hi): return (f >= lo) & (f < hi)
    total_mask = band_mask(*TOTAL_BAND)
    masks = {k: band_mask(*rng) for k, rng in BANDS.items()}

    total = trapezoid(Pxx[:, total_mask], f[total_mask], axis=-1) + 1e-12
    bp_mu  = trapezoid(Pxx[:, masks['mu']],    f[masks['mu']],    axis=-1)
    bp_b1  = trapezoid(Pxx[:, masks['beta1']], f[masks['beta1']], axis=-1)
    bp_b2  = trapezoid(Pxx[:, masks['beta2']], f[masks['beta2']], axis=-1)

    rel_mu = bp_mu / total
    rel_b1 = bp_b1 / total
    rel_b2 = bp_b2 / total

    ratio  = np.log((bp_mu + 1e-12) / (bp_b1 + bp_b2 + 1e-12))
    return bp_mu, bp_b1, bp_b2, rel_mu, rel_b1, rel_b2, ratio


def features_event_minus_baseline(seg_evt: np.ndarray, seg_base: np.ndarray, fs: float) -> np.ndarray:
    """
    Build baseline-relative feature vector for one trial:
      For each channel:
       - log(bp_mu_evt) - log(bp_mu_base)
       - log(bp_b1_evt) - log(bp_b1_base)
       - log(bp_b2_evt) - log(bp_b2_base)
       - (rel_mu_evt - rel_mu_base)
       - (rel_b1_evt - rel_b1_base)
       - (rel_b2_evt - rel_b2_base)
       - (ratio_evt - ratio_base)
      + an asymmetry delta: (C3 - C4) computed on ratio deltas if 3 channels are present.
    Returns shape: (C*7 + 1,)
    """
    evt = welch_bandpowers(seg_evt, fs)
    bas = welch_bandpowers(seg_base, fs)

    # logs for absolute bandpowers
    log_evt = [np.log(evt[i] + 1e-12) for i in range(3)]          # mu, b1, b2
    log_bas = [np.log(bas[i] + 1e-12) for i in range(3)]
    log_d   = [log_evt[i] - log_bas[i] for i in range(3)]

    # rel and ratio are differences
    rel_d   = [evt[i] - bas[i] for i in range(3, 6)]               # rel_mu, rel_b1, rel_b2
    ratio_d = [evt[6] - bas[6]]                                    # ratio

    feats_per_ch = log_d + rel_d + ratio_d                         # 7 per channel
    feats = np.stack(feats_per_ch, axis=-1)                        # (C, 7)
    vec = feats.flatten()                                          # (C*7,)

    # Asymmetry delta using ratio_d channel-wise if we have C3/Cz/C4
    if feats.shape[0] == 3:
        # use ratio delta (column index 6) for asymmetry C3 - C4
        asym_delta = feats[0, 6] - feats[2, 6]
    else:
        asym_delta = 0.0
    vec = np.concatenate([vec, np.array([asym_delta], dtype=np.float32)], axis=0)
    return vec.astype(np.float32)


# ----------------------------
# === Feature builders      ===
# ----------------------------

def build_features_from_file(path: str,
                             label_char: str,
                             prefer_ch: List[str],
                             fallback_ch: List[str],
                             seg_len_s: float,
                             bandpass: Tuple[float, float],
                             aug_jitters: int,
                             max_jitter: int,
                             rng: np.random.Generator,
                             group_counter_start: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, List[str]]:
    """
    Returns:
      X (N,F),
      y (N,),
      groups (N,)  -- same group id for all jitters from the same base event,
      next_group_counter (int),
      fs (float),
      used_channel_names (List[str])
    """
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    fs = float(raw.info['sfreq'])
    picks = pick_eeg(raw, prefer_ch, fallback_ch)
    ch_names = [raw.ch_names[i] for i in picks]
    _, ev = find_event_key_for_code(raw, EVENT_CODE)
    seg_len = int(round(seg_len_s * fs))

    X_list, y_list, g_list = [], [], []
    group_counter = group_counter_start

    for onset, _, _ in ev:
        base_start = int(onset)  # event-aligned (stim on)
        starts = [base_start]
        for _ in range(aug_jitters):
            J = int(rng.integers(-max_jitter, max_jitter + 1))
            starts.append(base_start + J)

        for s_evt in starts:
            s_pre = s_evt - seg_len
            s_end = s_evt + seg_len
            if s_pre < 0 or s_end > raw.n_times:
                continue

            seg_base = raw.get_data(picks=picks, start=s_pre, stop=s_evt)  # (C,T)
            seg_evt  = raw.get_data(picks=picks, start=s_evt, stop=s_end)

            # identical preprocessing for base & event
            seg_base = apply_bandpass(seg_base, fs, bandpass)
            seg_evt  = apply_bandpass(seg_evt,  fs, bandpass)
            seg_base = per_trial_zscore(seg_base)
            seg_evt  = per_trial_zscore(seg_evt)

            vec = features_event_minus_baseline(seg_evt, seg_base, fs)
            X_list.append(vec)
            y_list.append(label_char)
            g_list.append(group_counter)

        group_counter += 1  # next base event id

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    groups = np.asarray(g_list, dtype=int)
    return X, y, groups, group_counter, fs, ch_names


def extract_target_features_for_alignment(file_path: str,
                                          prefer_ch: List[str],
                                          fallback_ch: List[str],
                                          seg_len_s: float,
                                          bandpass: Tuple[float, float]) -> Tuple[np.ndarray, float, List[str]]:
    """
    Build ONE baseline-relative feature per target event using J=0 (no jitter).
    Used only to compute target-domain statistics for CORAL.
    """
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    fs = float(raw.info['sfreq'])
    picks = pick_eeg(raw, prefer_ch, fallback_ch)
    ch_names = [raw.ch_names[i] for i in picks]
    _, ev = find_event_key_for_code(raw, EVENT_CODE)
    seg_len = int(round(seg_len_s * fs))

    feats = []
    for onset, _, _ in ev:
        s_evt = int(onset)
        s_pre = s_evt - seg_len
        s_end = s_evt + seg_len
        if s_pre < 0 or s_end > raw.n_times:
            continue
        seg_base = raw.get_data(picks=picks, start=s_pre, stop=s_evt)
        seg_evt  = raw.get_data(picks=picks, start=s_evt, stop=s_end)

        seg_base = apply_bandpass(seg_base, fs, bandpass)
        seg_evt  = apply_bandpass(seg_evt,  fs, bandpass)
        seg_base = per_trial_zscore(seg_base)
        seg_evt  = per_trial_zscore(seg_evt)

        vec = features_event_minus_baseline(seg_evt, seg_base, fs)
        feats.append(vec)

    X_tgt = np.asarray(feats, dtype=np.float32)
    return X_tgt, fs, ch_names


# ----------------------------
# === CORAL alignment       ===
# ----------------------------

def coral_fit_transform(X_src: np.ndarray, X_tgt: np.ndarray, reg: float = 1e-3) -> Tuple[np.ndarray, dict]:
    """
    Align source features to the target domain (unsupervised).
    Returns:
      X_src_aligned, params dict (for reproducibility/debug)
    """
    # center
    mu_s = X_src.mean(axis=0, keepdims=True)
    mu_t = X_tgt.mean(axis=0, keepdims=True)

    Xs = X_src - mu_s
    Xt = X_tgt - mu_t

    # covariances
    Cs = np.cov(Xs, rowvar=False)
    Ct = np.cov(Xt, rowvar=False)

    # regularize
    F = X_src.shape[1]
    Cs = Cs + reg * np.eye(F)
    Ct = Ct + reg * np.eye(F)

    Cs_inv_sqrt = fractional_matrix_power(Cs, -0.5)
    Ct_sqrt     = fractional_matrix_power(Ct,  0.5)

    A = Cs_inv_sqrt @ Ct_sqrt
    Xs_aligned = (Xs @ A) + mu_t

    info = {"mu_s": mu_s.squeeze(), "mu_t": mu_t.squeeze()}
    return Xs_aligned.astype(np.float32), info


# ----------------------------
# === CV + training         ===
# ----------------------------

def grouped_nested_cv_and_fit_final(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                    labels_order: List[str]) -> Tuple[Pipeline, dict, np.ndarray]:
    """
    Group-aware Nested CV:
      - Outer: 5-fold StratifiedGroupKFold (if available), else GroupKFold.
      - Inner: 3-fold StratifiedGroupKFold (if available), else GroupKFold.

    Returns final fitted estimator on full data, metrics, and outer-fold predictions.
    """
    def make_outer():
        if HAS_SGKF:
            return StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
        else:
            print("[WARN] StratifiedGroupKFold not available. Falling back to GroupKFold (no stratification).")
            return GroupKFold(n_splits=N_SPLITS_OUTER)

    def make_inner():
        if HAS_SGKF:
            return StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)
        else:
            return GroupKFold(n_splits=N_SPLITS_INNER)

    outer = make_outer()
    param_grid = {
        'kbest__k': None,  # filled below
        'svc__kernel': ['rbf', 'linear'],
        'svc__C': [0.5, 1.0, 2.0, 4.0],
        'svc__gamma': ['scale', 0.05, 0.1],
    }
    F = X.shape[1]
    ks = ['all'] + [k for k in [10, 14, 18, 22, 26] if (isinstance(k, int) and k <= F)]
    param_grid['kbest__k'] = ks

    y_true_all, y_pred_all = [], []
    fold_acc, fold_best_params = [], []

    split_iter = outer.split(X, y, groups) if HAS_SGKF else outer.split(X, groups=groups)
    for fold_idx, (tr, te) in enumerate(split_iter, 1):
        Xtr, Xte, ytr, yte, gtr = X[tr], X[te], y[tr], y[te], groups[tr]
        inner = make_inner()

        # Inner loop: grid search
        best_score, best_params = -np.inf, None
        inner_split = inner.split(Xtr, ytr, gtr) if HAS_SGKF else inner.split(Xtr, groups=gtr)

        for k in param_grid['kbest__k']:
            for ker in param_grid['svc__kernel']:
                for C in param_grid['svc__C']:
                    for g in param_grid['svc__gamma']:
                        if ker == 'linear' and g != 'scale':
                            continue
                        scores = []
                        for itr, ite in inner_split:
                            est = Pipeline([
                                ('scaler', StandardScaler()),
                                ('kbest', SelectKBest(score_func=f_classif, k=k)),
                                ('pca', PCA(n_components=0.95, svd_solver='full')),
                                ('svc', SVC(kernel=ker,
                                            C=C,
                                            gamma=g if ker == 'rbf' else 'scale',
                                            probability=True,
                                            class_weight='balanced',
                                            random_state=RANDOM_STATE))
                            ])
                            est.fit(Xtr[itr], ytr[itr])
                            p = est.predict(Xtr[ite])
                            scores.append(accuracy_score(ytr[ite], p))
                        sc = float(np.mean(scores))
                        if sc > best_score:
                            best_score = sc
                            best_params = dict(kbest__k=k, svc__kernel=ker, svc__C=C, svc__gamma=g)

        # Train best model on tr, evaluate on te
        best_est = Pipeline([
            ('scaler', StandardScaler()),
            ('kbest', SelectKBest(score_func=f_classif, k=best_params['kbest__k'])),
            ('pca', PCA(n_components=0.95, svd_solver='full')),
            ('svc', SVC(kernel=best_params['svc__kernel'],
                        C=best_params['svc__C'],
                        gamma=best_params['svc__gamma'] if best_params['svc__kernel']=='rbf' else 'scale',
                        probability=True, class_weight='balanced', random_state=RANDOM_STATE))
        ])
        best_est.fit(Xtr, ytr)
        y_hat = best_est.predict(Xte)

        acc = accuracy_score(yte, y_hat)
        fold_acc.append(acc)
        fold_best_params.append(best_params)
        y_true_all.append(yte)
        y_pred_all.append(y_hat)
        print(f"[CV] Fold {fold_idx} acc={acc:.3f}  best={best_params}")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_order)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    metrics = {
        "cv_acc_mean": float(np.mean(fold_acc)),
        "cv_acc_std": float(np.std(fold_acc)),
    }

    # Refit final model on ALL data using params from best outer fold
    best_overall = fold_best_params[int(np.argmax(fold_acc))]
    final_est = Pipeline([
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif, k=best_overall['kbest__k'])),
        ('pca', PCA(n_components=0.95, svd_solver='full')),
        ('svc', SVC(kernel=best_overall['svc__kernel'],
                    C=best_overall['svc__C'],
                    gamma=best_overall['svc__gamma'] if best_overall['svc__kernel']=='rbf' else 'scale',
                    probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    final_est.fit(X, y)

    # Save confusion matrix figure
    fig = plt.figure(figsize=(5.2,4.6))
    im = plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels_order)), labels_order)
    plt.yticks(range(len(labels_order)), labels_order)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f"Grouped CV Confusion (A/B)  acc={metrics['cv_acc_mean']:.3f} ± {metrics['cv_acc_std']:.3f}")
    for i in range(len(labels_order)):
        for j in range(len(labels_order)):
            plt.text(j, i, f"{cm_norm[i,j]:.2f}", ha='center', va='center', fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "cv_confusion_matrix_AB_grouped.png")
    plt.savefig(fig_path, dpi=160); plt.close(fig)
    print(f"[Saved] CV confusion matrix → {fig_path}")

    print("\n[CV] Classification report (outer folds, grouped):")
    print(classification_report(y_true_all, y_pred_all, labels=labels_order))

    return final_est, metrics, y_pred_all


# ----------------------------
# === Target prediction     ===
# ----------------------------

def predict_target_sequence(model: Pipeline,
                            file_path: str,
                            prefer_ch: List[str],
                            fallback_ch: List[str],
                            seg_len_s: float,
                            bandpass: Tuple[float, float],
                            scan_jitter: int,
                            unk_thresh: float) -> Tuple[List[str], List[int], float, List[str]]:
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    fs = float(raw.info['sfreq'])
    picks = pick_eeg(raw, prefer_ch, fallback_ch)
    ch_names = [raw.ch_names[i] for i in picks]
    _, ev = find_event_key_for_code(raw, EVENT_CODE)
    seg_len = int(round(seg_len_s * fs))

    preds, shifts_used = [], []
    for onset, _, _ in ev:
        best_prob, best_lab, best_shift = -1.0, 'UNK', 0
        for J in range(-scan_jitter, scan_jitter+1):
            s_evt = int(onset) + J
            s_pre = s_evt - seg_len
            s_end = s_evt + seg_len
            if s_pre < 0 or s_end > raw.n_times:
                continue
            seg_base = raw.get_data(picks=picks, start=s_pre, stop=s_evt)
            seg_evt  = raw.get_data(picks=picks, start=s_evt, stop=s_end)

            seg_base = apply_bandpass(seg_base, fs, bandpass)
            seg_evt  = apply_bandpass(seg_evt,  fs, bandpass)
            seg_base = per_trial_zscore(seg_base)
            seg_evt  = per_trial_zscore(seg_evt)

            vec = features_event_minus_baseline(seg_evt, seg_base, fs).reshape(1, -1)
            proba = model.predict_proba(vec)[0]
            idx = int(np.argmax(proba))
            pmax = float(proba[idx])
            lab = model.classes_[idx]
            if pmax > best_prob:
                best_prob, best_lab, best_shift = pmax, lab, J

        preds.append(best_lab if best_prob >= unk_thresh else 'UNK')
        shifts_used.append(best_shift)

    # Plot jitter shifts
    fig = plt.figure(figsize=(6,3))
    plt.hist(shifts_used, bins=2*scan_jitter+1)
    plt.title(f"Target jitter shifts (mean={np.mean(shifts_used):.1f}, sd={np.std(shifts_used):.1f})")
    plt.xlabel("Shift (samples)"); plt.ylabel("Count")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "target_shift_hist_AB.png")
    plt.savefig(fig_path, dpi=160); plt.close(fig)
    print(f"[Saved] Target jitter shift histogram → {fig_path}")

    return preds, shifts_used, fs, ch_names


def parse_ground_sequence(seq_str: str) -> List[str]:
    tokens = seq_str.strip().split()
    clean = []
    for t in tokens:
        t = t.strip().upper().strip(",;")
        if t in {'A','B','UNK'}:
            clean.append(t)
    return clean


def compare_sequence(pred: List[str], gt: List[str]) -> dict:
    n = min(len(pred), len(gt))
    if n == 0:
        return {}
    pred_n, gt_n = pred[:n], gt[:n]
    strict = sum(p == g for p, g in zip(pred_n, gt_n)) / n
    nz_idx = [i for i in range(n) if pred_n[i] != 'UNK']
    ign_unk = sum(pred_n[i] == gt_n[i] for i in nz_idx) / len(nz_idx) if nz_idx else 0.0
    return {
        "strict_acc": strict,
        "ignore_UNK_acc": ign_unk,
        "n_aligned": n,
        "n_nonUNK_pred": len(nz_idx),
    }


def save_sequence(pred: List[str], path_txt: str, path_csv: str):
    with open(path_txt, 'w') as f:
        f.write(" ".join(pred) + "\n")
    with open(path_csv, 'w') as f:
        f.write("idx,label\n")
        for i, lab in enumerate(pred):
            f.write(f"{i},{lab}\n")
    print(f"[Saved sequence] {path_txt}\n[Saved table]    {path_csv}")


def plot_timeline_vs_ground(pred: List[str], gt: List[str], out_path: str):
    """Stripe plot: 0=A,1=B,2=UNK."""
    map_lab = {'A':0,'B':1,'UNK':2}
    n = min(len(pred), len(gt))
    P = np.array([map_lab.get(x,2) for x in pred[:n]])
    G = np.array([map_lab.get(x,2) for x in gt[:n]])

    fig = plt.figure(figsize=(12,2.6))
    ax1 = plt.subplot(2,1,1)
    ax1.imshow(P[np.newaxis,:], aspect='auto', vmin=0, vmax=2)
    ax1.set_yticks([0]); ax1.set_yticklabels(['Pred'])
    ax1.set_xticks([])
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.imshow(G[np.newaxis,:], aspect='auto', vmin=0, vmax=2)
    ax2.set_yticks([0]); ax2.set_yticklabels(['GT'])
    ax2.set_xlabel("Event index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[Saved] Timeline vs ground → {out_path}")


# ----------------------------
# === Main flow            ===
# ----------------------------

def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # 0) (Optional) Pre-extract unlabeled target features (no jitter) for CORAL
    X_tgt_align, fs_tgt, chs_tgt = extract_target_features_for_alignment(
        TARGET_FILE, PRIMARY_PICKS, FALLBACK_PICKS, SEG_LEN_S, BANDPASS
    )
    if USE_CORAL:
        print(f"[CORAL] Target alignment pool: {X_tgt_align.shape}")

    # 1) Build training set from registrations A & B WITH GROUP IDS (baseline-relative)
    X_list, y_list, g_list = [], [], []
    fs_used = None
    used_ch_names = None
    per_class_counts = {}
    group_counter = 0
    print("[Training] Building baseline-relative features from registrations (A,B) with grouped CV ...")

    for lab, fpath in TEMPLATE_SOURCES.items():
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing training file for {lab}: {fpath}")
        Xi, yi, gi, group_counter, fs, chs = build_features_from_file(
            fpath, lab, PRIMARY_PICKS, FALLBACK_PICKS,
            SEG_LEN_S, BANDPASS, TRAIN_JITTERS, MAX_JITTER_SAMP, rng,
            group_counter_start=group_counter
        )
        X_list.append(Xi); y_list.append(yi); g_list.append(gi)
        per_class_counts[lab] = int(np.sum(yi == lab))
        if fs_used is None:
            fs_used = fs; used_ch_names = chs
        elif abs(fs_used - fs) > 1e-6:
            raise RuntimeError("Sampling rates differ across registration files.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)  # one group per base event; all jitters share it
    labels_order = ['A','B']
    print(f"[Train set] X={X.shape} (N,F), Lap={used_ch_names}, sf={fs_used:.1f} Hz | per-class: {per_class_counts} | groups={len(np.unique(groups))}")

    # 1b) Unsupervised CORAL: align training features to target domain (optional)
    if USE_CORAL and X_tgt_align.shape[0] >= 5:
        X_aligned, coral_info = coral_fit_transform(X, X_tgt_align, reg=CORAL_REG)
        print("[CORAL] Applied alignment of training features to target domain.")
    else:
        X_aligned = X
        coral_info = {}

    # 2) Grouped Nested CV & final fit (binary A/B)
    clf, cv_metrics, cv_preds = grouped_nested_cv_and_fit_final(X_aligned, y, groups, labels_order)
    print(f"\n[CV] Welch+SVM (A/B, GROUPED, baseline-rel{', CORAL' if USE_CORAL else ''}) "
          f"accuracy: {cv_metrics['cv_acc_mean']:.3f} (± {cv_metrics['cv_acc_std']:.3f})\n")

    # 3) Predict on movement-mix with jitter scan + UNK threshold
    if not os.path.isfile(TARGET_FILE):
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")
    print("[Target] Loading movement-mix target ...")
    preds, shifts, fs_tgt2, chs_tgt2 = predict_target_sequence(
        clf, TARGET_FILE, PRIMARY_PICKS, FALLBACK_PICKS,
        SEG_LEN_S, BANDPASS, TARGET_SCAN_JITTER, UNK_THRESHOLD
    )

    # 4) Summary + save
    uniq, counts = np.unique(preds, return_counts=True)
    print("\n=== Detected Sequence (Welch+SVM A/B, trigger-locked+jitter) ===")
    s = " ".join(preds); width = 80
    for i in range(0, len(s), width):
        print(s[i:i+width])
    print("\n[Counts]")
    for u, c in zip(uniq, counts):
        print(f"  {u}: {c}")

    out_txt = os.path.join(OUT_DIR, "welch_svm_AB_movement_mix_sequence.txt")
    out_csv = os.path.join(OUT_DIR, "welch_svm_AB_movement_mix_sequence.csv")
    save_sequence(preds, out_txt, out_csv)

    # 5) Optional: compare to ground truth (A/B)
    if GROUND_SEQUENCE_STR is not None:
        gt = parse_ground_sequence(GROUND_SEQUENCE_STR)
        stats = compare_sequence(preds, gt)
        if stats:
            print("\n[Target accuracy vs ground (A/B)]")
            print(f"  strict_acc     : {stats['strict_acc']:.3f} (aligned={stats['n_aligned']})")
            print(f"  ignore_UNK_acc : {stats['ignore_UNK_acc']:.3f} (non-UNK preds={stats['n_nonUNK_pred']})")
            timeline_png = os.path.join(OUT_DIR, "timeline_vs_ground_AB.png")
            plot_timeline_vs_ground(preds, gt, timeline_png)

    print("\nDone. Outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()

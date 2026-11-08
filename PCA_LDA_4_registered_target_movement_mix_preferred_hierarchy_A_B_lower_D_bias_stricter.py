# -*- coding: utf-8 -*-
"""
Hierarchical CSP+LDA with Auto-Tuning (AB-friendly gate, energy override)
------------------------------------------------------------------------
- Learns CSP sizes, gate thresholds, CD margin, and energy override from registrations.
- Balanced training to avoid class dominance.
- Predicts A/B/C/D on the target file with temporal smoothing.
- Saves detected sequence (txt + csv).

Deps: mne, numpy, scikit-learn, tqdm
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

import numpy as np
import mne
from mne.decoding import CSP
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.utils import check_random_state

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
TARGET_EVENT: Optional[Union[int, str]] = None  # None -> auto most frequent

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

# CV
N_FOLDS_CV = 5
RANDOM_STATE = 7
BALANCE_TRAIN = True

# Temporal smoothing
SMOOTH_WIN = 5          # odd; 1 disables smoothing
SMOOTH_INCLUDE_UNK = False

# Output
OUTPUT_DIR = os.path.join(DATA_DIR, "eeg_csp_lda_hier_autotune_output")
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
    sf = raw.info["sfreq"]
    tmin_s = int(np.floor(tmin * sf))
    tmax_s = int(np.ceil(tmax * sf))
    n_times = raw.n_times
    mask = (events[:, 0] + tmin_s >= 0) & (events[:, 0] + tmax_s < n_times)
    dropped = int(np.sum(~mask))
    if dropped > 0:
        print(f"  [Safe epoching] Dropping {dropped} events that clip the window (kept {int(np.sum(mask))}).")
    return events[mask]

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

def lap_rms_energy(x: np.ndarray) -> float:
    """x: (C,T) -> scalar RMS across channels/time."""
    return float(np.sqrt(np.mean(x**2)))

def smooth_labels(labels: List[str], win: int, include_unk: bool) -> List[str]:
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
        best = max(counts.items(), key=lambda kv: (kv[1], -abs((s+e)//2 - i)))[0]
        out[i] = best
    return out

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
    print("\n=== Detected Sequence (Hier-CSP+LDA • auto-tuned) ===")
    print(txt)
    print(f"\n[Saved sequence] {txt_path}")
    print(f"[Saved table]    {csv_path}")

# -----------------------
# --- MODEL HELPERS -----
# -----------------------

def make_csp(n_comp: int) -> CSP:
    # average_power + log=True gives stable log-power features
    return CSP(n_components=n_comp, reg="ledoit_wolf",
               cov_est="epoch", transform_into="average_power", log=True)

def fit_csp_lda(X: np.ndarray, y: np.ndarray, n_comp: int) -> Tuple[CSP, StandardScaler, LinearDiscriminantAnalysis]:
    csp = make_csp(n_comp)
    Z = csp.fit_transform(X, y)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Z)
    Zs = scaler.transform(Z)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(Zs, y)
    return csp, scaler, lda

def choose_best_ncomp(X: np.ndarray, y: np.ndarray, n_list: List[int], cv_folds: int) -> int:
    best_acc, best_n = -1.0, n_list[0]
    for n in n_list:
        csp = make_csp(n)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        # pipeline: CSP -> StandardScaler -> LDA via manual CV
        Z = csp.fit_transform(X, y)
        scaler = StandardScaler().fit(Z)
        Zs = scaler.transform(Z)
        acc = np.mean(cross_val_score(lda, Zs, y, cv=StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)))
        if acc > best_acc:
            best_acc, best_n = acc, n
    return best_n

def balance_downsample(X: np.ndarray, n_keep: int, rng) -> np.ndarray:
    if X.shape[0] <= n_keep:
        return X
    idx = rng.choice(X.shape[0], size=n_keep, replace=False)
    return X[idx]

# -----------------------
# -------- MAIN ---------
# -----------------------

def main():
    # Quiet MNE
    mne.set_log_level("WARNING")
    ensure_outdir(OUTPUT_DIR)
    rng = check_random_state(RANDOM_STATE)

    # --- Build training Laplacian epochs ---
    sfreq_ref = None
    ref_order = None
    per_label_epochs = {}
    per_label_names = None

    print("[Training] Gathering Laplacian epochs from registrations (A,B,C,D) ...")
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
        ev = valid_events_for_window(raw, events[events[:, 2] == code], TMIN, TMAX)

        names_needed = set()
        for base, neigh in CENTRAL_BASES.values():
            names_needed.add(base); names_needed.update(neigh)
        picks = mne.pick_channels(raw.info["ch_names"], include=list(names_needed))
        if len(picks) == 0:
            raise RuntimeError(f"[{lbl}] Central/neighbor channels missing in {os.path.basename(fpath)}.")

        epochs = mne.Epochs(raw, ev, event_id={chosen: code}, tmin=TMIN, tmax=TMAX,
                            baseline=None, picks=picks, preload=True, detrend=0,
                            reject_by_annotation=False, verbose="ERROR")
        if len(epochs) == 0:
            raise RuntimeError(f"No epochs in {os.path.basename(fpath)} for '{chosen}'.")

        data = epochs.get_data()
        ch_names = [epochs.ch_names[i] for i in range(len(epochs.ch_names))]
        lap_ep, lap_names = derive_laplacian_epochs(data, ch_names)

        if BASELINE_USE_PRE_ANNOT:
            bmean, bstd, secs, lap_names_b = compute_baseline_stats_on_laplacian(raw, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
            if lap_names_b != lap_names:
                idx_map = {n: i for i, n in enumerate(lap_names_b)}
                idx = [idx_map[n] for n in lap_names if n in idx_map]
                bmean, bstd = bmean[idx], bstd[idx]
            lap_ep = apply_baseline_norm(lap_ep, bmean, bstd, BASELINE_MODE)

        if ref_order is None:
            ref_order = lap_names
        else:
            idx_map = {n: i for i, n in enumerate(lap_names)}
            idx = [idx_map[n] for n in ref_order if n in idx_map]
            lap_ep = lap_ep[:, idx, :]

        per_label_epochs[lbl] = lap_ep
        per_label_names = ref_order

    print("\n[Sanity] Per-label epoch counts and shapes:")
    for k in ["A","B","C","D"]:
        x = per_label_epochs[k]
        print(f"  {k}: n={x.shape[0]}, C={x.shape[1]}, T={x.shape[2]}")

    # --- Balance training (per pair) ---
    if BALANCE_TRAIN:
        n_ab = min(per_label_epochs["A"].shape[0], per_label_epochs["B"].shape[0])
        n_cd = min(per_label_epochs["C"].shape[0], per_label_epochs["D"].shape[0])
        per_label_epochs["A"] = balance_downsample(per_label_epochs["A"], n_ab, rng)
        per_label_epochs["B"] = balance_downsample(per_label_epochs["B"], n_ab, rng)
        per_label_epochs["C"] = balance_downsample(per_label_epochs["C"], n_cd, rng)
        per_label_epochs["D"] = balance_downsample(per_label_epochs["D"], n_cd, rng)

    # --- Build datasets for gate and sub-tasks ---
    X_gate = np.concatenate([per_label_epochs["A"], per_label_epochs["B"],
                             per_label_epochs["C"], per_label_epochs["D"]], axis=0)
    y_gate = np.concatenate([
        np.array(["AB"] * per_label_epochs["A"].shape[0]),
        np.array(["AB"] * per_label_epochs["B"].shape[0]),
        np.array(["CD"] * per_label_epochs["C"].shape[0]),
        np.array(["CD"] * per_label_epochs["D"].shape[0]),
    ])

    X_ab = np.concatenate([per_label_epochs["A"], per_label_epochs["B"]], axis=0)
    y_ab = np.concatenate([
        np.array(["A"] * per_label_epochs["A"].shape[0]),
        np.array(["B"] * per_label_epochs["B"].shape[0]),
    ])

    X_cd = np.concatenate([per_label_epochs["C"], per_label_epochs["D"]], axis=0)
    y_cd = np.concatenate([
        np.array(["C"] * per_label_epochs["C"].shape[0]),
        np.array(["D"] * per_label_epochs["D"].shape[0]),
    ])

    # --- Auto-select CSP components via CV ---
    best_n_gate = choose_best_ncomp(X_gate, y_gate, n_list=[4, 6, 8], cv_folds=N_FOLDS_CV)
    best_n_ab   = choose_best_ncomp(X_ab,   y_ab,   n_list=[4, 6, 8], cv_folds=N_FOLDS_CV)
    best_n_cd   = choose_best_ncomp(X_cd,   y_cd,   n_list=[4, 6, 8], cv_folds=N_FOLDS_CV)
    print(f"\n[Auto] Best CSP components -> Gate:{best_n_gate}  AB:{best_n_ab}  CD:{best_n_cd}")

    # --- Train final models on full (balanced) data ---
    csp_gate, scaler_gate, lda_gate = fit_csp_lda(X_gate, y_gate, best_n_gate)
    csp_ab,   scaler_ab,   lda_ab   = fit_csp_lda(X_ab,   y_ab,   best_n_ab)
    csp_cd,   scaler_cd,   lda_cd   = fit_csp_lda(X_cd,   y_cd,   best_n_cd)

    # --- Get CV probabilities to tune thresholds/margins ---
    def cv_probs(X, y, n_comp):
        skf = StratifiedKFold(N_FOLDS_CV, shuffle=True, random_state=RANDOM_STATE)
        probs = np.zeros((len(y), 0))  # will stack along columns per class order
        cls_list = None
        all_probs = None
        for train, test in skf.split(X, y):
            csp = make_csp(n_comp)
            Z_tr = csp.fit_transform(X[train], y[train])
            scaler = StandardScaler().fit(Z_tr)
            Z_te = scaler.transform(csp.transform(X[test]))
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(scaler.transform(Z_tr), y[train])
            P = lda.predict_proba(Z_te)
            if cls_list is None:
                cls_list = lda.classes_.tolist()
                all_probs = np.zeros((len(y), len(cls_list)))
            all_probs[test] = P
        return all_probs, cls_list

    gate_probs, gate_classes = cv_probs(X_gate, y_gate, best_n_gate)  # columns correspond to classes
    ab_probs,   ab_classes   = cv_probs(X_ab,   y_ab,   best_n_ab)
    cd_probs,   cd_classes   = cv_probs(X_cd,   y_cd,   best_n_cd)

    # --- Energy override auto-tune (percentile sweep) ---
    ab_train = np.concatenate([per_label_epochs["A"], per_label_epochs["B"]], axis=0)
    cd_train = np.concatenate([per_label_epochs["C"], per_label_epochs["D"]], axis=0)
    ab_energy = np.array([lap_rms_energy(ep) for ep in ab_train])
    cd_energy = np.array([lap_rms_energy(ep) for ep in cd_train])

    # Heuristic: try percentiles 60..85, choose one that best separates AB (higher energy)
    p_grid = list(range(60, 86, 5))
    best_p = 70
    best_sep = -np.inf
    for p in p_grid:
        thr = np.percentile(ab_energy, p)
        # separation score: AB above thr (recall) minus CD above thr (false override)
        sep = (np.mean(ab_energy >= thr) - np.mean(cd_energy >= thr))
        if sep > best_sep:
            best_sep = sep
            best_p = p
    energy_thr = float(np.percentile(ab_energy, best_p))
    print(f"\n[Auto] Energy override percentile={best_p} -> threshold={energy_thr:.3f}  (sep={best_sep:.3f})")

    # --- Gate threshold & margin auto-tune on CV probs ---
    idx_ab_gate = gate_classes.index("AB")
    idx_cd_gate = gate_classes.index("CD")
    y_gate_bin = (y_gate == "CD").astype(int)  # 0=AB,1=CD for scoring

    def eval_gate(GA, GC, margin):
        preds = []
        for i in range(len(y_gate)):
            p_ab = gate_probs[i, idx_ab_gate]
            p_cd = gate_probs[i, idx_cd_gate]
            if p_ab >= GA:
                preds.append("AB")
            elif (p_cd >= GC) and ((p_cd - p_ab) >= margin):
                preds.append("CD")
            else:
                preds.append("UNK")
        # scores
        y_true = y_gate
        acc = np.mean([preds[i] == y_true[i] for i in range(len(y_true)) if preds[i] != "UNK"]) if np.any(np.array(preds)!="UNK") else 0.0
        # AB-friendliness: penalize false CD on AB more
        fp_cd_on_ab = sum(1 for i in range(len(y_true)) if (y_true[i]=="AB" and preds[i]=="CD"))
        fn_ab = sum(1 for i in range(len(y_true)) if (y_true[i]=="AB" and preds[i]!="AB"))
        score = acc - 0.5*(fp_cd_on_ab/len(y_true)) - 0.2*(fn_ab/len(y_true))
        return score, preds

    GA_grid = np.linspace(0.50, 0.80, 4)  # AB min prob
    GC_grid = np.linspace(0.65, 0.90, 4)  # CD min prob
    MG_grid = [0.05, 0.10, 0.15, 0.20]    # margin

    best_gate = (0.6, 0.8, 0.15)
    best_score = -1e9
    best_cv_preds = None
    for GA in GA_grid:
        for GC in GC_grid:
            for MG in MG_grid:
                sc, cv_pred = eval_gate(GA, GC, MG)
                if sc > best_score:
                    best_score = sc
                    best_gate = (float(GA), float(GC), float(MG))
                    best_cv_preds = cv_pred

    GATE_PROBA_MIN_AB, GATE_PROBA_MIN_CD, GATE_DOM_MARGIN = best_gate
    print(f"[Auto] Gate tuned: AB>= {GATE_PROBA_MIN_AB:.2f}  CD>= {GATE_PROBA_MIN_CD:.2f}  margin={GATE_DOM_MARGIN:.2f}")

    # --- Sub-thresholds auto-tune (A/B and C/D) ---
    def tune_sub_threshold(probs: np.ndarray, classes: List[str], y_true: np.ndarray):
        # one-vs-all max prob; try thresholds to maximize accuracy
        maxp = probs.max(axis=1)
        best_th, best_acc = 0.5, -1.0
        for th in np.linspace(0.55, 0.85, 7):
            pred = []
            for i in range(len(maxp)):
                if maxp[i] >= th:
                    pred.append(classes[int(np.argmax(probs[i]))])
                else:
                    pred.append("UNK")
            acc = np.mean([pred[i] == y_true[i] for i in range(len(y_true)) if pred[i] != "UNK"]) if np.any(np.array(pred)!="UNK") else 0.0
            acc -= 0.1 * (np.mean(np.array(pred)=="UNK"))  # mild penalty on too many UNKs
            if acc > best_acc:
                best_acc, best_th = acc, float(th)
        return best_th

    SUB_TH_AB = tune_sub_threshold(ab_probs, ab_classes, y_ab)
    SUB_TH_CD = tune_sub_threshold(cd_probs, cd_classes, y_cd)
    print(f"[Auto] Sub-thresholds tuned: AB>= {SUB_TH_AB:.2f}  CD>= {SUB_TH_CD:.2f}")

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

    if BASELINE_USE_PRE_ANNOT:
        bmean_t, bstd_t, secs_t, lap_names_b_t = compute_baseline_stats_on_laplacian(raw_t, BASELINE_MIN_SEC, BASELINE_MARGIN_SEC)
        if lap_names_b_t != lap_names_t:
            idx_map = {n: i for i, n in enumerate(lap_names_b_t)}
            idx = [idx_map[n] for n in lap_names_t if n in idx_map]
            bmean_t, bstd_t = bmean_t[idx], bstd_t[idx]
        lap_t = apply_baseline_norm(lap_t, bmean_t, bstd_t, BASELINE_MODE)

    # Reorder target Lap order to training ref
    if lap_names_t != per_label_names:
        idx_map = {n: i for i, n in enumerate(lap_names_t)}
        idx = [idx_map[n] for n in per_label_names if n in idx_map]
        lap_t = lap_t[:, idx, :]

    # --- Gate + subclassify with tuned params ---
    print("\n[Predict] Gate & subclassify (CSP+LDA • auto-tuned) ...")
    pred_final, max_probs = [], []

    gate_cls = lda_gate.classes_.tolist()  # ['AB','CD']
    i_ab = gate_cls.index("AB"); i_cd = gate_cls.index("CD")

    ab_cls = lda_ab.classes_.tolist()      # ['A','B']
    cd_cls = lda_cd.classes_.tolist()      # ['C','D']

    for i in tqdm(range(lap_t.shape[0]), total=lap_t.shape[0], desc="[Classify]"):
        ep = lap_t[i:i+1]
        # gate probs
        z_gate = scaler_gate.transform(csp_gate.transform(ep))
        p = lda_gate.predict_proba(z_gate)[0]
        p_ab, p_cd = float(p[i_ab]), float(p[i_cd])

        # energy override (AB-friendly)
        e = lap_rms_energy(ep[0])
        override_ab = (e >= energy_thr) and ((p_cd - p_ab) < (GATE_DOM_MARGIN + 0.05))

        # decide group
        if p_ab >= GATE_PROBA_MIN_AB or override_ab:
            group = "AB"
        elif (p_cd >= GATE_PROBA_MIN_CD) and ((p_cd - p_ab) >= GATE_DOM_MARGIN):
            group = "CD"
        else:
            pred_final.append("UNK"); max_probs.append(max(p_ab, p_cd)); continue

        # subclassify with tuned thresholds
        if group == "AB":
            z_ab = scaler_ab.transform(csp_ab.transform(ep))
            p_sub = lda_ab.predict_proba(z_ab)[0]
            j = int(np.argmax(p_sub))
            if float(p_sub[j]) >= SUB_TH_AB:
                pred_final.append(ab_cls[j]); max_probs.append(float(p_sub[j]))
            else:
                pred_final.append("UNK"); max_probs.append(float(p_sub[j]))
        else:
            z_cd = scaler_cd.transform(csp_cd.transform(ep))
            p_sub = lda_cd.predict_proba(z_cd)[0]
            j = int(np.argmax(p_sub))
            if float(p_sub[j]) >= SUB_TH_CD:
                pred_final.append(cd_cls[j]); max_probs.append(float(p_sub[j]))
            else:
                pred_final.append("UNK"); max_probs.append(float(p_sub[j]))

    # smoothing
    pred_sm = smooth_labels(pred_final, SMOOTH_WIN, include_unk=SMOOTH_INCLUDE_UNK) if SMOOTH_WIN > 1 else pred_final

    # report & save
    counts = Counter(pred_sm)
    print("\n[Counts after smoothing]")
    for k in ["A","B","C","D","UNK"]:
        if counts.get(k, 0) > 0:
            print(f"  {k}: {counts[k]}")
    if SAVE_TXT_CSV:
        save_sequence(pred_sm, max_probs, OUTPUT_DIR, prefix="hier_csp_lda_autotune")

    print("\nDone.")

if __name__ == "__main__":
    main()

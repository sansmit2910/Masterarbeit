# -*- coding: utf-8 -*-
"""
CNN + BiLSTM (binary C/D) for MI registrations → detect on MI-mix
- Train on C & D registrations with sliding windows
- 8–30 Hz, 250 Hz resample, common channel pick by name
- 1D CNN + BiLSTM, label smoothing, light augment, per-channel z-score
- Annotation-locked detection on target with ± jitter and per-event baseline norm
- Temperature + UNK gate (τ, margin) tuned on validation (C/D)
- Saves: training curve, weights (.pt), norms (.npz), sequence (txt/csv),
        max-prob plot, shift histogram
"""

import os, random, warnings
from typing import Tuple, List

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------- USER PATHS -----------------------
# MI registrations (C=imagine-move-1, D=imagine-move-2) and MI mix target
PATH_C = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif"
PATH_D = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif"
PATH_TARGET = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_cnnlstm_cd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: paste C/D GT string to compute detection accuracy (leave empty to skip)
GROUND_TRUTH_STR = "D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D D C D C ..."
#GT string like C D D C C D....
# ----------------------- CONFIG ---------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = True

# EEG preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)
NOTCH = None  # set 50 or 60 to enable notch

# Channels preference (by name)
PREFERRED_CHANS = ["C3", "Cz", "C4", "FCz"]

# Windowing
WIN_SEC  = 1.0
STEP_SEC = 0.25
AUG_NOISE_STD = 0.03
AUG_TIME_JIT_SAMP = 5

# Training
EPOCHS = 40
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
VAL_SIZE = 0.15

# Detection
ANNOT_KEY = "Stimulus/S  1"
EVENT_TMIN = 0.0
SEG_SEC   = 1.0
JITTER_SAMP = 50
BASELINE_SEC = 0.2
SMOOTH_MEDIAN = 5  # odd int; set 1 to disable

# Threshold grid (temperature, prob threshold, margin)
TEMP_GRID = [0.7, 0.9, 1.0, 1.1, 1.3]
TAU_GRID  = [0.3, 0.4, 0.5]
MARGIN_GRID = [0.0, 0.05, 0.10, 0.15]

LABS = ["C", "D"]  # label names in this MI setup

# ----------------------- HELPERS --------------------------
def log(msg):
    print(msg, flush=True)

def read_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH: raw.notch_filter(NOTCH, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ: raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def pick_names_or_all(raw: mne.io.BaseRaw, wanted: List[str]) -> List[int]:
    names_upper = [n.upper() for n in raw.ch_names]
    idx = []
    for w in wanted:
        wu = w.upper()
        if wu in names_upper:
            idx.append(names_upper.index(wu))
    if not idx:
        return mne.pick_types(raw.info, eeg=True).tolist()
    return sorted(list(dict.fromkeys(idx)))

def sliding_windows(data: np.ndarray, sf: float, win_sec: float, step_sec: float) -> np.ndarray:
    C, T = data.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    if win <= 0 or step <= 0 or win > T:
        return np.zeros((0, C, 0), dtype=np.float32)
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s+win]
    return X

def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=(0,2), keepdims=True)
    sd = X.std(axis=(0,2), keepdims=True) + 1e-8
    return mu, sd, (X - mu) / sd

def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

def train_val_split_idx(y, val_size=0.15, seed=SEED):
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)
    return tr, va

def events_from_annotations_or_stim(raw: mne.io.BaseRaw, prefer: str = ANNOT_KEY):
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception:
        events, event_id = np.empty((0,3), int), {}
    if events.size == 0:
        try:
            events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
            if len(events):
                vals, counts = np.unique(events[:,2], return_counts=True)
                chosen = int(vals[np.argmax(counts)])
                return events, chosen, f"code {chosen}"
        except Exception:
            pass
        raise RuntimeError("No annotations or stim events found.")
    if prefer in event_id:
        return events, int(event_id[prefer]), prefer
    vals, counts = np.unique(events[:,2], return_counts=True)
    chosen = int(vals[np.argmax(counts)])
    rev = {v:k for k,v in event_id.items()}
    return events, chosen, rev.get(chosen, f"code {chosen}")

def per_event_baseline_norm(seg: np.ndarray, base_len: int) -> np.ndarray:
    bl = max(1, min(base_len, seg.shape[1]))
    b = seg[:, :bl]
    mu = b.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True) + 1e-6
    return (seg - mu) / sd

# ----------------------- DATASETS -------------------------
class EEGWinSet(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.train = train

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]  # (C,T)
        if self.train:
            if AUG_TIME_JIT_SAMP > 0:
                s = random.randint(-AUG_TIME_JIT_SAMP, AUG_TIME_JIT_SAMP)
                if s < 0:
                    x = np.pad(x, ((0,0),(0,-s)), mode='edge')[:, -x.shape[1]:]
                elif s > 0:
                    x = np.pad(x, ((0,0),(s,0)), mode='edge')[:, :x.shape[1]]
            if AUG_NOISE_STD > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * AUG_NOISE_STD
        y = self.y[i]
        return x, y

# ----------------------- MODEL ---------------------------
class CNNBiLSTM_CD(nn.Module):
    def __init__(self, n_ch: int, n_cls: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_ch, 16, kernel_size=15, padding=7)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(input_size=32, hidden_size=32, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.drop  = nn.Dropout(0.25)
        self.fc    = nn.Linear(64, n_cls)

    def forward(self, x):
        # x: (B,C,T)
        x = self.pool(F.elu(self.bn1(self.conv1(x))))
        x = self.pool(F.elu(self.bn2(self.conv2(x))))  # (B,32,T/4)
        x = x.transpose(1, 2)  # (B,T',32)
        x, _ = self.lstm(x)    # (B,T',64)
        x = x.mean(dim=1)      # temporal avg
        x = self.drop(x)
        return self.fc(x)      # (B,2)

# ----------------------- TRAIN / EVAL --------------------
def make_windows_from_file(path: str) -> Tuple[np.ndarray, float, List[str]]:
    raw = read_raw(path)
    picks = pick_names_or_all(raw, PREFERRED_CHANS)
    names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)
    sf = raw.info["sfreq"]
    X = sliding_windows(data, sf, WIN_SEC, STEP_SEC)  # (N,C,T)
    return X, sf, names

def build_training():
    log("| [START] Load raws & pick common channels")
    X_C, sfC, namesC = make_windows_from_file(PATH_C)
    X_D, sfD, namesD = make_windows_from_file(PATH_D)
    if sfC != sfD:
        raise RuntimeError("Sampling frequencies differ after resample.")
    sf = sfC
    if X_C.shape[0] == 0 or X_D.shape[0] == 0:
        raise RuntimeError("No windows extracted from C or D. Check preprocessing/paths.")
    y_C = np.zeros((X_C.shape[0],), dtype=int)  # 0->C
    y_D = np.ones((X_D.shape[0],), dtype=int)   # 1->D
    X = np.concatenate([X_C, X_D], axis=0)
    y = np.concatenate([y_C, y_D], axis=0)
    if VERBOSE:
        log(f"|   [C] {os.path.basename(PATH_C)} | picks={namesC} | sf={sf:.1f} Hz | windows={len(y_C)}")
        log(f"|   [D] {os.path.basename(PATH_D)} | picks={namesD} | sf={sf:.1f} Hz | windows={len(y_D)}")
        log(f"| [TRAIN] Total windows: {X.shape} (N,C,T), per-class: {{'C': {len(y_C)}, 'D': {len(y_D)}}}")
        log("| [END]   Load raws & pick common channels")
    return X, y, sf, namesC

def label_smoothing_targets(y, n_cls=2, eps=LABEL_SMOOTH):
    t = np.full((len(y), n_cls), eps/(n_cls-1), dtype=np.float32)
    t[np.arange(len(y)), y] = 1.0 - eps
    return torch.from_numpy(t)

def train_model(X: np.ndarray, y: np.ndarray, sf: float, ch_names: List[str]):
    # standardize on training subset only
    tr_idx, va_idx = train_val_split_idx(y, val_size=VAL_SIZE, seed=SEED)
    mu, sd, Xtr_std = standardize_fit(X[tr_idx])
    Xva_std = standardize_apply(X[va_idx], mu, sd)

    ds_tr = EEGWinSet(Xtr_std, y[tr_idx], train=True)
    ds_va = EEGWinSet(Xva_std, y[va_idx], train=False)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    n_ch = X.shape[1]
    model = CNNBiLSTM_CD(n_ch=n_ch, n_cls=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_va = -1
    tr_hist, va_hist = [], []

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_correct, tr_total = 0, 0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model(xb)
            targets = label_smoothing_targets(yb.cpu().numpy()).to(DEVICE)
            loss = F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            tr_correct += int((logits.argmax(1) == yb).sum().item())
            tr_total   += int(xb.size(0))
        tr_acc = tr_correct / max(1, tr_total)
        tr_hist.append(tr_acc)

        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                logits = model(xb)
                va_correct += int((logits.argmax(1) == yb).sum().item())
                va_total   += int(xb.size(0))
        va_acc = va_correct / max(1, va_total)
        va_hist.append(va_acc)
        log(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={tr_acc:.4f} va={va_acc:.4f}")

        # Save pure weights + norms
        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_weights.pt"))
            np.savez(os.path.join(OUT_DIR, "norms.npz"),
                     mu=mu, sd=sd, sf=sf,
                     ch_names=np.array(ch_names, dtype=object))

    # training curve
    plt.figure(figsize=(6,3))
    plt.plot(tr_hist, label="train acc")
    plt.plot(va_hist, label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout()
    path_curve = os.path.join(OUT_DIR, "final_training_curve_cnnlstm_cd.png")
    plt.savefig(path_curve, dpi=150); plt.close()
    log(f"[Saved] Final training curve → {path_curve}")

    # Reload best weights + norms (PyTorch 2.6-safe)
    state_path = os.path.join(OUT_DIR, "best_weights.pt")
    try:
        state_dict = torch.load(state_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    norms = np.load(os.path.join(OUT_DIR, "norms.npz"), allow_pickle=True)
    mu = norms["mu"]; sd = norms["sd"]; sf = float(norms["sf"]); ch_names = list(norms["ch_names"])
    return model, mu, sd, sf, ch_names, (tr_hist, va_hist), (Xva_std, y[va_idx])

def choose_thresholds(model, Xva: np.ndarray, yva: np.ndarray):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xva.astype(np.float32)).to(DEVICE)
        logits = model(xb).cpu().numpy()
    best = (1.0, 0.5, 0.0, 0.0)  # T, tau, margin, score
    for T in TEMP_GRID:
        probs = F.softmax(torch.from_numpy(logits / T), dim=1).numpy()
        conf  = probs.max(axis=1)
        diff  = np.abs(probs[:,1] - probs[:,0])
        pred  = probs.argmax(axis=1)
        for tau in TAU_GRID:
            for m in MARGIN_GRID:
                q = pred.copy()
                q[(conf < tau) | (diff < m)] = -1  # UNK=-1
                mask = q >= 0
                score = 0.0 if not np.any(mask) else balanced_accuracy_score(yva[mask], q[mask])
                if score > best[3]:
                    best = (T, tau, m, score)
    log(f"[Threshold] Best temperature T={best[0]:.2f}, τ={best[1]:.2f}, margin={best[2]:.2f} (val balanced acc={best[3]:.3f})")
    return best[0], best[1], best[2]

# ----------------------- DETECTION -----------------------
def detect_sequence(model, mu, sd, sf, ch_names, Tbest, TAU, MARGIN):
    raw = read_raw(PATH_TARGET)
    # pick channels by saved names
    picks = []
    for nm in ch_names:
        if nm in raw.ch_names:
            picks.append(raw.ch_names.index(nm))
    if not picks:
        picks = pick_names_or_all(raw, PREFERRED_CHANS)
        ch_names = [raw.ch_names[p] for p in picks]

    data = raw.get_data(picks=picks)  # (C,T)
    events, code, desc = events_from_annotations_or_stim(raw, ANNOT_KEY)
    ev = events[events[:,2] == code]
    if len(ev) == 0:
        raise RuntimeError("No target events after filtering by chosen code.")

    seg_samp = int(round(SEG_SEC * sf))
    base_len = int(round(BASELINE_SEC * sf))
    base_off = int(round(EVENT_TMIN * sf))

    # Zero-shift batch to estimate target μ/σ AFTER baseline norm, THEN re-std by train mu/sd
    Z0 = []
    for e in ev:
        center = e[0] + base_off
        start = center
        end = start + seg_samp
        if start < 0 or end > data.shape[1]:
            continue
        seg = data[:, start:end]
        seg = per_event_baseline_norm(seg, base_len)
        Z0.append(seg)
    if len(Z0) == 0:
        raise RuntimeError("No valid zero-shift segments in target.")
    Z0 = np.stack(Z0, axis=0)  # (N0,C,T)
    Z0_std = standardize_apply(Z0, mu, sd)
    tgt_mu = Z0_std.mean(axis=(0,2), keepdims=True)
    tgt_sd = Z0_std.std(axis=(0,2), keepdims=True) + 1e-8

    labels = []
    max_probs = []
    best_shifts = []
    jitters = np.arange(-JITTER_SAMP, JITTER_SAMP+1, dtype=int)

    model.eval()
    with torch.no_grad():
        for e in ev:
            center = e[0] + base_off
            cand_segments, cand_shifts = [], []
            for s in jitters:
                start = center + s
                end = start + seg_samp
                if start < 0 or end > data.shape[1]:
                    continue
                seg = data[:, start:end]
                seg = per_event_baseline_norm(seg, base_len)
                seg = standardize_apply(seg[np.newaxis, ...], mu, sd)
                seg = (seg - tgt_mu) / tgt_sd
                cand_segments.append(seg[0])
                cand_shifts.append(s)
            if not cand_segments:
                labels.append("UNK"); max_probs.append(0.0); best_shifts.append(0); continue

            Xb = torch.from_numpy(np.stack(cand_segments, axis=0).astype(np.float32)).to(DEVICE)
            logits = model(Xb) / Tbest
            prob = F.softmax(logits, dim=1).cpu().numpy()
            conf = prob.max(axis=1)
            diff = np.abs(prob[:,1] - prob[:,0])
            pred = prob.argmax(axis=1)

            m = int(np.argmax(conf))
            pmax = conf[m]; dmax = diff[m]; h = pred[m]; s = cand_shifts[m]
            if (pmax < TAU) or (dmax < MARGIN):
                labels.append("UNK")
            else:
                labels.append(LABS[h])  # 0->C, 1->D
            max_probs.append(float(pmax))
            best_shifts.append(int(s))

    # optional smoothing (keep UNK)
    if SMOOTH_MEDIAN >= 3 and (SMOOTH_MEDIAN % 2 == 1):
        enc_map = {LABS[0]: 0, LABS[1]: 1}
        enc = np.array([enc_map.get(l, -1) for l in labels], dtype=int)
        mask = enc >= 0
        if np.any(mask):
            from scipy.signal import medfilt
            enc_s = enc.copy()
            enc_s[mask] = medfilt(enc[mask], kernel_size=SMOOTH_MEDIAN)
            inv = {0: LABS[0], 1: LABS[1]}
            labels = ["UNK" if v<0 else inv[v] for v in enc_s]

    # print & save
    print("\n=== Detected Sequence (CNN+LSTM C/D, annotation-locked + jitter) ===")
    line=[]
    for i,l in enumerate(labels,1):
        line.append(l)
        if len(line)>=90:
            print(" ".join(line)); line=[]
    if line: print(" ".join(line))
    counts = {LABS[0]: labels.count(LABS[0]), LABS[1]: labels.count(LABS[1]), "UNK": labels.count("UNK")}
    print("\n[Counts]")
    print(f"  {LABS[0]}:", counts[LABS[0]])
    print(f"  {LABS[1]}:", counts[LABS[1]])
    print("  UNK:", counts["UNK"])

    # write files
    seq_txt = os.path.join(OUT_DIR, "cnnlstm_cd_mi_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels) + "\n")
    seq_csv = os.path.join(OUT_DIR, "cnnlstm_cd_mi_mix_sequence.csv")
    with open(seq_csv, "w", encoding="utf-8") as f:
        f.write("event_idx,label,shift,max_prob\n")
        for i,(l,s,p) in enumerate(zip(labels, best_shifts, max_probs),1):
            f.write(f"{i},{l},{s},{p:.6f}\n")

    # plots
    plt.figure(figsize=(8,3))
    plt.plot(max_probs, lw=1)
    plt.axhline(y=TAU, ls="--", alpha=0.5)
    plt.xlabel("Event #"); plt.ylabel("Max prob")
    plt.title("Per-event max posterior (MI mix)")
    plt.tight_layout()
    out_prob = os.path.join(OUT_DIR, "mi_mix_maxprob_cnnlstm_cd.png")
    plt.savefig(out_prob, dpi=150); plt.close()

    plt.figure(figsize=(6,3))
    bins = np.arange(-JITTER_SAMP-0.5, JITTER_SAMP+1.5, 1)
    plt.hist(best_shifts, bins=bins, edgecolor="black")
    plt.xlabel("Best shift (samples)"); plt.ylabel("Count")
    plt.title("Target shift histogram (MI mix)")
    plt.tight_layout()
    out_shift = os.path.join(OUT_DIR, "target_shift_hist_cnnlstm_cd.png")
    plt.savefig(out_shift, dpi=150); plt.close()

    # accuracy vs GT if provided (expects C/D tokens)
    if GROUND_TRUTH_STR.strip():
        gt_tokens = [t.strip() for t in GROUND_TRUTH_STR.strip().split()]
        n = min(len(gt_tokens), len(labels))
        gt = np.array(gt_tokens[:n])
        pr = np.array(labels[:n])
        acc = (gt == pr).mean()
        print(f"[Accuracy vs GT] {acc*100:.2f}% (truncated to common length)")
    else:
        print("[Accuracy vs GT] (skipped — no GT pasted)")

    print(f"[Saved] Sequence → {seq_txt}")
    print(f"[Saved] Table    → {seq_csv}")
    print(f"[Saved] Probability trace → {out_prob}")
    print(f"[Saved] Shift histogram  → {out_shift}")

# ----------------------- MAIN ----------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    X, y, sf, ch_names = build_training()
    model, mu, sd, sf, ch_names, _hist, val_pack = train_model(X, y, sf, ch_names)

    # quick validation report
    Xva_std, yva = val_pack
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xva_std.astype(np.float32)).to(DEVICE)).cpu().numpy()
    yhat = logits.argmax(axis=1)
    print("\n[Validation] Classification report:")
    print(classification_report(yva, yhat, target_names=LABS, digits=3, zero_division=0))
    print("[Validation] Confusion matrix:")
    print(confusion_matrix(yva, yhat))
    print(f"[Validation] Accuracy: {accuracy_score(yva, yhat):.3f}")

    # choose thresholds (temperature, tau, margin) on validation slice
    Tbest, TAU, MARGIN = choose_thresholds(model, Xva_std, yva)

    # detect on MI mix using chosen thresholds
    detect_sequence(model, mu, sd, sf, ch_names, Tbest, TAU, MARGIN)

if __name__ == "__main__":
    main()

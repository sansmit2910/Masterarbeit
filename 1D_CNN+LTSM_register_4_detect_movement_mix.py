# -*- coding: utf-8 -*-
"""
1D CNN + BiLSTM for EEG (A,B,C,D) -> detect sequence in movement-mix target

- Train on four registrations (A,B,C,D) using sliding windows (continuous, not event-locked)
- Quick grouped CV (leave-one-registration-out like) for sanity check
- Final train on all windows
- Detect 4-class sequence in movement-mix target:
    * Use annotations (prefers 'Stimulus/S  1'); fallback to most frequent event code
    * Jitter scan ±JITTER samples; pick shift with highest posterior
    * Optional target-domain z-normalization from zero-shift windows
    * Optional temperature scaling from a train/val split
    * Optional UNK thresholding

Dependencies: mne, numpy, torch, scikit-learn, matplotlib
"""

import os
import math
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score, log_loss
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# === USER FILE PATHS ===
# --------------------------
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"
OUT_DIR     = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_cnnlstm_abcd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)   # mu/beta band
NOTCH_HZ = None          # 50 or 60 if needed; else None
REF_AVG  = True

# Channels: prefer motor set; fallback to all EEG intersection
PREFERRED_CHANSETS = [
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"]
]

# Sliding-window params (training)
WIN_SEC  = 1.0
STEP_SEC = 0.25

# Training params
BATCH_SIZE = 128
EPOCHS     = 25
LR         = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT      = 0.5

# CV
DO_QUICK_GROUPED_CV = True

# Detection params
EVENT_DESC_PREF = "Stimulus/S  1"   # annotation label
TMIN_SEC        = 0.0               # segment start relative to event
SEG_SEC         = 1.0               # segment length
JITTER_SAMP     = 50                # ± jitter (samples)
BASELINE_SEC    = 0.2               # baseline length (pre-seg) to subtract mean
USE_TARGET_ZNORM = True             # compute μ/σ on zero-shift target segments

# Thresholding / calibration
GRID_TEMPS = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
GRID_TAU   = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
UNK_LABEL  = "UNK"

# Optional: compute accuracy vs ground-truth sequence if provided (tokens space-separated)
GT_SEQUENCE_STR: Optional[str] = None  # paste ground-truth string if you want accuracy reported


# --------------------------
# === UTILITIES ===
# --------------------------
def log(msg: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "|", msg)

def load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if REF_AVG:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def pick_common_channels(raws: List[mne.io.BaseRaw]) -> List[int]:
    # Try preferred sets (exact names)
    names_all = [set([n.upper() for n in r.ch_names]) for r in raws]
    def idx_for(raw, ordered_names):
        idx = []
        upper = [n.upper() for n in raw.ch_names]
        for nm in ordered_names:
            if nm.upper() in upper:
                idx.append(upper.index(nm.upper()))
            else:
                return None
        return idx

    for wanted in PREFERRED_CHANSETS:
        ok = True
        for r in raws:
            if idx_for(r, wanted) is None:
                ok = False
                break
        if ok:
            # use the order of 'wanted'
            return idx_for(raws[0], wanted)

    # Else use intersection (keep order from first raw)
    common = set([n.upper() for n in raws[0].ch_names])
    for r in raws[1:]:
        common &= set([n.upper() for n in r.ch_names])
    if not common:
        raise RuntimeError("No common EEG channels across files.")
    ordered = [i for i, n in enumerate([n.upper() for n in raws[0].ch_names]) if n in common]
    return ordered

def sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win  = int(round(win_sec  * sf))
    step = int(round(step_sec * sf))
    if win <= 0 or step <= 0 or win > T:
        return np.zeros((0, C, max(win, 1)), dtype=np.float32)
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = data[:, s:s+win]
        seg = seg - seg.mean(axis=1, keepdims=True)  # 0-mean per-channel
        X[i] = seg
    return X

def standardize_inplace(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    # X: (N,C,T), mu: (C,1), sigma: (C,1)
    X -= mu
    X /= np.where(sigma > 1e-8, sigma, 1.0)

def compute_mu_sigma(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # per-channel over all windows/time
    # X: (N,C,T)
    if X.size == 0:
        return np.zeros((X.shape[1], 1), dtype=np.float32), np.ones((X.shape[1], 1), dtype=np.float32)
    mu = X.mean(axis=(0, 2), keepdims=False)[:, None]
    sd = X.std(axis=(0, 2), keepdims=False)[:, None]
    mu = mu.astype(np.float32)
    sd = sd.astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd

def extract_events(raw: mne.io.BaseRaw, prefer_desc: str = EVENT_DESC_PREF):
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) == 0 or events.size == 0:
        # fallback to stim channel
        try:
            events = mne.find_events(raw, verbose="ERROR")
        except Exception:
            events = np.empty((0, 3), dtype=int)
        if events.size == 0:
            raise RuntimeError("No events/annotations found in target file.")
        unique, counts = np.unique(events[:, 2], return_counts=True)
        chosen = unique[np.argmax(counts)]
        return events, chosen, f"code {int(chosen)}"
    if prefer_desc in event_id:
        chosen = event_id[prefer_desc]
        return events, chosen, prefer_desc
    unique, counts = np.unique(events[:, 2], return_counts=True)
    chosen = unique[np.argmax(counts)]
    rev = {v: k for k, v in event_id.items()}
    desc = rev.get(chosen, f"code {int(chosen)}")
    return events, chosen, desc

# --------------------------
# === DATASET ===
# --------------------------
class WinDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray], augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (C,T)
        if self.augment:
            # tiny time-shift (± 10 samples)
            max_shift = 10
            s = np.random.randint(-max_shift, max_shift+1)
            if s != 0:
                if s > 0:
                    x = np.concatenate([x[:, s:], x[:, :s]], axis=1)
                else:
                    s = -s
                    x = np.concatenate([x[:, -s:], x[:, :-s]], axis=1)
            # slight Gaussian noise
            x = x + 0.01 * np.random.randn(*x.shape).astype(np.float32)

        x = torch.from_numpy(x.astype(np.float32))  # (C,T)
        # model expects (B, C, T); we keep it like this for Conv1d
        if self.y is None:
            return x
        return x, int(self.y[idx])

# --------------------------
# === MODEL ===
# --------------------------
class CNNBiLSTM(nn.Module):
    """
    Input: (B, C, T)
    Conv1d blocks over time -> (B, F, L)
    Permute -> (B, L, F) for BiLSTM
    Global mean over time of BiLSTM outputs -> linear -> logits
    """
    def __init__(self, n_ch: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_ch, 32, kernel_size=15, padding=7, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9,  padding=4, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(DROPOUT)

        self.lstm  = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,
                             batch_first=True, bidirectional=True, dropout=0.0)
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc    = nn.Linear(128, n_classes)

    def forward(self, x):  # x:(B,C,T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,32,T/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,64,T/4)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)  # (B,L,64)
        out, _ = self.lstm(x)   # (B,L,128) biLSTM
        out = out.mean(dim=1)   # global mean over time
        out = self.drop2(out)
        logits = self.fc(out)   # (B,K)
        return logits

# --------------------------
# === TRAIN / EVAL ===
# --------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device)   # (B,C,T)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=0.05)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * xb.size(0)
        pred = logits.argmax(1)
        correct += int((pred == yb).sum().item())
        total += xb.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    all_logits = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss_sum += float(loss.item()) * xb.size(0)
        pred = logits.argmax(1)
        correct += int((pred == yb).sum().item())
        total += xb.size(0)
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    all_logits = np.concatenate(all_logits) if all_logits else np.empty((0,2))
    return (loss_sum / max(total,1), correct / max(total,1), y_true, y_pred, all_logits)

def temperature_scale(logits: np.ndarray, y_true: np.ndarray, temps=GRID_TEMPS) -> float:
    best_T, best_nll = 1.0, 1e9
    for T in temps:
        probs = softmax_np(logits / T)
        nll = log_loss(y_true, probs, labels=np.arange(probs.shape[1]))
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return best_T

def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def quick_grouped_cv(X, y, g, class_names, device):
    if len(np.unique(g)) < 2:
        log("[CV] Skipped (need ≥2 groups).")
        return
    kf = GroupKFold(n_splits=min(4, len(np.unique(g))))
    accs = []
    for i, (tr, te) in enumerate(kf.split(X, y, g), 1):
        model = CNNBiLSTM(n_ch=X.shape[1], n_classes=len(class_names)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        tr_loader = DataLoader(WinDataset(X[tr], y[tr], augment=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        te_loader = DataLoader(WinDataset(X[te], y[te], augment=False), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        # a few epochs for sanity
        for _ in range(8):
            train_one_epoch(model, tr_loader, opt, device)
        _, acc, y_t, y_p, _ = eval_model(model, te_loader, device)
        accs.append(acc)
        log(f"[CV] Fold {i} acc={acc:.3f} (n={len(te)})")
    log(f"[CV] mean={np.mean(accs):.3f} ± {np.std(accs):.3f}")

# --------------------------
# === BUILD TRAIN SET ===
# --------------------------
def build_training_set(paths: Dict[str, str]):
    labels = sorted(paths.keys())  # ['A','B','C','D']
    raws = [load_raw(paths[k]) for k in labels]
    # Pick channels
    picks0 = pick_common_channels(raws)
    pick_names = [raws[0].ch_names[i] for i in picks0]
    log(f"[Channels] Using preferred/common picks: {pick_names}")
    sf = raws[0].info["sfreq"]

    allX, ally, allg = [], [], []
    for li, lab in enumerate(labels):
        raw = raws[li]
        Xw = sliding_windows(raw, picks0, WIN_SEC, STEP_SEC)
        y  = np.full((Xw.shape[0],), li, dtype=int)
        g  = np.full((Xw.shape[0],), li, dtype=int)
        log(f"  [{lab}] {os.path.basename(paths[lab])} | picks={pick_names} | sf={sf:.1f} Hz | windows={len(Xw)}")
        allX.append(Xw); ally.append(y); allg.append(g)

    X = np.concatenate(allX, axis=0) if allX else np.empty((0, len(picks0), int(WIN_SEC*sf)))
    y = np.concatenate(ally, axis=0) if ally else np.empty((0,), dtype=int)
    g = np.concatenate(allg, axis=0) if allg else np.empty((0,), dtype=int)

    # Standardize (train μ/σ)
    mu, sd = compute_mu_sigma(X)
    standardize_inplace(X, mu, sd)

    log(f"[TRAIN] Total windows: {X.shape} (N,C,T), per-class: " +
        str({labels[i]: int((y==i).sum()) for i in range(len(labels))}))
    return X, y, g, labels, picks0, sf, (mu, sd)

# --------------------------
# === FIT FINAL MODEL ===
# --------------------------
def fit_final_model(X, y, device):
    # simple stratified split by groups (simulate; use last 10% for val)
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_n = max(100, int(0.1 * n))
    val_idx = idx[:val_n]
    tr_idx  = idx[val_n:]

    tr_ds = WinDataset(X[tr_idx], y[tr_idx], augment=True)
    va_ds = WinDataset(X[val_idx], y[val_idx], augment=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = CNNBiLSTM(n_ch=X.shape[1], n_classes=len(np.unique(y))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    tr_hist, va_hist = [], []
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, device)
        va_loss, va_acc, y_t, y_p, logits = eval_model(model, va_loader, device)
        tr_hist.append(tr_acc); va_hist.append(va_acc)
        log(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={tr_acc:.4f} va={va_acc:.4f}")
    # Plot curve
    fig = plt.figure(figsize=(6,3))
    plt.plot(tr_hist, label="train acc"); plt.plot(va_hist, label="val acc")
    plt.ylim(0,1.02); plt.legend(); plt.title("Final training curve (acc)")
    plt.xlabel("epoch"); plt.ylabel("acc")
    pth = os.path.join(OUT_DIR, "final_training_curve_cnnlstm.png")
    plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(fig)
    log(f"[Saved] Final training curve → {pth}")

    # Validation report
    if len(y_t):
        print("\n[Validation] Classification report:")
        print(classification_report(y_t, y_p, digits=3))
        cm = confusion_matrix(y_t, y_p, labels=np.arange(len(np.unique(y))))
        print("[Validation] Confusion matrix:\n", cm)
        print(f"[Validation] Accuracy: {accuracy_score(y_t, y_p):.3f}")

        # Temperature scaling
        Tbest = temperature_scale(logits, y_t, GRID_TEMPS) if logits.size else 1.0
    else:
        Tbest = 1.0

    return model, {"T": Tbest}

# --------------------------
# === DETECTION ===
# --------------------------
@torch.no_grad()
def detect_sequence(model, meta, target_path, picks_ref, labels, train_mu_sd, device,
                    tmin_sec=TMIN_SEC, seg_sec=SEG_SEC, jitter_samp=JITTER_SAMP,
                    unk_thresholds=GRID_TAU):
    raw = load_raw(target_path)
    # Ensure we use the same channels (by name match); if missing, fallback to intersection order
    ref_names = [raw.info["ch_names"][i] for i in picks_ref]
    upper = [n.upper() for n in raw.ch_names]
    picks = []
    for nm in ref_names:
        try: picks.append(upper.index(nm.upper()))
        except ValueError: pass
    if len(picks) < len(picks_ref):
        # fallback to all EEG intersection with first raw's order assumption
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False)
    log(f"[Detect] Using target picks by name: {[raw.ch_names[i] for i in picks]}")

    events, chosen, desc = extract_events(raw, EVENT_DESC_PREF)
    ev = events[events[:, 2] == chosen]
    n_events = len(ev)
    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_off = int(round(tmin_sec * sf))
    base_samp = int(round(BASELINE_SEC * sf))

    data = raw.get_data(picks=picks)  # (C,T)
    C, T = data.shape

    # Target μ/σ from zero-shift windows (if enabled)
    mu_t, sd_t = None, None
    if USE_TARGET_ZNORM and n_events > 0:
        zeros = []
        for e in ev:
            start = e[0] + base_off
            end   = start + seg_samp
            if start < 0 or end > T: continue
            seg = data[:, start:end].copy()
            # subtract baseline (pre-seg) if available
            if base_samp > 0 and (start - base_samp) >= 0:
                bl = data[:, start-base_samp:start].mean(axis=1, keepdims=True)
                seg -= bl
            zeros.append(seg)
        if zeros:
            Z = np.stack(zeros, axis=0)  # (N0,C,T)
            mu_t, sd_t = compute_mu_sigma(Z)
            log(f"[Target μ/σ] computed on {len(zeros)} zero-shift events.")

    mu_tr, sd_tr = train_mu_sd

    # pick threshold τ by grid (maximize fraction of confident predictions on **zero-shift** only)
    # (fallback to 0.30 if no events)
    tau_best = 0.30
    if n_events > 0:
        conf_fracs = []
        for tau in unk_thresholds:
            cnt_conf = 0
            for e in ev:
                start = e[0] + base_off
                end   = start + seg_samp
                if start < 0 or end > T: continue
                seg = data[:, start:end].copy()
                if base_samp > 0 and (start - base_samp) >= 0:
                    bl = data[:, start-base_samp:start].mean(axis=1, keepdims=True)
                    seg -= bl
                # z-norm with target μ/σ if available
                if (mu_t is not None) and (sd_t is not None):
                    seg = seg - mu_t
                    seg = seg / np.where(sd_t>1e-8, sd_t, 1.0)
                else:
                    seg = seg - mu_tr
                    seg = seg / np.where(sd_tr>1e-8, sd_tr, 1.0)
                xb = torch.from_numpy(seg[None, ...].astype(np.float32)).to(device)  # (1,C,T)
                logits = model(xb).cpu().numpy()  # (1,K)
                Tscale = meta.get("T", 1.0)
                probs  = softmax_np(logits / Tscale)[0]
                if probs.max() >= tau:
                    cnt_conf += 1
            conf_fracs.append(cnt_conf / max(n_events,1))
        tau_best = unk_thresholds[int(np.argmax(conf_fracs))]
        log(f"[Threshold] Selected τ={tau_best:.2f} maximizing confident zero-shift fraction.")

    labels_out: List[str] = []
    probs_out:  List[np.ndarray] = []
    chosen_shifts: List[int] = []

    # Jitter loop
    shifts = np.arange(-jitter_samp, jitter_samp+1, dtype=int)
    Tscale = meta.get("T", 1.0)

    for k, e in enumerate(ev):
        best_p = -np.inf
        best_lab = UNK_LABEL
        best_shift = 0
        best_vec = None

        for s in shifts:
            start = e[0] + base_off + s
            end   = start + seg_samp
            if start < 0 or end > T: continue
            seg = data[:, start:end].copy()
            # baseline removal
            if base_samp > 0 and (start - base_samp) >= 0:
                bl = data[:, start-base_samp:start].mean(axis=1, keepdims=True)
                seg -= bl
            # z-norm
            if (mu_t is not None) and (sd_t is not None):
                seg = seg - mu_t
                seg = seg / np.where(sd_t>1e-8, sd_t, 1.0)
            else:
                seg = seg - mu_tr
                seg = seg / np.where(sd_tr>1e-8, sd_tr, 1.0)

            xb = torch.from_numpy(seg[None, ...].astype(np.float32)).to(device)
            logits = model(xb).cpu().numpy()
            probs  = softmax_np(logits / Tscale)[0]
            mp = float(probs.max())
            lab = labels[int(probs.argmax())]
            if mp > best_p:
                best_p = mp; best_lab = lab; best_shift = int(s); best_vec = probs

        if best_p < tau_best:
            labels_out.append(UNK_LABEL)
        else:
            labels_out.append(best_lab)
        probs_out.append(best_vec if best_vec is not None else np.full((len(labels),), np.nan))
        chosen_shifts.append(best_shift)

        if k < 5:
            print(f"[Detect] Event {k}: valid_shifts={np.sum([(e[0]+base_off+ss>=0 and e[0]+base_off+ss+seg_samp<=T) for ss in shifts])} "
                  f"(min={min(shifts)}, max={max(shifts)})")

    # Pretty print & counts
    print("\n=== Detected Sequence (CNN+LSTM, 4-class, annotation-locked+jitter) ===")
    line = []
    for i, tok in enumerate(labels_out, 1):
        line.append(tok)
        if len(line) == 90:
            print(" ".join(line)); line = []
    if line: print(" ".join(line))

    counts = {lab: labels_out.count(lab) for lab in labels}
    counts[UNK_LABEL] = labels_out.count(UNK_LABEL)
    print("\n[Counts]")
    for k in labels + [UNK_LABEL]:
        print(f"  {k}: {counts.get(k,0)}")

    # Save sequence + table
    seq_txt = os.path.join(OUT_DIR, "cnnlstm_abcd_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    seq_csv = os.path.join(OUT_DIR, "cnnlstm_abcd_movement_mix_sequence.csv")
    with open(seq_csv, "w", encoding="utf-8") as f:
        f.write("idx,label,shift," + ",".join([f"p_{lb}" for lb in labels]) + "\n")
        for i,(lab,sh,pv) in enumerate(zip(labels_out, chosen_shifts, probs_out)):
            if pv is None: pv = np.full((len(labels),), np.nan)
            f.write(f"{i},{lab},{sh}," + ",".join([f"{v:.6f}" if np.isfinite(v) else "" for v in pv]) + "\n")
    log(f"[Saved] Sequence  → {seq_txt}")
    log(f"[Saved] Table     → {seq_csv}")

    # Figures
    fig = plt.figure(figsize=(6,3))
    plt.hist(chosen_shifts, bins=np.arange(-jitter_samp-0.5, jitter_samp+1.5, 1), edgecolor="black")
    plt.title("Target jitter shift histogram")
    plt.xlabel("Shift (samples)"); plt.ylabel("Count")
    pth = os.path.join(OUT_DIR, "target_shift_hist_cnnlstm.png")
    plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(fig)
    log(f"[Saved] Shift histogram → {pth}")

    # Max-prob trace
    maxp = [float(np.nanmax(v)) if v is not None else np.nan for v in probs_out]
    fig = plt.figure(figsize=(7,3))
    plt.plot(maxp, lw=1); plt.axhline(tau_best, ls="--", color="gray")
    plt.title("Per-event maximum posterior (best shift)")
    plt.xlabel("Event #"); plt.ylabel("Max p")
    pth = os.path.join(OUT_DIR, "target_maxprob_cnnlstm.png")
    plt.tight_layout(); plt.savefig(pth, dpi=150); plt.close(fig)
    log(f"[Saved] Probability trace → {pth}")

    # Optional accuracy vs provided GT
    if GT_SEQUENCE_STR is not None:
        gt_tokens = GT_SEQUENCE_STR.split()
        L = min(len(gt_tokens), len(labels_out))
        acc = np.mean([gt_tokens[i] == labels_out[i] for i in range(L)]) if L>0 else 0.0
        log(f"[Accuracy vs GT] {acc*100:.2f}% (truncated to common length)")

    return labels_out, counts


# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("[START] Load raws & pick common channels")
    X, y, g, class_names, picks, sf, train_mu_sd = build_training_set(TEMPLATE_SOURCES)
    log("[END]   Load raws & pick common channels")

    # Quick grouped CV (sanity)
    if DO_QUICK_GROUPED_CV and len(np.unique(g)) >= 2:
        quick_grouped_cv(X, y, g, class_names, device)

    # Final fit
    model, meta = fit_final_model(X, y, device)

    # Detect
    events, chosen, desc = extract_events(load_raw(TARGET_FILE), EVENT_DESC_PREF)
    n_events = (events[:,2] == chosen).sum()
    log(f"[Target] Using events '{desc}'; N={n_events}")

    _ = detect_sequence(model, meta, TARGET_FILE, picks, class_names, train_mu_sd, device,
                        tmin_sec=TMIN_SEC, seg_sec=SEG_SEC, jitter_samp=JITTER_SAMP,
                        unk_thresholds=GRID_TAU)

    log(f"[Done] All outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()

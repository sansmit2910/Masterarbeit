# -*- coding: utf-8 -*-
"""
Time-Series Transformer (2-class MI: C/D)
- Train only on two MI registrations (C, D)
- Detect sequence on MI target recording with annotation-locked + jitter scan
- Optional accuracy vs provided ground-truth string (A/B/C/D tokens)

Requires: mne, numpy, torch, scikit-learn, matplotlib
"""

import os
import sys
import math
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# === USER PATHS (MI) ===
# --------------------------
MI_SOURCES: Dict[str, str] = {
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}

# If you have a specific MI target mix file, point to it here.
# Common naming pattern is "imagine-move-mix". Adjust if needed.
MI_TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_tsxfmr_cd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: paste GT as a single string of tokens separated by spaces
# (example) GT_STRING = "C D C D C D ..."
GT_STRING = "D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C D C DC D C D C D C D C D C D"
  # keep empty if you don't have GT for this target

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_STATE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preproc
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)
NOTCH_HZ = None

# Windows
WIN_SEC = 1.0
STEP_SEC = 0.25

# Target events
EVENT_DESC_PREF = "Stimulus/S  1"
TMIN_SEC = 0.0
SEG_SEC = 1.0
JITTER_SAMP = 50
BASELINE_SEC = 0.2  # mean-removal using pre-segment

# UNK threshold (probability); used only for detection
UNK_THRESH = 0.30

# Channels preference (by name); fallback to all EEG
PREFERRED_CHANSETS = [
    ["C3L", "CzL", "C4L"],
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"],
]

# Model / Training
D_MODEL = 64
N_HEAD = 4
FF_MULT = 4
N_LAYERS = 2
DROPOUT = 0.15
EPOCHS = 40
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8   # early stopping patience
VAL_FRAC = 0.15

# ---------------------------------
# === HELPERS: IO / Preprocessing
# ---------------------------------
def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def _pick_channels(raw: mne.io.BaseRaw) -> List[int]:
    upper = [ch.upper() for ch in raw.ch_names]
    for wanted in PREFERRED_CHANSETS:
        idx = []
        for w in wanted:
            if w.upper() in upper:
                idx.append(upper.index(w.upper()))
            else:
                idx = []
                break
        if idx:
            return idx
    # fallback
    return mne.pick_types(raw.info, eeg=True)

def _sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    starts = np.arange(0, max(0, T - win + 1), step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s+win]
    return X

def _build_dataset(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str], float, List[str]]:
    """
    Returns: X (N,C,T), y (N,), labs ['C','D'], sf, ch_names_used
    """
    labs = sorted(list(paths.keys()))  # ['C','D']
    if labs != ['C', 'D']:
        raise ValueError("This script is configured for MI=C/D only. Keys found: {}".format(labs))
    all_X, all_y = [], []
    sf = None
    ch_names_used = None

    print("| [START] Load raws & pick common channels")
    raws = {}
    for k in labs:
        raws[k] = _load_raw(paths[k])

    # Find common pick set by trying preferred sets on first raw and ensuring all exist on others
    first = raws[labs[0]]
    picks = _pick_channels(first)
    # Ensure same names exist in both
    names = [first.ch_names[p] for p in picks]
    # remap picks for all raws by names
    common_idx_by_raw = {}
    for k in labs:
        r = raws[k]
        upper = [ch.upper() for ch in r.ch_names]
        idx = []
        for nm in names:
            if nm.upper() in upper:
                idx.append(upper.index(nm.upper()))
            else:
                raise RuntimeError(f"[{k}] Missing channel {nm} for common set.")
        common_idx_by_raw[k] = idx

    print(f"|   [C] {os.path.basename(paths['C'])} | picks={names} | sf={raws['C'].info['sfreq']:.1f} Hz")
    print(f"|   [D] {os.path.basename(paths['D'])} | picks={names} | sf={raws['D'].info['sfreq']:.1f} Hz")

    for k, lab_idx in zip(labs, [0, 1]):
        r = raws[k]
        if sf is None:
            sf = r.info["sfreq"]
            ch_names_used = [r.ch_names[i] for i in common_idx_by_raw[k]]
        Xk = _sliding_windows(r, common_idx_by_raw[k], WIN_SEC, STEP_SEC)
        yk = np.full((len(Xk),), lab_idx, dtype=np.int64)
        print(f"|   [{k}] windows={len(Xk)}")
        all_X.append(Xk)
        all_y.append(yk)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"| [TRAIN] Total windows: {X.shape} (N,C,T), per-class: "
          f"{{'C': {(y==0).sum()}, 'D': {(y==1).sum()}}}")
    print("| [END]   Load raws & pick common channels")
    return X, y, labs, sf, ch_names_used

def _events_from_target(raw: mne.io.BaseRaw, prefer_desc=EVENT_DESC_PREF):
    # Prefer annotations; fallback to stim channel if necessary
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) and events.size:
        if prefer_desc in event_id:
            chosen = event_id[prefer_desc]
        else:
            u, c = np.unique(events[:,2], return_counts=True)
            chosen = u[np.argmax(c)]
        return events[events[:,2]==chosen], chosen, {v:k for k,v in event_id.items()}.get(chosen, str(chosen))
    # fallback
    evs = mne.find_events(raw, shortest_event=1, verbose="ERROR")
    if evs.size == 0:
        raise RuntimeError("No events/annotations found in target file.")
    u, c = np.unique(evs[:,2], return_counts=True)
    chosen = u[np.argmax(c)]
    return evs[evs[:,2]==chosen], chosen, f"code {int(chosen)}"

# ---------------------------------
# === DATASETS
# ---------------------------------
class WindowsDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mu: Optional[np.ndarray]=None, sd: Optional[np.ndarray]=None):
        # X: (N,C,T)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        if mu is None or sd is None:
            mu = self.X.mean(axis=(0,2), keepdims=True)     # (1,C,1)
            sd = self.X.std(axis=(0,2), keepdims=True) + 1e-6
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        x = (self.X[i:i+1] - self.mu) / self.sd  # keep (1,C,T) for conv1d
        return x.squeeze(0), self.y[i]           # (C,T), y

# ---------------------------------
# === MODEL
# ---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_ch: int, n_class: int, d_model: int, nhead: int, ff_mult: int, nlayers: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(n_ch, d_model, kernel_size=1)  # (B,C,T)->(B,D,T)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model*ff_mult,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        z = self.proj(x)           # (B,D,T)
        z = z.transpose(1, 2)      # (B,T,D)
        z = self.pe(z)             # (B,T,D)
        z = self.encoder(z)        # (B,T,D)
        z = z.mean(dim=1)          # (B,D)
        logits = self.head(z)      # (B,K)
        return logits

# ---------------------------------
# === TRAINING
# ---------------------------------
def _class_weights(y: np.ndarray, n_class: int) -> torch.Tensor:
    cls = np.arange(n_class, dtype=np.int64)
    present = np.unique(y)
    cls = np.array(sorted(present), dtype=np.int64)
    w = compute_class_weight(class_weight="balanced", classes=cls, y=y)
    # Map to full length
    full = np.ones((n_class,), dtype=np.float32)
    for c, ww in zip(cls, w):
        full[c] = float(ww)
    return torch.tensor(full, dtype=torch.float32)

def train_model(X: np.ndarray, y: np.ndarray, sf: float, ch_names: List[str]):
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)

    # global normalization from train split only
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(sss.split(X, y))
    mu = X[tr_idx].mean(axis=(0,2), keepdims=True).astype(np.float32)
    sd = (X[tr_idx].std(axis=(0,2), keepdims=True) + 1e-6).astype(np.float32)

    ds_tr = WindowsDS(X[tr_idx], y[tr_idx], mu, sd)
    ds_va = WindowsDS(X[va_idx], y[va_idx], mu, sd)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    n_ch = X.shape[1]
    n_class = 2  # C/D

    model = TimeSeriesTransformer(n_ch=n_ch, n_class=n_class, d_model=D_MODEL,
                                  nhead=N_HEAD, ff_mult=FF_MULT, nlayers=N_LAYERS, dropout=DROPOUT).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)
    w = _class_weights(y[tr_idx], n_class).to(DEVICE)
    best_state = None
    best_va = np.inf
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE)    # (B,C,T)
            yb = yb.to(DEVICE)    # (B,)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=w)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(ds_tr)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=w)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(ds_va)
        sched.step(va_loss)

        print(f"[Final] Epoch {epoch:02d}/{EPOCHS} | tr={tr_loss:.4f} va={va_loss:.4f}")

        if va_loss + 1e-6 < best_va:
            best_va = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Validation report
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb = xb.to(DEVICE)
            lg = model(xb).cpu().numpy()
            all_logits.append(lg)
            all_y.append(yb.numpy())
    all_logits = np.vstack(all_logits)
    all_y = np.concatenate(all_y)
    ypred = all_logits.argmax(1)
    print("\n[Validation] Classification report:")
    print(classification_report(all_y, ypred, target_names=["C","D"], digits=3))
    print("[Validation] Confusion matrix:")
    print(confusion_matrix(all_y, ypred, labels=[0,1]))
    print(f"[Validation] Accuracy: {accuracy_score(all_y, ypred):.3f}")

    # Plot training curve (simple last epoch marker)
    plt.figure(figsize=(5,3))
    plt.title("Final validation logits histogram")
    plt.hist(all_logits.max(1), bins=30, edgecolor="black")
    plt.xlabel("Max softmax probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "final_val_maxprob_hist_tsxfmr_cd.png"), dpi=140)
    plt.close()

    meta = dict(mu=mu, sd=sd, sf=sf, ch_names=ch_names)
    return model, meta

# ---------------------------------
# === DETECTION
# ---------------------------------
@torch.no_grad()
def _predict_best_shift(model: nn.Module, segs: np.ndarray) -> Tuple[int, float, np.ndarray, int]:
    """
    segs: (M, C, T) candidate segments across shifts
    Returns: (best_class_idx, best_prob, proba_vec_of_best, best_shift_index)
    """
    if segs.shape[0] == 0:
        return -1, np.nan, np.array([np.nan, np.nan]), -1
    X = torch.from_numpy(segs.astype(np.float32)).to(DEVICE)  # (M,C,T)
    logits = model(X)                                         # (M,2)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    m_idx, m_cls = np.unravel_index(np.nanargmax(probs), probs.shape)
    return int(m_cls), float(probs[m_idx, m_cls]), probs[m_idx], int(m_idx)

def detect_sequence(model: nn.Module,
                    meta: dict,
                    target_path: str,
                    labs: List[str],
                    unk_thresh: float = UNK_THRESH,
                    jitter_samp: int = JITTER_SAMP,
                    tmin_sec: float = TMIN_SEC,
                    seg_sec: float = SEG_SEC,
                    baseline_sec: float = BASELINE_SEC):
    raw = _load_raw(target_path)
    # Align picks by name
    upper = [ch.upper() for ch in raw.ch_names]
    pick_idx = []
    for nm in meta["ch_names"]:
        if nm.upper() not in upper:
            raise RuntimeError(f"Target missing channel {nm} required from training set.")
        pick_idx.append(upper.index(nm.upper()))

    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_off = int(round(tmin_sec * sf))
    base_pre = int(round(baseline_sec * sf))

    events, chosen, desc = _events_from_target(raw, EVENT_DESC_PREF)
    ev = events[:,0]
    print(f"\n[Target] Using events '{desc}'; N={len(ev)}")

    data = raw.get_data(picks=pick_idx)  # (C,T)
    C, T = data.shape
    labels_out, probs_out, shifts = [], [], []

    # Normalize with training mu/sd
    mu = meta["mu"]; sd = meta["sd"]

    for i, center in enumerate(ev):
        # pre-baseline mean removal (on raw, to mimic ERP-like cleaning)
        # We'll do it per-shift window below

        segs = []
        shift_vals = []
        for s in range(-jitter_samp, jitter_samp+1):
            start = center + base_off + s
            end = start + seg_samp
            if start < 0 or end > T:
                continue
            x = data[:, start:end].copy()  # (C,T)
            # baseline mean removal using pre-window if available
            if baseline_sec > 0 and (start - base_pre) >= 0:
                base = data[:, start-base_pre:start].mean(axis=1, keepdims=True)
                x = x - base
            # global normalization
            x = (x - mu.squeeze(0)) / sd.squeeze(0)
            segs.append(x)
            shift_vals.append(s)

        if len(segs) == 0:
            labels_out.append("UNK"); probs_out.append(np.array([np.nan, np.nan])); shifts.append(0)
            continue

        segs = np.stack(segs, axis=0)  # (M,C,T)
        m_cls, m_prob, m_vec, m_i = _predict_best_shift(model, segs)
        if m_cls < 0 or m_prob < unk_thresh:
            labels_out.append("UNK")
        else:
            labels_out.append(labs[m_cls])
        probs_out.append(m_vec)
        shifts.append(shift_vals[m_i])

    # Print sequence (wrapped)
    print("\n=== Detected Sequence (TS-Transformer C/D, annotation-locked + jitter) ===")
    line = []
    for t in labels_out:
        line.append(t)
        if len(line) >= 80:
            print(" ".join(line)); line = []
    if line: print(" ".join(line))

    # Counts
    counts = {lab: labels_out.count(lab) for lab in labs}
    counts["UNK"] = labels_out.count("UNK")
    print("\n[Counts]")
    for k in labs + ["UNK"]:
        print(f"  {k}: {counts.get(k, 0)}")

    # Save outputs
    seq_txt = os.path.join(OUT_DIR, "tsxfmr_cd_mi_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    csv_path = os.path.join(OUT_DIR, "tsxfmr_cd_mi_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("event_idx,label,best_shift,prob_C,prob_D\n")
        for i, (lab, sft, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if pv is None or np.any(np.isnan(pv)):
                row = [str(i), lab, str(sft), "", ""]
            else:
                row = [str(i), lab, str(sft), f"{pv[0]:.6f}", f"{pv[1]:.6f}"]
            f.write(",".join(row) + "\n")
    # Shift histogram
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.hist(shifts, bins=np.arange(-jitter_samp-0.5, jitter_samp+1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best shift per event)")
    ax.set_xlabel("Shift (samples)"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_shift_hist_tsxfmr_cd.png"), dpi=150)
    plt.close(fig)

    # Probability trace (max prob per event)
    maxp = []
    for pv in probs_out:
        if pv is None or np.any(np.isnan(pv)):
            maxp.append(np.nan)
        else:
            maxp.append(float(np.max(pv)))
    plt.figure(figsize=(7,2.4))
    plt.plot(maxp, lw=1)
    plt.title("Per-event max class probability")
    plt.xlabel("Event index"); plt.ylabel("max p")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mi_maxprob_tsxfmr_cd.png"), dpi=150)
    plt.close()

    return labels_out

# ---------------------------------
# === ACCURACY vs GT (optional)
# ---------------------------------
def accuracy_vs_gt(detected: List[str], gt_str: str) -> Optional[float]:
    gt = [t.strip() for t in gt_str.replace(",", " ").split() if t.strip()]
    if len(gt) == 0:
        print("[Accuracy vs GT] (skipped — no GT pasted)")
        return None
    n = min(len(gt), len(detected))
    if n == 0:
        print("[Accuracy vs GT] (skipped — empty sequences)")
        return None
    match = sum(1 for a,b in zip(detected[:n], gt[:n]) if a == b)
    acc = match / n
    print(f"[Accuracy vs GT] {acc*100:.2f}% (truncated to common length)")
    return acc

# ---------------------------------
# === MAIN ===
# ---------------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Build dataset from MI registrations C & D
    X, y, labs, sf, ch_names = _build_dataset(MI_SOURCES)   # labs=['C','D']

    # 2) Train Transformer on C/D
    model, meta = train_model(X, y, sf, ch_names)

    # 3) Detect sequence on MI target
    detected = detect_sequence(model, meta, MI_TARGET_FILE, labs, unk_thresh=UNK_THRESH)

    # 4) Optional: compare with provided GT string
    if GT_STRING.strip():
        accuracy_vs_gt(detected, GT_STRING)

    print("\n[Done] All outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()

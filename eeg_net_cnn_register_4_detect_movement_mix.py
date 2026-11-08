# -*- coding: utf-8 -*-
"""
GNet (Small CNN) EEG Classifier
- Train on four registration recordings (A,B,C,D) with sliding windows
- Leave-one-registration-out CV (GroupKFold with 4 groups)
- Detect sequence on movement-mix target using annotation-locked windows + jitter scan
- Save: confusion matrix, training curves, jitter histogram, probability trace, sequence (txt,csv)

Author: you + ChatGPT
"""

import os
import sys
import time
import math
import json
import random
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle as sk_shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

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

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_gnet_abcd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
RANDOM_STATE = 42
DISPLAY_PLOTS = False  # set True to popup figures in addition to saving

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)  # motor bands
NOTCH_HZ = None         # set to 50 or 60 if needed, else None

# Channels (we’ll try these first; else fall back to all common EEG channels)
PREFERRED_CHANSETS = [
    ["C3L", "CzL", "C4L", "FCzL"],
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"],  # last resort set
]

# Sliding windows for training on registrations
WIN_SEC  = 1.0    # window length
STEP_SEC = 0.25   # step

# Detection on target (annotation-locked)
EVENT_DESC_PREF = "Stimulus/S  1"  # annotations name in your files
TMIN_SEC   = 0.0    # relative to annotation onset
SEG_SEC    = 1.0    # window length for detection
JITTER_SAMP = 30    # +/- samples around event center to scan
UNK_THRESH  = 0.45  # if max softmax < thresh => UNK
SMOOTH_KERNEL = 5   # median filter on label indices (odd; 0/1 disables)

# CNN training
EPOCHS = 18
BATCH  = 128
LR     = 1e-3
WD     = 1e-4
VAL_FRAC = 0.1     # inside each training fold

LOG = logging.getLogger("GNet")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# === UTILS ===
# --------------------------
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _norm_names(names: List[str]) -> List[str]:
    return [n.strip().upper().replace(" ", "") for n in names]

def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick(picks="eeg")  # EEG only
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def _pick_common_channels(raws: List[mne.io.BaseRaw]) -> List[int]:
    """Pick the same ordered channel indices in every Raw."""
    # Try preferred sets first
    raws_upper = [ _norm_names(r.ch_names) for r in raws ]
    for wanted in PREFERRED_CHANSETS:
        w = _norm_names(wanted)
        ok = all(all(ch in ru for ch in w) for ru in raws_upper)
        if ok:
            # Use the ordering in the first raw
            idx_first = [raws_upper[0].index(ch) for ch in w]
            LOG.info("[Channels] Using preferred set: %s", [raws[0].ch_names[i] for i in idx_first])
            return idx_first

    # Otherwise, use intersection of EEG chs across all raws
    sets = [set(ru) for ru in raws_upper]
    inter = set.intersection(*sets)
    if len(inter) < 2:
        raise RuntimeError("No sufficient common EEG channels across files.")
    inter = list(sorted(inter))  # stable order
    idx = [raws_upper[0].index(ch) for ch in inter]
    LOG.info("[Channels] Using common intersection (%d): %s", len(idx), [raws[0].ch_names[i] for i in idx])
    return idx

def _sliding_windows(data: np.ndarray, win: int, step: int) -> np.ndarray:
    """data: (C,T) -> (N,C,win)"""
    C, T = data.shape
    if win > T:
        return np.zeros((0, C, 0), dtype=np.float32)
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = data[:, s:s+win]
        # per-window z-norm per channel
        mu = seg.mean(axis=-1, keepdims=True)
        sd = seg.std(axis=-1, keepdims=True)
        sd[sd < 1e-6] = 1e-6
        X[i] = (seg - mu) / sd
    return X

def build_training_set(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int], float]:
    """Return X:(N,C,T), y:(N,), groups:(N,), class_names, picks, sfreq"""
    LOG.info("[START] Load raws & pick common channels")
    raws = [ _load_raw(paths[k]) for k in sorted(paths.keys()) ]
    picks = _pick_common_channels(raws)
    sf = raws[0].info["sfreq"]
    win = int(round(WIN_SEC * sf))
    step = int(round(STEP_SEC * sf))

    X_all, y_all, g_all = [], [], []
    class_names = sorted(paths.keys())
    for gi, lab in enumerate(class_names):
        raw = raws[gi]
        data = raw.get_data(picks=picks)  # (C,T)
        X = _sliding_windows(data, win=win, step=step)
        y = np.full((X.shape[0],), gi, dtype=np.int64)
        g = np.full((X.shape[0],), gi, dtype=np.int64)

        LOG.info("  [%s] %s | picks=%s | sf=%.1f Hz | windows=%d",
                 lab, os.path.basename(paths[lab]),
                 [raw.ch_names[p] for p in picks], sf, X.shape[0])

        X_all.append(X); y_all.append(y); g_all.append(g)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    g_all = np.concatenate(g_all, axis=0)
    counts = {class_names[i]: int((y_all==i).sum()) for i in range(len(class_names))}
    LOG.info("[TRAIN] Total windows: %s, per-class: %s", X_all.shape, counts)
    LOG.info("[END]   Load raws & pick common channels")
    return X_all, y_all, g_all, class_names, picks, sf

def extract_events_from_annotations(raw: mne.io.BaseRaw, prefer_desc=EVENT_DESC_PREF):
    """Return (events, chosen_code, desc_used). Prefer annotations; fallback to stim if needed."""
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except Exception:
        events, event_id = np.empty((0,3), dtype=int), {}

    if events.size == 0 or len(event_id) == 0:
        # fallback to stim channels
        try:
            events = mne.find_events(raw, shortest_event=1, verbose="ERROR")
        except Exception as e:
            raise RuntimeError(f"No events or annotations found: {e}")
        if events.size == 0:
            raise RuntimeError("No events/annotations found in target file.")
        # choose most frequent code
        uniq, cnt = np.unique(events[:,2], return_counts=True)
        chosen = int(uniq[np.argmax(cnt)])
        return events, chosen, f"code {chosen}"

    if prefer_desc in event_id:
        chosen = event_id[prefer_desc]
        return events, int(chosen), prefer_desc
    # else: choose most frequent
    uniq, cnt = np.unique(events[:,2], return_counts=True)
    chosen = int(uniq[np.argmax(cnt)])
    rev = {v:k for k,v in event_id.items()}
    desc = rev.get(chosen, f"code {chosen}")
    return events, chosen, desc

# --------------------------
# === MODEL (GNet) ===
# --------------------------
class GNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        # (B, C, T)
        self.conv1 = nn.Conv1d(in_ch, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.25)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):  # x:(B,C,T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)   # (B,64)
        x = self.fc(x)                # (B,K)
        return x

def train_one(model, loader, opt, device):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
    return tot / max(1, n)

@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        tot += loss.item() * xb.size(0)
        n += xb.size(0)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(yb.cpu().numpy())
    if n == 0:
        return float("inf"), np.array([], int), np.array([], int)
    y_true = np.concatenate(y_true) if len(y_true) else np.array([], int)
    y_pred = np.concatenate(y_pred) if len(y_pred) else np.array([], int)
    return tot / n, y_true, y_pred

def arrays_to_loader(X, y, batch=BATCH, shuffle_flag=False):
    tensX = torch.from_numpy(X.astype(np.float32))
    tY = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(tensX, tY)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle_flag, num_workers=0, pin_memory=False)

def cross_validate_gnet(X: np.ndarray, y: np.ndarray, groups: np.ndarray, class_names: List[str]) -> Tuple[float, np.ndarray, np.ndarray]:
    """Leave-one-registration-out using GroupKFold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    y_true_all, y_pred_all = [], []
    accs = []
    fold = 0
    for tr_idx, te_idx in kf.split(X, y, groups):
        fold += 1
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        # small internal val split from training set
        n_val = max(1, int(round(len(Xtr) * VAL_FRAC)))
        n_tr  = len(Xtr) - n_val
        # torch dataset split
        full_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        tr_ds, va_ds = random_split(full_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(RANDOM_STATE+fold))
        tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False)
        te_loader = arrays_to_loader(Xte, yte, shuffle_flag=False)

        model = GNet(in_ch=X.shape[1], n_classes=len(class_names)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        best_va = float("inf")
        patience, bad = 6, 0

        for ep in range(1, EPOCHS+1):
            tr_loss = train_one(model, tr_loader, opt, device)
            va_loss, _, _ = eval_one(model, va_loader, device)
            if va_loss < best_va - 1e-4:
                best_va = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

        # load best
        model.load_state_dict(best_state)

        # test
        _, ytrue, ypred = eval_one(model, te_loader, device)
        acc = accuracy_score(ytrue, ypred) if len(ytrue) else 0.0
        accs.append(acc)
        y_true_all.append(ytrue)
        y_pred_all.append(ypred)
        LOG.info("[CV] Fold %d acc=%.3f (n=%d)", fold, acc, len(ytrue))

    y_true = np.concatenate(y_true_all) if len(y_true_all) else np.array([], int)
    y_pred = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([], int)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0

    LOG.info("\n[CV] Classification report (leave-one-registration-out):")
    if len(y_true):
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    else:
        print("No CV predictions.")

    # Confusion matrix
    if len(y_true):
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        fig, ax = plt.subplots(figsize=(6,5))
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(len(class_names))); ax.set_xticklabels(class_names)
        ax.set_yticks(np.arange(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (normalized)")
        for (i,j), v in np.ndenumerate(cmn):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cm_path = os.path.join(OUT_DIR, "cv_confusion_matrix_gnet_abcd.png")
        fig.tight_layout(); fig.savefig(cm_path, dpi=150)
        if DISPLAY_PLOTS: plt.show()
        plt.close(fig)
        LOG.info("[Saved] CV confusion matrix → %s", cm_path)

    LOG.info("[CV] GNet accuracy: %.3f (folds mean=%.3f ± %.3f)", acc, np.mean(accs) if accs else 0.0, np.std(accs) if accs else 0.0)
    return acc, y_true, y_pred

def fit_final_model(X: np.ndarray, y: np.ndarray, class_names: List[str]):
    """Train final GNet on all data (with small holdout for early stopping)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    n_val = max(1, int(round(len(X) * VAL_FRAC)))
    n_tr  = len(X) - n_val
    tr_ds, va_ds = random_split(full_ds, [n_tr, n_val], generator=torch.Generator().manual_seed(RANDOM_STATE+777))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    model = GNet(in_ch=X.shape[1], n_classes=len(class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    hist = {"tr_loss": [], "va_loss": []}
    best_va = float("inf")
    patience, bad = 8, 0
    best_state = None

    for ep in range(1, EPOCHS+1):
        tr_loss = train_one(model, tr_loader, opt, device)
        va_loss, _, _ = eval_one(model, va_loader, device)
        hist["tr_loss"].append(tr_loss)
        hist["va_loss"].append(va_loss)
        LOG.info("[Final] Epoch %02d/%02d | tr=%.4f va=%.4f", ep, EPOCHS, tr_loss, va_loss)
        if va_loss < best_va - 1e-4:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            LOG.info("[Final] Early stop.")
            break

    # restore best and save curve
    if best_state:
        model.load_state_dict(best_state)
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(hist["tr_loss"], label="train")
    ax.plot(hist["va_loss"], label="val")
    ax.set_title("Training curve (final model)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "final_training_curve_gnet.png")
    fig.savefig(out, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)
    LOG.info("[Saved] Final training curve → %s", out)
    return model

# --------------------------
# === DETECTION ===
# --------------------------
@torch.no_grad()
def detect_sequence(model: nn.Module,
                    target_path: str,
                    class_names: List[str],
                    picks: List[int],
                    tmin_sec: float,
                    seg_sec: float,
                    jitter_samp: int,
                    unk_thresh: float):
    raw = _load_raw(target_path)
    sf = raw.info["sfreq"]
    seg = int(round(seg_sec * sf))
    base = int(round(tmin_sec * sf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # reorder picks by name to match training order
    # make mapping from training names to current raw names
    # (we assume same picks indices apply—same channel names exist)
    data = raw.get_data(picks=picks)  # (C,T)
    C, T = data.shape

    # events
    events, code, desc = extract_events_from_annotations(raw, EVENT_DESC_PREF)
    ev = events[events[:,2] == code]
    N = len(ev)
    LOG.info("[Target] Using events '%s'; N=%d", desc, N)

    labels, maxps, shifts = [], [], []
    jitters = np.arange(-jitter_samp, jitter_samp+1, dtype=int)

    for i in range(N):
        center = ev[i,0] + base
        cand_logits = []
        cand_shift  = []
        for sh in jitters:
            beg = center + sh
            end = beg + seg
            if beg < 0 or end > T:  # skip out-of-bounds
                continue
            segx = data[:, beg:end]  # (C,seg)
            # z-norm per channel
            mu = segx.mean(axis=-1, keepdims=True); sd = segx.std(axis=-1, keepdims=True)
            sd[sd<1e-6] = 1e-6
            segx = (segx - mu) / sd
            xb = torch.from_numpy(segx.astype(np.float32)).unsqueeze(0).to(device)  # (1,C,seg)
            logit = model(xb)  # (1,K)
            cand_logits.append(logit.cpu().numpy()[0])
            cand_shift.append(int(sh))
        if not cand_logits:
            labels.append("UNK"); maxps.append(0.0); shifts.append(0); continue
        cand_logits = np.stack(cand_logits, axis=0)    # (M,K)
        proba = F.softmax(torch.from_numpy(cand_logits), dim=1).numpy()
        m_idx, m_cls = np.unravel_index(np.argmax(proba), proba.shape)
        m_p = float(proba[m_idx, m_cls])
        if m_p < unk_thresh:
            labels.append("UNK")
        else:
            labels.append(class_names[int(m_cls)])
        maxps.append(m_p)
        shifts.append(cand_shift[m_idx])

    # optional median smoothing (map to indices; keep -1 for UNK)
    idx_map = {c:i for i,c in enumerate(class_names)}
    idx_seq = np.array([ idx_map.get(l, -1) for l in labels ], dtype=int)
    if SMOOTH_KERNEL and SMOOTH_KERNEL >= 3 and SMOOTH_KERNEL % 2 == 1:
        from scipy.signal import medfilt
        sm = idx_seq.copy()
        mask = sm >= 0
        if np.any(mask):
            sm[mask] = medfilt(sm[mask], kernel_size=SMOOTH_KERNEL)
        labels = [ class_names[i] if i>=0 else "UNK" for i in sm ]

    # Save sequence text + csv
    seq_txt = os.path.join(OUT_DIR, "gnet_abcd_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels) + "\n")
    seq_csv = os.path.join(OUT_DIR, "gnet_abcd_movement_mix_sequence.csv")
    with open(seq_csv, "w", encoding="utf-8") as f:
        f.write("event_idx,label,shift,max_prob\n")
        for i,(l,s,p) in enumerate(zip(labels, shifts, maxps)):
            f.write(f"{i},{l},{s},{p:.6f}\n")

    # Figures
    # 1) Jitter histogram
    fig, ax = plt.subplots(figsize=(6,3.4))
    ax.hist(shifts, bins=np.arange(-jitter_samp-0.5, jitter_samp+1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best per event)"); ax.set_xlabel("Shift (samples)"); ax.set_ylabel("Count")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "target_shift_hist_gnet.png"); fig.savefig(out, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)

    # 2) Max probability per event
    fig, ax = plt.subplots(figsize=(8,3.4))
    ax.plot(maxps, lw=1, marker="o", ms=2)
    ax.axhline(UNK_THRESH, ls="--", color="k", alpha=0.6)
    ax.set_title("Per-event maximum posterior probability")
    ax.set_xlabel("Event #"); ax.set_ylabel("Max p")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "target_maxprob_gnet.png"); fig.savefig(out, dpi=150)
    if DISPLAY_PLOTS: plt.show()
    plt.close(fig)

    # Console pretty print
    LOG.info("\n=== Detected Sequence (GNet CNN, 4-class, annotation-locked+jitter) ===")
    wrap = []
    line = []
    for t in labels:
        line.append(t)
        if len(line) >= 80:
            wrap.append(" ".join(line)); line=[]
    if line: wrap.append(" ".join(line))
    print("\n".join(wrap))

    counts = {c: labels.count(c) for c in class_names}
    counts["UNK"] = labels.count("UNK")
    LOG.info("\n[Counts]")
    for k in class_names + ["UNK"]:
        LOG.info("  %s: %d", k, counts.get(k, 0))

    LOG.info("[Saved sequence] %s", seq_txt)
    LOG.info("[Saved table]    %s", seq_csv)
    return labels, counts

# --------------------------
# === MAIN ===
# --------------------------
def main():
    set_seed(RANDOM_STATE)
    # 1) Build training set
    X, y, groups, class_names, picks, sf = build_training_set(TEMPLATE_SOURCES)  # X:(N,C,T)

    # 2) Leave-one-registration-out CV
    _ = cross_validate_gnet(X, y, groups, class_names)

    # 3) Fit final model on all registrations
    model = fit_final_model(X, y, class_names)

    # 4) Detect on target
    _ = detect_sequence(
        model=model,
        target_path=TARGET_FILE,
        class_names=class_names,
        picks=picks,
        tmin_sec=TMIN_SEC,
        seg_sec=SEG_SEC,
        jitter_samp=JITTER_SAMP,
        unk_thresh=UNK_THRESH
    )

    LOG.info("\n[Done] All outputs → %s", OUT_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOG.exception(str(e))
        sys.exit(1)

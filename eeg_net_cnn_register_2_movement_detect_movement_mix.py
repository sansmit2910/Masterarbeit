# -*- coding: utf-8 -*-
"""
EEGNet (PyTorch) — A/B registration -> detect sequence on movement-mix
- Train on registrations A & B (sliding windows)
- Validate with stratified split, class-weighted loss, early stopping
- Detect on movement-mix with annotation-locked windows + jitter scan
- Mark UNK if max probability < threshold
- Save sequence (txt/csv), jitter histogram, and training curve

Requires: torch, mne, numpy, scipy, scikit-learn, matplotlib
"""

import os
import math
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import medfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Paths & Output
# --------------------------
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    # (C and D intentionally not used here)
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_eegnet_ab_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Config
# --------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Preproc
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)
NOTCH_HZ = None  # set e.g. 50 or 60 if you need it

# Sliding windows (registrations)
WIN_SEC = 1.0
STEP_SEC = 0.25

# Detection on target
EVENT_DESC_PREF = "Stimulus/S  1"
TMIN_SEC = 0.0
SEG_SEC = 1.0
JITTER_SAMP = 30
UNK_THRESH = 0.55
SMOOTH_KERNEL = 5  # median filter length (odd), applied to indices (A=0, B=1), UNK kept separate

# Channels (prefer motor set)
PREFERRED_CHANSETS = [
    ["C3L", "CzL", "C4L", "FCzL"],
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"],
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# I/O and preprocessing
# --------------------------
def load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def pick_center_channels(raw: mne.io.BaseRaw) -> List[int]:
    ch_upper = [c.upper() for c in raw.ch_names]
    for wanted in PREFERRED_CHANSETS:
        idx = []
        ok = True
        for w in wanted:
            try:
                idx.append(ch_upper.index(w.upper()))
            except ValueError:
                ok = False
                break
        if ok and len(idx):
            return idx
    # fallback to all EEG
    return mne.pick_types(raw.info, eeg=True)

def sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    if win <= 0 or step <= 0 or win > T:
        return np.zeros((0, C, 0))
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s + win]
    return X  # (N, C, win)

def zscore_per_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # x: (N, C, T)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)

# --------------------------
# EEGNet (PyTorch)
# --------------------------
class EEGNet(nn.Module):
    """
    Minimal EEGNet (Lawhern et al.) for (B, 1, C, T) -> 2 classes
    """
    def __init__(self, n_ch: int, n_samples: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, pool1: int = 4, pool2: int = 8, dropout: float = 0.5):
        super().__init__()
        self.n_ch = n_ch
        self.n_samples = n_samples

        # Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Convolution (spatial)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Separable Convolution (depthwise temporal + pointwise)
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, kernel_length),
                                         padding=(0, kernel_length // 2), groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # Final linear
        # compute flattened dimension
        with torch.no_grad():
            tmp = torch.zeros(1, 1, n_ch, n_samples)
            tmp = self.forward_features(tmp)
            flat = tmp.shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def forward_features(self, x):
        # x: (B, 1, C, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

# --------------------------
# Dataset
# --------------------------
class EEGWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, C, T), y: (N,)
        X = zscore_per_window(X).astype(np.float32)
        self.X = X
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Return (1, C, T) for EEGNet
        x = self.X[idx][np.newaxis, :, :]  # (1, C, T)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y)

# --------------------------
# Data building (A & B only)
# --------------------------
def build_train_ab(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], float]:
    labels = ["A", "B"]
    used = {k: paths[k] for k in labels}
    raws = {}
    sf = None
    for lab, p in used.items():
        raw = load_raw(p)
        raws[lab] = raw
        if sf is None:
            sf = raw.info["sfreq"]

    # unified picks across A and B
    picksA = pick_center_channels(raws["A"])
    picksB = pick_center_channels(raws["B"])
    # intersect channel names (robust)
    chanA = [raws["A"].ch_names[i] for i in picksA]
    chanB = [raws["B"].ch_names[i] for i in picksB]
    common = [ch for ch in chanA if ch in chanB]
    if len(common) < 3:  # guardrail
        common = [c for c in raws["A"].ch_names if c in raws["B"].ch_names and c.upper().startswith("C")]
    if len(common) == 0:
        raise RuntimeError("No common EEG channels between A and B.")

    # map common picks
    def picks_from_names(raw, names):
        up = [c.upper() for c in raw.ch_names]
        out = []
        for n in names:
            out.append(up.index(n.upper()))
        return out

    picksA = picks_from_names(raws["A"], common)
    picksB = picks_from_names(raws["B"], common)

    # windows
    XA = sliding_windows(raws["A"], picksA, WIN_SEC, STEP_SEC)
    XB = sliding_windows(raws["B"], picksB, WIN_SEC, STEP_SEC)

    print(f"[TRAIN] A windows={XA.shape[0]} | B windows={XB.shape[0]} | picks={len(common)} | sf={sf:.1f} Hz")

    # labels
    yA = np.zeros((XA.shape[0],), dtype=np.int64)
    yB = np.ones((XB.shape[0],), dtype=np.int64)

    X = np.concatenate([XA, XB], axis=0)
    y = np.concatenate([yA, yB], axis=0)

    return X, y, common, picksA, sf

# --------------------------
# Training utilities
# --------------------------
def train_eegnet_ab(X: np.ndarray, y: np.ndarray, ch_names: List[str],
                    batch_size: int = 128, lr: float = 1e-3, max_epochs: int = 40,
                    patience: int = 6, weight_decay: float = 1e-4):
    n_ch = X.shape[1]
    n_samp = X.shape[2]

    # stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    (tr_idx, va_idx) = next(sss.split(X, y))

    ds_tr = EEGWindowDataset(X[tr_idx], y[tr_idx])
    ds_va = EEGWindowDataset(X[va_idx], y[va_idx])

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # class weights for imbalance
    n0 = int((y[tr_idx] == 0).sum())
    n1 = int((y[tr_idx] == 1).sum())
    w0 = 0.5 * (n0 + n1) / max(n0, 1)
    w1 = 0.5 * (n0 + n1) / max(n1, 1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE)

    model = EEGNet(n_ch=n_ch, n_samples=n_samp, n_classes=2,
                   F1=8, D=2, F2=16, kernel_length=64, pool1=4, pool2=8, dropout=0.5).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_va = -np.inf
    best_state = None
    train_hist, val_hist = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE)  # (B,1,C,T)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
        tr_acc = correct / max(total, 1)
        tr_loss = loss_sum / max(total, 1)

        # val
        model.eval()
        v_correct, v_total, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = crit(logits, yb)
                v_loss_sum += float(loss.item()) * yb.size(0)
                pred = logits.argmax(dim=1)
                v_correct += int((pred == yb).sum().item())
                v_total += int(yb.size(0))
        va_acc = v_correct / max(v_total, 1)
        va_loss = v_loss_sum / max(v_total, 1)

        train_hist.append(tr_acc)
        val_hist.append(va_acc)

        print(f"[Train] Epoch {epoch:02d}/{max_epochs} | tr_acc={tr_acc:.4f} va_acc={va_acc:.4f}")

        # early stopping on val acc
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve = epoch - np.argmax(val_hist) - 1
        if no_improve >= patience:
            print("[Train] Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # plot training curve
    fig = plt.figure(figsize=(6, 3))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EEGNet A/B Training Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(OUT_DIR, "eegnet_ab_training_curve.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] Training curve → {curve_path}")

    # final val report
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb = xb.to(DEVICE)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print("\n[Validation] Classification report:")
    print(classification_report(y_true, y_pred, target_names=["A", "B"], digits=3))
    print("[Validation] Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"[Validation] Accuracy: {accuracy_score(y_true, y_pred):.3f}")

    return model

# --------------------------
# Events from annotations
# --------------------------
def extract_events_from_annotations(raw: mne.io.BaseRaw, prefer_desc: str = EVENT_DESC_PREF):
    # Always use annotations (no stim channel in your files)
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) == 0 or events.size == 0:
        raise RuntimeError("No events/annotations found.")
    if prefer_desc in event_id:
        chosen = event_id[prefer_desc]
        used_desc = prefer_desc
    else:
        # fallback: most frequent
        uniq, cnt = np.unique(events[:, 2], return_counts=True)
        chosen = uniq[np.argmax(cnt)]
        rev = {v: k for k, v in event_id.items()}
        used_desc = rev.get(chosen, f"code {int(chosen)}")
    return events, chosen, used_desc

# --------------------------
# Predict probabilities
# --------------------------
def predict_proba_eegnet(model: EEGNet, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    # X: (N, C, T) floats
    model.eval()
    ds = EEGWindowDataset(X, np.zeros((X.shape[0],), dtype=np.int64))  # dummy labels
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            logits = model(xb)          # (B,2)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
    return np.vstack(probs)  # (N,2)

# --------------------------
# Detection on movement-mix
# --------------------------
def detect_sequence_ab(model: EEGNet, target_path: str, ch_names_train: List[str],
                       tmin_sec: float, seg_sec: float, jitter_samp: int,
                       unk_thresh: float, smooth_kernel: int = SMOOTH_KERNEL):
    raw = load_raw(target_path)

    # map training channels onto target (by name)
    up = [c.upper() for c in raw.ch_names]
    picks = []
    for nm in ch_names_train:
        try:
            picks.append(up.index(nm.upper()))
        except ValueError:
            pass
    if len(picks) < len(ch_names_train):
        raise RuntimeError("Target file is missing some training channels.")
    data = raw.get_data(picks=picks)  # (C, T)
    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_offset = int(round(tmin_sec * sf))
    C, T = data.shape

    # events from annotations
    events, chosen_code, desc = extract_events_from_annotations(raw, EVENT_DESC_PREF)
    ev = events[events[:, 2] == chosen_code]
    n_events = len(ev)
    print(f"\n[Target] Using annotations '{desc}'; N={n_events}")

    labels_out: List[str] = []
    probs_out: List[np.ndarray] = []
    shifts: List[int] = []

    jitters = np.arange(-jitter_samp, jitter_samp + 1)

    for i in range(n_events):
        center = ev[i, 0] + base_offset

        segs = []
        svals = []
        for sh in jitters:
            beg = center + sh
            end = beg + seg_samp
            if beg < 0 or end > T:
                continue
            segs.append(data[:, beg:end])
            svals.append(sh)
        if not segs:
            labels_out.append("UNK")
            probs_out.append(np.array([np.nan, np.nan], dtype=np.float32))
            shifts.append(0)
            continue

        X_evt = np.stack(segs, axis=0)  # (M, C, T)
        P = predict_proba_eegnet(model, X_evt)  # (M,2)
        m_idx, m_cls = np.unravel_index(np.argmax(P), P.shape)
        best_prob = float(P[m_idx, m_cls])
        best_lab = "A" if m_cls == 0 else "B"

        if best_prob < unk_thresh:
            labels_out.append("UNK")
        else:
            labels_out.append(best_lab)
        probs_out.append(P[m_idx])
        shifts.append(svals[m_idx])

    # Optional smoothing (keep UNK intact)
    idx_map = {"A": 0, "B": 1}
    idx_seq = np.array([idx_map.get(l, -1) for l in labels_out], dtype=int)
    if smooth_kernel and smooth_kernel >= 3 and smooth_kernel % 2 == 1:
        sm = idx_seq.copy()
        mask = sm >= 0
        if np.any(mask):
            sm[mask] = medfilt(sm[mask], kernel_size=smooth_kernel)
        labels_out = ["A" if v == 0 else ("B" if v == 1 else "UNK") for v in sm]

    # Summary & saves
    counts = {"A": labels_out.count("A"), "B": labels_out.count("B"), "UNK": labels_out.count("UNK")}
    print("\n=== Detected Sequence (EEGNet A/B, annotation-locked + jitter) ===")
    line, wrapped = [], []
    for tok in labels_out:
        line.append(tok)
        if len(line) >= 80:
            wrapped.append(" ".join(line)); line = []
    if line: wrapped.append(" ".join(line))
    print("\n".join(wrapped))

    print("\n[Counts]")
    for k in ["A", "B", "UNK"]:
        print(f"  {k}: {counts[k]}")

    # Save sequence
    seq_txt = os.path.join(OUT_DIR, "eegnet_ab_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    print(f"[Saved sequence] {seq_txt}")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "eegnet_ab_movement_mix_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("event_idx,label,shift,prob_A,prob_B\n")
        for i, (lab, sh, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if np.any(np.isnan(pv)):
                f.write(f"{i},{lab},{sh},,\n")
            else:
                f.write(f"{i},{lab},{sh},{pv[0]:.6f},{pv[1]:.6f}\n")
    print(f"[Saved table]    {csv_path}")

    # Jitter histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(shifts, bins=np.arange(-jitter_samp - 0.5, jitter_samp + 1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best shift per event)")
    ax.set_xlabel("Shift (samples)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_hist = os.path.join(OUT_DIR, "target_shift_hist_eegnet_ab.png")
    fig.savefig(out_hist, dpi=150)
    plt.close(fig)
    print(f"[Saved] Target jitter shift histogram → {out_hist}")

    return labels_out, counts

# --------------------------
# Main
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[START] Build A/B training set …")
    X, y, ch_names, picksA, sf = build_train_ab(TEMPLATE_SOURCES)
    if X.shape[0] == 0:
        raise RuntimeError("No training windows built from A/B.")

    print(f"[TRAIN] Total windows: {X.shape} | per-class: {{'A': {int((y==0).sum())}, 'B': {int((y==1).sum())}}}")
    model = train_eegnet_ab(X, y, ch_names)

    print("\n[Target] Detecting on movement-mix …")
    _ = detect_sequence_ab(model, TARGET_FILE, ch_names, TMIN_SEC, SEG_SEC, JITTER_SAMP, UNK_THRESH)

    print("\n[Done] Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()

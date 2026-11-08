# -*- coding: utf-8 -*-
"""
GNet (Small 1D-CNN) for EEG:
- Train on four registrations (A,B,C,D) using sliding windows
- Stratified 4-fold CV with class-weighted loss
- Detect 4-class sequence in movement-mix (annotation-locked + jitter)
- Save confusion matrix, training curve, sequence files, and jitter histogram
"""

import os
import sys
import time
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (6.0, 32.0)        # slightly wider than 8-30
NOTCH_HZ = None               # set 50 or 60 if needed

# Windowing for training from registration recordings
WIN_SEC = 1.0                 # window length (seconds)
STEP_SEC = 0.25               # step between windows (seconds)

# Detection on target (movement-mix)
EVENT_DESC_PREF = "Stimulus/S  1"  # annotations label
TMIN_SEC = 0.10               # start 100 ms after trigger
SEG_SEC = 1.0                 # segment length for detection (seconds)
JITTER_SAMP = 75              # ±samples (≈ ±300 ms @ 250 Hz)
UNK_THRESH = 0.45             # if max class prob < this -> UNK

# Channels: prefer the motor set first; else fall back to intersection
PREFERRED_CHANSETS = [
    ["C3", "Cz", "C4", "FCz"],
    ["C3", "Cz", "C4"],  # fallback
]

# Training
BATCH_SIZE = 128
LR = 1e-3
EPOCHS_CV = 14
EPOCHS_FINAL = 18
PATIENCE = 5                   # early stopping patience (epochs)
USE_AUGMENT_A = False          # add tiny noise to A during training

LOG_PREFIX = time.strftime("%Y-%m-%d %H:%M:%S")

# --------------------------
# === Torch model ===
# --------------------------
class GNet(nn.Module):
    """Tiny 1D CNN over time with channel as input channels."""
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 16, kernel_size=15, padding=7, bias=False)
        self.bn1   = nn.BatchNorm1d(16)
        self.dw2   = nn.Conv1d(16, 16, kernel_size=7, padding=3, groups=16, bias=False)
        self.bn2   = nn.BatchNorm1d(16)
        self.pw3   = nn.Conv1d(16, 32, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.2)
        self.fc    = nn.Linear(32, n_classes)

    def forward(self, x):  # x: (N, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.dw2(x)))
        x = F.relu(self.bn3(self.pw3(x)))
        x = self.dropout(x)
        x = x.mean(dim=-1)            # global average pool over time  -> (N, 32)
        logits = self.fc(x)           # (N, K)
        return logits

# --------------------------
# === Helpers ===
# --------------------------
def logi(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | INFO | {msg}")

def loge(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {msg}")

def load_raw(path: str) -> mne.io.BaseRaw:
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

def pick_common_center_channels(raws: List[mne.io.BaseRaw]) -> List[int]:
    # Try preferred sets in order on the first raw
    base_names = [ch.upper() for ch in raws[0].ch_names]
    for wanted in PREFERRED_CHANSETS:
        idx = []
        ok = True
        for w in wanted:
            if w.upper() not in base_names:
                ok = False; break
        if ok:
            # ensure present in all raws
            present_all = True
            for raw in raws[1:]:
                rnames = [c.upper() for c in raw.ch_names]
                for w in wanted:
                    if w.upper() not in rnames:
                        present_all = False; break
                if not present_all: break
            if present_all:
                picks = [raws[0].ch_names.index(w) for w in wanted]
                logi(f"[Channels] Using preferred set: {wanted}")
                return picks
    # fall back to intersection of EEG names
    inter = set([c.upper() for c in raws[0].ch_names])
    for raw in raws[1:]:
        inter = inter.intersection(set([c.upper() for c in raw.ch_names]))
    inter = [c for c in raws[0].ch_names if c.upper() in inter]
    picks = [raws[0].ch_names.index(c) for c in inter]
    logi(f"[Channels] Using EEG intersection (n={len(picks)}).")
    return picks

def zscore_windows(X: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    # X: (N, C, T) -> per-window, per-channel z-norm over time
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

def sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    if win <= 0 or step <= 0 or win > T:
        raise ValueError("Bad window/step config.")
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s + win]
    return X

def build_dataset(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int], float]:
    logi("[START] Load raws & pick common channels")
    raws = []
    for label in sorted(paths.keys()):
        raws.append(load_raw(paths[label]))
    picks = pick_common_center_channels(raws)
    sf = raws[0].info["sfreq"]

    all_X, all_y, all_g = [], [], []
    classes = sorted(paths.keys())
    for li, label in enumerate(classes):
        raw = raws[li]
        Xw = sliding_windows(raw, picks, WIN_SEC, STEP_SEC)
        Xw = zscore_windows(Xw)
        y  = np.full((Xw.shape[0],), li, dtype=np.int64)
        g  = np.full((Xw.shape[0],), li, dtype=np.int64)
        ch_used = [raw.ch_names[p] for p in picks]
        logi(f"  [{label}] {os.path.basename(paths[label])} | picks={ch_used} | sf={sf:.1f} Hz | windows={Xw.shape[0]}")
        all_X.append(Xw); all_y.append(y); all_g.append(g)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    g = np.concatenate(all_g, axis=0)
    per_class = {classes[i]: int((y == i).sum()) for i in range(len(classes))}
    logi(f"[TRAIN] Total windows: {X.shape}, per-class: {per_class}")
    logi("[END]   Load raws & pick common channels")
    return X, y, g, classes, picks, sf

def class_weights_from_labels(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def maybe_augment_A(X: np.ndarray, y: np.ndarray, class_names: List[str], sigma: float = 0.02) -> np.ndarray:
    if not USE_AUGMENT_A:
        return X
    if "A" not in class_names:
        return X
    idx_a = class_names.index("A")
    mask = (y == idx_a)
    if not np.any(mask):
        return X
    X_aug = X.copy()
    noise = np.random.normal(0, sigma, size=X_aug.shape).astype(np.float32)
    X_aug[mask] += noise[mask]
    return X_aug

def train_one(model, dl, optimizer, device, w_t=None):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, weight=w_t)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.numel()
            loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

def eval_one(model, dl, device, w_t=None):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=w_t)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.numel()
            loss_sum += loss.item() * yb.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

def cross_validate_gnet(X, y, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(class_names)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    y_true_all, y_pred_all, fold_accs = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        # small valid split inside training for early stopping
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold)
        tr_in, va_in = next(skf2.split(Xtr, ytr))
        Xtr_in, ytr_in = Xtr[tr_in], ytr[tr_in]
        Xva_in, yva_in = Xtr[va_in], ytr[va_in]

        Xtr_in = maybe_augment_A(Xtr_in, ytr_in, class_names)

        w_t = class_weights_from_labels(ytr_in, n_classes).to(device)

        model = GNet(in_ch=X.shape[1], n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        tr_ds = TensorDataset(torch.tensor(Xtr_in, dtype=torch.float32),
                              torch.tensor(ytr_in, dtype=torch.long))
        va_ds = TensorDataset(torch.tensor(Xva_in, dtype=torch.float32),
                              torch.tensor(yva_in, dtype=torch.long))
        te_ds = TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                              torch.tensor(yte, dtype=torch.long))
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        best_va = -1.0
        no_imp = 0
        for ep in range(1, EPOCHS_CV + 1):
            tr_loss, tr_acc = train_one(model, tr_dl, optimizer, device, w_t=w_t)
            va_loss, va_acc = eval_one(model, va_dl, device, w_t=w_t)
            logi(f"[CV] Fold {fold:>1} Epoch {ep:02d} | tr={tr_acc:.4f} va={va_acc:.4f}")
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        # test
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in te_dl:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())
        logits_te = np.concatenate(all_logits, axis=0)
        yte_hat = logits_te.argmax(axis=1)

        acc = accuracy_score(yte, yte_hat)
        fold_accs.append(acc)
        logi(f"[CV] Fold {fold} acc={acc:.3f} (n={len(yte)})")

        y_true_all.append(yte)
        y_pred_all.append(yte_hat)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = accuracy_score(y_true, y_pred)
    logi("\n[CV] Classification report (Stratified 4-fold):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    logi(f"[Saved] CV confusion matrix → {save_confusion_matrix(cm, class_names, os.path.join(OUT_DIR, 'cv_confusion_matrix_gnet_abcd.png'))}")
    logi(f"[CV] GNet accuracy: {acc:.3f} (folds mean={np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f})")
    return acc

def save_confusion_matrix(cm: np.ndarray, class_names: List[str], path: str) -> str:
    cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), v in np.ndenumerate(cmn):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def final_train_gnet(X, y, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(class_names)

    # stratified split for validation curve
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(skf.split(X, y))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    Xtr = maybe_augment_A(Xtr, ytr, class_names)

    w_t = class_weights_from_labels(ytr, n_classes).to(device)

    model = GNet(in_ch=X.shape[1], n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(ytr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                          torch.tensor(yva, dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    best_va = -1.0
    no_imp = 0
    hist_tr, hist_va = [], []

    for ep in range(1, EPOCHS_FINAL + 1):
        tr_loss, tr_acc = train_one(model, tr_dl, optimizer, device, w_t=w_t)
        va_loss, va_acc = eval_one(model, va_dl, device, w_t=w_t)
        hist_tr.append(tr_acc); hist_va.append(va_acc)
        logi(f"[Final] Epoch {ep:02d} | tr={tr_acc:.4f} va={va_acc:.4f}")
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    # restore best
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # save training curve
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(hist_tr, label="train acc")
    ax.plot(hist_va, label="val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Final training curve (GNet)")
    ax.legend()
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, "final_training_curve_gnet.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    logi(f"[Saved] Final training curve → {curve_path}")

    return model

def extract_events(raw: mne.io.BaseRaw, prefer_desc: str = EVENT_DESC_PREF):
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id) == 0 or events.size == 0:
        # fallback try stim detection
        try:
            events = mne.find_events(raw, verbose="ERROR")
        except Exception:
            events = np.empty((0, 3), dtype=int)
        if events.size == 0:
            raise RuntimeError("No events/annotations found in target.")
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

def detect_sequence(model: GNet,
                    target_path: str,
                    labs: List[str],
                    picks: List[int],
                    tmin_sec: float,
                    seg_sec: float,
                    jitter_samp: int,
                    unk_thresh: float):
    raw = load_raw(target_path)
    sf = raw.info["sfreq"]
    seg_samp = int(round(seg_sec * sf))
    base_offset = int(round(tmin_sec * sf))

    events, chosen_code, desc = extract_events(raw, EVENT_DESC_PREF)
    ev = events[events[:, 2] == chosen_code]
    n_events = len(ev)
    logi(f"[Target] Using events '{desc}'; N={n_events}")

    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    labels_out: List[str] = []
    probs_out: List[np.ndarray] = []
    shifts: List[int] = []

    with torch.no_grad():
        for idx in range(n_events):
            center = ev[idx, 0] + base_offset
            segs = []
            shift_vals = []
            for s in range(-jitter_samp, jitter_samp + 1):
                start = center + s
                end = start + seg_samp
                if start < 0 or end > T:
                    continue
                seg = data[:, start:end][np.newaxis, ...]  # (1, C, seg)
                segs.append(seg)
                shift_vals.append(s)
            if len(segs) == 0:
                labels_out.append("UNK")
                probs_out.append(np.full((len(labs),), np.nan))
                shifts.append(0)
                continue

            X_evt = np.concatenate(segs, axis=0).astype(np.float32)  # (M, C, seg)
            X_evt = zscore_windows(X_evt)

            xb = torch.tensor(X_evt, dtype=torch.float32, device=device)
            logits = model(xb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()  # (M, K)

            # best (shift, class) by max prob
            m_idx, m_cls = np.unravel_index(np.nanargmax(proba), proba.shape)
            max_prob = proba[m_idx, m_cls]
            pred_lab = labs[m_cls]

            if max_prob < unk_thresh:
                labels_out.append("UNK")
            else:
                labels_out.append(pred_lab)
            probs_out.append(proba[m_idx])
            shifts.append(shift_vals[m_idx])

    # summary & saves
    counts = {lab: labels_out.count(lab) for lab in labs}
    counts["UNK"] = labels_out.count("UNK")

    print("\n=== Detected Sequence (GNet CNN, 4-class, annotation-locked+jitter) ===")
    # wrap 80 tokens per line
    line, wrapped = [], []
    for t in labels_out:
        line.append(t)
        if len(line) >= 80:
            wrapped.append(" ".join(line)); line = []
    if line: wrapped.append(" ".join(line))
    print("\n".join(wrapped))
    print("\n[Counts]")
    for k in labs + ["UNK"]:
        logi(f"  {k}: {counts.get(k, 0)}")

    # sequence files
    seq_txt = os.path.join(OUT_DIR, "gnet_abcd_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f:
        f.write(" ".join(labels_out) + "\n")
    seq_csv = os.path.join(OUT_DIR, "gnet_abcd_movement_mix_sequence.csv")
    header = "event_idx,label,best_shift," + ",".join([f"prob_{lab}" for lab in labs])
    with open(seq_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, (lab, sft, pv) in enumerate(zip(labels_out, shifts, probs_out)):
            if pv is None or np.any(np.isnan(pv)):
                row = [str(i), lab, str(sft)] + [""] * len(labs)
            else:
                row = [str(i), lab, str(sft)] + [f"{p:.6f}" for p in pv]
            f.write(",".join(row) + "\n")
    logi(f"[Saved] Sequence → {seq_txt}")
    logi(f"[Saved] Table    → {seq_csv}")

    # jitter histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(shifts, bins=np.arange(-jitter_samp - 0.5, jitter_samp + 1.5), edgecolor="black")
    ax.set_title("Target Jitter Shift (best shift per event)")
    ax.set_xlabel("Shift (samples)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "target_shift_hist_gnet_abcd.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    logi(f"[Saved] Target jitter shift histogram → {hist_path}")

    return labels_out, counts

# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    X, y, groups, class_names, picks, sf = build_dataset(TEMPLATE_SOURCES)

    # 1) Cross-validate with stratified folds + class-weighted CE
    _ = cross_validate_gnet(X, y, class_names)

    # 2) Train final model on all data (with a small internal val split)
    model = final_train_gnet(X, y, class_names)

    # 3) Detect sequence on target file (annotation-locked + jitter)
    _ = detect_sequence(
        model=model,
        target_path=TARGET_FILE,
        labs=class_names,
        picks=picks,
        tmin_sec=TMIN_SEC,
        seg_sec=SEG_SEC,
        jitter_samp=JITTER_SAMP,
        unk_thresh=UNK_THRESH
    )

    logi(f"[Done] All outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()

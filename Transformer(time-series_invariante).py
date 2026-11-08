# -*- coding: utf-8 -*-
"""
Time-Series Transformer for EEG (A,B,C,D) + Movement-Mix detection
- Train on 4 registrations with sliding windows (label = file)
- GroupKFold CV (leave-one-registration-out)
- Detect sequence on movement-mix target (annotation-locked + jitter scan)
- Save figures (confusion, training curve, shift hist, prob trace)
- Compute accuracy vs embedded ground-truth (strict and masked-UNK)

Requires: mne, numpy, torch, scikit-learn, matplotlib
"""

import os, sys, math, time, warnings, random
from typing import Dict, List, Tuple

import numpy as np
import mne
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# === USER PATHS ===
# --------------------------
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_transformer_abcd_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)     # mu/beta for motor tasks
NOTCH_HZ = None

# Channels to prefer (fallback=all EEG)
PREFERRED_CHANSETS = [
    ["C3L","CzL","C4L","FCzL"],   # if Laplacian exist
    ["C3","Cz","C4","FCz"],       # standard
]

# Sliding windows for training from continuous regs
WIN_SEC  = 1.0
STEP_SEC = 0.25

# Detection on target
EVENT_DESC_PREF = "Stimulus/S  1"
TMIN_SEC  = 0.0     # align at annotation onset
SEG_SEC   = 1.0
JITTER_SAMP = 50    # ± samples
UNK_THRESH  = 0.30  # unknown threshold on max prob

# Model / training
D_MODEL   = 64
N_HEAD    = 4
N_LAYERS  = 3
D_FF      = 128
DROPOUT   = 0.15
BATCH     = 128
EPOCHS    = 40
LR        = 1e-3
WEIGHT_DECAY = 1e-2
EARLY_PATIENCE = 8

# --------------------------
# === GROUND TRUTH (movement-mix) ===
# (from your message; used for end accuracy calc)
# --------------------------
GT_STR = r"""B A A A B B B A A A A A B B B B B B B B B A A A A A A B B B A A B B B B A A A A A A B B B B B B B B B A A A A A A A A A A A B B B B B B B B B B B B B B A A A A A A B B B A A A A A A A A A A A B B B B B B A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B A A A A A A A A A A B B B B B B B B B B A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B"""
GT_LIST = GT_STR.split()

# --------------------------
# === HELPERS ===
# --------------------------
def log(msg): print(time.strftime("%Y-%m-%d %H:%M:%S |"), msg)

def _load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path): raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ: raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ: raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def _pick_channels(raw: mne.io.BaseRaw) -> List[int]:
    names_upper = [n.upper() for n in raw.ch_names]
    for wanted in PREFERRED_CHANSETS:
        idx=[]
        for w in wanted:
            try: idx.append(names_upper.index(w.upper()))
            except ValueError: idx=[]; break
        if idx: return idx
    return mne.pick_types(raw.info, eeg=True)

def _sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=picks)  # (C, T)
    C, T = data.shape
    win  = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    starts = np.arange(0, T - win + 1, step, dtype=int)
    X = np.empty((len(starts), C, win), dtype=np.float32)
    for i,s in enumerate(starts): X[i] = data[:, s:s+win]
    return X

def _build_dataset(paths: Dict[str,str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float, List[str]]:
    log("| [START] Load raws & pick common channels")
    raws = {k:_load_raw(v) for k,v in paths.items()}
    # base picks from first file
    first = list(raws.keys())[0]
    base_picks = _pick_channels(raws[first])
    ch_names = [raws[first].ch_names[i] for i in base_picks]
    sf = raws[first].info["sfreq"]
    allX=[]; ally=[]; allg=[]
    labs = sorted(list(paths.keys()))
    for gi,lab in enumerate(labs):
        raw = raws[lab]
        # try to match channel names exactly
        try:
            idx = [raw.ch_names.index(n) for n in ch_names]
        except ValueError:
            idx = _pick_channels(raw)
            ch_names = [raw.ch_names[i] for i in idx]  # update names if needed
        X = _sliding_windows(raw, idx, WIN_SEC, STEP_SEC)
        y = np.full((X.shape[0],), gi, dtype=np.int64)
        g = np.full((X.shape[0],), gi, dtype=np.int64)
        log(f"|   [{lab}] {os.path.basename(paths[lab])} | picks={ [raw.ch_names[i] for i in idx] } | sf={sf:.1f} Hz | windows={X.shape[0]}")
        allX.append(X); ally.append(y); allg.append(g)
    X = np.concatenate(allX, 0)
    y = np.concatenate(ally, 0)
    g = np.concatenate(allg, 0)
    per_class = {labs[i]: int((y==i).sum()) for i in range(len(labs))}
    log(f"| [TRAIN] Total windows: {X.shape} (N,C,T), per-class: {per_class}")
    log("| [END]   Load raws & pick common channels")
    return X, y, g, labs, sf, ch_names

def _events_from_raw(raw: mne.io.BaseRaw, prefer_desc: str):
    # Prefer annotations
    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        if len(event_id):
            if prefer_desc in event_id: chosen = event_id[prefer_desc]
            else:
                u,c = np.unique(events[:,2], return_counts=True)
                chosen = u[np.argmax(c)]
            return events, chosen, {v:k for k,v in event_id.items()}.get(chosen, f"code {int(chosen)}")
    except Exception:
        pass
    # Fallback to stim
    try:
        events = mne.find_events(raw, verbose="ERROR", shortest_event=1)
        if events.size:
            u,c = np.unique(events[:,2], return_counts=True)
            chosen = u[np.argmax(c)]
            return events, chosen, f"code {int(chosen)}"
    except Exception:
        pass
    raise RuntimeError("No events/annotations found in target file.")

# --------------------------
# === DATASETS ===
# --------------------------
class WindowSet(Dataset):
    def __init__(self, X, y, mu=None, sd=None):
        if mu is None:
            mu = X.mean(axis=(0,2), keepdims=True)  # (1,C,1)
            sd = X.std(axis=(0,2), keepdims=True) + 1e-6
        self.X = (X - mu) / sd
        self.y = y
        self.mu = mu; self.sd = sd
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])  # (C,T)
        y = torch.tensor(int(self.y[i]), dtype=torch.long)
        return x, y

# --------------------------
# === MODEL ===
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)
    def forward(self, x):  # x: (B,L,d)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    Tokens = time-steps; features = channels (C)
    Embed(C -> d_model), add sinusoidal PE over time, TransformerEncoder, mean-pool over time, MLP head.
    """
    def __init__(self, n_channels: int, n_classes: int, seq_len: int,
                 d_model: int=64, nhead:int=4, num_layers:int=3, dim_ff:int=128, dropout:float=0.15):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_ff, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
    def forward(self, x):  # x: (B,C,T)
        x = x.transpose(1,2)        # (B,T,C)
        x = self.input_proj(x)      # (B,T,d)
        x = self.posenc(x)          # add PE
        x = self.encoder(x)         # (B,T,d)
        x = self.norm(x).mean(dim=1)  # mean-pool over time
        return self.head(x)         # (B,K)

# --------------------------
# === TRAIN / EVAL ===
# --------------------------
def train_one(model, loader, opt, crit):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_one(model, loader, crit):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ys=[]; yp=[]
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        ys.append(yb.cpu().numpy()); yp.append(pred.cpu().numpy())
    acc = correct/total if total>0 else 0.0
    ys = np.concatenate(ys) if ys else np.array([],dtype=int)
    yp = np.concatenate(yp) if yp else np.array([],dtype=int)
    return loss_sum/max(1,total), acc, ys, yp

def class_weights_from_y(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Safe class weights for CrossEntropyLoss:
    - length = n_classes
    - for absent classes -> weight=0 (ignored by the loss)
    - for present classes -> inverse-frequency (balanced) scaling
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = counts.sum() / (len(counts) * counts[present])
    # normalize to mean=1 (optional)
    m = weights[present].mean() if present.any() else 1.0
    if m > 0:
        weights = weights / m
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def _clone_state_dict(model: nn.Module) -> dict:
    "Deep-copy state dict tensors to CPU so they don't mutate during training."
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def run_cv(X, y, g, labs, sf):
    kf = GroupKFold(n_splits=len(np.unique(g)))
    cm_total = np.zeros((len(labs), len(labs)), dtype=int)
    fold_acc = []
    for fold,(tr,te) in enumerate(kf.split(X,y,g),1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        mu = Xtr.mean(axis=(0,2), keepdims=True); sd = Xtr.std(axis=(0,2), keepdims=True)+1e-6
        dtr = WindowSet(Xtr, ytr, mu, sd)
        dte = WindowSet(Xte, yte, mu, sd)
        trld = DataLoader(dtr, batch_size=BATCH, shuffle=True, drop_last=False)
        valld= DataLoader(dte, batch_size=BATCH, shuffle=False)

        model = TimeSeriesTransformer(n_channels=X.shape[1], n_classes=len(labs), seq_len=X.shape[2],
                                      d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, dim_ff=D_FF, dropout=DROPOUT).to(DEVICE)
        w = class_weights_from_y(ytr, len(labs))
        crit = nn.CrossEntropyLoss(weight=w)
        opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # ---- FIX: initialize a valid best_state BEFORE training ----
        best_val = -1.0
        best_state = _clone_state_dict(model)

        patience = EARLY_PATIENCE
        for ep in range(1, EPOCHS+1):
            tl, ta = train_one(model, trld, opt, crit)
            vl, va, ys, yp = eval_one(model, valld, crit)
            if va > best_val + 1e-12:
                best_val = va
                best_state = _clone_state_dict(model)   # deep clone
                patience = EARLY_PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    break

        model.load_state_dict(best_state)   # always valid now
        _, va, ys, yp = eval_one(model, valld, crit)
        cm = confusion_matrix(ys, yp, labels=np.arange(len(labs)))
        cm_total += cm
        fold_acc.append(va)
        log(f"[CV] Fold {fold} acc={va:.3f} (n={len(ys)})")
    acc = np.trace(cm_total)/np.sum(cm_total) if cm_total.sum()>0 else 0.0
    log("\n[CV] Confusion matrix (rows=true, cols=pred):")
    print(cm_total)
    log(f"[CV] Mean acc={np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f} | overall acc={acc:.3f}")
    # Save normalized CM
    fig, ax = plt.subplots(figsize=(6,5))
    cmn = cm_total / np.maximum(cm_total.sum(axis=1,keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs); ax.set_yticklabels(labs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Transformer CV (LOO) — row-normalized CM")
    for (i,j),v in np.ndenumerate(cmn): ax.text(j,i,f"{v:.2f}",ha="center",va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"cv_confusion_matrix_transformer_abcd.png"), dpi=150)
    plt.close(fig)

def train_final(X, y, labs) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    # simple stratified split by tail 10% per class for val
    idxs = np.arange(len(y))
    val_mask = np.zeros_like(idxs, dtype=bool)
    for cls in np.unique(y):
        cls_idx = idxs[y==cls]
        k = max(1, int(0.1*len(cls_idx)))
        val_mask[cls_idx[-k:]] = True
    tr = idxs[~val_mask]; va = idxs[val_mask]

    Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
    mu = Xtr.mean(axis=(0,2), keepdims=True); sd = Xtr.std(axis=(0,2), keepdims=True)+1e-6
    dtr = WindowSet(Xtr, ytr, mu, sd); dva = WindowSet(Xva, yva, mu, sd)
    trld = DataLoader(dtr, batch_size=BATCH, shuffle=True)
    valld= DataLoader(dva, batch_size=BATCH, shuffle=False)

    model = TimeSeriesTransformer(n_channels=X.shape[1], n_classes=len(labs), seq_len=X.shape[2],
                                  d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS, dim_ff=D_FF, dropout=DROPOUT).to(DEVICE)
    w = class_weights_from_y(ytr, len(labs))
    crit = nn.CrossEntropyLoss(weight=w)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_va=-1.0
    best_state=_clone_state_dict(model)   # ---- initialize a safe best ----
    patience=EARLY_PATIENCE
    tr_hist=[]; va_hist=[]
    for ep in range(1, EPOCHS+1):
        tl, ta = train_one(model, trld, opt, crit)
        vl, va, ys, yp = eval_one(model, valld, crit)
        tr_hist.append(ta); va_hist.append(va)
        print(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={ta:.4f} va={va:.4f}")
        if va>best_va + 1e-12:
            best_va=va; best_state=_clone_state_dict(model); patience=EARLY_PATIENCE
        else:
            patience-=1
            if patience<=0: break

    model.load_state_dict(best_state)

    # save curve
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(tr_hist, label="train acc")
    ax.plot(va_hist, label="val acc")
    ax.set_title("Final training curve (Transformer)")
    ax.set_xlabel("epoch"); ax.set_ylabel("acc")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,"final_training_curve_transformer.png"), dpi=150)
    plt.close(fig)
    return model, mu, sd

# --------------------------
# === DETECTION ===
# --------------------------
@torch.no_grad()
def detect_sequence(model, mu, sd, labs: List[str], target_path: str,
                    tmin_sec: float, seg_sec: float, jitter_samp: int, unk_thresh: float,
                    train_ch_names: List[str]):
    raw = _load_raw(target_path)
    # match training channel names & order exactly when possible
    picks = []
    for n in train_ch_names:
        if n in raw.ch_names:
            picks.append(raw.ch_names.index(n))
    if len(picks) != len(train_ch_names):
        # fallback: use preferred picks (keeps channel count consistent with mu)
        picks = _pick_channels(raw)
        picks = picks[:mu.shape[1]]
    sf = raw.info["sfreq"]
    seg = int(round(seg_sec*sf))
    base = int(round(tmin_sec*sf))

    events, chosen, label_desc = _events_from_raw(raw, EVENT_DESC_PREF)
    ev = events[events[:,2] == chosen]
    log(f"[Target] Using events '{label_desc}'; N={len(ev)}")

    data = raw.get_data(picks=picks)  # (C,T)
    C, T = data.shape

    labels_out=[]; probs_out=[]; best_shifts=[]; maxprobs=[]
    for k in range(len(ev)):
        center = ev[k,0] + base
        segs = []; shift_vals=[]
        for s in range(-jitter_samp, jitter_samp+1):
            beg = center + s
            end = beg + seg
            if beg<0 or end>T: continue
            segs.append(data[:, beg:end]); shift_vals.append(s)
        if not segs:
            labels_out.append("UNK"); probs_out.append(None); best_shifts.append(0); maxprobs.append(0.0)
            continue
        Xevt = np.stack(segs,0).astype(np.float32)       # (M,C,seg)
        Xevt = (Xevt - mu) / sd                          # normalize like train
        xb = torch.from_numpy(Xevt).to(DEVICE)           # (M,C,T)
        logits = model(xb)                               # (M,K)
        prob = F.softmax(logits, dim=1).cpu().numpy()    # (M,K)
        m_idx, m_cls = np.unravel_index(np.argmax(prob), prob.shape)
        mp = float(prob[m_idx, m_cls]); pred_lab = labs[m_cls]
        lab = pred_lab if mp >= unk_thresh else "UNK"
        labels_out.append(lab)
        probs_out.append(prob[m_idx])
        best_shifts.append(shift_vals[m_idx]); maxprobs.append(mp)

    # save seq
    seq_txt = os.path.join(OUT_DIR, "transformer_abcd_movement_mix_sequence.txt")
    with open(seq_txt,"w",encoding="utf-8") as f: f.write(" ".join(labels_out)+"\n")
    # save CSV
    csv = os.path.join(OUT_DIR, "transformer_abcd_movement_mix_sequence.csv")
    with open(csv,"w",encoding="utf-8") as f:
        f.write("event_idx,label,best_shift," + ",".join([f"prob_{l}" for l in labs]) + "\n")
        for i,(lab,sh,pv) in enumerate(zip(labels_out,best_shifts,probs_out)):
            if pv is None: f.write(f"{i},{lab},{sh}," + ",".join([""]*len(labs)) + "\n")
            else: f.write(f"{i},{lab},{sh}," + ",".join(f"{x:.6f}" for x in pv) + "\n")

    # visuals
    fig,ax = plt.subplots(figsize=(6,3.2))
    ax.hist(best_shifts, bins=np.arange(-jitter_samp-0.5, jitter_samp+1.5))
    ax.set_title("Target jitter shift (best per event)")
    ax.set_xlabel("samples"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"target_shift_hist_transformer.png"), dpi=150); plt.close(fig)

    fig,ax = plt.subplots(figsize=(8,3.2))
    ax.plot(maxprobs, marker="o", lw=1, ms=2)
    ax.axhline(UNK_THRESH, ls="--", color="red", label="UNK τ")
    ax.set_title("Per-event max posterior (best shift)")
    ax.set_xlabel("event #"); ax.set_ylabel("max p"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"movement_mix_maxprob_transformer.png"), dpi=150); plt.close(fig)

    # print nicely
    print("\n=== Detected Sequence (Transformer, 4-class, annotation-locked+jitter) ===")
    line=[]
    for t in labels_out:
        line.append(t)
        if len(line)>=80: print(" ".join(line)); line=[]
    if line: print(" ".join(line))
    print("\n[Counts]")
    from collections import Counter
    cnt = Counter(labels_out)
    for k in labs + ["UNK"]:
        print(f"  {k}: {cnt.get(k,0)}")

    return labels_out

# --------------------------
# === ACCURACY vs GT ===
# --------------------------
def accuracy_vs_gt(pred_list: List[str], gt_list: List[str]):
    n = min(len(pred_list), len(gt_list))
    pr = pred_list[:n]
    gt = gt_list[:n]
    strict = 100.0 * sum(p==g for p,g in zip(pr,gt)) / max(1,n)
    # ignore UNK
    nz = [(p,g) for p,g in zip(pr,gt) if p!="UNK"]
    masked = 100.0 * sum(int(p==g) for p,g in nz) / max(1,len(nz))
    log(f"[Accuracy vs GT] Strict={strict:.2f}% | Masked(ignore UNK)={masked:.2f}% (n={n}, non-UNK={len(nz)})")
    from collections import Counter
    print("GT counts:", Counter(gt))
    print("Pred counts:", Counter(pr))
    with open(os.path.join(OUT_DIR,"accuracy_vs_gt.txt"),"w",encoding="utf-8") as f:
        f.write(f"Compared n={n}\nStrict={strict:.2f}%\nMasked(ignore UNK)={masked:.2f}% (non-UNK={len(nz)})\n")
    return strict, masked

# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Build dataset
    X, y, g, labs, sf, ch_names = _build_dataset(TEMPLATE_SOURCES)  # X:(N,C,T)

    # 2) CV (LOO)
    run_cv(X, y, g, labs, sf)

    # 3) Train final model (with small internal val for early stopping & curve)
    model, mu, sd = train_final(X, y, labs)

    # 4) Detect sequence on target (annotation-locked + jitter) with exact channel name order
    pred_seq = detect_sequence(model, mu, sd, labs, TARGET_FILE, TMIN_SEC, SEG_SEC, JITTER_SAMP, UNK_THRESH, ch_names)

    # 5) Accuracy vs GT (embedded)
    accuracy_vs_gt(pred_seq, GT_LIST)

    print("\n[Done] Outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()

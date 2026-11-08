# -*- coding: utf-8 -*-
"""
EEGNet (MI: C/D) — Train on two MI recordings (C & D), detect sequence on MI target.
- Annotation-locked windows to "Stimulus/S  1" (one per event).
- Jitter augmentation at train; jitter scan at detect.
- Target μ/σ normalization; temperature scaling; half-shift logit calibration (logit_B - logit_A analogue → D vs C).
- Optional class-prior nudging and simple Viterbi smoothing.
- Saves sequence (.txt/.csv), probability trace, shift histogram, and prints accuracy vs GT (if provided).

IMPORTANT FIXES (carried over):
- Proper Conv2d input shape (B,1,C,T) via to_tensor() -> (1,C,T).
- Stable training (balanced sampler), careful threshold selection on validation.
"""

import os, time, math, random
from typing import List, Tuple, Dict
import numpy as np
import mne
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

# ----------------------
# PATHS (edit these if needed)
# ----------------------
TEMPLATE_SOURCES = {
    # MI training files
    "C": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-1_raw.fif",
    "D": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-2_raw.fif",
}
# MI target (mix) file
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-imagine-move-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eegnet_mi_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# CONFIG
# ----------------------
# EEG preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)       # try (10, 26) if needed
NOTCH_HZ = None              # 50 or 60 to enable
PREFERRED_CH_NAMES = ["C3", "Cz", "C4", "FCz"]  # fallback to common EEG if missing

# Event/epoch config
EVENT_DESC = "Stimulus/S  1"
SEG_SEC = 1.0                # window length
BASELINE_SEC = 0.2           # baseline anchoring before event (normalization only)
JITTER_SAMP = 50             # ± jitter (samples) at detection
TRAIN_JITTER = 10            # ± jitter for augmentation during training
AUG_PER_EVENT = 2            # randomized jittered copies per event (0=no aug)

# Model/training
SEED = 42
BATCH = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
DEVICE = "cpu"  # change to "cuda" if available

# Calibration & thresholding (validated on held-out)
TEMP_GRID = [0.6, 0.8, 1.0]
TAU_GRID  = [0.30, 0.40, 0.50]
MARGIN_GRID = [0.00, 0.05, 0.10]

# Class prior (target) — for MI often close to balanced
USE_CLASS_PRIOR = True
TARGET_PRIOR = np.array([0.50, 0.50], dtype=np.float64)  # [C, D]

# Smoothing (Viterbi)
USE_VITERBI = True
HMM_SELF = 0.92    # stickiness (higher → fewer flips)
HMM_UNK  = 0.02    # prob of UNK when below thresholds

# Plot behavior
DISPLAY = False  # set True to plt.show()

# Optional ground truth for MI target (paste if you have it; else leave "").
# Example (shortened style): "C D C D C D ..."  — accuracy computed only if non-empty.
MI_GT_SEQ_STR = ""

LOG_PREFIX = time.strftime("%Y-%m-%d %H:%M:%S") + " |"

# ----------------------
# UTILS
# ----------------------
def log(msg: str): print(f"{LOG_PREFIX} {msg}")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path): raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH_HZ:
        raw.notch_filter(NOTCH_HZ, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def pick_by_name(raws: List[mne.io.BaseRaw], names: List[str]) -> List[str]:
    names_up = [n.upper() for n in names]
    for r in raws:
        r_up = [c.upper() for c in r.ch_names]
        if not all(n in r_up for n in names_up):
            # fallback: intersection of EEG channels across files
            inter = set(raws[0].ch_names)
            for rr in raws[1:]:
                inter &= set(rr.ch_names)
            common = []
            for c in sorted(inter):
                try:
                    if raws[0].get_channel_types(picks=[c])[0] == "eeg":
                        common.append(c)
                except Exception:
                    pass
            return common[:min(8, len(common))]  # cap to 8 channels to keep model small
    return names

def events_from_ann(raw: mne.io.BaseRaw, desc=EVENT_DESC) -> np.ndarray:
    ev, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(ev_id) == 0 or ev.size == 0:
        raise RuntimeError("No annotations found; please ensure events exist.")
    if desc in ev_id:
        code = ev_id[desc]
        return ev[ev[:, 2] == code]
    uniq, cnt = np.unique(ev[:, 2], return_counts=True)
    code = uniq[np.argmax(cnt)]
    return ev[ev[:, 2] == code]

def extract_event_windows(raw: mne.io.BaseRaw, pick_names: List[str], jitter_samp: int,
                          aug_per_event: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_dummy, groups) for one file. y is filled by caller; groups group by event id."""
    sf = raw.info["sfreq"]
    seg = int(round(SEG_SEC * sf))
    base = int(round(BASELINE_SEC * sf))
    picks = [raw.ch_names.index(nm) for nm in pick_names]
    data = raw.get_data(picks=picks)  # (C,T)
    C, T = data.shape
    ev = events_from_ann(raw, EVENT_DESC)
    X_list, groups = [], []
    gid = 0
    for e in ev:
        center = int(e[0])
        # zero-shift (always)
        s0 = center - base
        e0 = s0 + seg
        if s0 >= 0 and e0 <= T:
            X_list.append(data[:, s0:e0])
            groups.append(gid)
        # jitter aug
        for _ in range(aug_per_event):
            sh = np.random.randint(-jitter_samp, jitter_samp + 1)
            s = center - base + sh
            e1 = s + seg
            if s >= 0 and e1 <= T:
                X_list.append(data[:, s:e1])
                groups.append(gid)
        gid += 1
    X = np.stack(X_list, axis=0) if len(X_list) else np.zeros((0, C, seg))
    return X, np.zeros((len(X_list),), dtype=np.int64), np.array(groups, dtype=np.int64)

def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

# ----------------------
# EEGNet (tiny, standard)
# ----------------------
class EEGNetSmall(nn.Module):
    def __init__(self, C: int, T: int, n_classes=2, dropout=0.25, F1=8, D=2, F2=None, kernel_len=64):
        super().__init__()
        if F2 is None: F2 = F1 * D
        # 1) Temporal conv
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_len), padding=(0, kernel_len//2), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)
        # 2) Depthwise spatial conv across channels
        self.dw    = nn.Conv2d(F1, F1*D, kernel_size=(C, 1), groups=F1, bias=False)
        self.bn2   = nn.BatchNorm2d(F1*D)
        self.act1  = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)
        # 3) Separable conv: depthwise temporal then pointwise
        self.sep_dw  = nn.Conv2d(F1*D, F1*D, kernel_size=(1, 16), padding=(0, 8), groups=F1*D, bias=False)
        self.sep_pw  = nn.Conv2d(F1*D, F2, kernel_size=(1, 1), bias=False)
        self.bn3     = nn.BatchNorm2d(F2)
        self.act2    = nn.ELU()
        self.pool2   = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2   = nn.Dropout(dropout)
        # flatten dim
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self.conv1(x); x = self.bn1(x)
            x = self.dw(x);    x = self.bn2(x); x = self.act1(x); x = self.pool1(x); x = self.drop1(x)
            x = self.sep_dw(x); x = self.sep_pw(x); x = self.bn3(x); x = self.act2(x); x = self.pool2(x); x = self.drop2(x)
            flat = x.shape[1]*x.shape[2]*x.shape[3]
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):  # x: (B,1,C,T)
        x = self.conv1(x); x = self.bn1(x)
        x = self.dw(x);    x = self.bn2(x); x = self.act1(x); x = self.pool1(x); x = self.drop1(x)
        x = self.sep_dw(x); x = self.sep_pw(x); x = self.bn3(x); x = self.act2(x); x = self.pool2(x); x = self.drop2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def to_tensor(sample: np.ndarray) -> torch.Tensor:
    """(C,T) -> (1,C,T) so batch becomes (B,1,C,T)."""
    if sample.ndim != 2:
        raise ValueError(f"Expected (C,T), got {sample.shape}")
    x = torch.from_numpy(sample.astype(np.float32))
    return x.unsqueeze(0)

# ----------------------
# Data & training
# ----------------------
class NPDataset(Dataset):
    def __init__(self, X, y): self.X=X; self.y=y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return to_tensor(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

def build_trainset_cd(paths: Dict[str,str], pick_names: List[str]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,float,List[str]]:
    raws = [load_raw(paths["C"]), load_raw(paths["D"])]
    sf = raws[0].info["sfreq"]
    pick_names = pick_by_name(raws, pick_names)
    log(f"[Channels] Using preferred set (by name): {pick_names}")

    # C
    Xc, _, gc = extract_event_windows(raws[0], pick_names, TRAIN_JITTER, AUG_PER_EVENT)
    yc = np.zeros((len(Xc),), dtype=np.int64)  # 0 -> C
    log(f"  [C] {os.path.basename(paths['C'])} | picks={pick_names} | sf={sf:.1f} Hz | windows={len(Xc)}")

    # D
    Xd, _, gd = extract_event_windows(raws[1], pick_names, TRAIN_JITTER, AUG_PER_EVENT)
    yd = np.ones((len(Xd),), dtype=np.int64)   # 1 -> D
    log(f"  [D] {os.path.basename(paths['D'])} | picks={pick_names} | sf={sf:.1f} Hz | windows={len(Xd)}")

    X_all = np.concatenate([Xc, Xd], axis=0)
    y_all = np.concatenate([yc, yd], axis=0)
    g_all = np.concatenate([gc + 100000, gd + 200000], axis=0)

    log(f"[TRAIN] Total windows: {X_all.shape} (N,C,T), per-class: "
        f"{{'C': {(y_all==0).sum()}, 'D': {(y_all==1).sum()}}}")
    return X_all, y_all, g_all, sf, pick_names

def train_model(X:np.ndarray,y:np.ndarray,groups:np.ndarray,sf:float)->Tuple[nn.Module,Dict]:
    set_seed()
    N,C,T = X.shape
    mu_tr, sd_tr = zscore_fit(X)
    Xn = zscore_apply(X, mu_tr, sd_tr)

    # grouped split (events)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    (tr_idx, va_idx) = list(gss.split(Xn, y, groups))[0]
    Xtr, ytr = Xn[tr_idx], y[tr_idx]
    Xva, yva = Xn[va_idx], y[va_idx]

    # balanced sampler
    class_count = np.array([ (ytr==0).sum(), (ytr==1).sum() ], dtype=np.float32)
    weights = 1.0 / np.clip(class_count, 1, None)
    sample_w = np.take(weights, ytr)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(ytr), replacement=True)

    ds_tr = NPDataset(Xtr, ytr)
    ds_va = NPDataset(Xva, yva)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, sampler=sampler, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, drop_last=False)

    model = EEGNetSmall(C, T, n_classes=2, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best = {"acc":-1, "state":None}

    for ep in range(1, EPOCHS+1):
        model.train(); tr_correct=0; tr_total=0
        for xb,yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)   # xb: (B,1,C,T)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
            tr_correct += int((logits.argmax(1)==yb).sum().item())
            tr_total   += len(xb)
        # val
        model.eval(); correct=0; n_va=0
        with torch.no_grad():
            for xb,yb in dl_va:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred==yb).sum().item()); n_va += len(xb)
        tr_acc = tr_correct/max(1,tr_total)
        va_acc = correct/max(1,n_va)
        log(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={tr_acc:.4f} va={va_acc:.4f}")
        if va_acc >= best["acc"]:
            best["acc"]=va_acc
            best["state"]=model.state_dict()
        if best["acc"]>=0.999 and ep>=10: break

    model.load_state_dict(best["state"])
    # final val metrics
    model.eval()
    with torch.no_grad():
        logits = []
        ys = []
        for xb,yb in dl_va:
            xb=xb.to(DEVICE)
            lo = model(xb).cpu().numpy()
            logits.append(lo); ys.append(yb.numpy())
        logits = np.concatenate(logits,0)
        ys = np.concatenate(ys,0)
        probs = softmax_temp_np(logits, T=1.0)
        ypred = probs.argmax(1)
        rep = classification_report(ys, ypred, target_names=["C","D"], digits=3)
        cm  = confusion_matrix(ys, ypred, labels=[0,1])
        acc = accuracy_score(ys, ypred)
        log("[Validation] Classification report:\n"+rep)
        log("[Validation] Confusion matrix:\n"+str(cm))
        log(f"[Validation] Accuracy: {acc:.3f}")

    # choose temperature, tau, margin on val for balanced accuracy
    T_best, tau_best, m_best, bacc_best = 1.0, 0.5, 0.0, -1
    for T in TEMP_GRID:
        pr = softmax_temp_np(logits, T=T)
        for tau in TAU_GRID:
            for m in MARGIN_GRID:
                yhat = probs_to_labels(pr, tau=tau, margin=m)
                mask = yhat!=2
                if mask.sum()==0: continue
                acc_bal = balanced_acc(ys[mask], yhat[mask])
                if acc_bal > bacc_best:
                    T_best, tau_best, m_best, bacc_best = T, tau, m, acc_bal
    log(f"[Threshold] Best temperature T={T_best:.2f}, τ={tau_best:.2f}, margin={m_best:.2f} (val balanced acc={bacc_best:.3f})")

    return model, {"mu_tr":mu_tr, "sd_tr":sd_tr, "T":T_best, "tau":tau_best, "margin":m_best}

# ----------------------
# Softmax / helpers
# ----------------------
def softmax_temp_np(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/np.clip(e.sum(1,keepdims=True),1e-9,None)

def balanced_acc(y_true, y_pred):
    tp0 = ((y_true==0)&(y_pred==0)).sum(); fn0=((y_true==0)&(y_pred!=0)).sum()
    tp1 = ((y_true==1)&(y_pred==1)).sum(); fn1=((y_true==1)&(y_pred!=1)).sum()
    r0 = tp0/max(1,tp0+fn0); r1 = tp1/max(1,tp1+fn1)
    return 0.5*(r0+r1)

def probs_to_labels(p: np.ndarray, tau=0.5, margin=0.0) -> np.ndarray:
    top = p.argmax(1)
    mx  = p.max(1)
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    sep = sorted_p[:,0]-sorted_p[:,1]
    unk = (mx<tau) | (sep<margin)
    out = top.copy()
    out[unk] = 2  # UNK
    return out

# ----------------------
# Calibration (fixed half-shift on target)
# ----------------------
def predict_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(DEVICE)  # (N,1,C,T)
        lo = model(xb).cpu().numpy()
    return lo

def fit_target_logit_shift(model: nn.Module, raw_tgt: mne.io.BaseRaw, pick_names: List[str],
                           mu_tgt: np.ndarray, sd_tgt: np.ndarray, max_iter=200):
    """
    Estimate scalar Δ so that mean sigmoid((logit_D - logit_C) + Δ) ≈ TARGET_PRIOR[D].
    Returns: (delta, mean_before, mean_after, N0)
    """
    sf = raw_tgt.info["sfreq"]
    seg = int(round(SEG_SEC*sf)); base=int(round(BASELINE_SEC*sf))
    ev = events_from_ann(raw_tgt, EVENT_DESC)
    picks=[raw_tgt.ch_names.index(n) for n in pick_names]
    data=raw_tgt.get_data(picks=picks)
    C,T=data.shape
    X0=[]
    for e in ev:
        s=e[0]-base; e0=s+seg
        if s>=0 and e0<=T: X0.append(data[:,s:e0])
    if not len(X0): return 0.0, 0.5, 0.5, 0
    X0=np.stack(X0,0)
    X0n=zscore_apply(X0, mu_tgt, sd_tgt)
    logits=predict_logits(model, X0n)
    l = logits[:,1]-logits[:,0]   # D minus C
    p_tgt = float(TARGET_PRIOR[1]) # desired mean P(D)
    s = 1/(1+np.exp(-l)); mean_before=float(s.mean())
    delta=0.0
    for _ in range(max_iter):
        s = 1/(1+np.exp(-(l+delta)))
        m = s.mean()
        g = (m - p_tgt)
        h = (s*(1-s)).mean()+1e-9
        step = g/h
        delta -= 0.5*step  # half-step to avoid overshoot
        if abs(step)<1e-4: break
    delta=float(np.clip(delta,-3.0,3.0))
    mean_after=float((1/(1+np.exp(-(l+delta)))).mean())
    return delta, mean_before, mean_after, int(len(l))

def apply_half_shift(logits: np.ndarray, delta: float) -> np.ndarray:
    """C' = C - Δ/2, D' = D + Δ/2 ⇒ (D'-C') = (D-C) + Δ"""
    lo = logits.copy()
    lo[:,0] -= 0.5*delta   # C
    lo[:,1] += 0.5*delta   # D
    return lo

# ----------------------
# Detection
# ----------------------
def detect_sequence(model:nn.Module, meta:Dict, pick_names:List[str]) -> Tuple[List[str], List[int], List[float]]:
    raw_tgt = load_raw(TARGET_FILE)
    sf = raw_tgt.info["sfreq"]
    seg = int(round(SEG_SEC*sf)); base=int(round(BASELINE_SEC*sf))
    ev = events_from_ann(raw_tgt, EVENT_DESC)
    log(f"[Detect] Using target picks by name: {pick_names}")
    log(f"[Detect] seg={SEG_SEC:.1f}s ({seg} samp), jitter=±{JITTER_SAMP} samp, baseline={BASELINE_SEC:.1f}s anchored to event")

    # target normalization from zero-shift
    picks=[raw_tgt.ch_names.index(n) for n in pick_names]
    data = raw_tgt.get_data(picks=picks)
    C,T = data.shape
    X0=[]
    for e in ev:
        s=e[0]-base; e0=s+seg
        if s>=0 and e0<=T: X0.append(data[:,s:e0])
    X0=np.stack(X0,0) if len(X0) else np.zeros((0,C,seg))
    mu_tgt, sd_tgt = zscore_fit(X0) if len(X0) else (meta["mu_tr"], meta["sd_tr"])
    log(f"[Target μ/σ] computed on {len(X0)} zero-shift events.")

    # Optional class-prior (as centered log-prior) to logits
    log_prior = np.log(np.clip(TARGET_PRIOR,1e-6,1.0))
    log_prior -= log_prior.mean()

    # Fit scalar Δ on target and apply as half-shift
    delta, m_before, m_after, n0 = fit_target_logit_shift(model, raw_tgt, pick_names, mu_tgt, sd_tgt)
    log(f"[Calib] Target logit shift Δ = {delta:+.4f} (half-shift) | mean P(D) before={m_before:.3f} → after={m_after:.3f} on N0={n0}")

    labels=[]; best_shifts=[]; max_probs=[]
    jitter = np.arange(-JITTER_SAMP, JITTER_SAMP+1)
    for i,e in enumerate(ev):
        segs=[]; valid_sh=[]
        for sh in jitter:
            s=e[0]-base+sh; e1=s+seg
            if s<0 or e1>T: continue
            segs.append(data[:,s:e1]); valid_sh.append(sh)
        if not segs:
            labels.append("UNK"); best_shifts.append(0); max_probs.append(0.0); continue

        Xc=np.stack(segs,0)
        Xn=zscore_apply(Xc, mu_tgt, sd_tgt)
        logits=predict_logits(model, Xn)
        logits=apply_half_shift(logits, delta)
        if USE_CLASS_PRIOR:
            logits = logits + log_prior.reshape(1,2)
        probs = softmax_temp_np(logits, meta["T"])
        # select shift by highest max-prob
        idx = int(np.argmax(probs.max(1)))
        pbest = probs[idx]; top=int(pbest.argmax()); mx=float(pbest.max())
        sep = float(np.sort(pbest)[::-1][0]-np.sort(pbest)[::-1][1])
        lbl = ["C","D"][top]
        if (mx<meta["tau"]) or (sep<meta["margin"]):
            lbl="UNK"
        labels.append(lbl); best_shifts.append(int(valid_sh[idx])); max_probs.append(mx)

    if USE_VITERBI:
        labels = viterbi_smooth(labels, max_probs, states=("C","D","UNK"))

    return labels, best_shifts, max_probs

def viterbi_smooth(labels: List[str], confid: List[float], states=("C","D","UNK")) -> List[str]:
    S = list(states); idx={s:i for i,s in enumerate(S)}
    n=len(labels)
    trans = np.full((3,3), (1-HMM_SELF)/2.0)
    np.fill_diagonal(trans, HMM_SELF)
    trans[idx["UNK"],:] = [0.04,0.04,0.92]
    trans = np.log(np.clip(trans,1e-9,1.0))

    emit = np.zeros((n,3))
    for t,l in enumerate(labels):
        if l=="UNK":
            emit[t]=np.log([HMM_UNK, HMM_UNK, 1-2*HMM_UNK])
        else:
            c=max(1e-6, min(1-1e-6, confid[t]))
            if l==S[0]: emit[t]=np.log([c, 1-c, 1e-3])  # C
            else:       emit[t]=np.log([1-c, c, 1e-3])  # D

    dp = np.full((n,3), -np.inf); bp = np.full((n,3), -1, dtype=int)
    dp[0]=emit[0]
    for t in range(1,n):
        for s in range(3):
            cand = dp[t-1]+trans[:,s]
            j=int(np.argmax(cand))
            dp[t,s]=cand[j]+emit[t,s]; bp[t,s]=j
    path=[int(np.argmax(dp[-1]))]
    for t in range(n-1,0,-1):
        path.append(int(bp[t, path[-1]]))
    path=path[::-1]
    return [S[i] for i in path]

# ----------------------
# Evaluation vs GT (optional)
# ----------------------
def compute_accuracy_vs_gt(pred_seq: List[str], gt_seq_str: str) -> float:
    gt = gt_seq_str.split()
    n = min(len(gt), len(pred_seq))
    if n==0: return 0.0
    correct = sum(int(pred_seq[i]==gt[i]) for i in range(n))
    return 100.0*correct/float(n)

# ----------------------
# PLOTS & SAVES
# ----------------------
def save_prob_plot(probs: List[float], out: str):
    plt.figure(figsize=(9,3))
    plt.plot(probs, lw=1)
    plt.axhline(0.5, ls="--", alpha=0.6)
    plt.title("Per-event max probability")
    plt.xlabel("Event #"); plt.ylabel("max p")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if DISPLAY: plt.show()
    plt.close()

def save_shift_hist(shifts: List[int], out: str):
    plt.figure(figsize=(8,3))
    bins = np.arange(-JITTER_SAMP-0.5, JITTER_SAMP+1.5, 1)
    plt.hist(shifts, bins=bins, edgecolor="black")
    plt.title("Best jitter shift per event (samples)")
    plt.xlabel("Shift"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    if DISPLAY: plt.show()
    plt.close()

# ----------------------
# MAIN
# ----------------------
def main():
    set_seed()
    log("[START] Load raws & pick common channels")
    raws = [load_raw(TEMPLATE_SOURCES["C"]), load_raw(TEMPLATE_SOURCES["D"])]
    X, y, g, sf, pick_names = build_trainset_cd(TEMPLATE_SOURCES, PREFERRED_CH_NAMES)
    log("[END]   Load raws & pick common channels")

    # Train
    model, meta = train_model(X, y, g, sf)

    # Detect
    labels, shifts, probs = detect_sequence(model, meta, pick_names)

    # Print sequence (wrapped)
    print("\n=== Detected Sequence (EEGNet C/D, annotation-locked + jitter) ===")
    line=[]; wrap=[]
    for tok in labels:
        line.append(tok)
        if len(line)>=90: wrap.append(" ".join(line)); line=[]
    if line: wrap.append(" ".join(line))
    print("\n".join(wrap))

    # counts
    cntC=labels.count("C"); cntD=labels.count("D"); cntU=labels.count("UNK")
    print("\n[Counts]\n  C: {}\n  D: {}\n  UNK: {}".format(cntC,cntD,cntU))

    # Save outputs
    seq_txt = os.path.join(OUT_DIR, "eegnet_cd_mi_mix_sequence.txt")
    with open(seq_txt,"w",encoding="utf-8") as f: f.write(" ".join(labels)+"\n")
    seq_csv = os.path.join(OUT_DIR, "eegnet_cd_mi_mix_sequence.csv")
    with open(seq_csv,"w",encoding="utf-8") as f:
        f.write("event_idx,label,shift,max_prob\n")
        for i,(l,s,p) in enumerate(zip(labels,shifts,probs),1):
            f.write(f"{i},{l},{s},{p:.6f}\n")
    prob_png = os.path.join(OUT_DIR, "eegnet_cd_mi_mix_maxprob.png")
    save_prob_plot(probs, prob_png)
    hist_png = os.path.join(OUT_DIR, "target_shift_hist_eegnet_cd.png")
    save_shift_hist(shifts, hist_png)
    print(f"[Saved] Sequence → {seq_txt}")
    print(f"[Saved] Table    → {seq_csv}")
    print(f"[Saved] Probability trace → {prob_png}")
    print(f"[Saved] Shift histogram → {hist_png}")

    # Accuracy vs ground truth (optional)
    if MI_GT_SEQ_STR.strip():
        acc = compute_accuracy_vs_gt(labels, MI_GT_SEQ_STR)
        print(f"[Accuracy vs GT] {acc:.2f}% (truncated to common length)")

if __name__=="__main__":
    main()

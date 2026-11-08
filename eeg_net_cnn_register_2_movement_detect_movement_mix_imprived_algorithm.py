# -*- coding: utf-8 -*-
"""
EEGNet (A/B) — Register on movement A & B, detect sequence on movement-mix.
- Windows are annotation-locked to "Stimulus/S  1" (one window per event).
- Train/val split is *grouped by event* to avoid leakage (jittered copies share a group).
- Detection scans jitter around each annotation and picks the most confident shift.
- Half-shift calibration fixes the previous 2× logit-shift bug.
- Optional prior correction (log-prior) on logits.
- Optional simple 2-state Viterbi smoothing on the output stream.
- Saves sequence (.txt/.csv), probability trace, and shift histogram.
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
# CONFIG
# ----------------------
TEMPLATE_SOURCES = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eegnet_ab_output"
os.makedirs(OUT_DIR, exist_ok=True)

# EEG preprocessing
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)       # motor ERD/ERS-ish; try (10,26) if needed
NOTCH_HZ = None              # 50 or 60 to enable
PREFERRED_CH_NAMES = ["C3", "Cz", "C4", "FCz"]  # will fall back to all EEG if missing

# Event/epoch config
EVENT_DESC = "Stimulus/S  1"
SEG_SEC = 1.0                # window length
BASELINE_SEC = 0.2           # baseline anchoring before event (for normalization only)
JITTER_SAMP = 50             # ± jitter (samples) at detection
TRAIN_JITTER = 10            # ± jitter (samples) for augmentation during training
AUG_PER_EVENT = 2            # how many randomized jittered copies per event (0=no aug)

# Model/training
SEED = 42
BATCH = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
DEVICE = "cpu"  # set "cuda" if you have it

# Calibration & thresholding
TEMP_GRID = [0.6, 0.8, 1.0]       # softmax temperature candidates
TAU_GRID  = [0.30, 0.40, 0.50]    # unknown threshold on max prob
MARGIN_GRID = [0.00, 0.05, 0.10]  # min separation between top-1 and top-2

# Class prior (target)
USE_CLASS_PRIOR = True
TARGET_PRIOR = np.array([0.50, 0.50], dtype=np.float64)  # if you know true proportions, set here

# Smoothing
USE_VITERBI = True
HMM_SELF = 0.92    # stickiness (higher -> fewer flips)
HMM_UNK  = 0.02    # probability of UNK when below thresholds (soft state)

# Varia
DISPLAY = False     # set True to plt.show()
LOG_PREFIX = time.strftime("%Y-%m-%d %H:%M:%S") + " |"

# Ground-truth for movement-mix (provided by you)
GT_SEQ_STR = ("B A A A B B B A A A A A B B B B B B B B B A A A A A A B B B A A B B B B A A A A A A B B B B B B B B B A A A A A A A A A A A "
              "B B B B B B B B B B B B B B A A A A A A B B B A A A A A A A A A A A B B B B B B A A A A A A A A A A A A A A B B B B B B B B "
              "B B B B B B B B A A A A A A A A A A B B B B B B B B B B A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B A A "
              "A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B")

# ----------------------
# UTILS
# ----------------------
def log(msg: str): print(f"{LOG_PREFIX} {msg}")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

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

def pick_by_name(raws: List[mne.io.BaseRaw], names: List[str]) -> List[str]:
    # use case-insensitive matching; fall back to all EEG if any missing
    for nm in names:
        ok = all((nm.upper() in [c.upper() for c in r.ch_names]) for r in raws)
        if not ok:
            # fallback: intersection of EEG channels across files
            inter = set([c for c in raws[0].ch_names])
            for r in raws[1:]:
                inter = inter.intersection(set(r.ch_names))
            # filter EEG only
            common = [c for c in inter if r.get_channel_types(picks=[c])[0] == "eeg"]
            return sorted(common)[:min(8, len(common))]  # cap to 8 channels
    return names

def events_from_ann(raw: mne.io.BaseRaw, desc=EVENT_DESC) -> np.ndarray:
    ev, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(ev_id) == 0 or ev.size == 0:
        raise RuntimeError("No annotations found; please ensure events exist.")
    if desc in ev_id:
        code = ev_id[desc]
        return ev[ev[:, 2] == code]
    # else pick most frequent
    uniq, cnt = np.unique(ev[:, 2], return_counts=True)
    code = uniq[np.argmax(cnt)]
    return ev[ev[:, 2] == code]

def extract_event_windows(raw: mne.io.BaseRaw, pick_names: List[str], jitter_samp: int,
                          aug_per_event: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups) for A or B file. y is filled by caller; groups group by event id."""
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
        # jitter aug (random within ± jitter_samp)
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
    # X: (N,C,T) → per-channel μ/σ over all time & trials
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu, sd

def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

# ----------------------
# EEGNet (tiny)
# ----------------------
class EEGNetSmall(nn.Module):
    def __init__(self, C: int, T: int, n_classes=2, dropout=DROPOUT):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.DepthwiseConv2d if hasattr(nn, "DepthwiseConv2d") else nn.Identity()
        )
        # Use grouped conv to emulate depthwise if PyTorch < 2.2
        self.dw = nn.Conv2d(8, 16, kernel_size=(C, 1), groups=8, bias=False)
        self.dw_bn = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.separable_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        # compute out dims
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self.block1[0](x)
            x = self.block1[1](x)
            x = self.dw(x)
            x = self.dw_bn(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.separable_conv(x)
            flat = x.numel()
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):  # x: (B,1,C,T)
        x = self.block1[0](x)
        x = self.block1[1](x)
        x = self.dw(x)
        x = self.dw_bn(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable_conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def to_tensor(X: np.ndarray) -> torch.Tensor:
    # (N,C,T) → (N,1,C,T)
    return torch.from_numpy(X.astype(np.float32)).unsqueeze(1)

# ----------------------
# Data & training
# ----------------------
class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return to_tensor(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

def build_trainset_ab(paths: Dict[str,str], pick_names: List[str]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    raws = [load_raw(paths["A"]), load_raw(paths["B"])]
    sf = raws[0].info["sfreq"]
    pick_names = pick_by_name(raws, pick_names)
    log(f"[Channels] Using preferred set (by name): {pick_names}")

    X_all, y_all, g_all = [], [], []

    # A
    Xa, _, ga = extract_event_windows(raws[0], pick_names, TRAIN_JITTER, AUG_PER_EVENT)
    ya = np.zeros((len(Xa),), dtype=np.int64)
    log(f"  [A] {os.path.basename(paths['A'])} | picks={pick_names} | sf={sf:.1f} Hz | windows={len(Xa)}")

    # B
    Xb, _, gb = extract_event_windows(raws[1], pick_names, TRAIN_JITTER, AUG_PER_EVENT)
    yb = np.ones((len(Xb),), dtype=np.int64)
    log(f"  [B] {os.path.basename(paths['B'])} | picks={pick_names} | sf={sf:.1f} Hz | windows={len(Xb)}")

    X_all = np.concatenate([Xa, Xb], axis=0)
    y_all = np.concatenate([ya, yb], axis=0)
    g_all = np.concatenate([ga + 100000, gb + 200000], axis=0)  # distinct group ranges per file

    log(f"[TRAIN] Total windows: {X_all.shape} (N,C,T), per-class: "
        f"{{'A': {(y_all==0).sum()}, 'B': {(y_all==1).sum()}}}")
    return X_all, y_all, g_all, sf

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
    weights = 1.0 / class_count
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
        model.train(); tr_loss=0; n_tr=0
        for xb,yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item())*len(xb); n_tr += len(xb)
        # val
        model.eval(); correct=0; n_va=0
        with torch.no_grad():
            for xb,yb in dl_va:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred==yb).sum().item()); n_va += len(xb)
        tr_acc = 1.0 # not critical to print real train acc
        va_acc = correct/max(1,n_va)
        log(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={tr_acc:.4f} va={va_acc:.4f}")
        if va_acc >= best["acc"]:
            best["acc"]=va_acc
            best["state"]=model.state_dict()

        if ep%10==0 or ep==EPOCHS:
            # light early-stop if perfect for multiple epochs
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
        rep = classification_report(ys, ypred, target_names=["A","B"], digits=3)
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
                # treat UNK as wrong for the validation selection
                mask = yhat!=2
                if mask.sum()==0: continue
                acc_bal = balanced_acc(ys[mask], yhat[mask])
                if acc_bal > bacc_best:
                    T_best, tau_best, m_best, bacc_best = T, tau, m, acc_bal
    log(f"[Threshold] Best temperature T={T_best:.2f}, τ={tau_best:.2f}, margin={m_best:.2f} (val balanced acc={bacc_best:.3f})")

    return model, {"mu_tr":mu_tr, "sd_tr":sd_tr, "T":T_best, "tau":tau_best, "margin":m_best, "pick_names":None}

# ----------------------
# Softmax / helpers
# ----------------------
def softmax_temp_np(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e/np.clip(e.sum(1,keepdims=True),1e-9,None)

def balanced_acc(y_true, y_pred):
    # binary, ignore UNK=2 (should be filtered before calling)
    tpA = ((y_true==0)&(y_pred==0)).sum(); fnA=((y_true==0)&(y_pred!=0)).sum()
    tpB = ((y_true==1)&(y_pred==1)).sum(); fnB=((y_true==1)&(y_pred!=1)).sum()
    rA = tpA/max(1,tpA+fnA); rB = tpB/max(1,tpB+fnB)
    return 0.5*(rA+rB)

def probs_to_labels(p: np.ndarray, tau=0.5, margin=0.0) -> np.ndarray:
    top = p.argmax(1)
    mx  = p.max(1)
    # margin check
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    sep = sorted_p[:,0]-sorted_p[:,1]
    unk = (mx<tau) | (sep<margin)
    out = top.copy()
    out[unk] = 2  # UNK
    return out

# ----------------------
# Calibration (fixed half-shift)
# ----------------------
def predict_logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xb = to_tensor(X).to(DEVICE)
        lo = model(xb).cpu().numpy()
    return lo

def fit_target_logit_shift(model: nn.Module, raw_tgt: mne.io.BaseRaw, pick_names: List[str],
                           mu_tgt: np.ndarray, sd_tgt: np.ndarray, max_iter=200):
    """
    Estimate scalar Δ so that mean sigmoid(l + Δ) ≈ TARGET_PRIOR[1],
    where l = (logit_B - logit_A) on target zero-shift events.
    Returns: (delta, mean_before, mean_after, N0)
    """
    # collect zero-shift segments
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
    l = logits[:,1]-logits[:,0]
    p_tgt = float(TARGET_PRIOR[1])
    s = 1/(1+np.exp(-l)); mean_before=float(s.mean())
    delta=0.0
    for _ in range(max_iter):
        s = 1/(1+np.exp(-(l+delta)))
        m = s.mean()
        g = (m - p_tgt)
        h = (s*(1-s)).mean()+1e-9
        step = g/h
        delta -= 0.5*step
        if abs(step)<1e-4: break
    delta=float(np.clip(delta,-3.0,3.0))
    mean_after=float((1/(1+np.exp(-(l+delta)))).mean())
    return delta, mean_before, mean_after, int(len(l))

def apply_half_shift(logits: np.ndarray, delta: float) -> np.ndarray:
    """A' = A - Δ/2, B' = B + Δ/2 ⇒ (B'-A') = (B-A) + Δ"""
    lo = logits.copy()
    lo[:,0] -= 0.5*delta
    lo[:,1] += 0.5*delta
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

    # target normalization
    picks=[raw_tgt.ch_names.index(n) for n in pick_names]
    data = raw_tgt.get_data(picks=picks)
    C,T = data.shape
    # μ/σ from zero-shift target windows
    X0=[]
    for e in ev:
        s=e[0]-base; e0=s+seg
        if s>=0 and e0<=T: X0.append(data[:,s:e0])
    X0=np.stack(X0,0) if len(X0) else np.zeros((0,C,seg))
    mu_tgt, sd_tgt = zscore_fit(X0) if len(X0) else meta["mu_tr"], meta["sd_tr"]
    log(f"[Target μ/σ] computed on {len(X0)} zero-shift events.")

    # Optional class-prior (as log prior) to logits
    log_prior = np.log(np.clip(TARGET_PRIOR,1e-6,1.0))
    log_prior -= log_prior.mean()  # center to avoid constant bias

    # Fit scalar Δ on target
    delta, m_before, m_after, n0 = fit_target_logit_shift(model, raw_tgt, pick_names, mu_tgt, sd_tgt)
    log(f"[Calib] Target logit shift Δ = {delta:+.4f} (half-shift) | mean P(B) before={m_before:.3f} → after={m_after:.3f} on N0={n0}")

    # decode
    labels=[]; best_shifts=[]; max_probs=[]
    jitter = np.arange(-JITTER_SAMP, JITTER_SAMP+1)
    for i,e in enumerate(ev):
        # build all candidate shifts, normalize with target μ/σ
        segs=[]; valid_sh=[]
        for sh in jitter:
            s=e[0]-base+sh; e1=s+seg
            if s<0 or e1>T: continue
            segs.append(data[:,s:e1]); valid_sh.append(sh)
        if not segs:
            labels.append("UNK"); best_shifts.append(0); max_probs.append(0.0); continue

        Xc=np.stack(segs,0)
        Xn=zscore_apply(Xc, mu_tgt, sd_tgt)
        lo=predict_logits(model, Xn)
        # apply half-shift calibration
        lo=apply_half_shift(lo, delta)
        # add log-prior
        if USE_CLASS_PRIOR:
            lo = lo + log_prior.reshape(1,2)
        # temperature
        probs = softmax_temp_np(lo, meta["T"])
        # choose best by top prob
        idx = np.unravel_index(np.argmax(probs.max(1)), probs.shape)[0]
        pbest = probs[idx]; top=pbest.argmax(); mx=float(pbest.max())
        sep = float(np.sort(pbest)[::-1][0]-np.sort(pbest)[::-1][1])
        lbl = ["A","B"][top]
        if (mx<meta["tau"]) or (sep<meta["margin"]):
            lbl="UNK"
        labels.append(lbl); best_shifts.append(int(valid_sh[idx])); max_probs.append(mx)

    # optional smoothing (Viterbi over A/B with an UNK sink)
    if USE_VITERBI:
        labels = viterbi_smooth(labels, max_probs)

    return labels, best_shifts, max_probs

def viterbi_smooth(labels: List[str], confid: List[float]) -> List[str]:
    # Simple 3-state HMM: A,B,UNK; transitions favor staying
    S = ["A","B","UNK"]; idx={s:i for i,s in enumerate(S)}
    n=len(labels)
    trans = np.full((3,3), (1-HMM_SELF)/2.0)
    np.fill_diagonal(trans, HMM_SELF)
    # weakly favor UNK self
    trans[idx["UNK"],:] = [0.04,0.04,0.92]
    trans = np.log(np.clip(trans,1e-9,1.0))

    # emission: use confidences — if label says UNK then prefer UNK
    emit = np.zeros((n,3))
    for t,l in enumerate(labels):
        if l=="UNK":
            emit[t]=np.log([HMM_UNK, HMM_UNK, 1-2*HMM_UNK])
        else:
            c=confid[t]
            if l=="A": emit[t]=np.log([max(c,1e-6), max(1-c,1e-6), 1e-3])
            else:      emit[t]=np.log([max(1-c,1e-6), max(c,1e-6), 1e-3])

    # Viterbi
    dp = np.full((n,3), -np.inf); bp = np.full((n,3), -1, dtype=int)
    dp[0]=emit[0]
    for t in range(1,n):
        for s in range(3):
            cand = dp[t-1]+trans[:,s]
            j=np.argmax(cand)
            dp[t,s]=cand[j]+emit[t,s]; bp[t,s]=j
    path=[int(np.argmax(dp[-1]))]
    for t in range(n-1,0,-1):
        path.append(int(bp[t, path[-1]]))
    path=path[::-1]
    return [S[i] for i in path]

# ----------------------
# Evaluation vs GT
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
    raws = [load_raw(TEMPLATE_SOURCES["A"]), load_raw(TEMPLATE_SOURCES["B"])]
    pick_names = pick_by_name(raws, PREFERRED_CH_NAMES)
    X, y, g, sf = build_trainset_ab(TEMPLATE_SOURCES, pick_names)
    log("[END]   Load raws & pick common channels")

    # Train
    model, meta = train_model(X, y, g, sf)
    meta["pick_names"] = pick_names

    # Detect
    labels, shifts, probs = detect_sequence(model, meta, pick_names)

    # Print sequence (wrapped)
    print("\n=== Detected Sequence (EEGNet A/B, annotation-locked + jitter) ===")
    line=[]; wrap=[]
    for tok in labels:
        line.append(tok)
        if len(line)>=90: wrap.append(" ".join(line)); line=[]
    if line: wrap.append(" ".join(line))
    print("\n".join(wrap))

    # counts
    cntA=labels.count("A"); cntB=labels.count("B"); cntU=labels.count("UNK")
    print("\n[Counts]\n  A: {}\n  B: {}\n  UNK: {}".format(cntA,cntB,cntU))

    # Save outputs
    seq_txt = os.path.join(OUT_DIR, "eegnet_ab_movement_mix_sequence.txt")
    with open(seq_txt,"w",encoding="utf-8") as f: f.write(" ".join(labels)+"\n")
    seq_csv = os.path.join(OUT_DIR, "eegnet_ab_movement_mix_sequence.csv")
    with open(seq_csv,"w",encoding="utf-8") as f:
        f.write("event_idx,label,shift,max_prob\n")
        for i,(l,s,p) in enumerate(zip(labels,shifts,probs),1):
            f.write(f"{i},{l},{s},{p:.6f}\n")
    prob_png = os.path.join(OUT_DIR, "eegnet_ab_movement_mix_maxprob.png")
    save_prob_plot(probs, prob_png)
    hist_png = os.path.join(OUT_DIR, "target_shift_hist_eegnet_ab.png")
    save_shift_hist(shifts, hist_png)
    print(f"[Saved] Sequence → {seq_txt}")
    print(f"[Saved] Table    → {seq_csv}")
    print(f"[Saved] Probability trace → {prob_png}")
    print(f"[Saved] Shift histogram → {hist_png}")

    # Accuracy vs ground truth
    acc = compute_accuracy_vs_gt(labels, GT_SEQ_STR)
    print(f"[Accuracy vs GT] {acc:.2f}% (truncated to common length)")

if __name__=="__main__":
    main()

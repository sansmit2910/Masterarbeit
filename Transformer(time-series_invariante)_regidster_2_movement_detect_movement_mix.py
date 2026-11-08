# -*- coding: utf-8 -*-
"""
Time-Series Transformer (A/B only) for EEG
- Train on movement A,B (sliding windows)
- Validate; search temperature + UNK threshold + margin
- Detect A/B sequence in movement-mix target (annotation-locked + jitter)
- Accuracy vs optional pasted GT

Key fixes for 'all B':
  • Target re-centering (μ from zero-shift target segments; σ from train)
  • Prior correction (subtract log class prior learned on train)
  • Target logit debias (symmetrize A/B on zero-shift target before jitter)

Reqs: mne, numpy, matplotlib, torch, scikit-learn
"""

import os, time, math, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import mne
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
)

# --------------------------
# === USER FILE PATHS ===
# --------------------------
TEMPLATE_SOURCES: Dict[str, str] = {
    "A": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-1-reg.fif",
    "B": r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-2-reg.fif",
}
TARGET_FILE = r"C:\Users\User\Desktop\masters_thesis_directory\preprocessed_raw_sub-001-movement-mix_raw.fif"

OUT_DIR = r"C:\Users\User\Desktop\masters_thesis_directory\eeg_tsxfmr_ab_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# === CONFIG ===
# --------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess
RESAMPLE_HZ = 250.0
BANDPASS = (8.0, 30.0)   # motor band
NOTCH = None             # 50/60 or None
PREF_CHANSETS = [["C3", "Cz", "C4", "FCz"], ["C3", "Cz", "C4"]]

# Sliding windows
WIN_SEC  = 1.0
STEP_SEC = 0.25

# Detection (annotation-locked)
EVENT_DESC = "Stimulus/S  1"
TMIN_SEC   = 0.0
SEG_SEC    = 1.0
JITTER_SAMP = 50          # ± samples

# UNK search grid (picked on validation)
UNK_GRID    = [0.30, 0.40, 0.50]
TEMP_GRID   = [0.6, 0.8, 1.0]
MARGIN_GRID = [0.0, 0.1, 0.2]

# Training
BATCH = 128
EPOCHS = 40
LR = 2e-3
WD = 1e-3
DROPOUT = 0.2
PATIENCE = 7

# Optional: paste GT string to score accuracy
GT_STRING = ""  # e.g., "B A A A B B ..."

# --------------------------
# === UTILS ===
# --------------------------
def log(msg): print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}")

def safe_mean_std(X: np.ndarray, eps=1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N, C, T)
    mu = X.mean(axis=(0, 2), keepdims=True)  # (1,C,1)
    sd = X.std(axis=(0, 2), keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def load_raw(path: str) -> mne.io.BaseRaw:
    if not os.path.isfile(path): raise FileNotFoundError(path)
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    try: raw.pick(eeg=True, exclude="bads")
    except TypeError: raw.pick_types(eeg=True, meg=False, stim=False, eog=False, exclude="bads")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if NOTCH: raw.notch_filter(NOTCH, fir_design="firwin", verbose="ERROR")
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin", verbose="ERROR")
    if RESAMPLE_HZ: raw.resample(RESAMPLE_HZ, npad="auto", verbose="ERROR")
    return raw

def sliding_windows(raw: mne.io.BaseRaw, picks: List[int], win_sec: float, step_sec: float) -> np.ndarray:
    sf = raw.info["sfreq"]
    X = raw.get_data(picks=picks)  # (C, T)
    C, T = X.shape
    win = int(round(win_sec * sf))
    step = int(round(step_sec * sf))
    starts = np.arange(0, T - win + 1, step, dtype=int)
    out = np.empty((len(starts), C, win), dtype=np.float32)
    for i, s in enumerate(starts): out[i] = X[:, s:s+win]
    return out  # (N,C,win)

def intersect_channel_names(paths: Dict[str, str], pref: List[List[str]]) -> Tuple[List[str], float]:
    raws = [load_raw(p) for p in paths.values()]
    for wanted in pref:
        ok = True
        for raw in raws:
            ups = [n.upper() for n in raw.ch_names]
            for w in wanted:
                if w.upper() not in ups: ok=False; break
            if not ok: break
        if ok: return wanted, raws[0].info["sfreq"]
    # fallback: intersection of EEG names
    sets=[]
    for r in raws:
        try: names = r.copy().pick(eeg=True).ch_names
        except Exception:
            idx = mne.pick_types(r.info, eeg=True); names = [r.ch_names[i] for i in idx]
        sets.append(set([n.upper() for n in names]))
    inter = list(sorted(set.intersection(*sets)))
    if not inter: raise RuntimeError("No common EEG channels across A and B.")
    order = ["C3","Cz","C4","FCz","CP3","CPz","CP4","FC3","FC4"]
    ordered = [n for n in order if n.upper() in inter] + [n for n in inter if n not in [x.upper() for x in order]]
    return ordered, raws[0].info["sfreq"]

def build_dataset(paths: Dict[str, str]):
    labs = ["A", "B"]
    ch_names, sf = intersect_channel_names(paths, PREF_CHANSETS)
    all_X, all_y, all_g = [], [], []
    log("| [START] Load raws & pick common channels")
    for i, lab in enumerate(labs):
        raw = load_raw(paths[lab])
        ups = [n.upper() for n in raw.ch_names]
        picks = [ups.index(cn.upper()) for cn in ch_names]
        Xw = sliding_windows(raw, picks, WIN_SEC, STEP_SEC)
        all_X.append(Xw)
        all_y.append(np.full((Xw.shape[0],), i, dtype=np.int64))
        all_g.append(np.full((Xw.shape[0],), i, dtype=np.int64))
        log(f"|   [{lab}] {os.path.basename(paths[lab])} | picks={ch_names} | sf= {sf:.1f} Hz | windows={Xw.shape[0]}")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    g = np.concatenate(all_g, axis=0)
    log(f"| [TRAIN] Total windows: {X.shape} (N,C,T), per-class: {{'A': {(y==0).sum()}, 'B': {(y==1).sum()}}}")
    log("| [END]   Load raws & pick common channels")
    return X, y, g, labs, sf, ch_names

# --------------------------
# === MODEL ===
# --------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal PE for batch-first sequences (B,T,D)."""
    def __init__(self, d_model: int, dropout: float, max_len: int = 4000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)
    def forward(self, x):  # x: (B,T,D)
        T = x.size(1); return self.dropout(x + self.pe[:, :T, :])

class TSXfmrAB(nn.Module):
    """Time-Series Transformer for (C,T) -> logits(2) with batch_first=True."""
    def __init__(self, n_ch: int, win_len: int, n_classes: int = 2, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=DROPOUT):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(n_ch, d_model, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        Lp = (win_len + 2*4 - 9) // 2 + 1
        self.pos = PositionalEncoding(d_model, dropout, max_len=Lp+5)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, x):  # x: (B,C,T)
        z = self.pre(x)            # (B,D,Lp)
        z = z.transpose(1, 2)      # (B,Lp,D)
        z = self.pos(z)            # (B,Lp,D)
        z = self.encoder(z)        # (B,Lp,D)
        z = self.norm(z.mean(dim=1))
        logits = self.head(z) / torch.clamp(self.temperature, min=1e-3)
        return logits

# --------------------------
# === DATASETS ===
# --------------------------
class WinDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mu: Optional[np.ndarray]=None, sd: Optional[np.ndarray]=None):
        self.X = X.astype(np.float32)
        if mu is None or sd is None: mu, sd = safe_mean_std(self.X)
        self.mu = mu.astype(np.float32); self.sd = sd.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        xi = (self.X[i] - self.mu[0]) / self.sd[0]
        # per-window channel de-mean (helps robustness)
        xi = xi - xi.mean(axis=-1, keepdims=True)
        return torch.from_numpy(xi), torch.tensor(self.y[i])

# --------------------------
# === TRAIN / VAL ===
# --------------------------
def train_model(X: np.ndarray, y: np.ndarray, sf: float, ch_names: List[str]):
    N, C, T = X.shape
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    mu, sd = safe_mean_std(Xtr)
    ds_tr = WinDataset(Xtr, ytr, mu, sd)
    ds_va = WinDataset(Xva, yva, mu, sd)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    model = TSXfmrAB(n_ch=C, win_len=T, n_classes=2).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    best_state, best_val = None, 1e9
    patience = 0
    tr_curve, va_curve = [], []

    for ep in range(1, EPOCHS+1):
        model.train(); total=0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()*xb.size(0)
        tr_loss = total/len(ds_tr)

        model.eval(); total=0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += F.cross_entropy(model(xb), yb).item()*xb.size(0)
        va_loss = total/len(ds_va)
        tr_curve.append(tr_loss); va_curve.append(va_loss); sched.step(va_loss)
        print(f"[Final] Epoch {ep:02d}/{EPOCHS} | tr={tr_loss:.4f} va={va_loss:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss; best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break

    # curves
    fig, ax = plt.subplots(figsize=(6,3.2))
    ax.plot(tr_curve, label="train"); ax.plot(va_curve, label="val")
    ax.set_title("Training curve (loss)"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "final_training_curve_tsxfmr_ab.png"), dpi=150); plt.close(fig)

    if best_state is not None: model.load_state_dict(best_state)

    # validation report
    probs_va, yva_pred = infer_proba(model, ds_va)
    print("\n[Validation] Classification report:")
    print(classification_report(ds_va.y, yva_pred, target_names=["A","B"], digits=3))
    cm = confusion_matrix(ds_va.y, yva_pred, labels=[0,1])
    print("[Validation] Confusion matrix:\n", cm)
    print(f"[Validation] Accuracy: {accuracy_score(ds_va.y, yva_pred):.3f}")

    # search thresholds
    best_combo=(1.0,0.5,0.0); best_bal=-1.0
    logits_va = infer_logits(model, ds_va)
    for T in TEMP_GRID:
        lp = logits_va / T
        pv = F.softmax(torch.from_numpy(lp), dim=1).numpy()
        conf = pv.max(axis=1); margin = np.sort(pv, axis=1)[:,-1] - np.sort(pv, axis=1)[:,-2]
        for tau in UNK_GRID:
            for m in MARGIN_GRID:
                yhat = pv.argmax(1)
                yhat = np.where((conf<tau)|(margin<m), -1, yhat)
                mask = yhat>=0
                if mask.sum()==0: continue
                bal = balanced_accuracy_score(ds_va.y[mask], yhat[mask])
                if bal>best_bal: best_bal=bal; best_combo=(T,tau,m)
    T_best, TAU_best, MARGIN_best = best_combo
    print(f"[Threshold] Best temperature T={T_best:.2f}, τ={TAU_best:.2f}, margin={MARGIN_best:.2f} (val balanced acc={best_bal:.3f})")

    # store meta (incl. priors)
    pA = float((y==0).mean()); pB = float((y==1).mean())
    meta = {
        "mu": mu, "sd": sd, "sf": sf, "ch_names": ch_names,
        "temperature": T_best, "unk_tau": TAU_best, "unk_margin": MARGIN_best,
        "log_prior": np.log(np.array([max(pA,1e-6), max(pB,1e-6)], dtype=np.float32))
    }
    return model, meta

@torch.no_grad()
def infer_logits(model: nn.Module, ds: Dataset) -> np.ndarray:
    model.eval(); dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
    outs=[]
    for xb, _ in dl:
        xb = xb.to(DEVICE)
        outs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outs)

@torch.no_grad()
def infer_proba(model: nn.Module, ds: Dataset):
    logits = infer_logits(model, ds)
    probs  = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    return probs, probs.argmax(1)

# --------------------------
# === EVENTS / DETECTION ===
# --------------------------
def extract_events(raw: mne.io.BaseRaw, prefer_desc=EVENT_DESC):
    ev, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    if len(event_id)>0 and ev.size>0:
        if prefer_desc in event_id:
            code = event_id[prefer_desc]; used = prefer_desc
        else:
            uniq, cnt = np.unique(ev[:,2], return_counts=True)
            code = uniq[np.argmax(cnt)]; used = {v:k for k,v in event_id.items()}.get(code, f"code {int(code)}")
        return ev[ev[:,2]==code], used
    ev = mne.find_events(raw, verbose="ERROR")
    if ev.size==0: raise RuntimeError("No events/annotations found.")
    uniq, cnt = np.unique(ev[:,2], return_counts=True)
    code = uniq[np.argmax(cnt)]
    return ev[ev[:,2]==code], f"code {int(code)}"

def detect_sequence(model: nn.Module, meta: dict, target_path: str, labs: List[str]):
    raw = load_raw(target_path)
    ups = [n.upper() for n in raw.ch_names]
    picks=[]
    for cn in meta["ch_names"]:
        if cn.upper() not in ups: raise RuntimeError(f"Target missing channel {cn}.")
        picks.append(ups.index(cn.upper()))
    sf = raw.info["sfreq"]; seg_samp = int(round(SEG_SEC*sf)); base_off = int(round(TMIN_SEC*sf))
    events, used_desc = extract_events(raw, EVENT_DESC)
    print(f"\n[Target] Using events '{used_desc}'; N={len(events)}")
    data = raw.get_data(picks=picks).astype(np.float32)  # (C,T)
    C, T = data.shape

    # ---------- Target re-centering (zero-shift) ----------
    zero_segments=[]
    for e in events:
        s = e[0] + base_off; e2 = s + seg_samp
        if s>=0 and e2<=T: zero_segments.append(data[:, s:e2])
    if not zero_segments:
        raise RuntimeError("No valid zero-shift segments in target.")
    zs = np.stack(zero_segments, axis=0)  # (E,C,T)
    mu_targ = zs.mean(axis=(0,2), keepdims=True).astype(np.float32)  # (1,C,1)
    print(f"[Target μ] computed on {zs.shape[0]} zero-shift events.")

    # Helper to standardize a (C,T) segment with target μ and train σ
    mu_tr, sd_tr = meta["mu"], meta["sd"]
    def zscore_seg(seg_CT: np.ndarray) -> np.ndarray:
        z = (seg_CT[None,...] - mu_targ) / sd_tr
        # per-window channel de-mean
        z = z - z.mean(axis=-1, keepdims=True)
        return z.astype(np.float32)  # (1,C,T)

    # ---------- Prior correction + target logit debias ----------
    # Estimate mean logits on zero-shift segments
    model.eval()
    logits_z = []
    with torch.no_grad():
        for seg in zero_segments:
            xb = torch.from_numpy(zscore_seg(seg)).to(DEVICE)  # (1,C,T)
            logit = model(xb) / meta["temperature"]
            logits_z.append(logit.squeeze(0).cpu().numpy())
    logits_z = np.stack(logits_z, axis=0)  # (E,2)

    # subtract train log-prior (remove learned prior bias)
    log_prior = meta["log_prior"]
    logits_z_priorfree = logits_z - log_prior[None,:]

    # compute symmetric bias to balance A/B on target zero-shift
    mean_z = logits_z_priorfree.mean(axis=0)  # (2,)
    # delta pushes classes to have equal central tendency
    delta = (mean_z[1] - mean_z[0]) / 2.0
    bias_vec = np.array([+delta, -delta], dtype=np.float32)  # add to logits to counter skew

    print(f"[Calib] Train priors log p = {log_prior}, mean zero-shift logits(after prior)={mean_z}, applied bias={bias_vec}")

    # ---------- Jitter sweep with calibrated logits ----------
    labels_out, probs_out, shifts_out, conf_out = [], [], [], []
    Ttemp = meta["temperature"]; tau = meta["unk_tau"]; margin_req = meta["unk_margin"]

    for i in range(len(events)):
        center = events[i,0] + base_off
        best = {"conf": -1.0, "margin": -1.0, "vec": None, "shift": 0, "idx": 0}
        for sh in range(-JITTER_SAMP, JITTER_SAMP+1):
            beg = center + sh; end = beg + seg_samp
            if beg<0 or end>T: continue
            seg = data[:, beg:end]
            xb = torch.from_numpy(zscore_seg(seg)).to(DEVICE)
            with torch.no_grad():
                logits = (model(xb) / Ttemp).cpu().numpy()[0]
            # apply prior correction and target bias
            logits = logits - log_prior + bias_vec
            pv = F.softmax(torch.from_numpy(logits), dim=0).numpy()
            conf = float(pv.max()); marg = float(np.sort(pv)[-1]-np.sort(pv)[-2])
            if conf > best["conf"]:
                best.update(conf=conf, margin=marg, vec=pv.copy(), shift=sh, idx=int(pv.argmax()))
        pred = "UNK" if (best["conf"]<tau or best["margin"]<margin_req) else labs[best["idx"]]
        labels_out.append(pred); probs_out.append(best["vec"]); shifts_out.append(best["shift"]); conf_out.append(best["conf"])

    # Print + Save
    print("\n=== Detected Sequence (TS-Transformer A/B, annotation-locked + jitter) ===")
    line, chunk = [], []
    for t in labels_out:
        line.append(t)
        if len(line) >= 90: chunk.append(" ".join(line)); line=[]
    if line: chunk.append(" ".join(line))
    print("\n".join(chunk))

    counts = {lab: labels_out.count(lab) for lab in labs}; counts["UNK"] = labels_out.count("UNK")
    print("\n[Counts]"); [print(f"  {k}: {counts.get(k,0)}") for k in labs+["UNK"]]

    seq_txt = os.path.join(OUT_DIR, "tsxfmr_ab_movement_mix_sequence.txt")
    with open(seq_txt, "w", encoding="utf-8") as f: f.write(" ".join(labels_out) + "\n")
    csv_path = os.path.join(OUT_DIR, "tsxfmr_ab_movement_mix_sequence.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("event_idx,label,shift,prob_A,prob_B,confidence\n")
        for i,(lab,sh,pv,cf) in enumerate(zip(labels_out, shifts_out, probs_out, conf_out)):
            pa = "" if pv is None else f"{pv[0]:.6f}"; pb = "" if pv is None else f"{pv[1]:.6f}"
            f.write(f"{i},{lab},{sh},{pa},{pb},{cf:.6f}\n")
    print(f"[Saved] Sequence → {seq_txt}")
    print(f"[Saved] Table    → {csv_path}")

    # Plots
    fig, ax = plt.subplots(figsize=(6,3))
    bins = np.arange(-JITTER_SAMP-0.5, JITTER_SAMP+1.5)
    ax.hist(shifts_out, bins=bins, edgecolor="black")
    ax.set_title("Target jitter shift (best per event)"); ax.set_xlabel("shift (samples)"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "target_shift_hist_tsxfmr_ab.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(conf_out, lw=1); ax.axhline(meta["unk_tau"], ls="--")
    ax.set_title("Best-shift confidence per event"); ax.set_xlabel("event #"); ax.set_ylabel("max probability")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "movement_mix_maxprob_tsxfmr_ab.png"), dpi=150); plt.close(fig)

    return labels_out

# --------------------------
# === ACCURACY vs GT ===
# --------------------------
def parse_tokens(s: str) -> List[str]:
    return [t.strip().upper() for t in s.split() if t.strip()]

def acc_vs_gt(pred_labels: List[str], gt_str: str, valid_labels: List[str]):
    if not gt_str.strip():
        print("[Accuracy vs GT] (skipped — no GT pasted)"); return
    gt = parse_tokens(gt_str)
    pr = [p for p in pred_labels if p in valid_labels]  # drop UNK
    L = min(len(pr), len(gt))
    if L == 0: print("[Accuracy vs GT] 0.00% (empty overlap)"); return
    correct = sum(pr[i]==gt[i] for i in range(L))
    print(f"[Accuracy vs GT] {correct/L*100:.2f}% (truncated to common length)")

# --------------------------
# === MAIN ===
# --------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1) Build dataset from A,B
    X, y, g, labs, sf, ch_names = build_dataset(TEMPLATE_SOURCES)

    # 2) Train + validation (stores priors etc.)
    model, meta = train_model(X, y, sf, ch_names)

    # 3) Detect on movement-mix target (with re-centering + prior/bias calibration)
    pred_labels = detect_sequence(model, meta, TARGET_FILE, labs)

    # 4) Optional accuracy vs pasted GT
    acc_vs_gt(pred_labels, GT_STRING, labs)

    print(f"[Done] All outputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()

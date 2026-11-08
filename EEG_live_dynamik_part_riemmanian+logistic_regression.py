#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic MI test — Riemann (Log-Euclidean Tangent Space) + Logistic Regression
- Registration A/B at 0.25 Hz for 7 minutes each (≈105 beeps per motif)
- LSL EEG input (BrainVision→RDA→LSL bridge)
- Beeps + LSL markers (TTL/LabRecorder hooks can be added as needed)
- Bandpass 8–30 Hz + 50 Hz notch (default for DE), pre-beep baseline (−0.5…0 s)
- Features: covariance (Ledoit–Wolf) → Log-Euclidean tangent vector (upper-tri, √2 off-diags)
- Multi-window extraction (4 hops) + posterior averaging per decision
- Logistic Regression with tiny GridSearchCV; optional isotonic calibration
- Confidence + margin + entropy gating for UNK
- Hands-free online EMA adaptation of the reference on confident decisions
- Progress bars via tqdm; logs to ./bci_output
"""

import os, sys, time, threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---------- headless-safe tiny popups ----------
HEADLESS = os.environ.get("DISPLAY", "") == "" and sys.platform != "win32"
try:
    import tkinter as tk
    from tkinter import messagebox as mb
except Exception:
    tk, mb = None, None
    HEADLESS = True

# ---------- LSL ----------
from pylsl import StreamInlet, StreamInfo, StreamOutlet, local_clock
try:
    from pylsl import resolve_byprop  # preferred
except Exception:
    resolve_byprop = None
try:
    from pylsl import resolve_streams   # fallback
except Exception:
    resolve_streams = None

# ---------- signal processing / math ----------
from scipy.signal import filtfilt, iirnotch, resample_poly, butter
from scipy.linalg import eigh, logm, expm

# ---------- ML ----------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.covariance import LedoitWolf
from joblib import dump

# ---------- progress bars ----------
try:
    from tqdm import tqdm
    def make_pbar(total, desc): return tqdm(total=total, desc=desc, ncols=80, leave=True)
except Exception:
    tqdm = None
    def make_pbar(total, desc):
        class _PB:
            def __init__(self, total, desc): self.i=0; self.total=total; self.desc=desc
            def update(self, n=1):
                self.i += n
                if self.i % 5 == 0 or self.i==self.total:
                    print(f"[{self.desc}] {self.i}/{self.total}")
            def close(self): pass
        return _PB(total, desc)

# ---------- audio ----------
import pygame, pygame.sndarray

# ------------------------- Config -------------------------
@dataclass
class Config:
    # LSL
    lsl_type: str = "EEG"
    lsl_name: Optional[str] = None
    fs_guess: float = 250.0

    # Buffers
    ring_seconds: float = 60.0
    seg_sec: float = 1.25
    baseline_sec: float = 0.5
    jitter_samp: int = 60

    # Filters
    bandpass_low: float = 8.0
    bandpass_high: float = 30.0
    notch_hz: Optional[float] = 50.0
    notch_q: float = 35.0
    resample_to: Optional[int] = None  # e.g., 250

    # Registration pacing & duration
    reg_beep_hz: float = 0.25
    reg_duration_min: float = 7.0

    # Live
    live_beep_hz: float = 0.2
    live_max_minutes: float = 5.0

    # Decision thresholds
    conf_tau: float = 0.55
    conf_margin: float = 0.05
    entropy_max: float = 0.65

    # Audio
    beep_ms: int = 60
    beep_freq: int = 880
    volume: float = 0.4

    # Output
    out_dir: str = "./bci_output"
    save_log: str = "live_log.csv"
    save_model: str = "ab_riemann_logreg.joblib"

    # Online adaptation (reference EMA)
    ref_adapt_alpha: float = 0.02

CFG = Config()

# ------------------------- Small UI -------------------------
def popup(title: str, text: str):
    if HEADLESS or tk is None or mb is None:
        print(f"[POPUP] {title}: {text}")
        return
    root = tk.Tk(); root.withdraw()
    try:
        mb.showinfo(title, text)
    finally:
        root.destroy()

def show_detection_popup(label: str, pmax: float):
    if HEADLESS or tk is None or mb is None:
        print(f"[LIVE] Decision: {label}  (p={pmax:.2f})")
        return
    root = tk.Tk(); root.withdraw()
    try:
        mb.showinfo("Live decision", f"{label}  (p={pmax:.2f})")
    finally:
        root.destroy()

# ------------------------- LSL buffer & reader -------------------------
class LSLBuffer:
    def __init__(self, fs: float, n_chan: int, seconds: float):
        self.fs = fs
        self.n_chan = n_chan
        self.maxlen = int(fs * seconds) + 10_000
        self.buf = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()

    def append(self, ts: np.ndarray, X: np.ndarray):
        with self.lock:
            for t, row in zip(ts, X):
                self.buf.append((t, row.copy()))

    def get_window(self, t0: float, dur: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        t1 = t0 + dur
        with self.lock:
            arr = list(self.buf)
        if not arr: return None
        ts = np.array([r[0] for r in arr], dtype=np.float64)
        mask = (ts >= t0) & (ts < t1)
        if not np.any(mask): return None
        X = np.stack([r[1] for r in arr])[mask]
        tt = ts[mask]
        return tt, X

class LSLReader(threading.Thread):
    def __init__(self, inlet: StreamInlet, ring: LSLBuffer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.inlet = inlet; self.ring = ring; self.stop_event = stop_event
    def run(self):
        while not self.stop_event.is_set():
            samples, timestamps = self.inlet.pull_chunk(timeout=0.05)
            if timestamps:
                X = np.asarray(samples, dtype=np.float32)
                ts = np.asarray(timestamps, dtype=np.float64)
                self.ring.append(ts, X)

# ------------------------- Markers & audio -------------------------
def make_marker_outlet(name="ExpMarkers", type_="Markers", srate=0.0, chn=1):
    info = StreamInfo(name, type_, chn, srate, 'string', 'marker_' + str(time.time()))
    return StreamOutlet(info)

def do_beep_and_mark(snd, outlet, ttl, code: str):
    outlet.push_sample([code], local_clock())
    if snd: snd.play()
    return local_clock()

def make_beep_sound(freq=880, ms=60, volume=0.4, fs=44100):
    try:
        pygame.mixer.init(frequency=fs, size=-16, channels=1)
    except Exception as e:
        print(f"[AUDIO] mixer init failed ({e}); continuing silently.")
        return None
    t = np.arange(int(ms/1000*fs))
    wave = (0.5*np.sin(2*np.pi*freq*t/fs)).astype(np.float32)
    # make_sound expects int16 on most backends
    snd = pygame.sndarray.make_sound((wave * (2**15 - 1)).astype(np.int16))
    snd.set_volume(volume)
    return snd

# ------------------------- Filters & preprocessing -------------------------
def butter_bp(fs, lo, hi, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='bandpass')
    return b, a

def apply_bp(x, fs, lo, hi, order=4):
    b, a = butter_bp(fs, lo, hi, order)
    return filtfilt(b, a, x, axis=-1)

def apply_notch(x, fs, f0=50.0, q=35.0):
    if f0 is None: return x
    b, a = iirnotch(f0/(fs/2), q)
    return filtfilt(b, a, x, axis=-1)

def baseline_stats(ring, t_trigger: float, fs: float, bl_sec: float = 0.5):
    w = ring.get_window(t_trigger - bl_sec, bl_sec)
    if w is None: return None, None
    _, X = w
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return mu, sd

def preprocess_epoch(epoch: np.ndarray, fs: float, mu_sd=None):
    x = epoch.copy()  # (C,T)
    x = apply_bp(x, fs, CFG.bandpass_low, CFG.bandpass_high)
    if CFG.notch_hz: x = apply_notch(x, fs, CFG.notch_hz, CFG.notch_q)
    if CFG.resample_to and CFG.resample_to != fs:
        x = resample_poly(x, CFG.resample_to, int(fs), axis=-1)
        fs = CFG.resample_to
    if mu_sd is not None and mu_sd[0] is not None:
        mu, sd = mu_sd
        x = x.T
        x = (x - mu.reshape(1,-1)) / sd.reshape(1,-1)
        return x.T
    T = x.shape[1]
    bl = max(1, min(int(round(CFG.baseline_sec*fs)), T))
    mu = x[:, :bl].mean(axis=1, keepdims=True)
    sd = x[:, :bl].std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd

# ------------------------- Riemannian mapping -------------------------
def shrink_cov(X: np.ndarray) -> np.ndarray:
    # X: (C,T), channel-centered inside LedoitWolf
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit((X - X.mean(axis=1, keepdims=True)).T)  # samples×features
    return lw.covariance_.astype(np.float64)

class RiemannMapper:
    # Maintains a Log-Euclidean reference (log-mean SPD). Tangent vectors computed at that Ref.
    def __init__(self, adapt_alpha: float = 0.0):
        self.L_ref = None
        self.adapt_alpha = adapt_alpha

    def set_reference_from_covs(self, covs: np.ndarray):
        # Log-Euclidean mean: average of matrix logs
        Ls = np.array([logm(C) for C in covs])
        self.L_ref = Ls.mean(axis=0)

    def tangent_vec(self, C: np.ndarray) -> np.ndarray:
        # v = log(Ref^{-1/2} C Ref^{-1/2}), vectorized upper-tri (√2 off-diags)
        Ref = expm(self.L_ref)
        d, U = eigh(Ref)
        d = np.clip(d, 1e-12, None)
        W = U @ np.diag(1.0/np.sqrt(d)) @ U.T
        S = W @ C @ W.T
        lam, V = eigh(S)
        lam = np.clip(lam, 1e-12, None)
        LogS = V @ np.diag(np.log(lam)) @ V.T
        c = LogS.shape[0]
        feat = np.zeros((c*(c+1)//2,), dtype=np.float32)
        k=0
        for i in range(c):
            for j in range(i, c):
                val = LogS[i, j]
                if i != j: val *= np.sqrt(2.0)
                feat[k] = float(val); k += 1
        return feat

    def adapt_with_cov(self, C: np.ndarray):
        if self.adapt_alpha <= 0: return
        L = logm(C)
        if self.L_ref is None: self.L_ref = L
        else: self.L_ref = (1.0 - self.adapt_alpha)*self.L_ref + self.adapt_alpha*L

def features_tangent(epoch_std: np.ndarray, mapper: RiemannMapper) -> np.ndarray:
    C = shrink_cov(epoch_std)
    return mapper.tangent_vec(C)

# ------------------------- Epoch extraction -------------------------
def best_jitter_epoch(ring: 'LSLBuffer', t0: float, fs: float) -> Optional[np.ndarray]:
    T = int(round(CFG.seg_sec * fs))
    for d in range(-CFG.jitter_samp, CFG.jitter_samp+1):
        t = t0 + d / fs
        w = ring.get_window(t, CFG.seg_sec)
        if w is None: continue
        tt, X = w
        if len(tt) >= T:
            return X[:T, :].T  # (C,T)
    return None

def extract_windows(ring, t0, fs, hop_sec=0.25, n_hops=4, use_baseline=True, mapper: Optional[RiemannMapper]=None):
    feats, covs = [], []
    for h in range(n_hops):
        t = t0 + h*hop_sec
        seg = best_jitter_epoch(ring, t, fs)
        if seg is None: continue
        mu_sd = None
        if use_baseline:
            mu, sd = baseline_stats(ring, t0, fs, bl_sec=0.5)
            if mu is not None: mu_sd = (mu.T, sd.T)  # to (C,1)
        xp = preprocess_epoch(seg, fs, mu_sd=mu_sd)
        C = shrink_cov(xp)
        covs.append(C)
        if mapper and mapper.L_ref is not None:
            feats.append(mapper.tangent_vec(C))
        else:
            L = logm(C); c = L.shape[0]
            f = np.zeros((c*(c+1)//2,), dtype=np.float32); k=0
            for i in range(c):
                for j in range(i, c):
                    val = L[i, j]
                    if i != j: val *= np.sqrt(2.0)
                    f[k] = float(val); k += 1
            feats.append(f)
    return np.array(feats, dtype=np.float32), covs

# ------------------------- Classifier -------------------------
def make_clf():
    base = LogisticRegression(penalty='l2', max_iter=2000, solver='lbfgs')
    pipe = Pipeline([('sc', StandardScaler(with_mean=True, with_std=True)),
                     ('logreg', base)])
    param_grid = {'logreg__C':[0.25, 0.5, 1.0, 2.0]}
    return GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, refit=True)

# ------------------------- Registration -------------------------
def do_registration(label: str, snd, outlet, ttl, ring, fs):
    duration_s = CFG.reg_duration_min * 60.0
    period = 1.0 / CFG.reg_beep_hz
    popup(f"Registration {label}",
          f"Starting registration for Motif {label}.\n"
          f"Please imagine at each beep.\nDuration: {CFG.reg_duration_min} min @ {CFG.reg_beep_hz:.2f} Hz")

    feats, ylab, covs_all = [], [], []
    next_t = time.time() + 1.0
    start_time = time.time()
    expected_beeps = int(duration_s / period)
    pbar = make_pbar(expected_beeps, f"Register {label}")
    used_beeps = 0

    while (time.time() - start_time) < duration_s:
        while time.time() < next_t: time.sleep(0.001)
        t_beep = do_beep_and_mark(snd, outlet, ttl, f"TRIG_{label}")

        F, covs = extract_windows(ring, t_beep, fs, hop_sec=0.25, n_hops=4, use_baseline=True, mapper=None)
        if F.size == 0:
            print(f"[Reg {label}] missed (no windows), retrying …")
            next_t = time.time() + period
            continue
        feats.append(F)
        covs_all.extend(covs)
        ylab.extend([0 if label=='A' else 1]*len(F))

        used_beeps += 1
        pbar.update(1)
        next_t = time.time() + period

    pbar.close()
    if len(feats) == 0:
        raise RuntimeError(f"No registration data captured for {label}")

    X = np.vstack(feats).astype(np.float32)
    y = np.array(ylab, dtype=np.int64)
    os.makedirs(CFG.out_dir, exist_ok=True)
    np.savez(os.path.join(CFG.out_dir, f"reg_{label}.npz"),
             X=X, y=y, fs=fs, rate=CFG.reg_beep_hz, duration_min=CFG.reg_duration_min)
    np.save(os.path.join(CFG.out_dir, f"reg_{label}_covs.npy"),
            np.stack(covs_all, axis=0))
    print(f"[Reg {label}] beeps={used_beeps}, windows={len(y)}")
    return X, y

# ------------------------- Live -------------------------
def entropy_of(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p*np.log(p)))

def live_loop(ring, fs, snd, outlet, ttl, clf, mapper: RiemannMapper):
    period = 1.0 / CFG.live_beep_hz
    popup("Live detection",
          "Starting live acquisition and detection.\n"
          "Please imagine A or B at each beep.\n"
          "Posterior-averaged over multiple short windows.")
    log_path = os.path.join(CFG.out_dir, CFG.save_log)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("t_beep,label,pA,pB,entropy\n")

    total_steps = int(CFG.live_max_minutes*60.0/period)
    pbar = make_pbar(total_steps, "Live")
    steps = 0
    next_t = time.time() + 2.0
    t_end = time.time() + CFG.live_max_minutes*60.0

    while time.time() < t_end and steps < total_steps:
        while time.time() < next_t: time.sleep(0.001)
        t_beep = do_beep_and_mark(snd, outlet, ttl, "TRIG_LIVE")

        label, pA, pB, ent = "UNK", 0.0, 0.0, 0.0
        F, covs = extract_windows(ring, t_beep, fs, hop_sec=0.25, n_hops=4, use_baseline=True, mapper=mapper)
        if F.size > 0:
            proba = clf.predict_proba(F).mean(axis=0)
            ent = entropy_of(proba)
            idx = int(np.argmax(proba))
            top = float(proba[idx])
            second = float(np.partition(proba, -2)[-2]) if proba.size > 1 else 0.0
            if (top >= CFG.conf_tau) and (top - second >= CFG.conf_margin) and (ent <= CFG.entropy_max):
                label = "B" if idx==1 else "A"
                # tiny EMA adaptation of reference on confident windows
                if len(covs) > 0 and mapper is not None and mapper.adapt_alpha > 0:
                    mapper.adapt_with_cov(covs[0])
            pA, pB = float(proba[0]), (float(proba[1]) if proba.size > 1 else 1.0-float(proba[0]))

        show_detection_popup(label, max(pA, pB))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{t_beep:.6f},{label},{pA:.4f},{pB:.4f},{ent:.4f}\n")

        steps += 1
        pbar.update(1)
        next_t = time.time() + period

    pbar.close()

# ------------------------- LSL resolution -------------------------
def resolve_eeg_inlet():
    streams = []
    if resolve_byprop is not None:
        if CFG.lsl_name:
            streams = resolve_byprop('name', CFG.lsl_name, timeout=5.0)
        else:
            streams = resolve_byprop('type', CFG.lsl_type, timeout=5.0)
    elif resolve_streams is not None:
        all_streams = resolve_streams(wait_time=5.0)
        if CFG.lsl_name:
            streams = [s for s in all_streams if s.name() == CFG.lsl_name]
        else:
            streams = [s for s in all_streams if s.type() == CFG.lsl_type]
    else:
        raise RuntimeError(
            "Your pylsl installation lacks resolve_byprop/resolve_streams. "
            "Upgrade pylsl, e.g. `pip install --upgrade pylsl`."
        )

    if not streams:
        raise RuntimeError("No LSL EEG stream found. Start your RDA→LSL bridge.")

    inlet = StreamInlet(streams[0], max_buflen=60, processing_flags=0)
    info = inlet.info()
    fs = info.nominal_srate()
    n_chan = info.channel_count()
    return inlet, fs, n_chan

# ------------------------- Main -------------------------
def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    popup("Welcome", "Welcome to the dynamic MI test!\n\n"
                     "Make sure your LSL EEG stream is running.\n"
                     "We will do 7 minutes A @0.25 Hz, then 7 minutes B @0.25 Hz.\n"
                     "Then live decisions.\n")

    inlet, fs, n_chan = resolve_eeg_inlet()
    print(f"Resolved EEG stream: fs={fs:.2f} Hz, channels={n_chan}")

    # Ring buffer + reader
    ring = LSLBuffer(fs=fs, n_chan=n_chan, seconds=CFG.ring_seconds)
    stop_event = threading.Event()
    reader = LSLReader(inlet, ring, stop_event)
    reader.start()

    # Audio + markers (TTL optional stub)
    snd = make_beep_sound(freq=CFG.beep_freq, ms=CFG.beep_ms, volume=CFG.volume)
    outlet = make_marker_outlet()
    ttl = None

    # Registration A then B
    XA, ya = do_registration('A', snd, outlet, ttl, ring, fs)
    XB, yb = do_registration('B', snd, outlet, ttl, ring, fs)

    # Build mapper from covariances gathered during registration
    covA = np.load(os.path.join(CFG.out_dir, 'reg_A_covs.npy'))
    covB = np.load(os.path.join(CFG.out_dir, 'reg_B_covs.npy'))
    mapper = RiemannMapper(adapt_alpha=CFG.ref_adapt_alpha)
    mapper.set_reference_from_covs(np.concatenate([covA, covB], axis=0))

    # Convert covs → tangent features with fixed mapper
    def covs_to_feats(covs):
        feats = [mapper.tangent_vec(C) for C in covs]
        return np.vstack(feats).astype(np.float32)

    XA = covs_to_feats(covA); XB = covs_to_feats(covB)

    # Train LogReg with tiny CV + (optional) isotonic calibration
    clf = make_clf()
    y_all = np.hstack([ya, yb])
    X_all = np.vstack([XA, XB])
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    clf.fit(Xtr, ytr)
    best = clf.best_estimator_ if hasattr(clf, 'best_estimator_') else clf
    cal = CalibratedClassifierCV(best, cv='prefit', method='isotonic')
    cal.fit(Xte, yte)
    clf = cal

    print("\nValidation (held-out) classification report:")
    print(classification_report(yte, clf.predict(Xte), digits=3))

    # Save model
    dump(clf, os.path.join(CFG.out_dir, CFG.save_model))
    print(f"Saved model → {os.path.join(CFG.out_dir, CFG.save_model)}")

    # Live detection (hands-free)
    live_loop(ring, fs, snd, outlet, ttl, clf, mapper=mapper)

    # Cleanup
    stop_event.set(); reader.join(timeout=1.0)
    print(f"Done. Outputs in: {CFG.out_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e, file=sys.stderr)
        sys.exit(1)

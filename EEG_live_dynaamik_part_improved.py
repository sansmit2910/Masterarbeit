# -*- coding: utf-8 -*-
"""
Option C++: actiCHamp via Recorder (RDA→LSL). Registrations for A,B with beeps,
Wavelet+SVM training, then 10 s live loop (last-trigger classification + popup).
Extras:
  - Optional LabRecorder control (start/stop recording) via CLI wrapper
  - Optional TTL pulse per beep (parallel-port or serial RTS/DTR)

You STILL run BrainVision Recorder yourself; this script never “drives” Recorder,
it only consumes the LSL EEG stream and emits markers/TTL.

Flip the flags in CONFIG to enable LabRecorder and/or TTL.
"""

import os, time, threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import messagebox as mb

from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump
import pywt
import pygame

# ============ CONFIG ============
@dataclass
class Config:
    # LSL stream selection
    lsl_type: str = "EEG"         # prefer match by 'type'
    lsl_name: Optional[str] = None  # or force by name (e.g., "BrainVisionRDA")

    # Buffers & timing
    ring_seconds: int = 120
    pull_timeout: float = 0.25
    seg_sec: float = 1.25
    jitter_samp: int = 60

    # Preprocessing (no downsample)
    band: tuple = (8.0, 30.0)
    notch_hz: Optional[float] = 50.0   # 50 or 60 if needed
    baseline_sec: float = 0.2

    # Registration pacing
    reg_beep_hz: float = 0.25
    reg_duration_min: float = 7.0
    reg_trials_A: int = 120
    reg_trials_B: int = 120

    # Live pacing
    live_beep_hz: float = 0.1    # every 10 s
    live_max_minutes: float = 30.0

    # UNK gate
    conf_tau: float = 0.55
    conf_margin: float = 0.05

    # Audio
    beep_freq: float = 880.0
    beep_ms: int = 120
    audio_sr: int = 44100

    # Output
    out_dir: str = "./bci_output"
    save_model: str = "ab_wavelet_svm.joblib"
    save_log: str = "live_log.csv"

    # ---------- OPTIONAL EXTRAS ----------
    # A) LabRecorder control (CLI wrapper). If your LabRecorder build supports CLI,
    #    set enable_labrecorder=True and point to the executable.
    enable_labrecorder: bool = False
    labrecorder_exe: str = r"C:\Program Files\LabRecorder\LabRecorder.exe"
    labrecorder_out_xdf: str = os.path.join(out_dir, "session.xdf")
    labrecorder_stream_hints: tuple = ("type=EEG", "type=Markers")  # which streams to capture

    # B) TTL trigger output per beep
    enable_ttl: bool = False
    # choose one backend: "parallel" OR "serial" OR "none"
    ttl_backend: str = "serial"  # "parallel" / "serial" / "none"
    # for parallel (requires InpOutx64.dll on Windows)
    lpt_baseaddr: int = 0x378    # classic LPT1 base
    # for serial (pyserial): toggle RTS or DTR for a short pulse
    serial_port: str = "COM5"
    serial_use_rts: bool = True  # True=RTS, False=DTR
    ttl_pulse_ms: int = 8        # pulse width

CFG = Config()
os.makedirs(CFG.out_dir, exist_ok=True)

# ============ Audio & Popups ============
def init_audio():
    pygame.mixer.pre_init(frequency=CFG.audio_sr, size=-16, channels=1, buffer=512)
    pygame.init()
    t = np.linspace(0, CFG.beep_ms/1000.0, int(CFG.audio_sr*CFG.beep_ms/1000.0), endpoint=False)
    wave = 0.5*np.sin(2*np.pi*CFG.beep_freq*t) * np.linspace(1, 0, len(t))
    snd = pygame.sndarray.make_sound((wave*32767).astype(np.int16))
    return snd

def popup(title, text):
    root = tk.Tk(); root.withdraw()
    mb.showinfo(title, text)
    root.destroy()

def show_detection_popup(label: str, pmax: float):
    root = tk.Tk(); root.withdraw()
    mb.showinfo("Detected motif", f"Detected: {label}  (p≈{pmax:.2f})")
    root.destroy()

# ============ LSL Marker outlet ============
def make_marker_outlet():
    info = StreamInfo(name="ExpMarkers", type="Markers", channel_count=1,
                      nominal_srate=0, channel_format='string', source_id='exp_markers')
    return StreamOutlet(info)

# ============ Optional: LabRecorder Control ============
# NOTE: LabRecorder GUI usually lacks a stable API. Some builds expose a CLI
# that accepts output path and stream filter arguments. We provide a best-effort
# wrapper; if your build lacks CLI, set enable_labrecorder=False and start it manually.
import subprocess, shlex

class LabRecorderCtl:
    def __init__(self, exe, out_xdf, stream_hints):
        self.exe = exe
        self.out = out_xdf
        self.hints = stream_hints
        self.p = None

    def start(self):
        if not os.path.isfile(self.exe):
            print("[LabRecorder] Executable not found; skipping.")
            return
        if self.p is not None:
            print("[LabRecorder] Already running.")
            return
        # Common CLI patterns differ; try the simplest: LabRecorder.exe <outfile> <hint1> <hint2> ...
        # If your build requires different flags, adjust here.
        args = [self.exe, self.out] + list(self.hints)
        print(f"[LabRecorder] Starting: {' '.join(shlex.quote(a) for a in args)}")
        try:
            self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"[LabRecorder] Launch failed: {e}")
            self.p = None

    def stop(self):
        if self.p is None:
            return
        print("[LabRecorder] Stopping ...")
        try:
            self.p.terminate()
            self.p.wait(timeout=5)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        self.p = None

# ============ Optional: TTL backends ============
class TTLBackend:
    def pulse(self, ms: int = 5):
        pass

class TTLNone(TTLBackend):
    def pulse(self, ms: int = 5):
        return

class TTLSerial(TTLBackend):
    def __init__(self, port: str, use_rts=True):
        import serial
        self.ser = serial.Serial(port=port, baudrate=115200, timeout=0)
        self.use_rts = use_rts
        # ensure low
        if use_rts: self.ser.rts = False
        else:       self.ser.dtr = False
    def pulse(self, ms: int = 5):
        if self.use_rts:
            self.ser.rts = True; time.sleep(ms/1000.0); self.ser.rts = False
        else:
            self.ser.dtr = True; time.sleep(ms/1000.0); self.ser.dtr = False

class TTLParallel(TTLBackend):
    def __init__(self, baseaddr: int):
        # Requires InpOutx64.dll in PATH or working dir, Windows only
        import ctypes
        try:
            self.io = ctypes.windll.inpoutx64
        except Exception as e:
            print(f"[TTLParallel] InpOutx64 not available: {e}")
            self.io = None
        self.base = baseaddr
    def pulse(self, ms: int = 5):
        if self.io is None: return
        try:
            self.io.Out32(self.base, 0xFF)  # all bits high
            time.sleep(ms/1000.0)
            self.io.Out32(self.base, 0x00)
        except Exception as e:
            print(f"[TTLParallel] pulse failed: {e}")

def make_ttl_backend():
    if not CFG.enable_ttl:
        return TTLNone()
    if CFG.ttl_backend.lower() == "serial":
        try:
            return TTLSerial(CFG.serial_port, use_rts=CFG.serial_use_rts)
        except Exception as e:
            print(f"[TTL] Serial init failed: {e}. Continuing without TTL.")
            return TTLNone()
    if CFG.ttl_backend.lower() == "parallel":
        try:
            return TTLParallel(CFG.lpt_baseaddr)
        except Exception as e:
            print(f"[TTL] Parallel init failed: {e}. Continuing without TTL.")
            return TTLNone()
    return TTLNone()

# ============ Beep + Markers + TTL ============
def do_beep_and_mark(sound, outlet: StreamOutlet, ttl: TTLBackend, label: str):
    ts = local_clock()
    sound.play()
    outlet.push_sample([label], timestamp=ts)
    ttl.pulse(CFG.ttl_pulse_ms)
    return ts

# ============ LSL Ring Buffer ============
class LSLBuffer:
    def __init__(self, max_seconds, n_chan, fs):
        self.n_chan = n_chan
        self.fs = fs
        self.max_samples = int(max_seconds * fs)
        self.samples = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
    def append(self, chunk: np.ndarray, stamps: np.ndarray):
        if chunk.size == 0: return
        n = chunk.shape[0]
        rows = np.concatenate([stamps.reshape(n,1), chunk], axis=1)
        with self.lock:
            for r in rows: self.samples.append(r)
    def get_window(self, t0, dur_sec):
        t1 = t0 + dur_sec
        with self.lock:
            if not self.samples: return None
            arr = np.array(self.samples)
        ts = arr[:,0]
        m = (ts >= t0) & (ts < t1)
        if not np.any(m): return None
        w = arr[m]
        return w[:,0], w[:,1:]

class LSLReader(threading.Thread):
    def __init__(self, inlet: StreamInlet, ring: LSLBuffer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.inlet = inlet; self.ring = ring; self.stop_event = stop_event
    def run(self):
        while not self.stop_event.is_set():
            chunk, stamps = self.inlet.pull_chunk(timeout=CFG.pull_timeout, max_samples=1024)
            if stamps:
                self.ring.append(np.array(chunk, np.float32), np.array(stamps, np.float64))

# ============ Preprocessing ============
def butter_bp(low, high, fs, order=4):
    from scipy.signal import butter
    ny = 0.5*fs
    b,a = butter(order, [low/ny, high/ny], btype='bandpass')
    return b,a

def notch_coefs(fs, f0, q=30.0):
    return iirnotch(w0=f0/(fs/2.0), Q=q)

def preprocess_epoch(epoch: np.ndarray, fs: float) -> np.ndarray:
    x = epoch.copy()
    C,T = x.shape
    b,a = butter_bp(CFG.band[0], CFG.band[1], fs)
    x = filtfilt(b,a,x,axis=1)
    if CFG.notch_hz:
        bn,an = notch_coefs(fs, CFG.notch_hz)
        x = filtfilt(bn,an,x,axis=1)
    if mu_sd is not None and mu_sd[0] is not None:
        mu, sd = mu_sd
        return (x - mu) / sd
    bl = max(1, min(int(round(CFG.baseline_sec*fs)), T))
    mu = x[:,:bl].mean(axis=1, keepdims=True)
    sd = x[:,:bl].std(axis=1, keepdims=True) + 1e-6
    return (x - mu)/sd

def wavelet_features(epoch: np.ndarray, wavelet='db4') -> np.ndarray:

from scipy.signal import welch

def _bandpower(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, x.shape[-1]))
    m = (f >= fmin) & (f <= fmax)
    # sum over selected freqs
    return np.log(Pxx[..., m].sum(axis=-1) + 1e-12)

def mi_band_features(epoch: np.ndarray, fs: float) -> np.ndarray:
    # epoch shape (C, T)
    mu = _bandpower(epoch, fs, 8.0, 13.0)     # (C,)
    beta = _bandpower(epoch, fs, 14.0, 30.0)  # (C,)
    ratio = beta - mu  # log(beta/mu)
    return np.concatenate([mu, beta, ratio], axis=0).astype(np.float32)

def features_all(epoch: np.ndarray, fs: float) -> np.ndarray:
    # Concatenate wavelet energies + MI bandpowers
    w = wavelet_features(epoch)
    b = mi_band_features(epoch, fs)
    return np.concatenate([w, b], axis=0).astype(np.float32)
    C,T = epoch.shape
    w = pywt.Wavelet(wavelet)
    max_lvl = min(6, pywt.dwt_max_level(T, w.dec_len))
    feats=[]
    for ci in range(C):
        coeffs = pywt.wavedec(epoch[ci], wavelet, level=max_lvl, mode='symmetric')
        energies = [np.log(np.sum(c**2)+1e-12) for c in coeffs]
        feats.extend(energies)
    return np.asarray(feats, np.float32)

# ============ Epoch extraction ============

def extract_windows(ring, t0, fs, win_sec=None, hop_sec=0.25, n_hops=4, use_baseline=True):
    win_sec = win_sec or CFG.seg_sec
    feats = []
    for h in range(n_hops):
        t = t0 + h*hop_sec
        seg = best_jitter_epoch(ring, t, fs)
        if seg is None: 
            continue
        mu_sd = None
        if use_baseline:
            mu, sd = baseline_stats(ring, t0, fs, bl_sec=0.5)
            if mu is not None: 
                mu_sd = (mu.T, sd.T)  # match (C,1) broadcasting
        xp = preprocess_epoch(seg, fs, mu_sd=mu_sd)
        feats.append(features_all(xp, fs))
    return np.array(feats, dtype=np.float32)  # (N, F)

def best_jitter_epoch(ring: LSLBuffer, t_trigger: float, fs: float):
    T = int(round(CFG.seg_sec*fs))
    best = None
    for s in range(-CFG.jitter_samp, CFG.jitter_samp+1):
        t0 = t_trigger + s/fs
        w = ring.get_window(t0, CFG.seg_sec)
        if w is None: continue
        ts, x = w
        if x.shape[0] >= T:
            best = x[:T,:].T  # (C,T)
            break
    return best

# ============ Registration ============
def do_registration(label: str, n_trials: int, snd, outlet, ttl, ring, fs):
    # n_trials is ignored; we run by duration (CFG.reg_duration_min) at CFG.reg_beep_hz
    # and collect as many usable trials as fit in the duration, repeating missed ones.
    duration_s = CFG.reg_duration_min * 60.0
    target_rate = CFG.reg_beep_hz
    period = 1.0 / target_rate
    start_time = time.time()
    popup(f"Registration {label}",
          f"Starting registration for Motif {label}.\\n"
          f"Please imagine at each beep.\\nDuration: {CFG.reg_duration_min} min @ {target_rate:.2f} Hz")
    feats=[]; y=[]
    next_t = time.time() + 1.0
    k = 0
    while (time.time() - start_time) < duration_s:
        while time.time() < next_t: time.sleep(0.001)
        t_beep = do_beep_and_mark(snd, outlet, ttl, f"TRIG_{label}")
        # Multi-window extraction for robustness / slight data augmentation
        F = extract_windows(ring, t_beep, fs, hop_sec=0.25, n_hops=4, use_baseline=True)
        if F.size == 0:
            print(f"[Reg {label}] missed (no windows), retrying ...")
            next_t = time.time() + period
            continue
        for row in F:
            feats.append(row)
            y.append(0 if label=='A' else 1)
        k += 1
        next_t = time.time() + period
    if not feats:
        raise RuntimeError(f"No registration data captured for {label}")
    X = np.vstack(feats).astype(np.float32)
    y = np.array(y, np.int64)
    np.savez(os.path.join(CFG.out_dir, f"reg_{label}.npz"), X=X, y=y, fs=fs, rate=target_rate, duration_min=CFG.reg_duration_min)
    print(f"[Reg {label}] collected trials: {k}, windows: {len(y)}")
    return X,y
    period = 1.0 / CFG.reg_beep_hz
    popup(f"Registration {label}",
          f"Starting registration for Motif {label}.\n"
          f"Please imagine at each beep.\nTrials: {n_trials}, rate: {CFG.reg_beep_hz:.2f} Hz.")
    feats=[]; y=[]
    next_t = time.time() + 1.0
    for k in range(1, n_trials+1):
        while time.time() < next_t: time.sleep(0.001)
        t_beep = do_beep_and_mark(snd, outlet, ttl, f"TRIG_{label}")
        seg = best_jitter_epoch(ring, t_beep, fs)
        if seg is None:
            print(f"[Reg {label}] missed #{k}, retrying ...")
            next_t = time.time() + period
            continue
        xp = preprocess_epoch(seg, fs)
        feats.append(wavelet_features(xp))
        y.append(0 if label=="A" else 1)
        print(f"[Reg {label}] {k:03d} captured.")
        next_t += period
    X = np.vstack(feats).astype(np.float32)
    y = np.array(y, np.int64)
    np.savez(os.path.join(CFG.out_dir, f"reg_{label}.npz"), X=X, y=y, fs=fs)
    return X,y

# ============ Classifier ============
def make_clf():
    base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=0)
    pipe = Pipeline([('sc', StandardScaler()), ('svm', base)])
    param_grid = {'svm__C':[0.5,1,2,4], 'svm__gamma':['scale', 0.01, 0.001]}
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, refit=True)
    return gs

# ============ Live loop ============
def live_loop(ring, fs, snd, outlet, ttl, clf):
    period = 1.0 / CFG.live_beep_hz
    popup("Live detection",
          "Starting live acquisition and detection.\n"
          "Please imagine A or B at each beep.\n"
          "Only the latest trigger window is classified.")
    log_path = os.path.join(CFG.out_dir, CFG.save_log)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("t_beep,label,pmax\n")
    t_end = time.time() + CFG.live_max_minutes*60.0
    next_t = time.time() + 2.0
    while time.time() < t_end:
        while time.time() < next_t: time.sleep(0.001)
        t_beep = do_beep_and_mark(snd, outlet, ttl, "TRIG_LIVE")
        seg = best_jitter_epoch(ring, t_beep, fs)
        label, pmax = "UNK", 0.0
        if seg is not None:
            xp = preprocess_epoch(seg, fs)
            feat = wavelet_features(xp)[None,:]
            proba = clf.predict_proba(feat)[0]
            idx = int(np.argmax(proba))
            top, second = float(proba[idx]), float(np.partition(proba, -2)[-2])
            if (top >= CFG.conf_tau) and (top - second >= CFG.conf_margin):
                label = "B" if idx==1 else "A"
            pmax = top
        show_detection_popup(label, pmax)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{t_beep:.6f},{label},{pmax:.4f}\n")
        print(f"[Live] {label}, p={pmax:.3f}")
        next_t += period

# ============ Main ============
def main():
    popup("Welcome", "Welcome to the dynamic phase of my master’s thesis project! :)")

    # Resolve EEG stream
    print("Resolving LSL EEG stream ...")
    streams = resolve_stream('name', CFG.lsl_name) if CFG.lsl_name else resolve_stream('type', CFG.lsl_type)
    if not streams: raise RuntimeError("No LSL EEG stream found. Is RDA + LSL bridge running?")
    inlet = StreamInlet(streams[0], max_buflen=CFG.ring_seconds*2)
    fs = inlet.info().nominal_srate()
    n_chan = inlet.info().channel_count()
    print(f"Connected to '{inlet.info().name()}', fs≈{fs} Hz, C={n_chan}")

    # Start ring + reader
    ring = LSLBuffer(CFG.ring_seconds, n_chan, fs)
    stop_event = threading.Event()
    reader = LSLReader(inlet, ring, stop_event); reader.start()

    # Marker, TTL, Audio
    outlet = make_marker_outlet()
    ttl = make_ttl_backend()
    snd = init_audio()

    # Optional: start LabRecorder
    lr = None
    if CFG.enable_labrecorder:
        lr = LabRecorderCtl(CFG.labrecorder_exe, CFG.labrecorder_out_xdf, CFG.labrecorder_stream_hints)
        lr.start()
        time.sleep(1.5)

    # Registration A then B
    XA,yA = do_registration("A", CFG.reg_trials_A, snd, outlet, ttl, ring, fs)
    XB,yB = do_registration("B", CFG.reg_trials_B, snd, outlet, ttl, ring, fs)

    # Train
    clf = make_clf(); clf.fit(np.vstack([XA,XB]), np.concatenate([yA,yB]))
    print("[Train] In-sample report:\n", classification_report(np.concatenate([yA,yB]),
                                                               clf.predict(np.vstack([XA,XB])),
                                                               target_names=["A","B"], zero_division=0, digits=3))
    dump(clf, os.path.join(CFG.out_dir, CFG.save_model))

    # Live detection
    live_loop(ring, fs, snd, outlet, ttl, clf)

    # Cleanup
    if lr: lr.stop()
    stop_event.set(); reader.join(timeout=1.0)
    print(f"Done. Outputs in: {CFG.out_dir}")

if __name__ == "__main__":
    main()

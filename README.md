# Feasibility Assessment of EEG-Based Motor Intention Inference Algorithms for Real-Time Rehabilitation Robotics

**Master's thesis, M.Sc. Biomedical Engineering, TH Lübeck** · Smit Khandelwal
Submitted 02.07.2026 · Supervisors: Prof. Dr.-Ing. Ulf Pilz (TH Lübeck), Prof. Dr. rer. nat. habil. Jonas Obleser (University of Lübeck, CBBM)

This repository contains the **Python code** behind the thesis *"Feasibility Assessment of EEG-Based Motor Intention Inference Algorithms for Real-Time Rehabilitation Robotics"*. It covers the full pipeline: cue/trigger delivery for EEG acquisition, pre-processing, ERD/ERS verification, offline benchmarking of nine decoding algorithms, and a real-time decoding experiment over Lab Streaming Layer (LSL).

> **Scope of this repository.** Only the scripts are published here. The EEG recordings (human data from a single volunteer), the electrode montage file, generated figures/outputs and the thesis document are **not** included. See [Data availability](#8-data-availability).

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Key results](#2-key-results)
3. [Experimental design](#3-experimental-design)
4. [Repository structure](#4-repository-structure)
5. [Script map](#5-script-map)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Data availability](#8-data-availability)
9. [Limitations](#9-limitations)
10. [Future work](#10-future-work)
11. [Citation](#11-citation)
12. [Acknowledgements](#12-acknowledgements)
13. [License and contact](#13-license-and-contact)

---

## 1. Project summary

**Motivation.** In severe stroke or spinal cord injury, patients who cannot generate any limb movement receive passive mobilisation, often combined with motor imagery (MI) in the *Combined Observation and Motor Imagery* (COMI) paradigm. Current rehabilitation robots mostly assist patients who can already generate some force. An EEG-based brain–computer interface (BCI) that detects the patient's *will-to-move* and triggers passive limb mobilisation at that moment could close this gap and exploit Hebbian-type plasticity through contingent intention and movement.

**Research question.** *Can EEG reliably decode a person's motor intention ("will-to-move") to trigger corresponding robotic movement in rehabilitation robotics systems?*

**Approach.** Because patient recruitment was not possible, MI in a single neurologically intact participant served as a surrogate for the will-to-move of a paralysed patient, following the single-subject proof-of-concept tradition in BCI research. Nine algorithms of ascending computational complexity were benchmarked offline (*Static* phase) across four training/inference configurations. The most suitable algorithm was then run in a real-time pipeline (*Dynamic* phase).

| Tier | Algorithm |
|---|---|
| Low | Template Matching / Normalised Cross-Correlation (NCC) |
| Low | Dynamic Time Warping (DTW) |
| Low | PCA + LDA |
| Medium | Common Spatial Patterns (CSP) + SVM |
| Medium | Wavelet Transform + SVM |
| Medium | Riemannian geometry-based classifier (covariance → tangent space → logistic regression) |
| High | EEGNet (compact CNN) |
| High | 1D-CNN + LSTM |
| High | Time-series Transformer |

---

## 2. Key results

All figures come from a **single participant** and are feasibility evidence, not clinical performance estimates.

**Static phase: inference accuracy per configuration**

| Algorithm | Case 1: A–D → movement mix | Case 2: A,B → movement mix | Case 3: C,D → MI mix | Case 4: C,D → movement mix | Mean runtime | Case 3 acc./runtime (%/s) |
|---|---|---|---|---|---|---|
| Template Matching / NCC | 54.75% | 55.61% | 49.59% | 3.98% | 6.56 s | 7.56 |
| DTW | 47.36% | 53.51% | 49.36% | 1.42% | 7.54 s | 6.55 |
| PCA + LDA | 23.32% | 54.78% | 49.79% | 0% | 5.67 s | 8.78 |
| CSP + SVM | 23.87% | 51.82% | 50.85% | 0% | 2.60 s | 19.57 |
| **Wavelet + SVM** | 16.67% | **71.84%** | **56.54%** | 0% | 10.76 s | 5.25 |
| Riemannian | 50.9% | 51.45% | 50.64% | 2.1% | 28.76 s | 1.76 |
| EEGNet | 38.7% | 51.96% | 50.28% | 1.3% | 98.66 s | 0.51 |
| 1D-CNN + LSTM | 28.32% | 53.93% | 50.94% | 0% | 89.95 s | 0.57 |
| Transformer | 41.9% | 54.74% | 51.45% | 0% | 120.15 s | 0.43 |

- **Chance level** was set at 33% (two target classes plus an `UNK` rejection output). In a one-sided binomial test on Case 3, all nine algorithms were significantly above chance (all p < 0.0001). Wavelet + SVM had the highest z-statistic (z = 7.59).
- **Cross-modal transfer failed for every algorithm.** Models trained on MI and applied to overt movement (Case 4) collapsed to 0–4% accuracy regardless of architecture. The thesis interprets this as a neurophysiological mismatch between MI- and execution-related ERD, not an algorithmic weakness.
- **Algorithm selection.** Wavelet + SVM was chosen for the Dynamic phase because of its absolute Case 3 accuracy combined with near-real-time runtime. PCA + LDA had a better accuracy/runtime ratio but was indistinguishable from chance on the two-class problem.

**Dynamic phase (real-time LSL pipeline, static decoder)**

| Block | Accuracy | Longest streak of consecutive correct inferences |
|---|---|---|
| Events 1–100 | 43% | 3 |
| Events 101–200 | 63% | 5 |

The 20-percentage-point gain (two-proportion z-test, z = 2.83, p = 0.002) occurred while the decoder was *not* updated, which the thesis attributes to progressive subject adaptation.

**Signal quality / verification.** Pre-processing showed effective 50 Hz suppression, successful ICA decomposition and 0% dropped epochs. ERD/ERS in the µ/α and β bands at C3/Cz/C4 for the overt-movement datasets showed the expected sensorimotor desynchronisation after trigger onset.

---

## 3. Experimental design

**Hardware / software.** BrainVision ActiChamp Plus (32-channel cap, 10–20 layout, 1000 Hz acquisition, later down-sampled to 250 Hz), BrainVision Recorder, TTL triggers over a parallel port, and Python (MNE-Python, scikit-learn, PyTorch, pyRiemann, PyWavelets). Lab Streaming Layer was used for the real-time phase. Data were acquired at the Center of Brain, Behavior and Metabolism (CBBM), University of Lübeck.

**Datasets (one participant, left hand)**

| ID | Content | Duration / trials |
|---|---|---|
| **A** | Audio-cued overt movement 1: opening/closing the hand | 6 min, ≈ 238 trials |
| **B** | Audio-cued overt movement 2: index-finger/thumb pinch | 6 min, ≈ 238 trials |
| **C** | Audio-cued motor imagery of movement 1 | 6 min, ≈ 238 trials |
| **D** | Audio-cued motor imagery of movement 2 | 6 min, ≈ 238 trials |
| Movement mix | Participant freely chose A or B at each cue; sequence verified by video | 3 min |
| MI mix | Participant alternated the imagined movements in a predetermined order | 3 min |

Cues were 50 ms, 1 kHz clicks at 0.66 Hz (one every ≈ 1.67 s), each time-locked to a TTL marker.

**Four training → inference configurations**

| Case | Training data | Inference target |
|---|---|---|
| 1 | A, B, C, D | Movement mix |
| 2 | A, B (movement) | Movement mix |
| 3 | C, D (MI) | MI mix, the case closest to the intended clinical use and the Dynamic phase |
| 4 | C, D (MI) | Movement mix, the cross-modal transfer test |

**Common decoding pipeline (all nine algorithms)**

1. Virtual Laplacian derivations over the motor cortex (C3L, CzL, C4L)
2. Baseline z-score normalisation from the pre-annotation segment
3. Trigger-aligned epoch extraction (≈ 0.5–1.0 s post-event)
4. Training under one of the four configurations
5. Low-confidence rejection mapped to an `UNK` label (posterior, margin or score-calibration thresholds, depending on the algorithm)

Most classifiers decode trigger-locked, with a small ± temporal jitter scan around each trigger, and keep the most confident shift.

**Pre-processing (MNE-Python).** Custom montage → 50 Hz notch → ICA (15 components, fitted on a 1–40 Hz copy; EOG components identified via AF7) → common average reference → resample to 250 Hz → save as `.fif`.

**ERD verification.** Epochs from −1 s to +2 s around each trigger, baseline −1…0 s, band-wise ERD% = 100·(P − P_baseline)/P_baseline in α (8–12 Hz), µ (8–13 Hz), β1 (12–20 Hz) and β2 (20–30 Hz), with Savitzky–Golay smoothing.

**Dynamic phase.** Raw EEG was streamed via BrainVision Remote Data Access → LSL.
- Epoching and filtering: −0.5…0 s pre-cue baseline, 8–30 Hz band-pass, 50 Hz notch, virtual Laplacian channels.
- Features: 1.25 s windows (4 overlapping windows per cue, 0.25 s hop) with Daubechies-4 wavelet log-energies, Welch µ/β band power and the β/µ ratio.
- Classifier: RBF-SVM (grid-searched C, γ) with isotonic calibration, and `UNK` gating on confidence, margin and entropy.
- Registration used beeps at 0.25 Hz for 7 minutes per class (≈ 105 cues per class). Live inference used beeps at 0.2 Hz.

---

## 4. Repository structure

```
Masterarbeit/
├── README.md
├── trigger_for_acquisition.py          # Cue (click) + TTL trigger generator
├── trigger_for_acquisition_v2.0.py
├── pre_processing_initial_aufnahmen_*.py   # Pre-processing script iterations
├── ERD_Visualization*.py               # ERD/ERS verification
├── raw_vs_smoothed_comparison_bands.py
├── visualize_eeg.py
│
├── final_phase_tests/                  # FINAL scripts behind the thesis results
│   ├── <Algorithm>_register_*_detect_*.py      # One script per algorithm × case
│   ├── Wavelet_transform+SVM_permutation_learning_model.py
│   ├── EEG_live_dynamik_part_wavelet+SVM.py    # Dynamic-phase pipeline
│   └── check_lsl_stream.py
│
├── step_0/                             # Early pattern-search prototypes
│
└── (root) exploratory scripts          # CNN_attempt_*, DTW_attempt_1, eeg_motif_detector*,
                                        # EEG_Pattern_Classification_*_trials_*, etc.
```

> **Note on iterations.** The repository preserves the working history of the project. The exploratory scripts in the repository root (`CNN_attempt_*.py`, `DTW_attempt_1.py`, `eeg_motif_detector*.py`, `claude_code_attempt_*.py`, `EEG_Pattern_Classification_using_Multiple_ML_models_trials_*.py`, `train+detect_…attempt_1.py`) are **not** the source of the numbers in the thesis. The thesis results come from the scripts in `final_phase_tests/`.

---

## 5. Script map

### 5.1 Acquisition
| Script | Purpose |
|---|---|
| `trigger_for_acquisition.py`, `trigger_for_acquisition_v2.0.py` | Plays a 50 ms, 1 kHz click (pygame) and sends a 2 ms TTL pulse (value 1, parallel port `0xCFF8` via PsychoPy) at 0.66 Hz, so each auditory cue has a matching hardware marker in BrainVision Recorder. |

### 5.2 Pre-processing
| Script | Purpose |
|---|---|
| `pre_processing_initial_aufnahmen_1…6.py` | Iterations of the MNE pipeline (montage → notch → ICA → CAR → 250 Hz → `.fif`). The highest-numbered version is the most recent. |

### 5.3 Verification (ERD/ERS)
| Script | Purpose |
|---|---|
| `ERD_Visualization.py`, `ERD_Visualization_3_bands.py`, `ERD_Visualization_3_bands_smoothed.py` | Band-wise ERD% curves at C3/Cz/C4, time-locked to the trigger. |
| `raw_vs_smoothed_comparison_bands.py` | Raw vs. Savitzky–Golay-smoothed ERD curves. |
| `visualize_eeg.py` | General signal inspection. |

### 5.4 Static phase: training + inference

Script names in `final_phase_tests/` encode the case: `register_4` → Case 1, `register_2_movement_detect_movement_mix` → Case 2, `register_2_MI_detect_MI_mix` → Case 3. Case 4 is the cross-modal test described in the thesis.

| Algorithm | Scripts (`final_phase_tests/`) |
|---|---|
| Template Matching / NCC | `Template_Matching+CrossCorrelation_movement_mix_all_{4_registered, 2_movements_registered, 2_MI_registered}.py` (+ `trial_2_…` variant) |
| DTW | `DTW_all_4_registered_test_movement_mix.py`, `DTW_all_2-movements_registered_test_movement_mix.py`, `DTW_all_2-MI_registered_test_MI_mix.py` |
| PCA + LDA | `PCA_LDA_4_registered_target_movement_mix*.py` (incl. preferred-hierarchy A/B and lower-D-bias variants), `PCA_LDA_2_registered_target_movement_mix.py`, `PCA_LDA_2_registered_MI__target_MI_mix.py` |
| CSP + SVM | `Common_Spatial_Patterns(CSP)+SVM_register_{4_detect_movement_mix, 2_movement_detect_movement_mix, 2_MI_detect_MI_mix}.py` |
| Wavelet + SVM | `Wavelet_Transform+SVM_register_{4, 2_movement, 2_MI}_….py` (+ `_corrected_working` versions), `Wavelet_transform+SVM_permutation_learning_model.py` |
| Riemannian | `Riemannian Geometry-Based Classifier_register_{4, 2_movement, 2_MI}_…_improved_algorithm_adapted_better_results.py` |
| EEGNet | `eeg_net_cnn_register_{4_detect_movement_mix, 2_movement_detect_movement_mix, 2_MI_detect_MI_mix}.py` (+ improved variants) |
| 1D-CNN + LSTM | `1D_CNN+LTSM_register_{4_detect_movement_mix, 2_movement_detect_movement_mix, 2_MI_detect_MI_mix}.py` |
| Transformer | `Transformer(time-series_invariante)*.py` |
| Custom algorithm | `my_algorithm_4_registered_detect_movement_mix.py` |

Each script writes the recognised event sequence (`.txt`/`.csv`), per-label counts (including `UNK`) and, depending on the method, diagnostic plots (confusion matrices, jitter histograms, posterior traces, training curves, CSP patterns).

### 5.5 Dynamic phase (real-time)
| Script | Purpose |
|---|---|
| `check_lsl_stream.py` | Checks that the BrainVision → LSL EEG stream is discoverable and flowing. |
| `EEG_live_dynamik_part_wavelet+SVM.py` | The pipeline used in the thesis: registration of C/D, SVM training, then live cue-locked classification with pop-up and CSV logging. |
| `EEG_live_dynaamik_part*.py`, `EEG_live_dynamik_part_riemmanian+logistic_regression.py`, `EEG_dynamic_riemann_logreg_auto.py` | Earlier/alternative live pipelines (not used for the reported results). |

---

## 6. Installation

Python 3.10+ is recommended. Install the main dependencies, for example:

```bash
git clone https://github.com/sansmit2910/Masterarbeit.git
cd Masterarbeit
pip install mne numpy scipy scikit-learn pyriemann PyWavelets torch matplotlib pandas tqdm pygame pylsl
# only needed for the acquisition/trigger scripts (parallel-port access):
pip install psychopy
```

The scripts were developed on Windows. The acquisition/trigger scripts require the BrainVision hardware setup and a parallel port.

Paths in the scripts are currently absolute Windows paths (e.g. `C:\Users\User\Desktop\masters_thesis_directory\…`). Adjust the path constants at the top of each script to your own data location before running.

---

## 7. Usage

Because the recordings are not part of this repository, the scripts are intended as a **reference implementation**. To reproduce the pipeline on your own BrainVision recordings (`.vhdr/.vmrk/.eeg`) acquired with the same paradigm:

1. Acquire the cued datasets (A–D plus the mix files) with `trigger_for_acquisition.py` running alongside BrainVision Recorder.
2. Pre-process with the latest `pre_processing_initial_aufnahmen_*.py`. This needs an electrode montage file for your cap and writes `.fif` files.
3. Verify the data quality with `ERD_Visualization_3_bands_smoothed.py`.
4. Run an algorithm script from `final_phase_tests/`, for example:
   ```bash
   python "final_phase_tests/Wavelet_Transform+SVM_register_2_MI_detect_MI_mix.py"
   ```
5. Compare the produced sequence against your ground truth (video-annotated for the movement mix, the instructed alternating order for the MI mix).

**Real-time pipeline** (requires BrainVision Recorder with Remote Data Access and an LSL bridge):

```bash
python final_phase_tests/check_lsl_stream.py
python "final_phase_tests/EEG_live_dynamik_part_wavelet+SVM.py"
```

---

## 8. Data availability

The EEG recordings come from a **single volunteer** who gave informed consent for use in this thesis. To protect participant privacy, **no raw or pre-processed recordings (and no video) are published** in this repository. The electrode montage file and generated figures are also not included. If you are working with similar data, the scripts expect BrainVision-format recordings and a custom montage file.

---

## 9. Limitations

- **Single participant**, neurologically intact. Results do not generalise to patient populations.
- **Small training sets** (≈ 6 min / ≈ 238 trials per class), which particularly disadvantages the deep-learning models.
- **Static decoder in the Dynamic phase.** The accuracy gain reflects subject adaptation, not model learning.
- **Ground truth for the MI mix** was an instructed alternating sequence, so actual compliance on every trial cannot be verified. Movement-mix ground truth came from video annotation.
- **ERD verification** was performed on the overt-movement datasets (A, B) only. The MI datasets (C, D) were not quantified.
- Laboratory setting only. Movement artefacts, electrode displacement and clinical conditions were not evaluated.
- Absolute accuracies (≈ 50–57% in the MI case) are above chance but well below the levels typically needed for clinical-grade control.

---

## 10. Future work

1. Multi-subject validation (N ≥ 20), including chronic stroke patients.
2. Larger training sets (≥ 200 trials per class) and validation on public MI benchmarks (BCI Competition IV 2a, PhysioNet EEG Motor Movement/Imagery).
3. Adaptive online decoding (e.g. continuously updated Riemannian covariance estimation or fine-tuned transformers) plus filter-bank / tangent-space alignment to reduce jitter-induced misregistration.
4. End-to-end latency profiling (target < ≈ 300 ms from acquisition to actuation).
5. Closed-loop robotic feedback, so that correct detections trigger real passive limb mobilisation.
6. Cross-session evaluation (≥ 3 sessions over ≥ 2 weeks) and explicit domain adaptation between MI and overt movement.

---

## 11. Citation

```bibtex
@mastersthesis{khandelwal2026feasibility,
  author  = {Khandelwal, Smit},
  title   = {Feasibility Assessment of EEG-Based Motor Intention Inference Algorithms for Real-Time Rehabilitation Robotics},
  school  = {Technische Hochschule Lübeck},
  year    = {2026},
  month   = {7}
}
```

---

## 12. Acknowledgements

Thanks to the Center of Brain, Behavior and Metabolism (CBBM) and Prof. Dr. Jonas Obleser at the University of Lübeck for access to the EEG equipment and laboratory and for guidance on the neurophysiological foundations and experimental framework. Thanks also to Marcus Heldmann for support with the experimental paradigm, data acquisition and thesis structure, and to Prof. Dr.-Ing. Ulf Pilz for supervision at TH Lübeck.

AI-assisted tools (Grammarly / Microsoft Copilot; Claude for structural review and language consistency) were used for language correction only, as declared in the thesis.

---

## 13. License and contact

- **Code:** add a `LICENSE` file (e.g. MIT). Until one is added, all rights are reserved by default.
- **Contact:** Smit Khandelwal · [smit.khandelwal29@gmail.com](mailto:smit.khandelwal29@gmail.com) · [LinkedIn](https://www.linkedin.com/in/smit-k-79a7bb225/)

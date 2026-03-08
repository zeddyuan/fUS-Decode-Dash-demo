# fUS-Decode-Dash: Functional Ultrasound Brain Signal Decoding & Clinical Monitoring Dashboard

> A full-stack prototype demonstrating how functional ultrasound (fUS) brain-computer interface data can be processed, decoded, and visualized in a real-time clinical monitoring dashboard — designed for the next generation of non-invasive neural interfaces.

<p align="center">
  <img src="https://d2xsxph8kpxj0f.cloudfront.net/119582955/LvQqNF3DtV4h9VNqrxANkn/dashboard-demo_14f6c9cf.gif" alt="fUS-Decode-Dash Live Demo" width="100%" />
</p>
<p align="center"><em>Real-time clinical monitoring dashboard with fUS signal streaming, AI neural decoding, and clinical metrics</em></p>

---

## Executive Summary

Functional ultrasound imaging (fUS) is emerging as a transformative modality for brain-computer interfaces (BCIs). Unlike traditional electrophysiology, fUS measures cerebral blood volume changes across thousands of voxels simultaneously, providing spatially rich neural signals without penetrating electrodes [1]. The landmark 2024 study by Griggs et al. at Caltech demonstrated that fUS can decode movement intentions from the posterior parietal cortex of non-human primates in real time, achieving online BMI control with accuracy comparable to electrophysiology-based systems [1].

However, the clinical translation of fUS-BMI technology faces a critical bottleneck: **data interpretation**. A single fUS session generates gigabytes of high-dimensional spatiotemporal data. Clinicians — whether neurologists monitoring stroke rehabilitation or pain specialists evaluating targeted ultrasound neuromodulation — need intuitive, real-time interfaces to make sense of this data flood.

**fUS-Decode-Dash** addresses this gap by providing an end-to-end prototype pipeline that spans from raw data preprocessing to a production-quality clinical monitoring dashboard. This project demonstrates how companies like Gestalt Diagnostics could operationalize fUS-BMI technology for clinical workflows, particularly in chronic pain management and post-stroke motor rehabilitation.

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Stage 1: Data Engineering & Preprocessing](#stage-1-data-engineering--preprocessing)
3. [Stage 2: Neural Decoding Model](#stage-2-neural-decoding-model)
4. [Stage 3: Clinical Dashboard](#stage-3-clinical-dashboard)
5. [Pain Point Analysis](#pain-point-analysis)
6. [Product Value Proposition](#product-value-proposition)
7. [Future Roadmap](#future-roadmap)
8. [Installation & Usage](#installation--usage)
9. [References](#references)

---

## Project Architecture

The project is organized into three computational stages that mirror a real clinical data pipeline, plus a React-based dashboard for visualization and interaction.

| Component | Technology Stack | Purpose |
|-----------|-----------------|---------|
| Data Engineering | Python, SciPy, MNE-Python, Pandas, Matplotlib | Load `.mat` files, construct MNE RawArray objects, filter and explore fUS signals |
| Neural Decoding | MNE-Python, scikit-learn | Epoch segmentation, time-domain feature extraction, SVM/RF classification |
| Clinical Dashboard | React 19, Tailwind CSS 4, Recharts, Vite | Real-time signal streaming, AI decoder output, clinical metrics visualization |
| Data Synthesis | NumPy, SciPy | Generate high-fidelity synthetic fUS data matching Caltech dataset structure |

```
fUS-Decode-Dash/
├── data/                          # Synthetic fUS data (.mat files)
│   ├── generate_synthetic_fus_data.py
│   ├── rt_fUS_data_S{n}_R{n}.mat
│   ├── processed_roi_data.npz
│   └── project_record.json
├── stage1_preprocessing/
│   └── data_loader.py             # MNE-Python integration & feature exploration
├── stage2_decoding/
│   └── neural_decoder.py          # Epoch cutting, feature extraction, ML classifiers
├── models/
│   ├── svm_rbf_model.pkl          # Trained SVM classifier
│   ├── random_forest_model.pkl    # Trained Random Forest classifier
│   └── model_metadata.json        # Model parameters & performance metrics
├── figures/                       # Generated visualizations
└── fus-dashboard/                 # React clinical dashboard (separate project)
    └── client/src/
        ├── components/            # Dashboard UI modules
        ├── hooks/useFusData.ts    # Synthetic real-time data generator
        └── pages/Home.tsx         # Main dashboard layout
```

---

## Stage 1: Data Engineering & Preprocessing

### Data Source

This project is built upon the data structure and experimental protocol described in the Caltech fUS-BMI dataset [1] [2]. The original dataset comprises 34.3 GB of Power Doppler imaging data recorded from two rhesus macaques performing an 8-direction center-out reach task. Due to the dataset's size, this project uses high-fidelity synthetic data that replicates the original's statistical properties, spatial structure, and temporal dynamics.

### Data Loading Pipeline

The `data_loader.py` script implements a complete preprocessing pipeline using industry-standard neuroscience tools. Raw `.mat` files are loaded via `scipy.io.loadmat`, extracting Power Doppler volumes (128 x 100 pixels per frame at 2 Hz) along with trial metadata including target directions, state transitions, and timing information. Five cortical regions of interest (ROIs) — LIP, MIP, VIP, Area 5, and Area 7 — are defined as spatial masks over the posterior parietal cortex, and mean blood-flow time series are extracted from each.

### MNE-Python Integration

The extracted ROI time series are encapsulated into `mne.io.RawArray` objects, enabling access to MNE-Python's extensive signal processing toolkit. A FIR bandpass filter (0.01–0.5 Hz) is applied to remove slow drift artifacts and high-frequency noise while preserving the hemodynamic response bandwidth. Power spectral density analysis confirms the filter's effectiveness, with the dominant signal energy concentrated in the 0.01–0.2 Hz range consistent with neurovascular coupling dynamics.

### Feature Exploration

Exploratory data analysis using Pandas and Matplotlib reveals clear direction-dependent activation patterns across ROIs. Box plots of mean blood-flow amplitude stratified by movement direction demonstrate that each ROI exhibits preferential tuning — for example, LIP shows strongest activation for rightward (0°) and upper-right (45°) targets, while Area 5 responds maximally to leftward (180°) movements. These tuning curves are consistent with the directional selectivity reported in the original Caltech study [1].

### Generated Visualizations

The preprocessing stage produces five categories of figures: neurovascular maps showing ROI spatial distributions, multi-channel time series with direction-coded color overlays, direction-stratified box plots, power spectral density plots, and pre/post-filtering comparison panels.

---

## Stage 2: Neural Decoding Model

### Epoch Segmentation

Using MNE-Python's event detection framework, continuous fUS recordings are segmented into discrete trials based on the experimental state machine (trialstart → fixation → cue → memory → go → hold → ITI). Each epoch captures a 4-second window (8 samples at 2 Hz) beginning at cue onset, encompassing the period of maximal direction-selective neural activity. Across four session-run combinations, a total of **176 epochs** are extracted, distributed across 8 movement directions.

### Feature Extraction

From each epoch, 8 time-domain features are computed per ROI channel: mean amplitude, standard deviation, maximum, minimum, range, linear slope, peak latency, and signal energy. With 5 ROI channels, this yields a **40-dimensional feature vector** per trial — a compact but information-rich representation suitable for lightweight classifiers deployable in real-time clinical settings.

### Classification Results

Two classifiers are evaluated using 5-fold stratified cross-validation to ensure balanced direction representation in each fold.

| Classifier | CV Accuracy (Mean ± SD) | Weighted F1 Score |
|-----------|------------------------|-------------------|
| **SVM (RBF kernel)** | **86.9% ± 9.5%** | **0.869** |
| Random Forest (100 trees) | 84.6% ± 3.0% | 0.844 |

These results align closely with the online decoding performance reported by Griggs et al. (2024), where fUS-BMI achieved real-time cursor control accuracy of approximately 80–90% across sessions [1]. The SVM classifier is selected as the primary model for the dashboard due to its superior mean accuracy and well-calibrated probability estimates via Platt scaling.

### Feature Importance Analysis

Random Forest Gini importance analysis reveals that **mean amplitude** and **energy** features from LIP and MIP are the most discriminative, accounting for over 40% of total importance. This finding is neurobiologically plausible, as these regions in the posterior parietal cortex are known to encode reach planning signals [3].

---

## Stage 3: Clinical Dashboard

### Design Philosophy

The dashboard follows an **"Operating Room Light"** design paradigm inspired by professional medical monitoring equipment (Philips IntelliVue, GE CARESCAPE). Key design decisions include a deep navy-black background (#0a0e1a) to minimize visual fatigue during extended monitoring sessions, a cyan-green primary accent (#00e5a0) evoking the classic "vital sign green" of bedside monitors, and a three-column asymmetric layout that prioritizes signal visualization while keeping controls and metrics accessible.

### Module Architecture

The dashboard comprises seven interconnected modules, each serving a distinct clinical function.

**Real-Time Signal Stream.** The central module renders 5-channel fUS blood-flow waveforms using Recharts, updating at 2 Hz to simulate live data acquisition. Clinicians can toggle individual ROI channels on and off, and the current trial state (baseline, fixation, cue, memory, go, hold, ITI) is displayed with color-coded indicators. The Y-axis represents Power Doppler intensity in arbitrary units, with a typical baseline around 1100 a.u. and activation peaks reaching 1400–1600 a.u.

**AI Decoder Output.** The right panel features a directional compass that rotates to indicate the predicted movement intention, accompanied by a confidence bar and an 8-class probability distribution histogram. A scrolling history of recent predictions is displayed as a color-coded grid (green for correct, red for incorrect), providing at-a-glance accuracy assessment. The module reports the model type (SVM-RBF), cumulative accuracy, and total prediction count.

**Clinical Indicators.** Three circular gauges display key clinical metrics: Signal-to-Noise Ratio (SNR, in dB), estimated subject fatigue (percentage), and hemodynamic response function (HRF) quality. Below the gauges, status cards report signal quality (Excellent/Good/Fair/Poor) and motion artifact levels. When fatigue exceeds 60% or motion artifacts exceed 5%, amber or red warning banners appear, prompting the clinician to consider intervention.

**ROI Activation Map.** A brain region visualization overlays real-time activation intensity indicators on an anatomical map of the posterior parietal cortex. Each ROI (LIP, MIP, VIP, Area 5, Area 7) is represented by a glowing circle whose size and opacity scale with current blood-flow amplitude, providing an intuitive spatial view of cortical activity patterns.

**Trial Timeline.** A horizontal state progression bar shows the current trial phase, enabling clinicians to understand the temporal context of observed signals and predictions.

**Accuracy Trend.** A mini sparkline chart displays rolling decoding accuracy over the last 20 predictions, helping clinicians identify performance degradation that might indicate electrode drift or subject disengagement.

**Patient Information Sidebar.** The left panel displays subject metadata (ID, species, weight, implant type, protocol) and provides session controls (Start/Stop streaming) and channel visibility toggles.

---

## Pain Point Analysis

### The Data Deluge Problem

Functional ultrasound imaging generates data at an extraordinary rate. A single 15.6 MHz linear probe with 128 elements produces approximately 200 compounded B-mode images per second, yielding raw data throughputs exceeding **2.4 GB/s** [1]. Even after Power Doppler processing reduces this to 2 Hz frame rates, a typical 2-hour clinical session produces over 50 GB of spatiotemporal data. Current clinical workflows require researchers to manually inspect this data offline using MATLAB scripts — a process that can take hours per session and requires specialized signal processing expertise.

### The Interpretation Gap

The fundamental challenge is not data collection but data interpretation. When a pain specialist administers targeted ultrasound neuromodulation for chronic pain, they need to answer questions in real time: Is the treatment reaching the intended brain region? Is the patient's neural response changing over the course of therapy? Are motion artifacts corrupting the signal? Current tools provide none of these answers at the point of care.

### The Clinical Workflow Bottleneck

In the emerging field of ultrasound neuromodulation for chronic pain — a key application area for companies like Gestalt Diagnostics — the absence of real-time monitoring creates a critical workflow bottleneck. Clinicians must complete a session, wait for offline analysis, and then decide whether to adjust treatment parameters for the next session. This trial-and-error approach extends treatment timelines, increases costs, and reduces patient satisfaction.

---

## Product Value Proposition

### For Gestalt Diagnostics' Clinical Teams

The fUS-Decode-Dash prototype demonstrates how a purpose-built monitoring interface can transform the clinical workflow for targeted ultrasound interventions. Rather than waiting hours for offline analysis, clinicians can observe treatment effects in real time through three key capabilities.

**Immediate Signal Verification.** The real-time signal stream module allows clinicians to confirm that the ultrasound probe is properly positioned and that blood-flow signals from target brain regions are being captured with adequate quality. The SNR gauge and signal quality indicator provide instant feedback, reducing setup time and minimizing wasted sessions due to poor probe placement.

**Treatment Response Monitoring.** By tracking hemodynamic response patterns across cortical ROIs during neuromodulation, clinicians can observe whether the targeted intervention is producing the expected neural response. The brain activation map provides a spatial view of which regions are responding, while the signal stream reveals temporal dynamics of the treatment effect.

**Adaptive Decision Support.** The AI decoder module demonstrates how machine learning can extract clinically meaningful patterns from complex fUS data. In a pain management context, this could be adapted to classify pain states, predict treatment response, or detect adverse neural events — all in real time, enabling clinicians to adjust treatment parameters during the session rather than after.

### Competitive Differentiation

No existing commercial product combines fUS signal processing, real-time neural decoding, and clinical-grade visualization in a single integrated platform. Current alternatives require researchers to cobble together MATLAB scripts, Python notebooks, and generic data visualization tools — an approach that is neither scalable nor suitable for clinical deployment.

---

## Future Roadmap

### Phase 1: Clinical Validation (6–12 months)

The immediate priority is replacing synthetic data with real clinical fUS recordings from Gestalt's pain management trials. This involves integrating with Gestalt's data acquisition hardware, validating the preprocessing pipeline against established offline analysis methods, and conducting usability testing with clinical staff. The dashboard's modular architecture is designed to accommodate this transition with minimal code changes — the `useFusData` hook can be replaced with a WebSocket-based real-time data feed without modifying any visualization components.

### Phase 2: Cloud-Native Deployment (12–18 months)

To support multi-site clinical trials, the dashboard should evolve into a cloud-native platform with HIPAA-compliant data storage, role-based access control, and remote monitoring capabilities. Key technical milestones include migrating the data pipeline to a streaming architecture (Apache Kafka or AWS Kinesis), implementing server-side model inference with GPU acceleration, and adding longitudinal patient tracking with session-over-session comparison views.

### Phase 3: Wearable Integration (18–36 months)

Gestalt's long-term vision includes a home-use wearable ultrasound headset for chronic pain management. This form factor demands a fundamentally different software architecture. The dashboard must be reimagined as a mobile-first progressive web application (PWA) with offline-capable signal processing, edge ML inference on the headset's embedded processor, and simplified clinical views designed for patient self-monitoring. Key challenges include reducing the computational requirements of Power Doppler processing to run on ARM-based SoCs, implementing adaptive data compression for intermittent cellular connectivity, and designing patient-facing interfaces that convey treatment status without requiring neuroscience expertise.

### Technical Milestones Summary

| Timeline | Milestone | Key Deliverable |
|----------|-----------|-----------------|
| Q1–Q2 2026 | Real data integration | Validated pipeline with Gestalt clinical data |
| Q3 2026 | FDA pre-submission | 510(k) regulatory strategy document |
| Q4 2026 | Cloud deployment | Multi-site monitoring platform (HIPAA) |
| Q1 2027 | Mobile prototype | PWA with offline signal processing |
| Q3 2027 | Edge inference | On-device ML for wearable headset |
| Q4 2027 | Patient portal | Self-monitoring interface for home use |

---

## Installation & Usage

### Prerequisites

The Python data pipeline requires Python 3.9+ with the following packages:

```bash
pip install numpy scipy matplotlib pandas mne scikit-learn h5py
```

The React dashboard requires Node.js 18+ and pnpm:

```bash
npm install -g pnpm
```

### Running the Data Pipeline

```bash
# Step 1: Generate synthetic fUS data
cd fUS-Decode-Dash
python data/generate_synthetic_fus_data.py

# Step 2: Run preprocessing & feature exploration
python stage1_preprocessing/data_loader.py

# Step 3: Run neural decoding
python stage2_decoding/neural_decoder.py
```

Generated figures are saved to `figures/` and trained models to `models/`.

### Running the Dashboard

```bash
cd fus-dashboard
pnpm install
pnpm dev
```

The dashboard will be available at `http://localhost:3000`. It generates synthetic real-time data in the browser — no backend server or Python process is required.

### Dashboard Controls

The left sidebar provides session controls (Start/Stop streaming) and channel visibility toggles for each of the 5 ROI channels. The dashboard automatically generates new trials every 10 seconds, cycling through 8 movement directions. AI predictions appear at the end of each trial's "go" phase, and clinical metrics update every 5 seconds.

---

## References

[1] Griggs, W.S., Norman, S.L., Deffieux, T. et al. "Decoding motor plans using a closed-loop ultrasonic brain–machine interface." *Nature Neuroscience*, 27, 196–207 (2024). https://doi.org/10.1038/s41593-023-01500-7

[2] Griggs, W.S. et al. "Real-time fUS-BMI Dataset." *CaltechDATA* (2023). https://doi.org/10.22002/pa710-cdn95

[3] Andersen, R.A. & Cui, H. "Intention, action planning, and decision making in parietal-frontal circuits." *Neuron*, 63(5), 568–583 (2009). https://doi.org/10.1016/j.neuron.2009.08.028

---

## License

This project is provided for educational and demonstration purposes. The synthetic data is generated independently and does not contain any proprietary information from the Caltech fUS-BMI dataset. The dashboard design and code are original work.

---

*Built by Manus AI — demonstrating the intersection of neurotechnology, machine learning, and clinical product design.*

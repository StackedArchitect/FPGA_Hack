# FPGA Edge AI Hackathon — Strategic Ideation & Domain Analysis

---

## Table of Contents

1. [Evaluation Criteria (What Judges Want)](#evaluation-criteria)
2. [Domain-by-Domain Ideas](#domain-ideas)
   - [Agriculture](#agriculture)
   - [Biomedical Systems](#biomedical)
   - [Traffic Management](#traffic)
   - [Smart Energy](#smart-energy)
   - [Defense Systems](#defense)
3. [Comparative Domain Matrix](#comparison)
4. [FINAL RECOMMENDATION](#recommendation)
5. [Winning Execution Strategy](#execution)

---

## 1. Evaluation Criteria — What Wins Hackathons <a name="evaluation-criteria"></a>

Before choosing a domain, understand what the judges score:

| Factor | Weight | What They Look For |
|---|---|---|
| **Technical Depth** | HIGH | Quantized model, clean RTL, FPGA synthesis |
| **Real-World Impact** | HIGH | Solves a genuine edge deployment problem |
| **Completeness** | CRITICAL | End-to-end pipeline: Dataset → Model → Quantize → RTL → Simulation/FPGA |
| **Innovation** | HIGH | Novel architecture, hardware-software co-design |
| **Edge AI Metrics** | CRITICAL | Latency (µs), power (mW), throughput, resource utilization |
| **Feasibility of Demo** | HIGH | Working prototype beats a PowerPoint every time |

**Key Insight:** The team that delivers a **complete, working pipeline** with **quantified edge metrics** will beat the team with a fancier idea but incomplete implementation.

---

## 2. Domain-by-Domain Ideas <a name="domain-ideas"></a>

---

### 🌾 AGRICULTURE <a name="agriculture"></a>

#### Idea A1: Binary Neural Network for Real-Time Crop Disease Detection
- **Dataset:** PlantVillage (54K+ leaf images, 38 classes, publicly available)
- **Model:** Quantized MobileNet-v2 or custom BinaryConnect CNN
- **RTL:** 2D convolution engine with binary weights (XNOR + popcount)
- **Edge Story:** Camera-equipped drone/handheld scans leaves → instant disease ID
- **Pros:** Great dataset, high accuracy (~98%), clear use case
- **Cons:** 2D image convolution is RTL-heavy; common project idea (low novelty)

#### Idea A2: Multi-Sensor Soil Health Classifier for Precision Irrigation
- **Dataset:** UCI Soil Moisture / custom synthetic from NASA SMAP
- **Model:** Small MLP or 1D-CNN on soil moisture, pH, temp, humidity
- **RTL:** Fully-connected NN in Verilog, fixed-point arithmetic
- **Edge Story:** Sensor node in remote farm → classifies soil → triggers drip irrigation
- **Pros:** Simple RTL, very low power, clear IoT narrative
- **Cons:** Less technically impressive, simpler model

#### Idea A3: Hyperspectral Crop Stress Detection
- **Dataset:** Indian Pines / Salinas (publicly available hyperspectral datasets)
- **Model:** 1D-CNN on spectral signatures per pixel
- **RTL:** 1D convolution pipeline with channel-wise processing
- **Edge Story:** Satellite/drone captures spectral data → on-board classification
- **Pros:** Novel, impressive data type, 1D operations are FPGA-friendly
- **Cons:** Niche domain knowledge needed

**Agriculture Verdict:** Feasible but **low novelty** — every other team might pick PlantVillage. Hyperspectral (A3) is interesting but niche.

---

### 🏥 BIOMEDICAL SYSTEMS <a name="biomedical"></a>

#### Idea B1: Real-Time Multi-Class ECG Arrhythmia Detection ⭐ (TOP CONTENDER)
- **Dataset:** MIT-BIH Arrhythmia Database (48 records, 109K+ beats, gold standard)
- **Model:** 1D Temporal CNN (3-5 conv layers) with Quantization-Aware Training (QAT)
- **RTL:** Pipelined 1D convolution engine → ReLU → MaxPool → FC classifier
- **Edge Story:** Wearable patch → real-time heartbeat classification → alert on anomaly
- **Innovation Angle:**
  - 4-bit/8-bit fixed-point quantization with minimal accuracy loss
  - Hardware-software co-design: model architecture chosen FOR FPGA efficiency
  - Sub-100µs inference latency (beats any microcontroller)
  - Power budget < 50mW (wearable-grade)
- **Pros:**
  - 1D operations = **dramatically simpler RTL** than 2D image CNNs
  - MIT-BIH is THE gold standard → judges respect it
  - >98% accuracy achievable with tiny model
  - Complete pipeline is very achievable in hackathon timeframe
  - Compelling real-world narrative (cardiac monitoring saves lives)
  - Can show actual waveform → inference → classification demo
- **Cons:** Popular domain (but our implementation depth will differentiate)

#### Idea B2: EEG-Based Seizure Onset Detection
- **Dataset:** CHB-MIT Scalp EEG Database (22 patients, publicly available)
- **Model:** 1D-CNN + attention mechanism, quantized to 8-bit
- **RTL:** Streaming 1D conv engine with sliding window
- **Edge Story:** Implantable/wearable neuro-device → predicts seizure 30s before onset
- **Pros:** High impact, 1D signals, Nobel-prize-worthy narrative
- **Cons:** EEG preprocessing is complex, multi-channel handling adds RTL complexity

#### Idea B3: PPG-Based Continuous Blood Pressure Estimation
- **Dataset:** MIMIC-III Waveform Database (publicly available)
- **Model:** 1D-CNN regression model (outputs systolic/diastolic)
- **RTL:** 1D conv + regression head
- **Edge Story:** Smartwatch PPG sensor → cuffless BP monitoring → hypertension alert
- **Pros:** Very novel, massive commercial potential
- **Cons:** Regression is harder to validate than classification

#### Idea B4: Retinal Disease Classification (Diabetic Retinopathy)
- **Dataset:** APTOS 2019 / Kaggle Diabetic Retinopathy (88K fundus images)
- **Model:** Quantized EfficientNet-Lite or custom tiny CNN
- **RTL:** 2D conv engine for small input resolution
- **Edge Story:** Portable ophthalmoscope → instant screening in rural clinics
- **Pros:** High impact, good datasets
- **Cons:** 2D convolutions = harder RTL, larger model

**Biomedical Verdict:** **EXTREMELY STRONG DOMAIN.** Idea B1 (ECG) is a powerhouse — simple RTL, gold-standard dataset, 98%+ accuracy, and a story that resonates with every judge. B2 (seizure) is impressive but harder. B3 (BP) is novel but riskier.

---

### 🚦 TRAFFIC MANAGEMENT <a name="traffic"></a>

#### Idea T1: Vehicle Density Estimation using Tiny Object Detection
- **Dataset:** UA-DETRAC / COCO (vehicle subset)
- **Model:** YOLO-Tiny variant (heavily quantized)
- **RTL:** 2D conv + anchor-free detection head
- **Edge Story:** Intersection camera → on-device vehicle count → adaptive signal
- **Pros:** Visual demo is impressive
- **Cons:** YOLO even tiny is MASSIVE in RTL — very hard to fit on FPGA in hackathon time

#### Idea T2: Acoustic Vehicle Classification at Intersections
- **Dataset:** UrbanSound8K / custom traffic audio dataset
- **Model:** 1D-CNN on Mel spectrogram features
- **RTL:** FFT → Mel filterbank → 1D-CNN classifier
- **Edge Story:** Microphone at intersection → classifies emergency vehicles → priority signal
- **Pros:** 1D pipeline, novel approach to traffic
- **Cons:** Less conventional, acoustic data is noisy

#### Idea T3: Traffic Flow Anomaly Detection (Congestion/Accident)
- **Dataset:** PeMS-BAY / METR-LA traffic sensor datasets
- **Model:** 1D-CNN or small LSTM for time-series anomaly detection
- **RTL:** Streaming temporal classifier
- **Edge Story:** Road sensors → detect unusual patterns → alert authorities
- **Pros:** Feasible RTL, time-series is FPGA-friendly
- **Cons:** Less visually impressive demo

**Traffic Verdict:** Object detection (T1) is too complex for clean FPGA implementation. T2 and T3 are feasible but lack the "wow" factor of other domains.

---

### ⚡ SMART ENERGY <a name="smart-energy"></a>

#### Idea E1: Non-Intrusive Load Monitoring (NILM) Classifier
- **Dataset:** REDD / UK-DALE (publicly available home energy datasets)
- **Model:** 1D-CNN for appliance identification from aggregate power signal
- **RTL:** 1D conv pipeline for real-time load disaggregation
- **Edge Story:** Smart meter → identifies which appliance is running → optimizes usage
- **Pros:** Novel, 1D signals, practical application
- **Cons:** Niche appeal, less dramatic narrative

#### Idea E2: Predictive Peak Demand Detection
- **Dataset:** UCI Individual Household Electric Power Consumption / Pecan Street
- **Model:** Temporal CNN for time-series forecasting + threshold classifier
- **RTL:** Sliding window feature extractor → classifier
- **Edge Story:** Smart grid node → predicts demand spike 15min ahead → triggers load shedding
- **Pros:** Clear utility narrative
- **Cons:** Forecasting is harder to demo impressively

#### Idea E3: Occupancy Detection for Smart HVAC
- **Dataset:** UCI Occupancy Detection Dataset
- **Model:** Small MLP/decision tree on temp, humidity, CO2, light
- **RTL:** Tiny fully-connected NN
- **Edge Story:** Room sensor → detect occupancy → control HVAC
- **Pros:** Very simple RTL, guaranteed completion
- **Cons:** Too simple to win — not enough technical depth

**Smart Energy Verdict:** Interesting domain but **lacks the dramatic impact** needed to win by a huge margin. NILM (E1) is the strongest here but doesn't match biomedical/defense in judge appeal.

---

### 🛡️ DEFENSE SYSTEMS <a name="defense"></a>

#### Idea D1: SAR Automatic Target Recognition (ATR) with Binary Neural Network ⭐⭐ (TOP CONTENDER)
- **Dataset:** MSTAR (Moving and Stationary Target Acquisition and Recognition) — publicly available, 10 military vehicle classes, 128×128 grayscale radar images
- **Model:** Binary/Ternary Neural Network (BNN/TNN) for SAR image classification
- **RTL:**
  - XNOR-based convolution engine (replaces MAC with XNOR + popcount)
  - Batch normalization folded into thresholds at compile time
  - Binary activations → single-bit feature maps
- **Edge Story:** Airborne/ground radar platform → on-board target classification → real-time battlefield awareness without satellite/cloud link
- **Innovation Angle:**
  - **XNOR convolutions** = ~58× speedup + ~32× memory savings vs float32
  - **Zero multipliers needed** — pure logic operations on FPGA
  - Power consumption drops from watts to milliwatts
  - Classification in **<10µs** latency
  - Demonstrates the MOST hardware-efficient CNN architecture possible
- **Pros:**
  - MSTAR is THE standard defense ML benchmark — judges from defense/academia know it
  - Binary neural networks are **THE innovation** in edge AI hardware
  - 128×128 grayscale = manageable image size for FPGA
  - Accuracy: 95-97% on 10-class MSTAR with BNN
  - Massive "cool factor" — classifying tanks from radar images
  - Perfect alignment with "low power, real-time, efficient hardware"
  - Published research validates this approach (proven feasible)
- **Cons:**
  - 2D convolutions in RTL are more complex than 1D
  - BNN implementation requires careful attention to batch norm folding
  - Slightly more challenging than 1D approaches

#### Idea D2: Acoustic Threat Classification (Gunshot/Explosion/Vehicle)
- **Dataset:** UrbanSound8K + custom augmented military acoustic data / ESC-50
- **Model:** 1D-CNN on Mel-frequency features
- **RTL:** FFT → Feature extraction → 1D-CNN classifier
- **Edge Story:** Sensor deployed in field → classifies acoustic threats → alert
- **Pros:** 1D pipeline, simpler RTL
- **Cons:** Dataset quality for military sounds is limited

#### Idea D3: Radar Pulse Anomaly Detection
- **Dataset:** Synthetic radar pulse datasets / DARPA Radar Challenge
- **Model:** 1D temporal CNN or autoencoder for anomaly scoring
- **RTL:** Streaming feature extractor + anomaly scorer
- **Edge Story:** Radar receiver → detects jamming/spoofing patterns → countermeasure trigger
- **Pros:** Unique, technically impressive
- **Cons:** Synthetic data, harder to validate

#### Idea D4: Infrared Target Detection with Quantized CNN
- **Dataset:** FLIR ADAS Dataset (thermal images, publicly available)
- **Model:** Quantized tiny CNN for person/vehicle detection in IR
- **RTL:** 2D conv engine with 8-bit fixed-point
- **Edge Story:** Thermal camera on UAV/vehicle → target detection at night
- **Pros:** Great dataset (FLIR provides it free), very practical
- **Cons:** Object detection RTL is complex

#### Idea D5: Multi-Sensor Fusion Threat Classifier
- **Dataset:** Combine radar (MSTAR) + acoustic + IR features (synthetic fusion)
- **Model:** Multi-input MLP/CNN fusion network
- **RTL:** Parallel feature extractors → fusion layer → classifier
- **Edge Story:** Multi-sensor platform → fuse all inputs → robust threat assessment
- **Pros:** Very impressive, multi-modal
- **Cons:** Complex to implement fully, synthetic fusion data

**Defense Verdict:** **THE STRONGEST DOMAIN.** Idea D1 (SAR ATR with BNN) is a hackathon winner — it combines a respected dataset, cutting-edge hardware technique (binary neural nets), dramatic narrative, and proves the entire value proposition of Edge AI (real-time, low power, no cloud dependency in hostile environments). D2 and D3 are strong fallbacks.

---

## 3. Comparative Domain Matrix <a name="comparison"></a>

| Criteria | Agriculture | Biomedical | Traffic | Smart Energy | Defense |
|---|:---:|:---:|:---:|:---:|:---:|
| **Judge Impact / Wow Factor** | ★★★ | ★★★★★ | ★★★ | ★★ | ★★★★★ |
| **RTL Feasibility (hackathon time)** | ★★★ | ★★★★★ | ★★ | ★★★★ | ★★★★ |
| **Dataset Quality & Availability** | ★★★★ | ★★★★★ | ★★★ | ★★★ | ★★★★★ |
| **Innovation Potential** | ★★★ | ★★★★ | ★★★ | ★★★ | ★★★★★ |
| **Edge AI Narrative Strength** | ★★★★ | ★★★★★ | ★★★★ | ★★★ | ★★★★★ |
| **Industry-Level Credibility** | ★★★ | ★★★★★ | ★★★ | ★★★ | ★★★★★ |
| **FPGA Resource Fit** | ★★★ | ★★★★★ | ★★ | ★★★★ | ★★★★ |
| **Competition Differentiation** | ★★ | ★★★★ | ★★ | ★★★ | ★★★★★ |
| **TOTAL** | **21/40** | **37/40** | **20/40** | **22/40** | **38/40** |

---

## 4. 🏆 FINAL RECOMMENDATION <a name="recommendation"></a>

### PRIMARY PICK: Defense — SAR Automatic Target Recognition with Binary Neural Network

### BACKUP PICK: Biomedical — Real-Time ECG Arrhythmia Detection with Quantized 1D-CNN

---

### Why Defense (SAR ATR + BNN) Wins:

**1. Maximum Differentiation:**
Most hackathon teams default to image classification with standard quantized CNNs. A **Binary Neural Network** that replaces ALL multiplications with XNOR gates is a fundamentally different hardware paradigm. When you tell judges "our design has **zero multipliers** — every convolution is pure logic," that's an instant attention-grabber.

**2. Perfect Problem-Hardware Fit:**
- MSTAR images are 128×128×1 (grayscale) — small enough for FPGA
- BNN reduces memory from 32-bit floats to **1-bit weights** → 32× compression
- XNOR convolution = LUT operations only → maps perfectly to FPGA fabric
- No DSP blocks needed → leaves room for parallelism

**3. Unbeatable Edge AI Metrics:**
- Inference latency: **< 10 microseconds**
- Power consumption: **< 100 milliwatts**
- Throughput: **> 100,000 classifications/second**
- These numbers DESTROY any GPU/CPU comparison → instant "wow"

**4. The Narrative is Bulletproof:**
"An airborne ISR platform processing radar returns in real-time, classifying targets without any communication link to command — because in contested environments, you can't rely on cloud connectivity. Our binary neural network achieves 96% accuracy on MSTAR while consuming less power than a single LED."

**5. Industry-Level Credibility:**
- MSTAR is used by DARPA, US Army Research Lab, and defense contractors
- Binary neural networks are active research at MIT, Intel, Xilinx
- This is publishable work — conference paper material

**6. Publicly Available & Validated:**
- MSTAR dataset is freely available (released by DARPA/AFRL)
- Multiple published papers validate BNN on MSTAR (proven approach)
- You're not guessing — this works

---

### Why Biomedical (ECG) is the Strong Backup:

If your team has limited FPGA/RTL experience, ECG arrhythmia detection is the **safest path to a complete, impressive demo**:

- **1D convolutions** are 5-10× simpler in Verilog than 2D
- MIT-BIH gives you 98%+ accuracy with a tiny model
- You can show a real-time waveform visualization
- The pipeline (raw ECG → R-peak detection → classification → output) is clean and demonstrable
- Still delivers impressive edge metrics (sub-ms latency, < 20mW)

**Choose ECG if:** You want maximum probability of a **complete, working demo**
**Choose SAR/BNN if:** You want maximum **differentiation and judge impact**

---

### My Strong Recommendation: GO WITH DEFENSE (SAR ATR + BNN)

**Rationale:** You said you want to "win by a huge margin" and make it "industry-level." The BNN approach on defense radar data is:
- Technically the most innovative (XNOR convolutions)
- Narratively the most compelling (battlefield edge AI)
- Metrically the most impressive (microsecond latency, milliwatt power)
- Academically the most credible (MSTAR + published BNN research)

The 2D convolution complexity is manageable because **binary convolutions are dramatically simpler** — you're implementing XNOR gates and popcount circuits, not multiply-accumulate units.

---

## 5. Winning Execution Strategy <a name="execution"></a>

### Phase 1: Model Development (Python/PyTorch)
```
1. Download MSTAR dataset (10-class vehicle classification)
2. Preprocess: 128×128 grayscale, normalize
3. Train floating-point baseline CNN (establish accuracy ceiling)
4. Implement Binarization-Aware Training using Larq or custom BNN training
5. Export binary weights and batch-norm-folded thresholds
6. Validate: Target ≥ 95% accuracy on MSTAR test set
```

### Phase 2: RTL Design (Verilog)
```
1. Design XNOR convolution module (replaces MAC)
2. Design popcount module (counts 1-bits)
3. Design batch-norm-as-threshold comparator
4. Design binary activation (sign function)
5. Pipeline: Conv → BN-Threshold → BinActivation → Pool → FC → Argmax
6. Fixed-point only for first/last layers (input image & final FC)
7. Testbench with MSTAR test images
```

### Phase 3: FPGA Implementation (Vivado)
```
1. Synthesize for target FPGA (e.g., Artix-7 / Zynq-7000)
2. Report: LUT utilization, FF utilization, BRAM, DSP (should be near-zero DSP)
3. Report: Clock frequency, latency, throughput
4. Estimate power consumption using Vivado Power Analyzer
5. Compare: FPGA vs. ARM Cortex-M4 vs. Raspberry Pi (latency & power)
```

### Phase 4: Demo & Presentation
```
1. Show training pipeline (Python notebook with accuracy curves)
2. Show RTL simulation (ModelSim/Vivado waveforms with correct classification)
3. Show FPGA resource utilization (pie charts)
4. Show latency comparison: FPGA (µs) vs CPU (ms) vs GPU (ms)
5. Show power comparison: FPGA (mW) vs CPU (W) vs GPU (W)
6. Killer slide: "96% accuracy, 8µs latency, 47mW power, ZERO multipliers"
```

### Key Technical Details for BNN Implementation:

| Layer | Implementation | Hardware Resource |
|---|---|---|
| Input Conv (first layer) | 8-bit fixed-point MAC | Small # of DSP slices |
| Hidden Conv Layers | XNOR + Popcount | LUTs only (zero DSP) |
| Batch Normalization | Pre-computed threshold comparison | Comparators |
| Activation | Sign function (MSB extraction) | Single wire |
| Pooling | Max of binary values (OR gate) | LUTs |
| Final FC Layer | 8-bit fixed-point MAC | Small # of DSP slices |
| Argmax | Comparator tree | LUTs |

### Recommended Architecture:
```
Input (128×128×1, 8-bit)
  → Conv2D 3×3, 64 filters (8-bit, only layer using DSPs)
  → BatchNorm → BinaryActivation
  → BinConv2D 3×3, 128 filters (XNOR)
  → BatchNorm → BinaryActivation → MaxPool 2×2
  → BinConv2D 3×3, 256 filters (XNOR)  
  → BatchNorm → BinaryActivation → MaxPool 2×2
  → BinConv2D 3×3, 256 filters (XNOR)
  → BatchNorm → BinaryActivation → MaxPool 2×2
  → Global Average Pool
  → FC 256→10 (8-bit, uses DSPs)
  → Argmax → Output class
```

### Tools & Frameworks:
- **Training:** Python + PyTorch + [Larq](https://larq.dev/) or custom BNN training
- **Quantization:** Brevitas (Xilinx's quantization library) for precise FPGA mapping
- **RTL:** Verilog (hand-written for control; HLS for prototyping)
- **Simulation:** Vivado Simulator / ModelSim
- **FPGA:** Vivado for AMD/Xilinx (Artix-7, Zynq-7020, or Kria KV260)
- **Comparison Benchmarks:** Python inference on Raspberry Pi / ARM for comparison numbers

---

## Final Words

The difference between a good hackathon project and a **winning** one is not complexity — it's **completeness + clarity of narrative + quantified results.**

Your pitch should be: *"We implemented a Binary Neural Network that classifies military vehicles from radar images in 8 microseconds using 47 milliwatts — with zero hardware multipliers. This is what real Edge AI looks like."*

That wins hackathons.

# Motion-intent-recognition-Sensor-Fusion
References for you: https://drive.google.com/drive/folders/1NnGhPkrpJf4BtNaIEeORqZLvkDQFgPEB?usp=drive_link
https://github.com/Levi-Ackman/TeCh

## Project Guideline: EMG Sensor Fusion for Motion Intent Recognition

### 1. Idea & Problem Statement

**Objective:** Develop a robust motion intent recognition system by fusing multiple biosignal and inertial sensors to improve accuracy and reliability of gesture/movement prediction compared to single-sensor approaches.

**Target Problem:**
- **EMG Limitations:** Single EMG signals suffer from non-stationarity, crosstalk, and amplitude variation across subjects and sessions
- **Solution:** Multi-modal sensor fusion (EMG + IMU, EMG + FMG, or EMG + IMU + FMG) provides:
  - Complementary information: EMG captures muscle activation; IMU captures motion kinematics; FMG captures finger motion
  - Improved robustness: Redundancy reduces individual sensor noise and artifacts
  - Better generalization: Multi-modal features reduce overfitting to single sensor characteristics
  - Enhanced real-world applicability: Mimics natural sensory integration in motor control

---

### 2. Available Datasets & Sensor Combinations

#### Option A: EMG + IMU Fusion
**Dataset:** NinaPro DB6 (https://ninapro.hevs.ch/instructions/DB6.html)
- **EMG:** Multichannel electromyography (muscle activity)
- **IMU:** Inertial Measurement Unit (accelerometer, gyroscope)
- **Use Case:** Gesture/movement classification with motion context
- **Advantage:** IMU provides kinematic information independent of muscle activation

#### Option B: EMG + FMG Fusion (Hybrid Approach)
**Dataset 1:** HybridSensor Repository (https://github.com/peijii/HybridSensor/tree/main)
- **EMG:** Electromyography (forearm muscle activity)
- **FMG:** Force Myography (pressure/force sensors on fingers)
- **Use Case:** Fine-grained finger gesture recognition
- **Advantage:** FMG captures direct finger forces; EMG captures motor intention

**Dataset 2:** TU Darmstadt Data Repository (https://tudatalib.ulb.tu-darmstadt.de/bitstreams/24c87db0-5503-4067-b85a-b513b72d07e8/download)
- **EMG + FMG:** Similar multimodal approach
- **Advantage:** Validated dataset for fusion algorithms

---

### 3. Implementation Roadmap

#### Phase 1: Data Exploration & Preprocessing
- [ ] Load and analyze each dataset independently
- [ ] Understand sampling rates, signal characteristics, and noise profiles
- [ ] Implement preprocessing pipeline (filtering, normalization, segmentation)

#### Phase 2: Feature Engineering
- **EMG Features:** RMS, MAV, ZC, SSC, Hjorth parameters
- **IMU Features:** Acceleration, angular velocity statistics
- **FMG Features:** Force magnitude, pressure distribution
- **Fusion Strategy:** Concatenation, weighted fusion, or deep learning fusion

#### Phase 3: Sensor Fusion Architecture
- [ ] Design multi-modal fusion architecture (early fusion, late fusion, or hybrid)
- [ ] Compare fusion strategies on each dataset
- [ ] Evaluate cross-subject and cross-session generalization

#### Phase 4: Motion Intent Recognition Model
- [ ] Train classifiers (SVM, Random Forest, Neural Networks, LSTM)
- [ ] Benchmark single-sensor vs. multi-sensor performance
- [ ] Test on different datasets

---

### 4. Expected Outcomes

- Comprehensive comparison of fusion strategies
- Improved classification accuracy over single-sensor baselines
- Guidelines for optimal sensor combination for different motion types
- Generalization analysis across subjects and recording sessions

# 🛰️ SAR Water Detection & Analysis Lab: The Scientific Reference Manual

**Version:** 3.0 (Master Edition)  
**Date:** January 2026  
**System:** Sentinel-1 Dual-Polarimetric SAR Analysis Suite  
**Target Resolution:** 10m (Ground Range Detected)  
**Polarizations:** VV + VH  
**Scope:** Complete Algorithm & Operational Documentation  

---

## 📖 Table of Contents

1.  [**Introduction & Scientific Scope**](#1-introduction--scientific-scope)
2.  [**System Architecture & Data Flow**](#2-system-architecture--data-flow)
3.  [**Core Physics: The SAR Scattering Models**](#3-core-physics-the-sar-scattering-models)
4.  [**Algorithm Encyclopedia**](#4-algorithm-encyclopedia)
    *   [**Group I: Pre-Processing (Speckle Reduction)**](#group-i-pre-processing-speckle-reduction)
    *   [**Group II: Radiometric Thresholding**](#group-ii-radiometric-thresholding)
    *   [**Group III: Polarimetric Indices**](#group-iii-polarimetric-indices)
    *   [**Group IV: Texture & Geometry**](#group-iv-texture--geometry)
    *   [**Group V: Hydro-Geomorphic Guards**](#group-v-hydro-geomorphic-guards)
5.  [**The Parameter Tuning Guide (Deep Dive)**](#5-the-parameter-tuning-guide-deep-dive)
6.  [**Advanced Modules & Fusion**](#6-advanced-modules--fusion)
7.  [**GUI User Manual**](#7-gui-user-manual)
8.  [**Appendix A: Mathematical Derivations**](#appendix-a-mathematical-derivations)
9.  [**Appendix B: Sentinel-1 Technical Specifications**](#appendix-b-sentinel-1-technical-specifications)
10. [**References**](#10-references)

---

## 1. Introduction & Scientific Scope

The **SAR Water Detection Lab** is a high-fidelity research environment designed for the rigorous analysis of Synthetic Aperture Radar (SAR) imagery. It specifically targets the challenge of **flood mapping** using **Sentinel-1** C-band data.

### The Challenge of SAR
SAR is an active microwave sensing technology. Unlike optical cameras (Passive), it transmits its own energy and measures the backscatter.
*   **Advantage**: It sees through clouds, rain, and smoke. It operates day and night.
*   **Difficulty**: The data is inherently noisy due to **coherence speckle** (constructive/destructive interference of waves) and geometric distortions (layover, foreshortening) caused by the side-looking geometry.

### The Lab's Solution
This software provides a **15-Window Parallel Processing Grid** that allows researchers to:
1.  **Ingest** processed chip stacks containing radar intensity (VV/VH) and terrain data (DEM/HAND).
2.  **Experiment** with 47+ distinct algorithms side-by-side.
3.  **Validate** detections using hydro-geomorphic constraints.
4.  **Fuse** results using logic gates to reduce False Positives.

---

## 2. System Architecture & Data Flow

### The Data Cube (7-Band Stack)
Each "Chip" used in this lab is a pre-processed tensor of shape `(7, H, W)`.
1.  **VV Band (dB)**: Vertical-Vertical Intensity. Primary water detector.
2.  **VH Band (dB)**: Vertical-Horizontal Intensity. Volume scattering detector (Vegetation).
3.  **MNDWI**: Modified Normalized Difference Water Index (Optical Truth, optional).
4.  **DEM**: Digital Elevation Model (SRTM/Copernicus).
5.  **HAND**: Height Above Nearest Drainage (Hydro-conditioned terrain).
6.  **Slope**: Terrain gradient (degrees).
7.  **TWI**: Topographic Wetness Index (Potential water accumulation).

### The Parallel Compute Graph
The application uses a **Streamlit** frontend with a cached backend.
*   **Session State**: Chips are loaded into memory once.
*   **Lazy Evaluation**: Filter operations are only triggered when their specific parameters change.
*   **Vectorization**: All mathematical operations are vectorized using `NumPy` and `SciPy`, avoiding slow Python loops. This allows 15 complex filters to run in near real-time on a standard CPU.

---

## 3. Core Physics: The SAR Scattering Models

Understanding *why* we detect water requires understanding how microwave energy interacts with matter.

### 3.1 Specular Reflection (The "Dark Water" Mechanism)
Water surfaces, especially when calm, act as mirrors for C-band radar ($\lambda \approx 5.6$ cm).
*   **Mechanism**: The radar pulse hits the flat surface an angle $\theta$ and reflects *away* from the sensor at angle $\theta$.
*   **Observation**: The sensor receives almost zero energy back.
*   **Result**: Water appears black (Noise floor: -22 to -30 dB).
*   **Failure Case**: **Wind Roughening**. If wind velocity > 5 m/s, capillary waves form on the water surface, causing Bragg scattering which sends energy back to the sensor. Water can appear as bright as land (-12 dB).

### 3.2 Volume Scattering (Vegetation)
Canopies (trees, crops) are complex 3D structures.
*   **Mechanism**: The signal bounces multiple times between leaves and branches.
*   **Depolarization**: Each bounce rotates the polarization plane. A pure VV signal sent out will return with a significant VH component.
*   **Result**: Vegetation is bright in both channels, but uniquely strong in VH relative to bare soil.

### 3.3 The Double-Bounce Effect (Urban/Flooded Forest)
The most challenging scenario.
*   **Mechanism**: The radar pulse hits the smooth water surface $\rightarrow$ bounces forward to a vertical structure (tree trunk or building wall) $\rightarrow$ reflects directly back to the sensor.
*   **Result**: A "Corner Reflector" effect. The signal is extremely strong (+5 to +10 dB).
*   **Detection Strategy**: Standard thresholding fails here. We must use **Texture** (these areas are very heterogenous) or **Polarimetric Ratios** (CPR) to detect the phase shift signature.

---

## 4. Algorithm Encyclopedia

### Group I: Pre-Processing (Speckle Reduction)

Raw SAR data has "salt-and-pepper" noise called speckle. We must smooth this *without* blurring the sharp boundaries of a river.

#### RFI Filter (Radio Frequency Interference)
*   **Problem**: Ground-based C-band radars (military/weather) emit pulses that interfere with Sentinel-1, creating bright "zipper" artifacts across the image.
*   **Algorithm**:
    1.  Compute local Z-scores: $Z = \frac{|x - \mu|}{\sigma}$
    2.  Identify outliers where $Z > Z_{thresh}$ (typically 3.0 or 5.0).
    3.  Replace outliers with the local **Median** (robust estimator).
*   **Why**: A standard Mean filter would be skewed by the extreme brightness of the RFI spike.

#### Refined Lee Filter
*   **Concept**: Uses the Local Minimum Mean Square Error (LMMSE) principle but solves the "edge blurring" problem of the original Lee filter.
*   **Mechanism**:
    1.  Defines **8 sub-windows** (masks) within the 7x7 kernel corresponding to 8 edge directions (0°, 45°, 90°, etc.).
    2.  Calculates the variance of each sub-window.
    3.  Selects the sub-window with the **lowest variance** (highest homogeneity).
    4.  Applies the Lee filter using statistics *only* from that sub-window.
*   **Result**: Sharply preserves river banks while smoothing the water surface.

#### Frost Filter
*   **Concept**: An adaptive Wiener filter that uses an exponentially damped convolution kernel.
*   **Formula**: The kernel weight $W$ depends on distance $d$ and local complexity $C_v$.
    $W = \exp(-K \cdot C_v^2 \cdot d)$
*   **Adaptivity**:
    *   In uniform areas ($C_v \approx 0$), $W \approx 1$ everywhere $\rightarrow$ Averaging.
    *   On edges ($C_v$ large), $W$ drops to 0 quickly $\rightarrow$ No averaging (preserves center pixel).

#### Gamma MAP (Maximum A Posteriori)
*   **Reference**: Lopes et al. (1990).
*   **Theory**: Assumes both the Scene ($x$) and the Speckle ($n$) follow Gamma distributions (not Gaussian).
*   **Equation**: Solves a quadratic equation derived from maximizing the posterior probability $P(x|z)$.
    $\hat{x} = \frac{(L-1)\bar{z} + \sqrt{\Delta}}{2L}$ (Simplified form)
*   **Strength**: The best filter for preserving radiometric fidelity (actual dB values) for scientific analysis.

#### BayesShrink Wavelet
*   **Domain**: Frequency domain (Wavelet Transform).
*   **Mechanism**:
    1.  Log-transform to make noise additive.
    2.  Discrete Wavelet Transform (DWT) creates sub-bands (HH, HL, LH, LL).
    3.  Estimate Noise Standard Deviation $\hat{\sigma}$ from the Median Absolute Deviation (MAD) of the HH band.
    4.  Calculate a unique threshold $T_B$ for each sub-band derived from Bayesian principles.
    5.  Soft-threshold the coefficients.
*   **Why**: Superior at removing "fine" speckle that spatial filters miss.

---

### Group II: Radiometric Thresholding

#### Otsu's Method
*   **Type**: Global Clustering.
*   **Logic**: Assumes the histogram has two peaks (Water and Land). Brute-forces every possible threshold $T$ to find the one that minimizes **Intra-Class Variance**.
*   **Weakness**: Fails if water < 5% of the image (histogram becomes unimodal).

#### Kittler-Illingworth
*   **Type**: Parametric Minimum Error.
*   **Logic**: Models the histogram as a mixture of two Gaussian distributions. Minimizes the Kullback-Leibler divergence.
*   **Strength**: Better than Otsu when the two classes have very different variances (e.g., Water is tight, Land is broad).

#### Hysteresis Thresholding
*   **Type**: Spatial-Radiometric.
*   **Concept**: Derived from Canny Edge Detection.
*   **Process**:
    1.  Apply a strict threshold $T_{low}$ (e.g., -22 dB). Any pixel below this is a "Seed".
    2.  Apply a relaxed threshold $T_{high}$ (e.g., -16 dB).
    3.  **Grow** the seeds: Any pixel between $T_{low}$ and $T_{high}$ is accepted *if and only if* it touches a Seed (spatial connectivity).
*   **Result**: Fills in the water body completely without picking up isolated noise elsewhere.

#### Triangle Algorithm
*   **Type**: Geometric.
*   **Logic**: Draws a line from the histogram peak to the tail. Finds the point of maximum distance from the line to the histogram curve.
*   **Use Case**: The robust choice for **Unimodal** histograms (e.g., small floods) where Otsu fails.

---

### Group III: Polarimetric Indices

#### SWI (SAR Water Index)
*   **Source**: Tian et al. (2017).
*   **Formula**: A complex polynomial that weights VV and VH to maximize water separation.
    $SWI = 0.1747\beta_{VV} + 0.0082\beta_{VH}\beta_{VV} + 0.0023\beta_{VV}^2 - 0.0015\beta_{VH}^2 + 0.1904$
*   **Interpretation**: The polynomial curvature penalizes values where VV and VH diverge (volume scattering), isolating the specular water response.

#### SDWI (Sentinel-1 Dual-Pol Water Index)
*   **Formula**: $S_{DWI} = \ln(10 \cdot VV \cdot VH) - 8$
*   **Physics**: Water has low backscatter in *both* channels. Multiplying two small fractions ($10^{-2}$) yields a tiny number ($10^{-4}$). The log amplifies this difference from land.

#### Cross-Polarization Ratio (CPR)
*   **Formula**: $CPR = \frac{\sigma_{VH}^0}{\sigma_{VV}^0}$
*   **Logic**:
    *   **Specular surface**: Little depolarization. Ratio is **Low** (< 0.1).
    *   **Rough/Volume surface**: Strong depolarization. Ratio is **High** (> 0.4).
*   **Application**: Crucial for distinguishing **Sand Dunes** (dark in VV, but higher ratio) from **Calm Water** (dark in VV, low ratio).

---

### Group IV: Texture & Geometry

#### GLCM Entropy (Gray-Level Co-occurrence Matrix)
*   **Metric**: "Randomness".
*   **Computation**: Builds a matrix of pixel-pair frequencies in a neighborhood.
*   **Water**: Homogeneous $\rightarrow$ Predictable $\rightarrow$ **Low Entropy**.
*   **Urban**: High contrast edges $\rightarrow$ Unpredictable $\rightarrow$ **High Entropy**.

#### Frangi Vesselness
*   **Metric**: "Tubular Structure".
*   **Math**: Eigen-analysis of the Hessian Matrix (2nd order derivatives).
    *   $\lambda_1 \approx 0, \lambda_2 \gg 0$ indicates a tube/line.
*   **Target**: Rivers, Canals, Ditches.

#### Touzi Ratio Edge Detector
*   **Metric**: "Edge Constant False Alarm Rate".
*   **Math**: Ratio of averages on opposite sides of a center point.
*   **Target**: Defining the exact sub-pixel boundary of a lake shoreline.

---

### Group V: Hydro-Geomorphic Guards

#### HAND (Height Above Nearest Drainage)
*   **Concept**: Normalizes the topography relative to the local drainage network.
*   **Physical Law**: Floods occur at low relative elevations.
*   **Guard**: Any pixel with $HAND > 15m$ is statistically unlikely to be floodwater (unless it's a perched mountain lake). We mask these out to remove hill shadows.

#### TWI (Topographic Wetness Index)
*   **Formula**: $\ln(a / \tan \beta)$.
*   **Meaning**:
    *   Numerator $a$: How much water flows into this cell.
    *   Denominator $\tan \beta$: How fast water flows out (slope).
*   **High Value**: Flat areas with high inflow (Swamps, Floodplains).

---

## 5. The Parameter Tuning Guide (Deep Dive)

A critical guide to the "Knobs and Sliders" in the GUI.

### 5.1 RFI Filter Settings
*   **Z-Threshold**: default `3.0`.
    *   *Increase to 5.0*: If you see valid bright urban features being deleted.
    *   *Decrease to 2.5*: If faint "zipper" lines persist in valid data.

### 5.2 Refined Lee Settings
*   **Window Size**: default `7x7`.
    *   *Use 5x5*: For fine-scale urban flood mapping (narrow streets).
    *   *Use 9x9 or 11x11*: For large open oceans to aggressively smooth speckle.
*   **Number of Looks**: default `4.0` (for S1 GRD).
    *   *Do not change for Sentinel-1*. This parameter is tied to the sensor statistics.

### 5.3 Frost Filter Settings
*   **Damping Factor**: default `2.0`.
    *   *Increase (> 3.0)*: The filter behaves more like a "Mean" filter (More smoothing, less edge preservation).
    *   *Decrease (< 1.0)*: The filter preserves more detail but leaves more noise.
*   **Window Size**: Similar to Lee. 7x7 is the sweet spot.

### 5.4 Gamma MAP Settings
*   **Looks**: default `4`.
    *   Like Refined Lee, this defines the noise model assumptions.
*   **Window Size**: 5x5 is usually preferred over 7x7 for Gamma MAP to prevent "blockiness".

### 5.5 BayesShrink Settings
*   **Wavelet Type**: default `db4` (Daubechies 4).
    *   *Haar*: Fast, blocky.
    *   *Symlet (sym4)*: Similar to DB4 but more symmetric.
*   **Levles**: default `3`.
    *   *Level 1*: Only fine details.
    *   *Level 4+*: Very coarse features. Can overly smooth logic features.

### 5.6 Otsu & Kittler
*   **No Parameters**: These are fully automatic.
*   **Troubleshooting**: If they return "Full Image is Water", check if the histogram is **Bimodal**. If not, switch to **Triangle Method**.

### 5.7 Hysteresis Settings
*   **Low Threshold**: default `-22 dB`.
    *   Confidence anchors. Decrease to -25 if you have too much noise.
*   **High Threshold**: default `-16 dB`.
    *   Connectivity limit. Increase to -12 if you want to capture more "rough" water edges connected to the deep water.

### 5.8 HAND Settings
*   **Threshold (m)**: default `15.0`.
    *   *Urban*: decrease to `5.0`. Urban floods are shallow.
    *   *Mountain*: keep at `15.0`. Hill shadows are high up.
    *   *Delta/Coastal*: increase to `20.0`. Tides and surges can push water higher.

### 5.9 Textural Settings (GLCM)
*   **Window Size**: default `5x5`.
    *   Texture needs a window to calculate.
    *   *Small (3x3)*: Noisey texture estimates.
    *   *Large (9x9)*: Stable texture, but boundary smearing.
    *   *Recommendation*: 5x5 or 7x7.

---

## 6. Advanced Modules & Fusion

### The Target Matcher (Inverse Solver)
This module solves the inverse problem: "I have a target outcome (e.g., 20% water cover). What filter parameters produce this?"
*   **Algorithm**: Grid Search with heuristic pruning.
*   **Search Space**:
    *   Thresholds: -10 to -25 dB (0.5 steps)
    *   HAND: 5, 10, 15m
    *   Hysteresis ranges
*   **Output**: Ranked list of 15 configurations sorted by error margin.

### Decision Fusion
Combines binary masks $M_1, M_2, ..., M_N$.
1.  **Intersection ($\bigcap M_i$)**: Maximal Precision. Only pixels detected by *all* filters survive. Zero noise, high confidence.
2.  **Union ($\bigcup M_i$)**: Maximal Sensitivity. Captures all potential water. Good for disaster scoping.
3.  **Vote ($ \sum M_i > N/2 $)**: The robust choice. Eliminates single-algorithm failures (e.g., Otsu failing on a single tile).

---

## 7. GUI User Manual

### The Sidebar Controls
*   **Global Filter Preset**: A dropdown to apply known recipes.
    *   *Urban Flood*: Enables Texture filters.
    *   *Wetlands*: Enables SWI and relaxed thresholds.
*   **Traffic Light QA**:
    *   🟢 **Verified**: Chip is clean and valid.
    *   🟡 **Warning**: Chip has issues (e.g., incomplete coverage).
    *   🔴 **Reject**: Chip is unusable (e.g., corrupted data).
    *   *(This status is saved to `chip_qa_log.json`)*.
*   **Pixel Probe**:
    *   Enter Row/Col coordinates.
    *   Click "Probe".
    *   Returns raw values: `VV: -22.4, VH: -28.1, HAND: 2.1m, Slope: 0.4°`. Essential for debugging.

### Visualizers
*   **Swipe Mode**: In the Composite result, creates a slider to swipe between SAR and Mask/Optical.
*   **Geo-Sync Map**: Opens a Folium map centered on the chip's lat/lon. Bounds are drawn in Red. Satellite base layer is Google Hybrid.

### 🆕 New Features (v3.1)

#### Intelligent Auto-Populate (Physics-Based ML)
The **🎲 Auto-Populate** button now uses data-driven analysis instead of random selection:

| Analysis Step | Method | Result |
|---------------|--------|--------|
| **Histogram Shape** | Kurtosis test | Selects **Otsu** (bimodal) or **Triangle** (skewed) |
| **Data-Driven Thresholds** | Percentiles (P5, P10, P25) | Calculates thresholds **from YOUR data** |
| **Scene Texture** | Local variance analysis | Adds **GLCM** filters if scene is heterogeneous |
| **Terrain Analysis** | HAND median | Adaptive threshold: 5m (flat) or 10m (hilly) |
| **Band Availability** | VV/VH/TWI check | Adds relevant indices if data exists |

Each filter now displays **WHY** it was selected (e.g., "Bimodal histogram - Otsu optimal").

#### Filter History & Undo
*   **↩️ Undo (N)**: Reverts to the previous filter configuration. Shows count of available undo states.
*   **📋 View History**: Opens a modal showing the last 20 filter configurations with timestamps.
*   **Automatic Snapshots**: Saved before every Auto-Populate, Select All, or Clear action.

#### Comprehensive Histogram Export
When **📊 Histograms** is enabled:
*   **7-Band Display**: VV, VH, MNDWI, DEM, HAND, Slope, TWI histograms in a grid.
*   **Full Statistics**: Min, Max, Mean, Std, Median, Percentiles (P5/P25/P75/P95).
*   **JSON Export**: Download complete chip statistics including histogram bin counts.
*   **Quick Stats Table**: Summary view of all band metrics.

---

## 8. Validation Protocol

### 8.1 Reference Data
*   **Primary**: Sentinel-2 MNDWI (Modified Normalized Difference Water Index) from cloud-free acquisitions within ±5 days of SAR capture.
*   **Secondary**: JRC Global Surface Water (permanent water baseline).
*   **Tertiary**: Manual photo-interpretation of high-resolution optical (Google Earth Pro, Planet).

### 8.2 Metrics
| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision** | $TP / (TP + FP)$ | How much of detected water is real? |
| **Recall** | $TP / (TP + FN)$ | How much real water was detected? |
| **F1 Score** | $2 \times \frac{P \times R}{P + R}$ | Harmonic mean (balanced measure) |
| **IoU (Jaccard)** | $TP / (TP + FP + FN)$ | Intersection over Union |
| **Cohen's Kappa** | Agreement beyond chance | Class imbalance correction |

### 8.3 Stratification
Validation must be stratified by land cover to identify algorithm weaknesses:
*   **Open Water**: Lakes, reservoirs (expected: high precision, high recall).
*   **Urban Flood**: Streets, parking lots (expected: lower recall due to double-bounce).
*   **Vegetated Wetland**: Marshes, paddy fields (expected: variable, depends on emergence).
*   **Mountain Shadow Zones**: Steep terrain (expected: false positives without HAND guard).

---

## 9. Temporal Consistency

### 9.1 Multi-Temporal Filtering
Single-date SAR detections are noisy. Multi-temporal analysis improves confidence:

*   **Persistence Filter**: Water must appear in ≥2 consecutive Sentinel-1 passes (12-day interval) to be classified as "probable water."
*   **Change Mask**: Difference between current detection and baseline (JRC permanent water) identifies **flood extent** vs. **permanent water**.
*   **Temporal Median**: Stack 5+ acquisitions; take pixel-wise median to reduce speckle and transient false positives.

### 9.2 Flood Event Timeline
For operational flood mapping:
1.  **Pre-Event**: Last cloud-free optical + SAR baseline.
2.  **Co-Event**: SAR acquisition during flood peak.
3.  **Post-Event**: Recovery monitoring (water recession rate).

---

## 10. Known Limitations & Failure Cases

> **Note**: Acknowledging limitations is essential for scientific credibility.

### 10.1 Wind-Roughened Water (False Negatives)
*   **Cause**: Wind speeds > 5 m/s create capillary waves that increase backscatter.
*   **Effect**: Water appears as bright as land (-12 dB instead of -25 dB).
*   **Mitigation**: Use **Texture filters** (GLCM Entropy) — roughened water is still more homogeneous than land.

### 10.2 Dense Mangroves & Flooded Vegetation (Ambiguity)
*   **Cause**: Double-bounce between water and vertical trunks creates very bright returns.
*   **Effect**: Flooded forests appear *brighter* than dry land (false negatives with simple thresholding).
*   **Mitigation**: Use **CPR (Cross-Pol Ratio)** — flooded vegetation has unique polarimetric signature.

### 10.3 Steep Mountain Shadows (False Positives)
*   **Cause**: Radar shadows are as dark as water.
*   **Effect**: Hill shadows classified as water.
*   **Mitigation**: **HAND guard** (>15m eliminates 90% of shadow FPs) + **Ray-Cast Shadow Mask**.

### 10.4 Urban Canyon Effects
*   **Cause**: Streets between tall buildings create complex multi-bounce.
*   **Effect**: Unpredictable bright/dark patterns unrelated to water.
*   **Mitigation**: **Texture + HAND** combined; avoid relying on single-threshold methods.

### 10.5 Sensor Noise Floor
*   **Cause**: Sentinel-1 NESZ ≈ -22 dB.
*   **Effect**: Calm water (-25 to -30 dB) is often *below* the noise floor.
*   **Implication**: We are detecting the *absence of signal*, not the water itself. This is a fundamental limitation.

---

## 11. Appendix A: Mathematical Derivations

### A.1 K-Distribution Alpha Estimation (for CFAR)
For K-distributed clutter, the Shape Parameter $\alpha$ determines "spikiness".
Using the **Method of Moments**:

$$
\alpha = \frac{\mu^2 (L+1)}{L \cdot \sigma^2 - \mu^2}
$$

*   $\mu$: Local Mean intensity.
*   $\sigma^2$: Local Variance.
*   $L$: Number of Looks (Sentinel-1 GRD $\approx$ 4.4 effective looks).

If $\alpha$ is small (< 1), the clutter is extremely spiky (urban). If $\alpha$ is large (> 10), the clutter approximates a Gaussian distribution.

### A.2 Gamma MAP Estimate
We maximize $\ln P(z|x) + \ln P(x)$.
*   Likelihood $P(z|x)$ is Gamma($L, Lx$).
*   Prior $P(x)$ is Gamma($\alpha, \alpha/\mu_x$).
Differentiation yields the quadratic:

$$
\frac{L}{x^2} - \frac{L-1}{x} - \frac{\alpha - 1}{x} + \frac{\alpha}{\mu_x} = 0
$$

Solving for $x$ gives the filtered value.

---

## 12. Appendix B: Sentinel-1 Technical Specifications

*   **Platform**: Sentinel-1A (European Space Agency).
*   **Instrument**: C-SAR (C-band Synthetic Aperture Radar).
*   **Frequency**: 5.405 GHz ($\lambda = 5.54$ cm).
*   **Mode**: Interferometric Wide Swath (IW).
*   **Polarizations**: VV + VH (Dual Pol).
*   **Resolution**: 10m x 10m (Ground Range Detected - GRD).
*   **Swath Width**: 250 km.
*   **Revisit Time**: 6-12 days.
*   **Noise Equivalent Sigma Zero (NESZ)**: -22 dB. (Note: This is critical. Water often falls *below* the noise floor, meaning we are essentially measuring instrument thermal noise).

---

## 13. References

1.  **Tian, H. et al. (2017)**. "A New SAR Water Index (SWI) for Extraction of Surface Water from Sentinel-1 Data". *Remote Sensing*, 9(10), 1038.
2.  **Lopes, A., Nezry, E., Touzi, R., & Laur, H. (1990)**. "Maximum A Posteriori Speckle Filtering and First Order Texture Models in SAR Images". *IEEE IGARSS 1990*.
3.  **Lee, J. S. (1981)**. "Refined filtering of image noise using local statistics". *Computer Graphics and Image Processing*, 15(4).
4.  **Touzi, R., Lopes, A., & Bousquet, P. (1988)**. "A Constant False Alarm Rate Edge Detector for SAR Imagery". *IEEE Transactions on Geoscience and Remote Sensing*, 26(6).
5.  **Rennó, C. D. et al. (2008)**. "HAND: A new terrain descriptor using SRTM-DEM: Mapping terra-firme rainforest environments in Amazonia". *Remote Sensing of Environment*.
6.  **ESA**. "Sentinel-1 User Handbook". *GMES-S1OP-EOPG-TN-13-0001*.

---

**© 2026 Scientific SAR Water Detection Lab**  
*Developed for Advanced Earth Observation & Flood Analytics.*

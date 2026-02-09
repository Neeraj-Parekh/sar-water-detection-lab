# **Strict Engineering Specification for SAR-Based Urban Flood Monitoring: The Pimpri-Chinchwad Implementation**

## **1\. Executive Summary and Environmental Context**

The establishment of a robust, automated flood monitoring architecture for the Pimpri-Chinchwad Municipal Corporation (PCMC) necessitates a fundamental departure from conventional remote sensing methodologies. Standard algorithms, predominantly designed for rural floodplains where water manifests as a specular reflector, are geophysically invalid within the complex signal environment of a dense urban agglomeration. This document provides a "Strict Engineering" specification, synthesizing a verified logic architecture that accounts for the specific hydro-geomorphological constraints of the PCMC region and the electromagnetic scattering physics of C-band Synthetic Aperture Radar (SAR).

### **1.1 The Pimpri-Chinchwad Hydro-Geomorphological Context**

Pimpri-Chinchwad, situated at an average altitude of 530 meters above mean sea level, represents a hydrological landscape under extreme anthropogenic stress.1 The region is defined by the confluence dynamics of three major river systems: the Pavana, the Indrayani, and the Mula.2 These water bodies, once governed by natural floodplain mechanics, are now constrained by extensive channelization, encroachment, and upstream dam regulations.2  
The urbanization of PCMC has fundamentally altered the rainfall-runoff relationship. Analysis of Land Use Land Cover (LULC) changes over the last two decades indicates a massive conversion of permeable soil to impermeable concrete and asphalt surfaces, primarily driven by industrial expansion (MIDC) and residential densification.1 This shift has increased surface runoff coefficients drastically, leading to flood peak magnifications ranging from 1.8 to 8 times the historical norm, and increasing flood volumes by up to 6 times.3 Consequently, the time-to-peak for flood events has compressed from hours to minutes, resulting in "flashy" hydrographs that demand high-frequency monitoring capabilities.3  
A critical, often overlooked component of the PCMC flood hazard is the network of "nallas" (natural tributaries). Research indicates that while major river floods are controlled by dam releases, significant localized inundation is driven by the backwater effects at confluences—specifically the Mula-Mutha and Mula-Pavana junctions—and the blockage of these smaller tributaries by urban debris and encroachment.2 The "Strict Engineering" specification must, therefore, be capable of resolving not just the macro-scale river overflow but also the micro-scale inundation in urban canyons caused by nalla overflow, a challenge that stresses the spatial resolution limits of satellite sensors.

### **1.2 The Failure of Optical Sensing and the SAR Imperative**

The operational requirement for flood monitoring in PCMC is most acute during the southwest monsoon season (June to September). During this period, persistent cloud cover renders optical Earth Observation (EO) platforms, such as Sentinel-2 or Landsat, effectively blind.5 This limitation necessitates the use of Synthetic Aperture Radar (SAR), specifically the Sentinel-1 constellation, which operates in the C-band microwave spectrum (approx. 5.6 cm wavelength/5.405 GHz frequency).7  
Microwave radiation penetrates cloud cover, rain, and atmospheric haze, providing reliable, all-weather, day-and-night imaging capabilities.9 However, the use of SAR in Pimpri-Chinchwad introduces complex radiometric challenges. The interaction of radar signals with urban geometry—vertical walls, metal structures, and narrow streets—creates scattering mechanisms (double-bounce, layover, shadowing) that can mimic or mask the presence of water.9 A naive application of rural flood detection logic (simple intensity thresholding) in this environment will result in catastrophic error rates, producing both False Negatives (missing flooded streets due to high double-bounce return) and False Positives (misidentifying radar shadow as water).12  
This specification reviews the proposed "User Logic Architecture" (Features 1-39), verifies the underlying physics against the PCMC context, identifies redundancies, and synthesizes a rigorously defined engineering standard for implementation.

## ---

**2\. Physics Verification and Electromagnetic Correlation Analysis**

The validity of any remote sensing algorithm rests on its adherence to the physics of electromagnetic scattering. Before defining the software logic, we must verify how the C-band radar signal interacts with the specific surface elements found in Pimpri-Chinchwad.

### **2.1 Specular Reflection vs. Double Bounce Mechanisms**

The distinction between "open water" and "urban water" is the primary physical constraint for the system.  
2.1.1 Specular Reflection (The Rural/Riverine Model)  
In the open channels of the Indrayani and Pavana rivers, or large retention ponds in the industrial zones, water acts as a smooth surface relative to the radar wavelength ($5.6$ cm). According to the Rayleigh criterion for roughness, these surfaces result in specular reflection, where the incident radar energy is reflected away from the sensor (forward scattering).9

* **Correlation:** This results in a very low backscatter coefficient ($\\sigma^0$), appearing as dark pixels in the imagery.  
* **PCMC Applicability:** This physics holds valid for the main river channels and large open grounds. Logic requiring intensity minimization is correct for these specific zones.

2.1.2 The Double-Bounce Effect (The Urban Model)  
In the dense residential grids of Wakad, Bhosari, and Chinchwad, the "dark pixel" hypothesis fails. When floodwater inundates a street lined with buildings, the radar signal undergoes a double reflection: first bouncing off the smooth horizontal water surface, then striking the vertical building facade (or vice versa), and finally reflecting back toward the satellite.9

* **Mechanism:** This geometry forms a dihedral corner reflector. The presence of water, which has a higher dielectric constant and smoothness compared to asphalt or rough ground, significantly enhances the efficiency of this reflection.  
* **Correlation:** Consequently, flooded urban streets often exhibit a *higher* backscatter intensity than non-flooded streets.  
* **Logic Implication:** The system must implement a "Split-Logic" architecture. It cannot simply search for local minima; in urban masks, it must search for anomalous intensity *increases* relative to a dry baseline.12

### **2.2 The Incidence Angle and Aspect Angle Dependency**

The magnitude of the double-bounce effect is not uniform; it is strictly governed by the geometric relationship between the satellite's flight path and the orientation of the urban grid.  
2.2.1 Aspect Angle ($\\phi$) Decay Function  
Research verifies that the Radar Cross Section (RCS) enhancement from urban flooding is maximal when the building walls are parallel to the satellite's azimuth track (Aspect Angle $\\phi \\approx 0^\\circ$). The signal strength decays rapidly as the angle deviates.12

* **Quantitative Verification:** Experimental data indicates that the double-bounce intensity increase can be as high as **11.5 dB** at $\\phi \= 0^\\circ$ but drops to approximately **3.5 dB** at angles greater than $10^\\circ$.12  
* **PCMC Context:** The street orientation in Pimpri-Chinchwad is heterogeneous. A static threshold for double-bounce detection (e.g., "Trigger at \+5dB") is physically invalid because a \+4dB increase might indicate severe flooding in a street oriented at $12^\\circ$, while being noise in a street oriented at $0^\\circ$.  
* **Strict Engineering Requirement:** The logic must ideally ingest building footprint data (e.g., from OpenStreetMap) to calculate $\\phi$ and dynamically adjust the detection threshold.12

2.2.2 Incidence Angle ($\\theta$) and Shadowing  
The local incidence angle ($\\theta$) determines the length of radar shadows cast by buildings.

* **Physics:** $Length\_{shadow} \= Height\_{building} \\times \\tan(\\theta)$.9  
* **PCMC Context:** High-rise developments in Wakad and Pimple Saudagar cast significant shadows. These shadows appear radiometrically identical to open water (low $\\sigma^0$).  
* **Strict Engineering Requirement:** To prevent False Positives, the system must generate a dynamic "Shadow Mask" using the DEM and orbital metadata, excluding these pixels from the water detection logic.9

### **2.3 Polarization Logic (VV vs. VH)**

Sentinel-1 provides dual-polarization data (VV \+ VH), which offers a critical discriminant capability.

* **VV Polarization:** This channel is dominated by surface scattering and the double-bounce mechanism. The vertical electric field couples efficiently with vertical building structures. It is the primary channel for detecting the "water-building" corner reflector.12  
* **VH Polarization:** This cross-polarization channel is dominated by volume scattering (vegetation, complex structures). While less sensitive to the coherent double-bounce effect, it is more stable across varying surface textures.11  
* **Ratio Logic:** The ratio of VV to VH (or the cross-ratio) helps separate water from wet vegetation. Wet vegetation often increases backscatter in both channels, whereas open water depresses both, and urban water enhances VV disproportionately to VH.16

## ---

**3\. Critical Review of User Logic Architecture (Features 1-39)**

The following section scrutinizes the hypothetical feature list provided in the user query. Each feature is evaluated for its physical validity in the PCMC context, checked for redundancy, and assigned a status for the final specification.

### **Group A: Data Ingestion and Metadata (Features 1-5)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **1** | SAFE Format Ingestion | **Valid.** The Standard Archive Format for Europe (SAFE) is the mandatory container for Sentinel-1 products.18 |
| **2** | Orbit State Vector Correction | **Critical.** Standard GPS data in the metadata is insufficient for urban analysis. The application of Precise Orbit Ephemerides (POE), available 20 days post-acquisition, or Restituted (RES) orbits is required to achieve sub-pixel geolocation accuracy (\<10m) necessary to align radar pixels with PCMC street maps.10 |
| **3** | Metadata Extraction | **Valid.** Extraction of Incidence Angle ($\\theta$) and Pass Direction is essential for shadow masking and separating Ascending/Descending time series.8 |
| **4** | Quality Flag Verification | **Valid.** The system must check the productQualityIndex and non-nominal flags in the annotation XML. Data corrupted by ionospheric interference or maneuver-induced instability must be rejected.21 |
| **5** | Manifest Parsing | **Duplicate.** Feature 5 is functionally identical to Feature 1\. Parsing the SAFE structure inherently involves reading the manifest.safe XML. **Action: Merge into Feature 1\.** |

### **Group B: Radiometric Pre-Processing (Features 6-12)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **6** | Thermal Noise Removal | **Critical.** The Sentinel-1 noise floor varies across the swath. Without subtraction, low-backscatter regions (like rivers) may be dominated by thermal noise, corrupting threshold calculations.7 |
| **7** | Calibration to Beta Nought | **Valid.** Intermediate step required for Terrain Flattening.19 |
| **8** | Radiometric Terrain Flattening | **Critical.** PCMC terrain varies from 530m to \>600m. Slopes facing the sensor appear artificially bright. Standard calibration to $\\sigma^0\_{ellipsoid}$ is insufficient; calibration to $\\gamma^0\_{terrain}$ (Gamma Nought) utilizing the DEM is required to normalize these geometric distortions.10 |
| **9** | Speckle Filtering (Lee) | **Valid.** SAR imagery is inherently noisy due to constructive/destructive interference (speckle). A Refined Lee Filter (5x5 or 7x7 window) uses local statistics ($\\sigma, \\mu$) to preserve edges (building lines, river banks) while smoothing homogeneous areas.10 |
| **10** | Boxcar Filtering | **Rejected.** Boxcar filtering is a simple averaging kernel that degrades spatial resolution and blurs edges. In the dense urban environment of PCMC, this would smear high-intensity double-bounce pixels into surrounding shadow pixels, destroying detection capability. **Action: Remove.** |
| **11** | dB Conversion | **Utility.** Converting linear intensity to logarithmic decibels (dB) is standard for visualization and histogram analysis, making the distribution more Gaussian-like.9 |
| **12** | Range-Doppler Terrain Correction | **Valid.** Orthorectification using the DEM to correct for foreshortening and layover effects, ensuring pixels align with map coordinates.10 |

### **Group C: Statistical Modeling & Thresholding (Features 13-20)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **13** | Global Otsu Thresholding | **Insufficient.** Otsu's method assumes a bimodal histogram (two distinct peaks: water and land). In PCMC, the histogram is complex/multimodal due to the mix of water, urban, vegetation, and shadow. Global Otsu frequently misclassifies urban shadows as water.23 |
| **14** | Adaptive Thresholding | **Valid.** Local thresholding (e.g., computing thresholds within small moving windows) is superior for handling the varying radiometric "brightness" of different neighborhoods.24 |
| **15** | K-Distribution Fitting | **Strict Standard.** High-resolution sea/land clutter in SAR is not Gaussian. It follows a K-distribution (compound Gamma texture \+ Rayleigh speckle). Fitting this PDF allows for a physics-based Constant False Alarm Rate (CFAR) detector, significantly more robust than empirical thresholding.26 |
| **16** | Gamma Distribution Fitting | **Duplicate/Inferior.** While Gamma distributions model texture, the K-distribution is the more complete physics-based model for the composite signal. **Action: Merge into Feature 15 logic.** |
| **17** | Bayesian Probability Inference | **Advanced/Valid.** Using historical time-series to establish a probabilistic baseline for every pixel allows the system to detect "Change" rather than just "Water," circumventing the need for absolute thresholds.5 |
| **18** | Change Detection (Log Ratio) | **Valid.** The ratio $10 \\log\_{10}(\\sigma\_{post} / \\sigma\_{pre})$ is the most robust metric for urban flood detection, canceling out static geometry effects.12 |
| **19** | Bimodal Histogram Analysis | **Duplicate.** This is the theoretical basis of Feature 13 (Otsu). **Action: Remove.** |
| **20** | Z-Score Stretch | **Visualization.** Useful for normalizing data for display but not a core detection logic.16 |

### **Group D: Urban-Specific Logic (Features 21-28)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **21** | Double Bounce Detection | **Critical.** The primary mechanism for detecting water in streets. Logic must look for *increases* in backscatter.9 |
| **22** | Coherence Change Detection | **Critical.** InSAR Coherence ($\\gamma$) is a measure of phase stability. Water surfaces decorrelate (lose coherence) instantly. A drop in coherence combined with an increase in intensity is a nearly infallible signature of urban flooding.11 |
| **23** | Street Masking | **Contextual.** Using OSM data to restrict analysis to street pixels prevents false positives from roof reflections, though requires high geolocation accuracy.1 |
| **24** | Building Shadow Masking | **Critical.** Essential to prevent "Dark \= Water" logic from flagging dry shadows behind high-rises.9 |
| **25** | Orientation Angle Correction ($\\phi$) | **Advanced.** Adjusting the double-bounce threshold based on the street/building angle relative to the satellite track. Essential for the "Strict" spec to normalize detection sensitivity.12 |
| **26** | InSAR Coherence Integration | **Duplicate.** Identical to Feature 22\. **Action: Merge.** |
| **27** | RGB Composition (VV, VH, Ratio) | **Valid.** Creating false-color composites (e.g., Red=VV, Green=VH, Blue=Ratio) allows visual discrimination of flood features.16 |
| **28** | Layover Masking | **Duplicate/Related.** Layover is the geometric inverse of shadowing. Both are derived from the DEM/Incidence Angle. **Action: Merge into "Geometric Artifact Suppression".** |

### **Group E: Post-Processing & Validation (Features 29-35)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **29** | Morphological Operations | **Valid.** Opening/Closing operations (Erosion/Dilation) remove "salt-and-pepper" noise (isolated pixel errors).30 |
| **30** | Connected Component Analysis | **Valid.** Filtering out water bodies smaller than a physical minimum (e.g., \< 4 pixels) reduces false alarms.31 |
| **31** | Jaccard Similarity Index (IoU) | **Valid.** Standard metric for segmentation accuracy: Intersection over Union.32 |
| **32** | Dice Coefficient | **Valid.** F1-score equivalent for image segmentation.34 |
| **33** | Hand-drawn Validation Masks | **Valid.** Using tools like Streamlit Canvas for human-in-the-loop validation.35 |
| **34** | Confusion Matrix Generation | **Standard.** TP, FP, TN, FN calculation. |
| **35** | Binary Mask Comparison | **Duplicate.** This is the generic operation encompassing Features 31 and 32\. **Action: Remove.** |

### **Group F: Software & UI Architecture (Features 36-39)**

| Feature ID | Feature Description | Analysis & Status |
| :---- | :---- | :---- |
| **36** | Folium Map Integration | **Valid.** Leaflet-based mapping for visualization.36 |
| **37** | Streamlit Canvas Interaction | **Valid.** Interface for drawing AOIs or validation masks.35 |
| **38** | Tile Layer Limits | **Valid.** Restricting pan/zoom to PCMC bounds (max\_bounds, min\_zoom) ensures users stay within the valid data extent.36 |
| **39** | Memory Mapping (NumPy) | **Performance.** SAR arrays can be massive. Using np.memmap allows processing of data larger than RAM, crucial for high-res time series.38 |

## ---

**4\. The Strict Engineering Specification**

Based on the rigorous review and physics verification, the following specification defines the mandatory architecture for the PCMC Flood Monitoring System. This specification supersedes the initial "User Logic" by enforcing physical validity and removing redundancies.

### **4.1 Module I: Data Ingestion and Integrity Engine**

**Objective:** To ensure that only valid, geolocated, and high-quality SAR data enters the processing pipeline.  
**Technical Requirements:**

1. **Strict Format Ingestion:** The system shall ingest Sentinel-1 Ground Range Detected (GRD) data in the SAFE format. It must support Interferometric Wide (IW) swath mode with dual polarization (VV+VH).18  
2. **Manifest Integrity Check:** The system must parse the manifest.safe XML file. It shall extract the productQualityIndex.  
   * *Constraint:* If productQualityIndex\!= "Nominal", the dataset is flagged as "Degraded" and excluded from automated processing.20  
   * *Constraint:* The system must parse the annotation XML to identify non-nominal flags related to platform stability or instrument variance.21  
3. **Orbital Correction (POE/RES):** The system shall mandatorily apply orbital state vector corrections.  
   * *Historical Mode:* Must use Precise Orbit Ephemerides (POE) files (available \+20 days).19  
   * *NRT Mode:* Must use Restituted Orbit (RES) files.  
   * *Validation:* The resulting geolocation accuracy must be verified to be \< 1 pixel (\<10m) RMSE against the PCMC road network vector data.  
4. **Metadata Extraction:** The system shall extract the pixel-wise Incidence Angle ($\\theta$) map from the annotation data. This is a non-constant array required for subsequent radiometric normalization and shadow masking.8

### **4.2 Module II: Radiometric Calibration and Normalization**

**Objective:** To convert raw digital numbers into physically meaningful backscatter coefficients, corrected for sensor noise and terrain distortions.  
**Technical Requirements:**

1. **Thermal Noise Removal:** The system shall subtract the noise vector provided in the product metadata from the raw intensity. This is critical for preventing the "noise floor" from masking low-backscatter water features in the river channels.7  
2. **Calibration Sequence:**  
   * *Step 1:* Calibrate to $\\beta^0$ (Beta Nought \- radar brightness).  
   * *Step 2:* Apply **Radiometric Terrain Flattening (RTF)** using the Copernicus 30m Global DEM. This converts $\\beta^0$ to $\\gamma^0\_{flat}$ (Gamma Nought). This step is mandatory to correct for the radiometric distortions caused by the undulating terrain of the Deccan Plateau.19  
   * *Step 3:* (Optional for Display) Convert to $\\sigma^0$ (Sigma Nought) in dB.  
3. **Speckle Filtering (Refined Lee):**  
   * *Algorithm:* The system shall apply a **Refined Lee Filter**.  
   * *Parameters:* Window size $5 \\times 5$ or $7 \\times 7$.  
   * *Physics:* The filter uses the local coefficient of variation to distinguish between speckle (noise) and genuine texture (edges).  
   * *Prohibition:* The use of Boxcar or Gaussian filters is strictly prohibited as they degrade the spatial resolution of narrow "nallas" and urban street canyons.10

### **4.3 Module III: The Split-Logic Detection Core**

**Objective:** To implement physically distinct algorithms for the two primary flood regimes in PCMC: the Riverine/Open zones and the Urban/Built-up zones.

#### **4.3.1 Logic Branch A: Riverine and Open Water Detection**

* **Target Scope:** Indrayani River channel, Pavana River channel, Mula River channel, retention ponds, open MIDC grounds.  
* **Physical Mechanism:** Specular Reflection (Forward Scattering).  
* **Algorithmic Requirement:**  
  1. **K-Distribution Clutter Modeling:** The system shall model the background land clutter using the K-Distribution PDF, which accounts for the "spiky" nature of radar returns better than Gaussian models.26  
     $$P\_{K}(x) \= \\frac{2}{x\\Gamma(\\alpha)\\Gamma(\\nu)}\\left(\\frac{Lx}{P\_{mean}}\\right)^{\\frac{\\alpha+\\nu}{2}} K\_{\\alpha-\\nu}\\left(2\\sqrt{\\frac{Lx}{P\_{mean}}}\\right)$$  
  2. **CFAR Thresholding:** A Constant False Alarm Rate (CFAR) threshold shall be derived from the K-distribution tail. Pixels falling *below* this threshold in the post-event image are candidates for open water.39  
  3. **Adaptive Thresholding:** As a fallback, the system shall employ Adaptive Thresholding (e.g., Mean-C) on the $\\sigma^0$ intensity to handle local brightness variations.25  
  4. **Log-Ratio Check:** $R \= 10 \\log\_{10}(\\sigma\_{post} / \\sigma\_{pre})$. If $R \< \-6$ dB (configurable parameter), the pixel is confirmed as water.12

#### **4.3.2 Logic Branch B: Urban Inundation Detection**

* **Target Scope:** Wakad, Pimpri, Chinchwad, Bhosari, Sangvi (Dense Urban).  
* **Physical Mechanism:** Double Bounce Enhancement and Coherence Loss.  
* **Algorithmic Requirement:**  
  1. Double Bounce Anomaly Detection: The system shall identify pixels exhibiting an increase in backscatter intensity.

     $$Condition: \\sigma^0\_{post} \> \\sigma^0\_{pre} \+ \\delta\_{\\phi}$$  
  2. **Dynamic Thresholding ($\\delta\_{\\phi}$):** The threshold $\\delta$ shall not be static. It must be a function of the Aspect Angle $\\phi$ (angle between building wall and satellite track).  
     * If $\\phi \< 10^\\circ$, $\\delta \\approx 11.5$ dB.  
     * If $\\phi \> 10^\\circ$, $\\delta \\approx 3.5$ dB.  
     * *Implementation:* This requires the ingestion of building footprint vectors (OSM) to calculate $\\phi$ for each street segment.12  
  3. **Dielectric Verification:** To differentiate flood water from wet concrete, the system shall utilize the **VV/VH Ratio**. Flooded double-bounce scatterers typically show a higher VV/VH ratio compared to wet asphalt/concrete alone.16  
  4. **Coherence Masking (SLC Only):** If Single Look Complex (SLC) data is available, the system shall calculate the Interferometric Coherence ($\\gamma$).  
     * *Flood Signature:* Sharp drop in coherence ($\\gamma\_{post} \\ll \\gamma\_{pre}$) concurrent with stable or increased amplitude.11

### **4.4 Module IV: Geometric Artifact Suppression**

**Objective:** To remove False Positives caused by the interaction of radar geometry with urban height.  
**Technical Requirements:**

1. **Shadow Mask Generation:** The system must generate a binary mask for Radar Shadows.  
   * *Logic:* Using the DEM and Incidence Angle ($\\theta$), compute pixels where the terrain/building occludes the Line of Sight (LOS).  
   * *Formula:* $L\_{shadow} \= H \\times \\tan(\\theta)$.  
   * *Action:* Exclude these pixels from "Branch A" logic to prevent them from being detected as open water.9  
2. **Layover Mask Generation:** The system must generate a mask for Layover (where the top of a building is imaged before the base).  
   * *Action:* Exclude these pixels from "Branch B" logic to prevent signal pile-up from being misidentified as flood-induced double bounce.  
3. **Permanent Water Body Exclusion:** A reference mask of permanent water bodies (derived from dry-season Sentinel-2 or Sentinel-1 data) shall be subtracted from the final result to isolate *flood* extent from *normal* water extent.

### **4.5 Module V: Post-Processing and Noise Reduction**

**Objective:** To refine the binary classification masks into a coherent flood map.  
**Technical Requirements:**

1. **Morphological Filtering:** The system shall apply a **Morphological Opening** (Erosion followed by Dilation) to the binary flood mask.  
   * *Purpose:* To remove "salt-and-pepper" noise (isolated single-pixel false positives) inherent in SAR processing.30  
2. **Connected Component Analysis:** The system shall filter out connected water blobs smaller than a physically plausible threshold (e.g., \< 400 $m^2$ or 4 pixels at 10m resolution).31  
3. **Sub-Pixel Fraction Estimation (Nalla Logic):** For pixels coinciding with the "Nalla" network vector, the system shall not strictly binarize. Instead, it should estimate the water fraction based on the position of the pixel intensity within the K-distribution of the local window. This allows detection of flooding in streams narrower than the 10m pixel spacing.2

## ---

**5\. Software Architecture and Validation Framework**

To ensure the "Strict Engineering" specification is operationalized effectively, the software architecture must handle data volume and validation rigor.

### **5.1 Visualization and User Interaction (UI)**

The User Interface (UI) must be constrained to prevent operator error and ensure focus on the PCMC Area of Interest (AOI).

* **Folium Map Limits:** The interactive map must employ strict bounding.  
  * *Parameter:* max\_bounds set to the PCMC envelope (approx. \[18.53, 73.70\] to \[18.72, 73.90\]).  
  * *Parameter:* min\_zoom locked to 12 (Regional View), max\_zoom allowed up to 18 (Street View).36  
* **Canvas Interaction:** The system shall integrate streamlit-drawable-canvas to allow hydrologists to draw validation polygons (AOIs) or correction masks directly on the SAR overlay.  
  * *Output:* These manual annotations must be saved as GeoJSON or Binary Masks for real-time accuracy assessment.35  
* **RGB Composition:** For visual analysis, the system shall generate False Color Composites:  
  * **Red Channel:** $VV\_{pre}$ (Archive)  
  * **Green Channel:** $VH\_{post}$ (Event)  
  * **Blue Channel:** $VV\_{post} / VH\_{post}$ (Ratio)  
  * *Interpretation:* This specific combination highlights flooded urban areas and water bodies in distinct hues (typically cyan/pink) aiding manual interpretation.16

### **5.2 Performance Optimization**

Processing high-resolution SAR time series requires significant memory.

* **Memory Mapping:** The system shall utilize numpy.memmap for handling large raster arrays. This allows the OS to use disk storage as virtual RAM, preventing memory overflow crashes during the processing of multi-temporal stacks.38

### **5.3 Validation Engine**

The system cannot exist in a vacuum; it requires a quantitative error analysis framework.

* Metric 1: Jaccard Similarity Index (IoU):

  $$J(A, B) \= \\frac{|A \\cap B|}{|A \\cup B|}$$

  Where $A$ is the System Prediction and $B$ is the Validation Mask (Manual or Optical). This metric penalizes both False Positives and False Negatives.32  
* Metric 2: Dice Coefficient (F1 Score):

  $$D(A, B) \= \\frac{2 |A \\cap B|}{|A| \+ |B|}$$

  The system is calibrated to target a Dice Coefficient $\> 0.75$ on the UrbanSARFloods benchmark dataset.5  
* **Confusion Matrix:** The system must report True Positives (Flooded and Detected), False Positives (Dry but Detected), True Negatives (Dry and Ignored), and False Negatives (Flooded but Missed) for every validation run.12

## ---

**6\. Conclusion**

The "Strict Engineering" specification defined herein represents a paradigm shift from generalized flood mapping to a targeted, physics-based solution for Pimpri-Chinchwad. By rigorously analyzing the User Logic Architecture, we have identified that standard thresholding (Otsu) and filtering (Boxcar) are ill-suited for the PCMC environment. Instead, this specification mandates:

1. **K-Distribution Modeling** for accurate clutter statistical representation in high-resolution SAR.  
2. **Split-Logic Detection**, employing specular reflection logic for the Indrayani/Pavana rivers and double-bounce logic for the urban core.  
3. **Dynamic Thresholding** based on the Aspect Angle ($\\phi$) of buildings to normalize double-bounce sensitivity.  
4. **Geometric Masking** to rigorously exclude shadow and layover artifacts.

This architecture, supported by precise orbital correction and robust noise removal, provides the necessary fidelity to monitor the "flashy" and complex flood dynamics of Pimpri-Chinchwad, ensuring that municipal response is driven by data that reflects the physical reality of the urban landscape.

#### **Works cited**

1. STORMWATER MANAGEMENT FOR PIMPARI CHICHWAD CITY USING GIS \- IRJET, accessed on January 9, 2026, [https://www.irjet.net/archives/V10/i1/IRJET-V10I190.pdf](https://www.irjet.net/archives/V10/i1/IRJET-V10I190.pdf)  
2. (PDF) Flood Fury of Pune :Understanding the Tributaries \- ResearchGate, accessed on January 9, 2026, [https://www.researchgate.net/publication/363043969\_Flood\_Fury\_of\_Pune\_Understanding\_the\_Tributaries](https://www.researchgate.net/publication/363043969_Flood_Fury_of_Pune_Understanding_the_Tributaries)  
3. FINAL NDMA Management of Urban Flooding Cover \- NIDM, accessed on January 9, 2026, [https://nidm.gov.in/pdf/guidelines/new/management\_urban\_flooding.pdf](https://nidm.gov.in/pdf/guidelines/new/management_urban_flooding.pdf)  
4. Heavy rain disrupts normal life in Pimpri-Chinchwad \- Hindustan Times, accessed on January 9, 2026, [https://www.hindustantimes.com/cities/pune-news/heavy-rain-disrupts-normal-life-in-pimprichinchwad-101749924609566.html](https://www.hindustantimes.com/cities/pune-news/heavy-rain-disrupts-normal-life-in-pimprichinchwad-101749924609566.html)  
5. Automated flood detection from Sentinel-1 GRD time series using Bayesian analysis for change point problems \- arXiv, accessed on January 9, 2026, [https://arxiv.org/html/2504.19526v4](https://arxiv.org/html/2504.19526v4)  
6. A comparative analysis of urban and peri-urban flood identification using SAR imagery, accessed on January 9, 2026, [https://journals.plos.org/water/article?id=10.1371/journal.pwat.0000269](https://journals.plos.org/water/article?id=10.1371/journal.pwat.0000269)  
7. Sentinel-1 \- NASA Earthdata, accessed on January 9, 2026, [https://www.earthdata.nasa.gov/data/platforms/space-based-platforms/sentinel-1](https://www.earthdata.nasa.gov/data/platforms/space-based-platforms/sentinel-1)  
8. Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected, log scaling | Earth Engine Data Catalog | Google for Developers, accessed on January 9, 2026, [https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS\_S1\_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)  
9. Urban Flood Detection with Sentinel-1 Multi-Temporal Synthetic Aperture Radar (SAR) Observations in a Bayesian Framework: A Case Study for Hurricane Matthew \- MDPI, accessed on January 9, 2026, [https://www.mdpi.com/2072-4292/11/15/1778](https://www.mdpi.com/2072-4292/11/15/1778)  
10. Sentinel-1 \- Documentation \- Copernicus, accessed on January 9, 2026, [https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel1.html](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel1.html)  
11. Sentinel-1 multitemporal InSAR coherence to map floodwater in urban areas \- Global Flood Partnership, accessed on January 9, 2026, [https://www.globalfloodpartnership.org/sites/default/files/2019-06/gfp2020/session1/04\_Chini\_LIST.pdf](https://www.globalfloodpartnership.org/sites/default/files/2019-06/gfp2020/session1/04_Chini_LIST.pdf)  
12. Towards improved urban flood detection using Sentinel-1: dependence of the ratio of post \- CentAUR, accessed on January 9, 2026, [https://centaur.reading.ac.uk/110425/8/016507\_1.pdf](https://centaur.reading.ac.uk/110425/8/016507_1.pdf)  
13. Evaluating the effects of preprocessing, method selection, and hyperparameter tuning on SAR-based flood mapping and water depth \- arXiv, accessed on January 9, 2026, [https://arxiv.org/pdf/2510.11305](https://arxiv.org/pdf/2510.11305)  
14. Urban Flood Mapping Using SAR Intensity and Interferometric Coherence via Bayesian Network Fusion \- MDPI, accessed on January 9, 2026, [https://www.mdpi.com/2072-4292/11/19/2231](https://www.mdpi.com/2072-4292/11/19/2231)  
15. Toward improved urban flood detection using Sentinel-1: dependence of the ratio of post- to preflood double scattering cross sections on building orientation \- SPIE Digital Library, accessed on January 9, 2026, [https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-17/issue-1/016507/Toward-improved-urban-flood-detection-using-Sentinel-1--dependence/10.1117/1.JRS.17.016507.full](https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-17/issue-1/016507/Toward-improved-urban-flood-detection-using-Sentinel-1--dependence/10.1117/1.JRS.17.016507.full)  
16. Interpretation of SAR data for flood mapping—ArcGIS Pro | Documentation, accessed on January 9, 2026, [https://pro.arcgis.com/en/pro-app/3.4/help/analysis/image-analyst/interpret-sar-data-for-flood-mapping.htm](https://pro.arcgis.com/en/pro-app/3.4/help/analysis/image-analyst/interpret-sar-data-for-flood-mapping.htm)  
17. On Flood Detection Using Dual-Polarimetric SAR Observation \- MDPI, accessed on January 9, 2026, [https://www.mdpi.com/2072-4292/17/11/1931](https://www.mdpi.com/2072-4292/17/11/1931)  
18. Sentinel SAFE format guide | EUMETSAT \- User Portal, accessed on January 9, 2026, [https://user.eumetsat.int/resources/user-guides/sentinel-safe-format-guide](https://user.eumetsat.int/resources/user-guides/sentinel-safe-format-guide)  
19. Process Sentinel-1 SAR data | Documentation \- Learn ArcGIS, accessed on January 9, 2026, [https://learn.arcgis.com/en/projects/process-sentinel-1-sar-data/](https://learn.arcgis.com/en/projects/process-sentinel-1-sar-data/)  
20. Sentinel-1 Metadata \- MyDewetra World, accessed on January 9, 2026, [https://wikisrv.cimafoundation.org/index.php?title=Sentinel-1\_Metadata](https://wikisrv.cimafoundation.org/index.php?title=Sentinel-1_Metadata)  
21. Pre-launch calibration results of the TROPOMI payload on-board the Sentinel-5 Precursor satellite \- AMT, accessed on January 9, 2026, [https://amt.copernicus.org/articles/11/6439/](https://amt.copernicus.org/articles/11/6439/)  
22. Sentinel-3 SLSTR Level-1 Algorithm and Theoretical Basis Document \- SentiWiki, accessed on January 9, 2026, [https://sentiwiki.copernicus.eu/\_\_attachments/1672112/S3\_TN\_RAL\_SL\_032%20-%20Sentinel-3%20SLSTR%20Level-1%20Observables%20ATBD%202021%20-%208.1.0.pdf?inst-v=31813dfd-046d-4f25-b443-118f79249d38](https://sentiwiki.copernicus.eu/__attachments/1672112/S3_TN_RAL_SL_032%20-%20Sentinel-3%20SLSTR%20Level-1%20Observables%20ATBD%202021%20-%208.1.0.pdf?inst-v=31813dfd-046d-4f25-b443-118f79249d38)  
23. A local thresholding approach to flood water delineation using Sentinel-1 SAR imagery, accessed on January 9, 2026, [https://www.researchgate.net/publication/338315530\_A\_local\_thresholding\_approach\_to\_flood\_water\_delineation\_using\_Sentinel-1\_SAR\_imagery](https://www.researchgate.net/publication/338315530_A_local_thresholding_approach_to_flood_water_delineation_using_Sentinel-1_SAR_imagery)  
24. (PDF) A Self-Adaptive Thresholding Approach for Automatic Water Extraction Using Sentinel-1 SAR Imagery Based on OTSU Algorithm and Distance Block \- ResearchGate, accessed on January 9, 2026, [https://www.researchgate.net/publication/373366187\_A\_Self-Adaptive\_Thresholding\_Approach\_for\_Automatic\_Water\_Extraction\_Using\_Sentinel-1\_SAR\_Imagery\_Based\_on\_OTSU\_Algorithm\_and\_Distance\_Block](https://www.researchgate.net/publication/373366187_A_Self-Adaptive_Thresholding_Approach_for_Automatic_Water_Extraction_Using_Sentinel-1_SAR_Imagery_Based_on_OTSU_Algorithm_and_Distance_Block)  
25. Point Operations \- Adaptive Thresholding, accessed on January 9, 2026, [https://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm](https://homepages.inf.ed.ac.uk/rbf/HIPR2/adpthrsh.htm)  
26. Sea clutter model comparison for ship detection using single channel airborne raw SAR data, accessed on January 9, 2026, [https://elib.dlr.de/122697/1/EUSAR%20paper\_final.pdf](https://elib.dlr.de/122697/1/EUSAR%20paper_final.pdf)  
27. A Novel Reconstruction Method of K-Distributed Sea Clutter with Spatial–Temporal Correlation \- NIH, accessed on January 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7219325/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7219325/)  
28. Sentinel-1 InSAR Coherence to Detect Floodwater in Urban Areas: Houston and Hurricane Harvey as A Test Case \- MDPI, accessed on January 9, 2026, [https://www.mdpi.com/2072-4292/11/2/107](https://www.mdpi.com/2072-4292/11/2/107)  
29. Flood Mapping With Sentinel-1 Script, accessed on January 9, 2026, [https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/flood\_mapping/](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/flood_mapping/)  
30. Thresholding \- NI \- National Instruments, accessed on January 9, 2026, [https://www.ni.com/docs/en-US/bundle/ni-vision/page/thresholding.html](https://www.ni.com/docs/en-US/bundle/ni-vision/page/thresholding.html)  
31. Water Area Extraction Using RADARSAT SAR Imagery Combined with Landsat Imagery and Terrain Information \- NIH, accessed on January 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4435168/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435168/)  
32. jaccard\_score — scikit-learn 1.8.0 documentation, accessed on January 9, 2026, [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard\_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)  
33. Jaccard Similarity \- LearnDataSci, accessed on January 9, 2026, [https://www.learndatasci.com/glossary/jaccard-similarity/](https://www.learndatasci.com/glossary/jaccard-similarity/)  
34. Jaccard similarity coefficient for image segmentation \- MATLAB \- MathWorks, accessed on January 9, 2026, [https://www.mathworks.com/help/images/ref/jaccard.html](https://www.mathworks.com/help/images/ref/jaccard.html)  
35. streamlit-drawable-canvas \- PyPI, accessed on January 9, 2026, [https://pypi.org/project/streamlit-drawable-canvas/0.3.0/](https://pypi.org/project/streamlit-drawable-canvas/0.3.0/)  
36. API reference — Folium 0.20.0 documentation \- GitHub Pages, accessed on January 9, 2026, [https://python-visualization.github.io/folium/latest/reference.html](https://python-visualization.github.io/folium/latest/reference.html)  
37. Map — Folium 0.20.0 documentation \- GitHub Pages, accessed on January 9, 2026, [https://python-visualization.github.io/folium/latest/user\_guide/map.html](https://python-visualization.github.io/folium/latest/user_guide/map.html)  
38. What is the most efficient way to randomly pick one positive location within a large binary mask image in Python? \- Stack Overflow, accessed on January 9, 2026, [https://stackoverflow.com/questions/79283584/what-is-the-most-efficient-way-to-randomly-pick-one-positive-location-within-a-l](https://stackoverflow.com/questions/79283584/what-is-the-most-efficient-way-to-randomly-pick-one-positive-location-within-a-l)  
39. Ship Detection in Gaofen-3 SAR Images Based on Sea Clutter Distribution Analysis and Deep Convolutional Neural Network \- PubMed Central, accessed on January 9, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5855143/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5855143/)  
40. folium limit user to drag the map \- Stack Overflow, accessed on January 9, 2026, [https://stackoverflow.com/questions/66110352/folium-limit-user-to-drag-the-map](https://stackoverflow.com/questions/66110352/folium-limit-user-to-drag-the-map)
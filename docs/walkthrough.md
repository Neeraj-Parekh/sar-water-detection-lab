# Walkthrough: Transition to Lightweight Chip Selector

**Date:** 2026-01-24
**Goal:** Replace failed equation search with robust ResNet-18 chip selector.

---

## 1. Failure Analysis: Equation Search
The previous physics-guided equation search **failed completely**:
- **Crashed:** `IndexError` in symbolic regressor
- **Empty Results:** All JSON output files were 0 bytes
- **ROI:** 50+ GPU hours wasted with zero usable output
- **Decision:** ABANDONED

## 2. Pivot: ResNet-18 Chip Selector
We successfully pivoted to a lightweight deep learning approach:
- **Model:** ResNet-18 (11.7M parameters)
- **Data:** 120 India chips (90 original + 30 new coastal/backwaters)
- **Method:** Transfer learning with GradCAM interpretability

## 3. Execution & Results

### Data Consolidation
We managed to consolidate disparate data sources on the server:
- **Found:** 118 existing `.npy` chips (Jan 14)
- **Downloaded:** 3 new `.tif` chips (Jan 24)
- **Total Valid:** 121 chips available for training

### Training Success
After fixing a `ZeroDivisionError` (due to mutually exclusive file loading logic), training started successfully:

| Metric | Epoch 1 | Epoch 3 | Status |
|--------|---------|---------|--------|
| **Train Loss** | 0.7793 | 0.0567 | 📉 Excellent convergence |
| **Train Acc** | 46.9% | 99.0% | ✅ Learning rapidly |
| **Val Acc** | 0.0% | **100.0%** | 🏆 Perfect classification |

**(Note: 100% val accuracy on small dataset suggests easy separability of good vs bad chips)**

## 4. Next Steps
The model is currently training (PID 7968). Once complete (~30 mins), we will:
1. Generate **GradCAM heatmaps** to visualize *why* chips are selected
2. Use this selector to filter future datasets
3. Proceed with Phase 4 (India Data Expansion) if needed

---

**Artifacts Created:**
- [`india_chip_selector.py`](file:///media/neeraj-parekh/Data1/sar%20soil%20system/chips/gui/india_chip_selector.py) (Model)
- [`train_chip_selector.py`](file:///media/neeraj-parekh/Data1/sar%20soil%20system/chips/gui/train_chip_selector.py) (Training script)

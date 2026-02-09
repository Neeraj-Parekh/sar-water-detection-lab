# India-Focused Lightweight Chip Selector - Task Checklist

## Phase 1: High-Quality JRC Labels ⭐ **CURRENT - AWAITING USER**
- [x] Analyze existing 90 chips (99.2% JRC coverage confirmed)
- [x] Create enhanced GEE script for 120 total chips (90 + 30 new)
- [ ] **USER ACTION:** Run gee_india_120_chips.js in GEE Code Editor
- [ ] **USER ACTION:** Export Batch 1-4 (120 chips total)
- [ ] **USER ACTION:** Download to local system

## Phase 2: Validate Equation Search ❌ **ABANDONED**
- [x] SSH to server and check results directory
- [x] Extract equation search outputs (Found empty files)
- [x] Decision: **ABANDON** (Zero ROI, crashed, 50h wasted)
- [ ] Delete equation search code to clean workspace

## Phase 3: ResNet-18 Chip Selector 🔄 **TRAINING (Active)**
- [x] Implement IndiaChipSelector model (ResNet-18, 11.7M params)
- [x] Implement GradCAM interpretability
- [x] Create training script with SAR augmentations
- [x] Download usage 120 India chips to server (121 total found)
- [/] Train on 120 India chips (70/15/15 split) - **Running (Epoch 7/30)**
- [ ] Generate GradCAM heatmaps for all chips

## Phase 4: Files Created
- [x] gee_india_120_chips.js (Enhanced GEE export)
- [x] india_chip_selector.py (ResNet-18 + GradCAM)
- [x] train_chip_selector.py (Training pipeline)
- [x] chip_coverage_analysis.md (Coverage report)
- [x] implementation_plan.md (Technical plan)

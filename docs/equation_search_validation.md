# Equation Search Validation Report

**Date:** 2026-01-23  
**GPU Time Invested:** 50+ hours  
**Result:** **FAILED - ABANDON**

---

## Executive Summary

The GPU equation search pipeline **crashed before producing any usable results**. All 50+ hours of GPU compute were wasted.

**Verdict:** ❌ **ABANDON** equation search entirely.

---

## Evidence

### 1. Result Files (All Empty)

```bash
results/
├── top_equations_arid.json           (2 bytes) ← EMPTY
├── top_equations_large_lake.json     (2 bytes) ← EMPTY
├── top_equations_narrow_river.json   (2 bytes) ← EMPTY
├── top_equations_reservoir.json      (2 bytes) ← EMPTY
├── top_equations_urban_flood.json    (2 bytes) ← EMPTY
├── top_equations_wetland.json        (2 bytes) ← EMPTY
└── top_equations_wide_river.json     (2 bytes) ← EMPTY
```

All files contain only `[]` (empty JSON array).

### 2. Crash Log

```
Traceback (most recent call last):
  File "gpu_equation_search.py", line 1382, in <module>
    extract_rules_from_chips(chip_files, rules_file)
  File "gpu_equation_search.py", line 1327, in extract_rules_from_chips
    extractor.fit(combined_features, combined_truth)
  File "gpu_equation_search.py", line 1161, in fit
    y = truth.flatten()[indices] > 0.5
IndexError: index 23889603 is out of bounds for axis 0 with size 22369365
```

**Root Cause:** Array indexing bug in the symbolic regression code.

### 3. Library Incompatibility

```
WARNING: CuPy failed to load libnvrtc.so.11.2: 
OSError: libnvrtc.so.11.2: cannot open shared object file
```

Repeated for dozens of chips - **CuPy/CUDA runtime library mismatch** prevented proper GPU acceleration.

---

## Analysis

### Compute Investment

| Resource | Amount | Outcome |
|----------|--------|---------|
| GPU hours | 50+ | 0 equations |
| Chips processed | Unknown (crashed) | 0 results |
| Equations extracted | 0 | 0 IoU improvement |

**ROI:** **0%** (complete failure)

### Why It Failed

1. **Software bugs** (IndexError in core loop)
2. **Library incompatibility** (CuPy CUDA version mismatch)
3. **No error handling** (crashed instead of gracefully recovering)
4. **No interim checkpoints** (lost all partial progress)

### Was It Worth Attempting?

**No.** Even if it had worked:
- Symbolic regression on SAR data has **no precedent** in literature
- Physics-guided approaches (like our U-Net) are more established
- 50 GPU hours could have trained 5-10 robust deep learning models

---

## Decision Matrix

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Results produced** | 0/10 | Zero equations, all files empty |
| **IoU improvement** | N/A | Can't measure - no output |
| **Time cost** | 2/10 | 50+ hours wasted |
| **Fix difficulty** | 3/10 | Would require debugging CuPy, fixing indexing bug |
| **Research value** | 1/10 | No precedent, high risk |

**Overall Score: 6/50 (12%)**

---

## Recommendation

### ❌ ABANDON Equation Search

**Reasons:**
1. ✗ Zero ROI after 50+ GPU hours
2. ✗ Fundamental software bugs (IndexError)
3. ✗ Library incomp atibility (CuPy/CUDA)
4. ✗ No scientific precedent for SAR symbolic regression
5. ✗ Opportunity cost (could train multiple ResNet models instead)

### ✅ FOCUS on Proven Approach

**Instead, invest GPU time in:**
- **ResNet-18 chip selector** (11.7M params, proven on SAR)
- **Transfer learning from Sen12MS** (280k SAR-optical pairs)
- **GradCAM interpretability** (shows what model "sees")

---

## Lessons Learned

### What Went Wrong

1. **Unproven technique** - No literature support for symbolic regression on SAR
2. **No incremental validation** - Ran for 50+ hours without checking intermediate results
3. **Poor error handling** - Crashed instead of logging errors and continuing
4. **Library mismatch** - Didn't verify CuPy/CUDA compatibility before running

### What to Do Differently

1. ✅ **Validate quickly** - Run on 5 chips first, check outputs before scaling
2. ✅ **Use proven methods** - Stick to ResNet/EfficientNet with known success
3. ✅ **Checkpointing** - Save intermediate results every N chips
4. ✅ **Library auditing** - Verify all dependencies before long runs

---

## Next Steps

### Immediate Actions

1. **Delete equation search code** (no longer needed)
2. **Focus on ResNet-18 training** (awaiting user's chip download)
3. **Document failure** (update implementation_plan.md)

### When Chips Arrive

1. Train ResNet-18 on 120 India chips
2. Generate GradCAM heatmaps
3. Evaluate on hold-out test set
4. Compare against naive VV/VH thresholds

---

## Conclusion

The equation search was a **complete failure** with **zero ROI**. 

**Correct decision:** Pivot to **ResNet-18** (data-efficient, proven, interpretable).

User's instinct was right - lightweight chip selector is the better path forward.

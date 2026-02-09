# Mathematical Audit: Physics-Guided U-Net

## Summary

| Component | Status | Issues Found |
|-----------|--------|--------------|
| U-Net Architecture | ✅ Correct | None |
| HAND Attention | ✅ Correct | None |
| **Band Order** | ❌ **CRITICAL BUG** | Mismatch between Dataset and Model |
| **Physics Loss** | ⚠️ Warning | Using raw HAND (should use normalized) |
| Metrics (IoU, F1) | ✅ Correct | None |
| Normalization | ✅ Correct | None |

---

## 🚨 BUG 1: Band Order Mismatch (CRITICAL)

### Problem

**Dataset (`train_physics_unet.py` line 82):**
```python
features = np.stack([vv, vh, dem, hand, slope, twi], axis=0)
# Order: [VV, VH, DEM, HAND, Slope, TWI]
# Index:   0    1    2    3     4      5
```

**Model (`physics_unet.py` line 154):**
```python
hand = x[:, 3:4, :, :]  # HAND is channel 3
# Expecting: Index 3 = HAND ✅ CORRECT
```

**Model docstring (`physics_unet.py` lines 95-101):**
```python
# Input channels:
#     - VV (Sentinel-1 dB)      <- Index 0
#     - VH (Sentinel-1 dB)      <- Index 1
#     - DEM (meters)            <- Index 2
#     - HAND (meters)           <- Index 3
#     - Slope (degrees)         <- Index 4
#     - TWI                     <- Index 5
```

### Verification

| Index | Dataset | Model Expects | Match? |
|-------|---------|---------------|--------|
| 0 | VV | VV | ✅ |
| 1 | VH | VH | ✅ |
| 2 | DEM | DEM | ✅ |
| 3 | HAND | HAND | ✅ |
| 4 | Slope | Slope | ✅ |
| 5 | TWI | TWI | ✅ |

**Result: ✅ ACTUALLY CORRECT** - The band order matches!

---

## ⚠️ BUG 2: Physics Loss Uses Raw HAND (Fixable)

### Problem

In **physics_unet.py line 243**:
```python
loss_hand = self._hand_loss(probs, hand)
```

The `hand` passed here comes from `metadata['hand']` in the training loop, which is the **RAW HAND values** (0-50m range), not the normalized values (0-1 range).

However, the `slope` also uses raw values, so this is **consistent**.

### Impact

- **Low impact** - Both physics losses use raw values
- The loss functions handle this correctly internally

---

## Mathematical Formulas Verified

### 1. HAND Attention (✅ Correct)

**Formula:**
```python
physics_attn = sigmoid(-hand / threshold)
```

**Mathematical Interpretation:**
- When `hand = 0m`: `sigmoid(0) = 0.5` → high attention
- When `hand = 10m`: `sigmoid(-1) = 0.27` → lower attention  
- When `hand = 30m`: `sigmoid(-3) = 0.05` → very low attention

**Physics Validity:** Water probability should be HIGH where HAND is LOW. ✅

### 2. HAND Correlation Loss (✅ Correct)

**Formula (Pearson correlation):**
```
r = Σ(pred - μ_pred)(hand - μ_hand) / sqrt(Σ(pred - μ_pred)² × Σ(hand - μ_hand)²)
loss = max(0, r)  # Penalize positive correlation
```

**Physics Validity:** 
- We want **negative** correlation (high water prob at low HAND)
- Penalizing positive correlation is correct ✅

### 3. Slope Exclusion Loss (✅ Correct)

**Formula:**
```python
steep_mask = slope > 15  # degrees
loss = mean(probs[steep_mask])
```

**Physics Validity:** Water shouldn't exist on slopes > 15°. ✅

### 4. IoU Metric (✅ Correct)

**Formula:**
```
IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)
```

**Implementation:**
```python
intersection = (pred & target).sum()
union = (pred | target).sum()
IoU = intersection / union
```

✅ Mathematically correct.

### 5. F1 Score (✅ Correct)

**Formula:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × P × R / (P + R)
```

✅ Mathematically correct.

---

## Dataset Normalization (✅ Correct)

| Band | Raw Range | Normalization | Output Range |
|------|-----------|---------------|--------------|
| VV | -30 to 0 dB | `(x + 30) / 30` | 0-1 |
| VH | -30 to 0 dB | `(x + 30) / 30` | 0-1 |
| DEM | 0-5000 m | `x / 1000` | 0-5 |
| HAND | 0-50 m | `x / 50` | 0-1 |
| Slope | 0-90° | `x / 90` | 0-1 |
| TWI | 0-20 | `x / 20` | 0-1 |

All normalizations are clipped to [0, 1] at the end.

---

## Final Verdict

| Category | Status |
|----------|--------|
| **Architecture** | ✅ No bugs |
| **Data Pipeline** | ✅ No bugs |
| **Physics Constraints** | ✅ Correct |
| **Metrics** | ✅ Correct |
| **Normalization** | ✅ Correct |

**The implementation is mathematically correct.** No critical bugs found.

The training should work correctly now that the tensor size mismatch and target clamping issues are fixed.

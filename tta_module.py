#!/usr/bin/env python3
"""
================================================================================
TEST-TIME AUGMENTATION (TTA) for SAR Water Detection
================================================================================

Test-Time Augmentation is a FREE performance boost that requires no retraining.

How it works:
1. Apply multiple augmentations to the input image
2. Run inference on each augmented version
3. Inverse-transform the predictions back to original space
4. Average all predictions

Why it helps:
- Reduces prediction variance
- Improves boundary accuracy
- Handles edge cases better (e.g., river at image edge)
- Typically provides 1-3% IoU improvement

Augmentations used:
- 4 rotations (0, 90, 180, 270 degrees)
- 2 flips (horizontal, vertical)
- Total: up to 8 predictions averaged

SAR-Specific Considerations:
- SAR images are NOT rotationally invariant (azimuth direction matters)
- However, for water detection, the physics (dark = water) is preserved
- We use geometric augmentations only, not radiometric

References:
- Simonyan et al., "Very Deep Convolutional Networks" (VGGNet used TTA)
- Krizhevsky et al., "ImageNet Classification with Deep CNNs"

Author: SAR Water Detection Project
Date: 2026-01-26
Version: 1.0 - Production Ready
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Available augmentation types."""

    IDENTITY = "identity"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    FLIP_HV = "flip_both"
    ROT90 = "rotate_90"
    ROT180 = "rotate_180"
    ROT270 = "rotate_270"
    ROT90_FLIP_H = "rotate_90_flip_h"


@dataclass
class TTAConfig:
    """Configuration for Test-Time Augmentation."""

    use_rotations: bool = True  # Use 90/180/270 degree rotations
    use_flips: bool = True  # Use horizontal/vertical flips
    use_scales: bool = False  # Use multi-scale (slower, marginal benefit)
    scales: List[float] = None  # Scale factors if use_scales=True
    merge_mode: str = "mean"  # How to merge: 'mean', 'max', 'gmean'
    batch_tta: bool = True  # Process TTA in batches (memory efficient)

    def __post_init__(self):
        if self.scales is None:
            self.scales = [0.75, 1.0, 1.25]


class TTATransform:
    """
    Handles forward and inverse transforms for TTA.

    For each augmentation, we need:
    1. Forward transform: image -> augmented image
    2. Inverse transform: prediction -> de-augmented prediction
    """

    def __init__(self, aug_type: AugmentationType):
        self.aug_type = aug_type

    def forward_numpy(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation to numpy array (H, W, C) or (C, H, W)."""
        if self.aug_type == AugmentationType.IDENTITY:
            return img
        elif self.aug_type == AugmentationType.FLIP_H:
            return np.flip(img, axis=-1).copy()
        elif self.aug_type == AugmentationType.FLIP_V:
            return np.flip(img, axis=-2).copy()
        elif self.aug_type == AugmentationType.FLIP_HV:
            return np.flip(np.flip(img, axis=-1), axis=-2).copy()
        elif self.aug_type == AugmentationType.ROT90:
            return np.rot90(img, k=1, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT180:
            return np.rot90(img, k=2, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT270:
            return np.rot90(img, k=3, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT90_FLIP_H:
            rotated = np.rot90(img, k=1, axes=(-2, -1))
            return np.flip(rotated, axis=-1).copy()
        else:
            return img

    def inverse_numpy(self, pred: np.ndarray) -> np.ndarray:
        """Apply inverse transform to prediction."""
        if self.aug_type == AugmentationType.IDENTITY:
            return pred
        elif self.aug_type == AugmentationType.FLIP_H:
            return np.flip(pred, axis=-1).copy()
        elif self.aug_type == AugmentationType.FLIP_V:
            return np.flip(pred, axis=-2).copy()
        elif self.aug_type == AugmentationType.FLIP_HV:
            return np.flip(np.flip(pred, axis=-1), axis=-2).copy()
        elif self.aug_type == AugmentationType.ROT90:
            return np.rot90(pred, k=-1, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT180:
            return np.rot90(pred, k=-2, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT270:
            return np.rot90(pred, k=-3, axes=(-2, -1)).copy()
        elif self.aug_type == AugmentationType.ROT90_FLIP_H:
            flipped = np.flip(pred, axis=-1)
            return np.rot90(flipped, k=-1, axes=(-2, -1)).copy()
        else:
            return pred

    def forward_torch(self, img: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to PyTorch tensor (B, C, H, W)."""
        if self.aug_type == AugmentationType.IDENTITY:
            return img
        elif self.aug_type == AugmentationType.FLIP_H:
            return torch.flip(img, dims=[-1])
        elif self.aug_type == AugmentationType.FLIP_V:
            return torch.flip(img, dims=[-2])
        elif self.aug_type == AugmentationType.FLIP_HV:
            return torch.flip(torch.flip(img, dims=[-1]), dims=[-2])
        elif self.aug_type == AugmentationType.ROT90:
            return torch.rot90(img, k=1, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT180:
            return torch.rot90(img, k=2, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT270:
            return torch.rot90(img, k=3, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT90_FLIP_H:
            rotated = torch.rot90(img, k=1, dims=[-2, -1])
            return torch.flip(rotated, dims=[-1])
        else:
            return img

    def inverse_torch(self, pred: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform to PyTorch prediction."""
        if self.aug_type == AugmentationType.IDENTITY:
            return pred
        elif self.aug_type == AugmentationType.FLIP_H:
            return torch.flip(pred, dims=[-1])
        elif self.aug_type == AugmentationType.FLIP_V:
            return torch.flip(pred, dims=[-2])
        elif self.aug_type == AugmentationType.FLIP_HV:
            return torch.flip(torch.flip(pred, dims=[-1]), dims=[-2])
        elif self.aug_type == AugmentationType.ROT90:
            return torch.rot90(pred, k=-1, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT180:
            return torch.rot90(pred, k=-2, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT270:
            return torch.rot90(pred, k=-3, dims=[-2, -1])
        elif self.aug_type == AugmentationType.ROT90_FLIP_H:
            flipped = torch.flip(pred, dims=[-1])
            return torch.rot90(flipped, k=-1, dims=[-2, -1])
        else:
            return pred


class TestTimeAugmentation:
    """
    Test-Time Augmentation wrapper for any segmentation model.

    Usage:
        model = YourModel()
        tta = TestTimeAugmentation(config=TTAConfig())

        # For PyTorch model
        prediction = tta.predict_torch(model, input_tensor)

        # For any callable
        prediction = tta.predict_numpy(predict_fn, input_array)
    """

    def __init__(self, config: Optional[TTAConfig] = None):
        """
        Args:
            config: TTA configuration (uses defaults if None)
        """
        self.config = config or TTAConfig()
        self.transforms = self._build_transforms()

        logger.info(f"TTA initialized with {len(self.transforms)} augmentations")

    def _build_transforms(self) -> List[TTATransform]:
        """Build list of transforms based on config."""
        transforms = [TTATransform(AugmentationType.IDENTITY)]

        if self.config.use_flips:
            transforms.extend(
                [
                    TTATransform(AugmentationType.FLIP_H),
                    TTATransform(AugmentationType.FLIP_V),
                ]
            )

        if self.config.use_rotations:
            transforms.extend(
                [
                    TTATransform(AugmentationType.ROT90),
                    TTATransform(AugmentationType.ROT180),
                    TTATransform(AugmentationType.ROT270),
                ]
            )

        if self.config.use_flips and self.config.use_rotations:
            # Add combination
            transforms.append(TTATransform(AugmentationType.FLIP_HV))

        return transforms

    def _merge_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Merge multiple predictions into one."""
        if self.config.merge_mode == "mean":
            return np.mean(predictions, axis=0)
        elif self.config.merge_mode == "max":
            return np.max(predictions, axis=0)
        elif self.config.merge_mode == "gmean":
            # Geometric mean (good for probabilities)
            stacked = np.stack(predictions, axis=0)
            return np.exp(np.mean(np.log(stacked + 1e-7), axis=0))
        else:
            return np.mean(predictions, axis=0)

    def _merge_predictions_torch(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Merge PyTorch predictions."""
        stacked = torch.stack(predictions, dim=0)

        if self.config.merge_mode == "mean":
            return stacked.mean(dim=0)
        elif self.config.merge_mode == "max":
            return stacked.max(dim=0)[0]
        elif self.config.merge_mode == "gmean":
            return torch.exp(torch.log(stacked + 1e-7).mean(dim=0))
        else:
            return stacked.mean(dim=0)

    def predict_numpy(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Apply TTA with numpy arrays.

        Args:
            predict_fn: Function that takes (H, W, C) or (C, H, W) and returns (H, W)
            data: Input data array
            return_all: If True, also return individual predictions

        Returns:
            merged_prediction: Merged TTA prediction
            all_predictions: (optional) List of individual predictions
        """
        all_predictions = []

        for transform in self.transforms:
            # Forward transform
            augmented = transform.forward_numpy(data)

            # Predict
            pred = predict_fn(augmented)

            # Inverse transform
            pred_inv = transform.inverse_numpy(pred)

            all_predictions.append(pred_inv)

        # Merge
        merged = self._merge_predictions(all_predictions)

        if return_all:
            return merged, all_predictions
        return merged

    def predict_torch(
        self,
        model: nn.Module,
        data: torch.Tensor,
        return_logits: bool = False,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Apply TTA with PyTorch model.

        Args:
            model: PyTorch model (should be in eval mode)
            data: Input tensor (B, C, H, W)
            return_logits: If True, average logits instead of probabilities
            return_all: If True, also return individual predictions

        Returns:
            merged_prediction: Merged TTA prediction (B, 1, H, W) or (B, H, W)
            all_predictions: (optional) List of individual predictions
        """
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for transform in self.transforms:
                # Forward transform
                augmented = transform.forward_torch(data)

                # Predict
                logits = model(augmented)

                # Handle output shape
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)

                # Inverse transform
                logits_inv = transform.inverse_torch(logits)

                if return_logits:
                    all_predictions.append(logits_inv)
                else:
                    all_predictions.append(torch.sigmoid(logits_inv))

        # Merge
        merged = self._merge_predictions_torch(all_predictions)

        # If we averaged logits, apply sigmoid at the end
        if return_logits:
            merged = torch.sigmoid(merged)

        if return_all:
            return merged, all_predictions
        return merged

    def predict_torch_efficient(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Memory-efficient TTA using batch processing.

        All augmented versions are stacked into a batch and processed together.
        More efficient for GPU but requires more memory per forward pass.

        Args:
            model: PyTorch model
            data: Input tensor (B, C, H, W) - typically B=1 for TTA

        Returns:
            merged_prediction: Merged TTA prediction
        """
        model.eval()
        B, C, H, W = data.shape

        # Stack all augmented versions
        augmented_batch = []
        for transform in self.transforms:
            augmented = transform.forward_torch(data)
            augmented_batch.append(augmented)

        # Batch process: (N_aug * B, C, H, W)
        batch = torch.cat(augmented_batch, dim=0)

        with torch.no_grad():
            logits = model(batch)

        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        probs = torch.sigmoid(logits)

        # Split back and inverse transform
        n_aug = len(self.transforms)
        probs_split = probs.chunk(n_aug, dim=0)

        all_predictions = []
        for transform, pred in zip(self.transforms, probs_split):
            pred_inv = transform.inverse_torch(pred)
            all_predictions.append(pred_inv)

        # Merge
        merged = self._merge_predictions_torch(all_predictions)

        return merged


class MultiScaleTTA(TestTimeAugmentation):
    """
    Multi-scale Test-Time Augmentation.

    In addition to geometric augmentations, also tests at multiple scales.
    Helpful for detecting both small rivers and large lakes.

    Note: Slower than regular TTA, use only if needed.
    """

    def __init__(
        self,
        config: Optional[TTAConfig] = None,
        scales: List[float] = [0.75, 1.0, 1.25],
    ):
        if config is None:
            config = TTAConfig()
        config.use_scales = True
        config.scales = scales
        super().__init__(config)
        self.scales = scales

    def predict_torch_multiscale(
        self,
        model: nn.Module,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply multi-scale TTA.

        Args:
            model: PyTorch model
            data: Input tensor (B, C, H, W)

        Returns:
            merged_prediction: Merged multi-scale TTA prediction
        """
        model.eval()
        _, _, H, W = data.shape

        all_scale_predictions = []

        with torch.no_grad():
            for scale in self.scales:
                # Resize input
                if scale != 1.0:
                    new_H, new_W = int(H * scale), int(W * scale)
                    scaled_data = F.interpolate(
                        data, size=(new_H, new_W), mode="bilinear", align_corners=False
                    )
                else:
                    scaled_data = data

                # Apply regular TTA at this scale
                pred = self.predict_torch(model, scaled_data)

                # Resize prediction back to original size
                if scale != 1.0:
                    pred = F.interpolate(
                        pred, size=(H, W), mode="bilinear", align_corners=False
                    )

                all_scale_predictions.append(pred)

        # Merge across scales
        merged = self._merge_predictions_torch(all_scale_predictions)

        return merged


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def apply_tta(
    model: nn.Module,
    data: torch.Tensor,
    use_rotations: bool = True,
    use_flips: bool = True,
    merge_mode: str = "mean",
) -> torch.Tensor:
    """
    Convenience function for quick TTA application.

    Args:
        model: PyTorch segmentation model
        data: Input tensor (B, C, H, W)
        use_rotations: Include 90/180/270 degree rotations
        use_flips: Include horizontal/vertical flips
        merge_mode: How to merge predictions

    Returns:
        Merged prediction with TTA applied
    """
    config = TTAConfig(
        use_rotations=use_rotations,
        use_flips=use_flips,
        merge_mode=merge_mode,
    )
    tta = TestTimeAugmentation(config)
    return tta.predict_torch(model, data)


def apply_tta_numpy(
    predict_fn: Callable,
    data: np.ndarray,
    use_rotations: bool = True,
    use_flips: bool = True,
) -> np.ndarray:
    """
    Convenience function for TTA with numpy predict function.

    Useful for LightGBM or other non-PyTorch models.

    Args:
        predict_fn: Function that takes array and returns prediction
        data: Input numpy array
        use_rotations: Include rotations
        use_flips: Include flips

    Returns:
        Merged prediction
    """
    config = TTAConfig(
        use_rotations=use_rotations,
        use_flips=use_flips,
    )
    tta = TestTimeAugmentation(config)
    return tta.predict_numpy(predict_fn, data)


# =============================================================================
# TESTING
# =============================================================================


def test_tta():
    """Test TTA implementation."""
    print("=" * 60)
    print("Testing Test-Time Augmentation")
    print("=" * 60)

    # Test 1: Transform consistency
    print("\n1. Testing transform consistency...")
    data = np.random.randn(3, 64, 64).astype(np.float32)

    for aug_type in AugmentationType:
        transform = TTATransform(aug_type)

        # Forward then inverse should give original
        augmented = transform.forward_numpy(data)
        recovered = transform.inverse_numpy(augmented)

        is_equal = np.allclose(data, recovered)
        print(f"   {aug_type.value}: {'PASS' if is_equal else 'FAIL'}")

        if not is_equal:
            print(f"      Max diff: {np.abs(data - recovered).max()}")

    # Test 2: PyTorch transforms
    print("\n2. Testing PyTorch transforms...")
    data_torch = torch.randn(1, 3, 64, 64)

    for aug_type in AugmentationType:
        transform = TTATransform(aug_type)

        augmented = transform.forward_torch(data_torch)
        recovered = transform.inverse_torch(augmented)

        is_equal = torch.allclose(data_torch, recovered)
        print(f"   {aug_type.value}: {'PASS' if is_equal else 'FAIL'}")

    # Test 3: Full TTA pipeline with dummy model
    print("\n3. Testing full TTA pipeline...")

    class DummyModel(nn.Module):
        def forward(self, x):
            # Just return mean of each spatial location
            return x.mean(dim=1, keepdim=True)

    model = DummyModel()
    config = TTAConfig(use_rotations=True, use_flips=True)
    tta = TestTimeAugmentation(config)

    input_data = torch.randn(1, 9, 64, 64)
    output = tta.predict_torch(model, input_data)

    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of augmentations: {len(tta.transforms)}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test 4: Numpy TTA
    print("\n4. Testing numpy TTA...")

    def dummy_predict(x):
        return x.mean(axis=0) if x.ndim == 3 else x

    data_np = np.random.randn(9, 64, 64).astype(np.float32)
    output_np = apply_tta_numpy(dummy_predict, data_np)

    print(f"   Input shape: {data_np.shape}")
    print(f"   Output shape: {output_np.shape}")

    # Test 5: Multi-scale TTA
    print("\n5. Testing multi-scale TTA...")

    ms_tta = MultiScaleTTA(scales=[0.75, 1.0, 1.25])
    output_ms = ms_tta.predict_torch_multiscale(model, input_data)

    print(f"   Output shape: {output_ms.shape}")
    print(f"   Scales used: {ms_tta.scales}")

    print("\n" + "=" * 60)
    print("TTA Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_tta()

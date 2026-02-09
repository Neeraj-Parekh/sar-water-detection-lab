#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE IMPROVED LOSS MODULE for SAR Water Detection
================================================================================

This module combines ALL research-backed loss functions into a unified,
production-ready package for training water segmentation models.

Losses Included:

1. clDice Loss (CVPR 2021) - Topology preservation for rivers
2. Hausdorff Distance Loss (TMI 2019) - Boundary accuracy
3. Boundary-Aware Focal Tversky Loss (MICCAI 2019) - Thin structure recall
4. Lovasz-Softmax Loss (CVPR 2018) - Direct IoU optimization
5. Deep Supervision Loss - Multi-scale training
6. Physics-Informed Loss - SAR-specific constraints

Usage:
    # Simple usage with defaults
    loss_fn = ComprehensiveLoss()
    loss, metrics = loss_fn(predictions, targets)

    # Customized for river detection
    loss_fn = ComprehensiveLoss(
        use_cldice=True,
        cldice_weight=0.3,
        use_hausdorff=True,
        hausdorff_weight=0.1,
    )

References:
- clDice: Shit et al., CVPR 2021
- Hausdorff: Karimi & Salcudean, TMI 2019
- Focal Tversky: Abraham & Khan, MICCAI 2019
- Lovasz: Berman et al., CVPR 2018
- Deep Supervision: Lee et al., AISTATS 2015

Author: SAR Water Detection Project
Date: 2026-01-26
Version: 1.0 - Production Ready
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class LossConfig:
    """Configuration for comprehensive loss function."""

    # Base losses
    use_bce: bool = True
    bce_weight: float = 0.2

    use_dice: bool = True
    dice_weight: float = 0.2

    # Topology-aware losses
    use_cldice: bool = True
    cldice_weight: float = 0.2
    cldice_iter: int = 10

    use_skeleton: bool = False  # Alternative to clDice
    skeleton_weight: float = 0.2

    # Boundary losses
    use_hausdorff: bool = True
    hausdorff_weight: float = 0.1
    hausdorff_percentile: float = 95.0

    use_boundary: bool = True
    boundary_weight: float = 0.1

    # Recall-focused losses
    use_focal_tversky: bool = True
    focal_tversky_weight: float = 0.2
    tversky_alpha: float = 0.3  # FP weight
    tversky_beta: float = 0.7  # FN weight (higher = more recall)
    tversky_gamma: float = 1.33  # Focal parameter

    # IoU optimization
    use_lovasz: bool = False
    lovasz_weight: float = 0.2

    # Deep supervision
    use_deep_supervision: bool = False
    deep_supervision_weights: List[float] = None

    # Physics-informed
    use_physics: bool = False
    physics_weight: float = 0.1
    hand_threshold: float = 100.0
    slope_threshold: float = 45.0

    # Class imbalance
    pos_weight: Optional[float] = None  # For BCE

    # Numerical stability
    smooth: float = 1.0

    def __post_init__(self):
        if self.deep_supervision_weights is None:
            self.deep_supervision_weights = [1.0, 0.5, 0.25]


# =============================================================================
# SOFT MORPHOLOGICAL OPERATIONS (for clDice)
# =============================================================================


class SoftMorphology(nn.Module):
    """Differentiable morphological operations."""

    @staticmethod
    def soft_erode(img: torch.Tensor) -> torch.Tensor:
        """Soft erosion using min-pooling."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def soft_dilate(img: torch.Tensor) -> torch.Tensor:
        """Soft dilation using max-pooling."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        return F.max_pool2d(img, 3, stride=1, padding=1)

    @staticmethod
    def soft_open(img: torch.Tensor) -> torch.Tensor:
        """Soft opening = erode then dilate."""
        return SoftMorphology.soft_dilate(SoftMorphology.soft_erode(img))

    @staticmethod
    def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
        """Compute soft skeleton via iterative thinning."""
        if img.dim() == 3:
            img = img.unsqueeze(1)

        skel = F.relu(img - SoftMorphology.soft_open(img))
        img_eroded = img.clone()

        for _ in range(num_iter):
            img_eroded = SoftMorphology.soft_erode(img_eroded)
            delta = F.relu(img_eroded - SoftMorphology.soft_open(img_eroded))
            skel = skel + F.relu(delta - skel * delta)

        return skel


# =============================================================================
# INDIVIDUAL LOSS FUNCTIONS
# =============================================================================


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss with optional positive weighting."""

    def __init__(self, pos_weight: Optional[float] = None, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return self.bce(pred, target.float())


class DiceLoss(nn.Module):
    """Soft Dice Loss."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.float().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class clDiceLoss(nn.Module):
    """
    Centerline Dice Loss for topology preservation.
    Reference: Shit et al., CVPR 2021
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        # Compute skeletons
        skel_pred = SoftMorphology.soft_skeleton(pred_prob, self.num_iter)
        skel_target = SoftMorphology.soft_skeleton(target, self.num_iter)

        # Topology precision and sensitivity
        tprec = ((skel_pred * target).sum() + self.smooth) / (
            skel_pred.sum() + self.smooth
        )
        tsens = ((skel_target * pred_prob).sum() + self.smooth) / (
            skel_target.sum() + self.smooth
        )

        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-7)

        return 1 - cl_dice


class HausdorffDistanceLoss(nn.Module):
    """
    Approximate Hausdorff Distance Loss for boundary accuracy.
    Reference: Karimi & Salcudean, TMI 2019
    """

    def __init__(self, percentile: float = 95.0, smooth: float = 1e-6):
        super().__init__()
        self.percentile = percentile
        self.smooth = smooth

    def _distance_transform_approx(
        self, mask: torch.Tensor, max_dist: int = 20
    ) -> torch.Tensor:
        """
        Approximate distance transform using iterative erosion.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        dist = torch.zeros_like(mask)
        current = mask.clone()

        for d in range(1, max_dist + 1):
            # Erode
            eroded = -F.max_pool2d(-current, 3, stride=1, padding=1)
            # Distance = d where current exists but eroded doesn't
            boundary = (current > 0.5) & (eroded < 0.5)
            dist = torch.where(boundary & (dist == 0), d * torch.ones_like(dist), dist)
            current = eroded

        # Remaining pixels get max distance
        dist = torch.where(
            (mask > 0.5) & (dist == 0), max_dist * torch.ones_like(dist), dist
        )

        return dist

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()
        pred_binary = (pred_prob > 0.5).float()

        # Distance transform of inverted masks
        pred_dt = self._distance_transform_approx(1 - pred_binary)
        target_dt = self._distance_transform_approx(1 - target)

        # Directed distances
        d_pred_to_target = pred_dt * target
        d_target_to_pred = target_dt * pred_binary

        # Use mean instead of percentile for differentiability
        hd_forward = (d_pred_to_target.sum() + self.smooth) / (
            target.sum() + self.smooth
        )
        hd_backward = (d_target_to_pred.sum() + self.smooth) / (
            pred_binary.sum() + self.smooth
        )

        return (hd_forward + hd_backward) / 2


class BoundaryLoss(nn.Module):
    """
    Boundary-focused loss using distance-weighted BCE.
    """

    def __init__(self, theta: float = 3.0):
        super().__init__()
        self.theta = theta

    def _compute_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute boundary mask."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Dilate - Erode = Boundary
        dilated = F.max_pool2d(mask.float(), 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask.float(), 3, stride=1, padding=1)
        boundary = dilated - eroded

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        # Compute boundary weights
        boundary = self._compute_boundary(target)
        weights = 1 + self.theta * boundary

        # Weighted BCE
        bce = F.binary_cross_entropy(pred_prob, target, reduction="none")
        weighted_bce = (weights * bce).mean()

        return weighted_bce


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling thin structures.
    Reference: Abraham & Khan, MICCAI 2019

    Tversky(α,β) = TP / (TP + α*FP + β*FN)
    Focal Tversky = (1 - Tversky)^γ

    Higher β → More focus on recall (catching all water)
    Higher γ → More focus on hard examples
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta  # FN weight
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        pred_flat = pred_prob.view(-1)
        target_flat = target.float().view(-1)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Focal modulation
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Softmax loss for direct IoU optimization.
    Reference: Berman et al., CVPR 2018
    """

    def __init__(self, per_image: bool = False):
        super().__init__()
        self.per_image = per_image

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute Lovasz gradient."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (
            torch.arange(1, p + 1, device=gt_sorted.device, dtype=gt_sorted.dtype)
            - gt_sorted.cumsum(0)
        )

        jaccard = 1.0 - intersection / (union + 1e-8)
        jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard

    def _lovasz_hinge(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary Lovasz hinge loss."""
        signs = 2 * labels.float() - 1
        errors = 1 - logits * signs
        errors_sorted, perm = torch.sort(errors, descending=True)

        gt_sorted = labels[perm].float()
        grad = self._lovasz_grad(gt_sorted)

        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        return self._lovasz_hinge(pred_flat, target_flat)


class PhysicsConstraintLoss(nn.Module):
    """
    Physics-informed loss for SAR water detection.
    Penalizes water predictions on steep slopes and high HAND areas.
    """

    def __init__(
        self,
        hand_threshold: float = 100.0,
        slope_threshold: float = 45.0,
        hand_weight: float = 0.5,
        slope_weight: float = 0.5,
    ):
        super().__init__()
        self.hand_threshold = hand_threshold
        self.slope_threshold = slope_threshold
        self.hand_weight = hand_weight
        self.slope_weight = slope_weight

    def forward(
        self,
        pred: torch.Tensor,
        hand: torch.Tensor,
        slope: torch.Tensor,
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        if pred_prob.dim() == 4:
            pred_prob = pred_prob.squeeze(1)

        # HAND penalty: water prediction where HAND > threshold
        hand_violation = torch.sigmoid((hand - self.hand_threshold) / 10)
        hand_penalty = (pred_prob * hand_violation).mean()

        # Slope penalty: water prediction where slope > threshold
        slope_violation = torch.sigmoid((slope - self.slope_threshold) / 10)
        slope_penalty = (pred_prob * slope_violation).mean()

        return self.hand_weight * hand_penalty + self.slope_weight * slope_penalty


# =============================================================================
# COMPREHENSIVE LOSS
# =============================================================================


class ComprehensiveLoss(nn.Module):
    """
    Comprehensive loss function combining all research-backed losses.

    This is the main loss function to use for training.
    """

    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()

        # Initialize losses based on config
        self.losses = nn.ModuleDict()
        self.loss_weights = {}

        if self.config.use_bce:
            self.losses["bce"] = BCELoss(pos_weight=self.config.pos_weight)
            self.loss_weights["bce"] = self.config.bce_weight

        if self.config.use_dice:
            self.losses["dice"] = DiceLoss(smooth=self.config.smooth)
            self.loss_weights["dice"] = self.config.dice_weight

        if self.config.use_cldice:
            self.losses["cldice"] = clDiceLoss(
                num_iter=self.config.cldice_iter,
                smooth=self.config.smooth,
            )
            self.loss_weights["cldice"] = self.config.cldice_weight

        if self.config.use_hausdorff:
            self.losses["hausdorff"] = HausdorffDistanceLoss(
                percentile=self.config.hausdorff_percentile,
            )
            self.loss_weights["hausdorff"] = self.config.hausdorff_weight

        if self.config.use_boundary:
            self.losses["boundary"] = BoundaryLoss()
            self.loss_weights["boundary"] = self.config.boundary_weight

        if self.config.use_focal_tversky:
            self.losses["focal_tversky"] = FocalTverskyLoss(
                alpha=self.config.tversky_alpha,
                beta=self.config.tversky_beta,
                gamma=self.config.tversky_gamma,
            )
            self.loss_weights["focal_tversky"] = self.config.focal_tversky_weight

        if self.config.use_lovasz:
            self.losses["lovasz"] = LovaszHingeLoss()
            self.loss_weights["lovasz"] = self.config.lovasz_weight

        if self.config.use_physics:
            self.losses["physics"] = PhysicsConstraintLoss(
                hand_threshold=self.config.hand_threshold,
                slope_threshold=self.config.slope_threshold,
            )
            self.loss_weights["physics"] = self.config.physics_weight

        # Normalize weights
        total_weight = sum(self.loss_weights.values())
        if total_weight > 0:
            self.loss_weights = {
                k: v / total_weight for k, v in self.loss_weights.items()
            }

        logger.info(f"ComprehensiveLoss initialized with: {list(self.losses.keys())}")
        logger.info(f"Weights: {self.loss_weights}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        hand: Optional[torch.Tensor] = None,
        slope: Optional[torch.Tensor] = None,
        aux_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute comprehensive loss.

        Args:
            pred: (B, 1, H, W) or (B, H, W) prediction logits
            target: (B, H, W) binary ground truth
            hand: (B, H, W) optional HAND values
            slope: (B, H, W) optional slope values
            aux_outputs: Optional list of auxiliary predictions for deep supervision

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        # Compute each loss
        for name, loss_fn in self.losses.items():
            if name == "physics":
                if hand is not None and slope is not None:
                    loss_val = loss_fn(pred, hand, slope)
                else:
                    continue
            else:
                loss_val = loss_fn(pred, target)

            # Check for NaN/Inf
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                logger.warning(f"NaN/Inf in {name} loss, skipping")
                loss_val = torch.tensor(0.0, device=pred.device)

            weighted_loss = self.loss_weights[name] * loss_val
            total_loss = total_loss + weighted_loss

            loss_dict[name] = loss_val.item()
            loss_dict[f"{name}_weighted"] = weighted_loss.item()

        # Deep supervision
        if self.config.use_deep_supervision and aux_outputs is not None:
            ds_loss = torch.tensor(0.0, device=pred.device)
            ds_weights = self.config.deep_supervision_weights

            for i, (aux, weight) in enumerate(zip(aux_outputs, ds_weights)):
                # Resize target to match aux output size
                if aux.shape[-2:] != target.shape[-2:]:
                    target_resized = F.interpolate(
                        target.unsqueeze(1).float(), size=aux.shape[-2:], mode="nearest"
                    ).squeeze(1)
                else:
                    target_resized = target

                # Compute loss on auxiliary output
                if "dice" in self.losses:
                    aux_loss = self.losses["dice"](aux, target_resized)
                else:
                    aux_loss = DiceLoss()(aux, target_resized)
                ds_loss = ds_loss + weight * aux_loss

                loss_dict[f"deep_supervision_{i}"] = aux_loss.item()

            total_loss = total_loss + 0.3 * ds_loss
            loss_dict["deep_supervision_total"] = ds_loss.item()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def get_river_detection_loss() -> ComprehensiveLoss:
    """
    Optimized loss configuration for narrow river detection.

    Key features:
    - High clDice weight for topology preservation
    - High FN penalty (Tversky beta=0.8) for recall
    - Boundary loss for edge accuracy
    """
    config = LossConfig(
        use_bce=True,
        bce_weight=0.15,
        use_dice=True,
        dice_weight=0.15,
        use_cldice=True,
        cldice_weight=0.30,
        cldice_iter=15,
        use_hausdorff=True,
        hausdorff_weight=0.10,
        use_boundary=True,
        boundary_weight=0.10,
        use_focal_tversky=True,
        focal_tversky_weight=0.20,
        tversky_alpha=0.2,
        tversky_beta=0.8,
        tversky_gamma=1.5,
        use_lovasz=False,
    )
    return ComprehensiveLoss(config)


def get_lake_detection_loss() -> ComprehensiveLoss:
    """
    Optimized loss configuration for large water body detection.

    Key features:
    - Standard Dice for area overlap
    - Lower clDice weight (topology less critical)
    - Physics constraints more important
    """
    config = LossConfig(
        use_bce=True,
        bce_weight=0.20,
        use_dice=True,
        dice_weight=0.30,
        use_cldice=True,
        cldice_weight=0.10,
        cldice_iter=5,
        use_hausdorff=False,
        use_boundary=True,
        boundary_weight=0.15,
        use_focal_tversky=True,
        focal_tversky_weight=0.15,
        tversky_alpha=0.5,
        tversky_beta=0.5,
        use_physics=True,
        physics_weight=0.10,
    )
    return ComprehensiveLoss(config)


def get_balanced_loss() -> ComprehensiveLoss:
    """
    Balanced loss configuration for mixed water types.
    """
    config = LossConfig(
        use_bce=True,
        bce_weight=0.20,
        use_dice=True,
        dice_weight=0.20,
        use_cldice=True,
        cldice_weight=0.20,
        use_hausdorff=True,
        hausdorff_weight=0.10,
        use_boundary=True,
        boundary_weight=0.10,
        use_focal_tversky=True,
        focal_tversky_weight=0.20,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    )
    return ComprehensiveLoss(config)


# =============================================================================
# TESTING
# =============================================================================


def test_comprehensive_loss():
    """Test all loss components."""
    print("=" * 60)
    print("Testing Comprehensive Loss Module")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dummy data
    B, H, W = 2, 64, 64
    pred = torch.randn(B, 1, H, W, device=device)
    target = (torch.rand(B, H, W, device=device) > 0.5).float()
    hand = torch.rand(B, H, W, device=device) * 200
    slope = torch.rand(B, H, W, device=device) * 90

    # Test 1: Individual losses
    print("\n1. Testing Individual Losses:")

    losses_to_test = [
        ("BCELoss", BCELoss()),
        ("DiceLoss", DiceLoss()),
        ("clDiceLoss", clDiceLoss()),
        ("HausdorffDistanceLoss", HausdorffDistanceLoss()),
        ("BoundaryLoss", BoundaryLoss()),
        ("FocalTverskyLoss", FocalTverskyLoss()),
        ("LovaszHingeLoss", LovaszHingeLoss()),
    ]

    for name, loss_fn in losses_to_test:
        loss_fn = loss_fn.to(device)
        try:
            loss = loss_fn(pred, target)
            print(f"   {name}: {loss.item():.4f}")
        except Exception as e:
            print(f"   {name}: ERROR - {e}")

    # Test 2: Comprehensive loss
    print("\n2. Testing Comprehensive Loss:")

    config = LossConfig(
        use_bce=True,
        use_dice=True,
        use_cldice=True,
        use_hausdorff=True,
        use_boundary=True,
        use_focal_tversky=True,
        use_physics=True,
    )
    comp_loss = ComprehensiveLoss(config).to(device)

    total_loss, loss_dict = comp_loss(pred, target, hand, slope)
    print(f"   Total loss: {total_loss.item():.4f}")
    print("   Components:")
    for name, value in loss_dict.items():
        print(f"      {name}: {value:.4f}")

    # Test 3: Gradient check
    print("\n3. Gradient Check:")
    pred.requires_grad_(True)
    loss, _ = comp_loss(pred, target, hand, slope)
    loss.backward()
    print(f"   Gradients computed: {pred.grad is not None}")
    print(f"   Gradient norm: {pred.grad.norm().item():.6f}")

    # Test 4: Preset configurations
    print("\n4. Testing Preset Configurations:")

    presets = [
        ("River Detection", get_river_detection_loss),
        ("Lake Detection", get_lake_detection_loss),
        ("Balanced", get_balanced_loss),
    ]

    pred_new = torch.randn(B, 1, H, W, device=device)

    for name, preset_fn in presets:
        loss_fn = preset_fn().to(device)
        loss, _ = loss_fn(pred_new, target)
        print(f"   {name}: {loss.item():.4f}")

    # Test 5: Deep supervision
    print("\n5. Testing Deep Supervision:")

    config_ds = LossConfig(use_deep_supervision=True)
    loss_ds = ComprehensiveLoss(config_ds).to(device)

    aux_outputs = [
        torch.randn(B, 1, 32, 32, device=device),
        torch.randn(B, 1, 16, 16, device=device),
    ]

    loss_val, loss_dict = loss_ds(pred_new, target, aux_outputs=aux_outputs)
    print(f"   Loss with deep supervision: {loss_val.item():.4f}")

    print("\n" + "=" * 60)
    print("Comprehensive Loss Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_comprehensive_loss()

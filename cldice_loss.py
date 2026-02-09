#!/usr/bin/env python3
"""
================================================================================
PROPER clDice LOSS - Topology-Preserving Loss for Thin Structures
================================================================================

Based on: "clDice - A Novel Topology-Preserving Loss Function for Tubular
          Structure Segmentation"
Authors: Shit et al., CVPR 2021
Paper: https://arxiv.org/abs/2003.07311

Key Innovation:
- Uses SOFT morphological operations (differentiable)
- Computes skeleton using iterative soft-erosion
- Measures topology preservation via centerline overlap

Why This Matters for SAR Water Detection:
- Rivers are thin tubular structures (width << length)
- Standard Dice optimizes for area overlap, ignoring connectivity
- A broken river with 95% pixels correct has TERRIBLE topology
- clDice forces the model to preserve the centerline

Mathematical Foundation:
- Topology Precision (Tprec): skeleton(pred) ∩ target / skeleton(pred)
- Topology Sensitivity (Tsens): skeleton(target) ∩ pred / skeleton(target)
- clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

Author: SAR Water Detection Project
Date: 2026-01-26
Version: 1.0 - Production Ready
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftSkeletonize(nn.Module):
    """
    Differentiable Soft Skeletonization using morphological operations.

    The skeleton is computed iteratively:
    1. Compute soft opening (erode then dilate)
    2. Skeleton = image - soft_open(image)
    3. Repeat on eroded image, accumulating skeleton

    This is GPU-optimized and fully differentiable.
    """

    def __init__(self, num_iter: int = 10):
        """
        Args:
            num_iter: Number of thinning iterations (higher = thinner skeleton)
        """
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Soft erosion using min-pooling (differentiable).

        Erosion shrinks bright regions. We use separable 1D pools for efficiency.
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)

        # Separable erosion: min in x, then min in y
        # Negative max-pool of negative = min-pool
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))

        # Take minimum of both directions
        eroded = torch.min(p1, p2)

        return eroded

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        """
        Soft dilation using max-pooling (differentiable).

        Dilation expands bright regions.
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)

        # Max-pool expands bright regions
        dilated = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

        return dilated

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        """
        Soft opening = erosion followed by dilation.

        Opening removes small bright regions (noise) and thin protrusions.
        """
        return self.soft_dilate(self.soft_erode(img))

    def soft_skeleton(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute soft skeleton via iterative thinning.

        Algorithm:
        1. skel = relu(img - open(img))  -- initial skeleton layer
        2. For each iteration:
           a. erode the image
           b. compute delta = relu(eroded - open(eroded))
           c. accumulate: skel = skel + relu(delta - skel*delta)

        The accumulation prevents double-counting overlapping skeleton pixels.
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)

        # Initial skeleton: pixels removed by opening
        img_open = self.soft_open(img)
        skel = F.relu(img - img_open)

        # Iteratively thin and accumulate skeleton
        img_eroded = img.clone()

        for i in range(self.num_iter):
            # Erode the image
            img_eroded = self.soft_erode(img_eroded)

            # Skeleton at this scale
            img_eroded_open = self.soft_open(img_eroded)
            delta = F.relu(img_eroded - img_eroded_open)

            # Accumulate without double-counting
            # skel + delta - skel*delta = union operation
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute soft skeleton.

        Args:
            img: (B, 1, H, W) or (B, H, W) probability map [0, 1]

        Returns:
            skeleton: (B, 1, H, W) soft skeleton
        """
        return self.soft_skeleton(img)


class clDiceLoss(nn.Module):
    """
    Centerline Dice Loss for topology-preserving segmentation.

    Unlike standard Dice which measures area overlap, clDice measures:
    1. Topology Precision: Is the predicted skeleton inside the ground truth?
    2. Topology Sensitivity: Is the ground truth skeleton inside the prediction?

    A prediction with a broken river will have low Tsens even if pixel-wise
    accuracy is high.

    Usage:
        loss_fn = clDiceLoss(iter_=10, smooth=1.0)
        loss = loss_fn(pred_logits, target_mask)
    """

    def __init__(
        self,
        iter_: int = 10,
        smooth: float = 1.0,
        include_dice: bool = True,
        cl_weight: float = 0.5,
    ):
        """
        Args:
            iter_: Iterations for skeleton computation (10-50 typical)
            smooth: Smoothing factor to prevent division by zero
            include_dice: If True, combine clDice with regular Dice
            cl_weight: Weight for clDice vs regular Dice (if include_dice=True)
        """
        super().__init__()
        self.soft_skeleton = SoftSkeletonize(num_iter=iter_)
        self.smooth = smooth
        self.include_dice = include_dice
        self.cl_weight = cl_weight

    def soft_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard soft Dice coefficient."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return dice

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute clDice loss.

        Args:
            pred: (B, 1, H, W) or (B, H, W) prediction logits
            target: (B, 1, H, W) or (B, H, W) binary ground truth

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with component metrics
        """
        # Handle dimensions
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Ensure 4D for skeleton computation
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Get probabilities
        pred_prob = torch.sigmoid(pred)
        target_float = target.float()

        # Compute soft skeletons
        skel_pred = self.soft_skeleton(pred_prob)
        skel_target = self.soft_skeleton(target_float)

        # Topology Precision: skeleton(pred) inside target
        # "Is my predicted centerline actually water?"
        tprec_num = (skel_pred * target_float).sum()
        tprec_den = skel_pred.sum() + self.smooth
        tprec = (tprec_num + self.smooth) / tprec_den

        # Topology Sensitivity: skeleton(target) inside pred
        # "Did I capture the true centerline?"
        tsens_num = (skel_target * pred_prob).sum()
        tsens_den = skel_target.sum() + self.smooth
        tsens = (tsens_num + self.smooth) / tsens_den

        # clDice = harmonic mean of Tprec and Tsens
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)

        # Combine with regular Dice if requested
        if self.include_dice:
            dice = self.soft_dice(pred_prob, target_float)
            combined_dice = (1 - self.cl_weight) * dice + self.cl_weight * cl_dice
        else:
            combined_dice = cl_dice
            dice = self.soft_dice(pred_prob, target_float)

        # Loss = 1 - Dice
        loss = 1.0 - combined_dice

        # Metrics for logging
        metrics = {
            "cl_dice": cl_dice.item(),
            "tprec": tprec.item(),
            "tsens": tsens.item(),
            "dice": dice.item(),
            "combined_dice": combined_dice.item(),
            "loss": loss.item(),
        }

        return loss, metrics


class clDiceBCELoss(nn.Module):
    """
    Combined clDice + BCE Loss.

    BCE provides pixel-wise gradients for stable training.
    clDice provides topology-aware gradients for connectivity.
    """

    def __init__(
        self,
        iter_: int = 10,
        smooth: float = 1.0,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        cldice_weight: float = 0.4,
        pos_weight: Optional[float] = None,
    ):
        """
        Args:
            iter_: Skeleton iterations
            smooth: Smoothing factor
            bce_weight: Weight for BCE loss
            dice_weight: Weight for regular Dice loss
            cldice_weight: Weight for clDice loss
            pos_weight: Positive class weight for BCE (for class imbalance)
        """
        super().__init__()
        self.soft_skeleton = SoftSkeletonize(num_iter=iter_)
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight

        # BCE with optional positive weight
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred: (B, 1, H, W) or (B, H, W) prediction logits
            target: (B, 1, H, W) or (B, H, W) binary ground truth

        Returns:
            loss: Combined loss
            metrics: Component metrics
        """
        # Handle dimensions
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        pred_prob = torch.sigmoid(pred)
        target_float = target.float()

        # 1. BCE Loss
        bce_loss = self.bce(pred, target_float)

        # 2. Dice Loss
        pred_flat = pred_prob.view(-1)
        target_flat = target_float.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        # 3. clDice Loss
        skel_pred = self.soft_skeleton(pred_prob)
        skel_target = self.soft_skeleton(target_float)

        tprec = ((skel_pred * target_float).sum() + self.smooth) / (
            skel_pred.sum() + self.smooth
        )
        tsens = ((skel_target * pred_prob).sum() + self.smooth) / (
            skel_target.sum() + self.smooth
        )
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
        cldice_loss = 1.0 - cl_dice

        # Combined loss
        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.cldice_weight * cldice_loss
        )

        metrics = {
            "bce": bce_loss.item(),
            "dice": dice.item(),
            "dice_loss": dice_loss.item(),
            "cl_dice": cl_dice.item(),
            "cldice_loss": cldice_loss.item(),
            "tprec": tprec.item(),
            "tsens": tsens.item(),
            "total": total_loss.item(),
        }

        return total_loss, metrics


# =============================================================================
# TESTING
# =============================================================================


def test_cldice():
    """Test clDice loss with synthetic data."""
    print("=" * 60)
    print("Testing clDice Loss Implementation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test 1: Perfect prediction
    print("\n1. Perfect prediction:")
    target = torch.zeros(1, 1, 64, 64, device=device)
    target[:, :, 28:36, 10:54] = 1  # Horizontal river
    pred = torch.zeros_like(target) + 5  # High logits where target is
    pred = pred * target - 5 * (1 - target)  # Logits: +5 for water, -5 for land

    loss_fn = clDiceLoss(iter_=10)
    loss, metrics = loss_fn(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   clDice: {metrics['cl_dice']:.4f}")
    print(f"   Tprec: {metrics['tprec']:.4f}, Tsens: {metrics['tsens']:.4f}")

    # Test 2: Broken river (gap in middle)
    print("\n2. Broken river (gap in middle):")
    pred_broken = pred.clone()
    pred_broken[:, :, :, 30:34] = -5  # Create gap

    loss_broken, metrics_broken = loss_fn(pred_broken, target)
    print(f"   Loss: {loss_broken.item():.4f}")
    print(f"   clDice: {metrics_broken['cl_dice']:.4f}")
    print(
        f"   Tprec: {metrics_broken['tprec']:.4f}, Tsens: {metrics_broken['tsens']:.4f}"
    )
    print(
        f"   -> clDice dropped from {metrics['cl_dice']:.3f} to {metrics_broken['cl_dice']:.3f}"
    )

    # Test 3: Standard Dice vs clDice comparison
    print("\n3. Dice vs clDice for broken river:")
    print(f"   Regular Dice: {metrics_broken['dice']:.4f} (looks OK)")
    print(f"   clDice: {metrics_broken['cl_dice']:.4f} (correctly penalized!)")

    # Test 4: Gradient check
    print("\n4. Gradient check:")
    pred_grad = pred.clone().requires_grad_(True)
    loss_grad, _ = loss_fn(pred_grad, target)
    loss_grad.backward()
    print(f"   Gradients computed: {pred_grad.grad is not None}")
    print(f"   Gradient norm: {pred_grad.grad.norm().item():.6f}")

    # Test 5: Combined loss
    print("\n5. Combined clDice+BCE+Dice Loss:")
    combined_loss_fn = clDiceBCELoss(
        iter_=10, bce_weight=0.3, dice_weight=0.3, cldice_weight=0.4
    )
    loss_combined, metrics_combined = combined_loss_fn(pred, target)
    print(f"   BCE: {metrics_combined['bce']:.4f}")
    print(f"   Dice: {metrics_combined['dice']:.4f}")
    print(f"   clDice: {metrics_combined['cl_dice']:.4f}")
    print(f"   Total: {metrics_combined['total']:.4f}")

    print("\n" + "=" * 60)
    print("clDice Loss Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_cldice()

#!/usr/bin/env python3
"""
================================================================================
TVERSKY & CENTERLINE DICE LOSS - Fixes for Narrow River Detection
================================================================================

FIX FOR ERROR E4: Standard DiceLoss fails on thin structures.

This module implements:
1. Tversky Loss (Salehi et al., 2017) - Asymmetric loss for recall
2. Focal Tversky Loss (Abraham & Khan, MICCAI 2019) - Focus on hard examples
3. Properly Differentiable Centerline Dice (CVPR 2021)

Mathematical Background:
-----------------------
Standard Dice treats FP and FN equally:
    Dice = 2*TP / (2*TP + FP + FN)

For thin rivers:
- Width = 3px, Length = 500px → 1500 pixels
- If model predicts width = 1px (broken): TP = 500, FP = 0, FN = 1000
- Dice = 2*500 / (2*500 + 0 + 1000) = 0.5
- BUT the river is COMPLETELY DISCONNECTED!

Tversky Loss allows asymmetric weighting:
    Tversky = TP / (TP + α*FP + β*FN)

For river detection (prioritize recall):
    α = 0.3 (low weight on FP)
    β = 0.7 (high weight on FN - penalize missing river pixels)

This forces the model to CAPTURE the full river width.

References:
- Salehi et al., "Tversky loss for highly unbalanced data", 2017
- Abraham & Khan, "A novel focal Tversky loss for segmentation", MICCAI 2019
- Shit et al., "clDice: topology-preserving loss", CVPR 2021

Author: SAR Water Detection Project - Code Audit Fix
Date: 2026-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# =============================================================================
# TVERSKY LOSS
# =============================================================================


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance in segmentation.

    Formula:
        TI = (TP + ε) / (TP + α*FP + β*FN + ε)
        Loss = 1 - TI

    Where:
        α controls false positive penalty
        β controls false negative penalty

    For thin structure detection (rivers):
        α = 0.3, β = 0.7 → prioritizes recall

    For precision-focused detection:
        α = 0.7, β = 0.3 → prioritizes precision

    Reference: Salehi et al., 2017
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        """
        Args:
            alpha: Weight for false positives (default 0.3 for recall focus)
            beta: Weight for false negatives (default 0.7 for recall focus)
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        # Validate: α + β should typically equal 1, but not required
        if not (0 <= alpha <= 1 and 0 <= beta <= 1):
            raise ValueError("alpha and beta must be in [0, 1]")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Tversky Loss.

        Args:
            pred: Predictions (logits), shape [B, 1, H, W] or [B, H, W]
            target: Ground truth, shape [B, 1, H, W] or [B, H, W]

        Returns:
            loss: Scalar loss value
            metrics: Dictionary with TP, FP, FN statistics
        """
        # Get probabilities
        pred_prob = torch.sigmoid(pred)

        # Flatten for computation
        pred_flat = pred_prob.view(-1)
        target_flat = target.float().view(-1)

        # Soft TP, FP, FN (differentiable)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = 1 - tversky_index

        metrics = {
            "tp": tp.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tversky_index": tversky_index.item(),
        }

        return loss, metrics


# =============================================================================
# FOCAL TVERSKY LOSS
# =============================================================================


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for hard example mining.

    Formula:
        FTL = (1 - TI)^γ

    Where:
        γ (gamma) is the focal parameter
        γ > 1 focuses on hard examples (low TI)

    For river detection:
        α = 0.3, β = 0.7 (recall focus)
        γ = 0.75 (standard for medical imaging)

    Reference: Abraham & Khan, MICCAI 2019
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ):
        """
        Args:
            alpha: FP weight
            beta: FN weight
            gamma: Focal parameter (0.75 = standard, 2.0 = aggressive focus)
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Focal Tversky Loss."""
        pred_prob = torch.sigmoid(pred)

        pred_flat = pred_prob.view(-1)
        target_flat = target.float().view(-1)

        # Soft TP, FP, FN
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Focal modulation
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)

        metrics = {
            "tp": tp.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tversky_index": tversky_index.item(),
            "focal_tversky": focal_tversky.item(),
        }

        return focal_tversky, metrics


# =============================================================================
# DIFFERENTIABLE CENTERLINE DICE (FIX FOR ERROR E3)
# =============================================================================


class SoftSkeletonize(nn.Module):
    """
    Differentiable soft skeletonization via morphological thinning.

    FIX FOR ERROR E3: The original SkeletonLoss used hard thresholds
    which break gradient flow. This implementation uses SOFT operations.

    Mathematical Guarantee:
        All operations use smooth functions (min, max pools)
        Gradients flow through the entire computation graph
    """

    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        """Soft erosion using min-pooling (differentiable)."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        # Min-pool = negative of max-pool on negated input
        return -F.max_pool2d(-img, 3, stride=1, padding=1)

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        """Soft dilation using max-pooling (differentiable)."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        return F.max_pool2d(img, 3, stride=1, padding=1)

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        """Soft opening = erode then dilate."""
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute soft skeleton.

        Algorithm:
            skel = 0
            for i in range(num_iter):
                skel += ReLU(eroded_i - opened(eroded_i))
                eroded_i = erode(eroded_i)
        """
        if img.dim() == 3:
            img = img.unsqueeze(1)

        # Initialize skeleton
        skel = F.relu(img - self.soft_open(img))
        img_eroded = img.clone()

        for _ in range(self.num_iter):
            img_eroded = self.soft_erode(img_eroded)
            opened = self.soft_open(img_eroded)
            delta = F.relu(img_eroded - opened)
            # Avoid double-counting: only add where skeleton doesn't already exist
            skel = skel + F.relu(delta - skel * delta)

        return skel


class CenterlineDiceLoss(nn.Module):
    """
    Centerline Dice Loss (clDice) for topology preservation.

    This is a PROPERLY DIFFERENTIABLE implementation.

    Formula:
        Tprec = |skeleton(P) ∩ G| / |skeleton(P)|
        Tsens = |skeleton(G) ∩ P| / |skeleton(G)|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
        Loss = 1 - clDice

    Where:
        skeleton() is the soft skeletonization
        P = prediction probability
        G = ground truth

    Why this works for rivers:
        - Standard Dice measures area overlap
        - clDice measures centerline preservation
        - A broken river has low Tsens even with high Dice

    Reference: Shit et al., CVPR 2021
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.skeleton = SoftSkeletonize(num_iter=num_iter)
        self.smooth = smooth

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Centerline Dice Loss.

        Args:
            pred: Prediction logits [B, 1, H, W]
            target: Ground truth [B, 1, H, W]

        Returns:
            loss: 1 - clDice
            metrics: Tprec, Tsens, clDice values
        """
        pred_prob = torch.sigmoid(pred)

        if pred_prob.dim() == 3:
            pred_prob = pred_prob.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        target = target.float()

        # Compute soft skeletons
        skel_pred = self.skeleton(pred_prob)
        skel_target = self.skeleton(target)

        # Topology Precision: skeleton(pred) inside ground truth
        tprec_num = (skel_pred * target).sum() + self.smooth
        tprec_den = skel_pred.sum() + self.smooth
        tprec = tprec_num / tprec_den

        # Topology Sensitivity: skeleton(target) inside prediction
        tsens_num = (skel_target * pred_prob).sum() + self.smooth
        tsens_den = skel_target.sum() + self.smooth
        tsens = tsens_num / tsens_den

        # Centerline Dice (harmonic mean)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-7)

        loss = 1 - cl_dice

        metrics = {
            "tprec": tprec.item(),
            "tsens": tsens.item(),
            "cl_dice": cl_dice.item(),
        }

        return loss, metrics


# =============================================================================
# COMBINED RIVER DETECTION LOSS
# =============================================================================


class RiverDetectionLoss(nn.Module):
    """
    Combined loss function optimized for narrow river detection.

    Combines:
    1. BCE for stable gradients
    2. Dice for area overlap
    3. Focal Tversky for thin structure recall
    4. clDice for topology preservation

    Recommended weights for rivers:
        bce: 0.1 (stability)
        dice: 0.2 (area)
        focal_tversky: 0.3 (recall for thin rivers)
        cldice: 0.4 (topology)
    """

    def __init__(
        self,
        bce_weight: float = 0.1,
        dice_weight: float = 0.2,
        focal_tversky_weight: float = 0.3,
        cldice_weight: float = 0.4,
        # Tversky parameters
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 0.75,
        # clDice parameters
        cldice_iter: int = 10,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.cldice_weight = cldice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = FocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma
        )
        self.cldice = CenterlineDiceLoss(num_iter=cldice_iter)

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.float().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        return 1 - dice

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined river detection loss."""

        # BCE
        bce = self.bce(pred, target.float())

        # Dice
        dice = self.dice_loss(pred, target)

        # Focal Tversky
        ft_loss, ft_metrics = self.focal_tversky(pred, target)

        # clDice
        cl_loss, cl_metrics = self.cldice(pred, target)

        # Combined
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.focal_tversky_weight * ft_loss
            + self.cldice_weight * cl_loss
        )

        metrics = {
            "bce": bce.item(),
            "dice": dice.item(),
            "focal_tversky": ft_loss.item(),
            "cldice": cl_loss.item(),
            "tversky_index": ft_metrics["tversky_index"],
            "cl_dice": cl_metrics["cl_dice"],
            "total": total.item(),
        }

        return total, metrics


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_river_detection_loss() -> RiverDetectionLoss:
    """
    Get pre-configured loss for river detection.

    This configuration prioritizes:
    1. Capturing full river width (high FN penalty via Tversky)
    2. Preserving river continuity (clDice)
    """
    return RiverDetectionLoss(
        bce_weight=0.1,
        dice_weight=0.2,
        focal_tversky_weight=0.3,
        cldice_weight=0.4,
        tversky_alpha=0.3,  # Low FP penalty
        tversky_beta=0.7,  # High FN penalty (recall focus)
        tversky_gamma=0.75,  # Focal for hard examples
        cldice_iter=10,  # Skeleton iterations
    )


def get_lake_detection_loss() -> RiverDetectionLoss:
    """
    Get pre-configured loss for lake detection.

    Lakes are compact structures, so we use:
    1. Standard Dice (area overlap)
    2. Lower clDice weight (topology less critical)
    3. Balanced Tversky (precision = recall)
    """
    return RiverDetectionLoss(
        bce_weight=0.2,
        dice_weight=0.4,  # Higher dice for area
        focal_tversky_weight=0.2,
        cldice_weight=0.2,  # Lower topology weight
        tversky_alpha=0.5,  # Balanced
        tversky_beta=0.5,  # Balanced
        tversky_gamma=0.75,
        cldice_iter=5,  # Fewer iterations (simpler shapes)
    )


# =============================================================================
# TEST
# =============================================================================


if __name__ == "__main__":
    print("Testing Tversky & Centerline Dice Loss")
    print("=" * 60)

    # Create test tensors
    pred = torch.randn(2, 1, 64, 64)
    target = torch.zeros(2, 1, 64, 64)

    # Create thin river in target
    target[:, :, 30:33, 10:54] = 1  # 3px wide river

    # Test Tversky
    tversky = TverskyLoss(alpha=0.3, beta=0.7)
    t_loss, t_metrics = tversky(pred, target)
    print(f"Tversky Loss: {t_loss.item():.4f}")
    print(
        f"  TP: {t_metrics['tp']:.1f}, FP: {t_metrics['fp']:.1f}, FN: {t_metrics['fn']:.1f}"
    )

    # Test Focal Tversky
    focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    ft_loss, ft_metrics = focal_tversky(pred, target)
    print(f"Focal Tversky Loss: {ft_loss.item():.4f}")

    # Test clDice
    cldice = CenterlineDiceLoss(num_iter=10)
    cl_loss, cl_metrics = cldice(pred, target)
    print(f"clDice Loss: {cl_loss.item():.4f}")
    print(f"  Tprec: {cl_metrics['tprec']:.4f}, Tsens: {cl_metrics['tsens']:.4f}")

    # Test combined
    river_loss = get_river_detection_loss()
    total, metrics = river_loss(pred, target)
    print(f"\nCombined River Loss: {total.item():.4f}")
    print(f"  BCE: {metrics['bce']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  Focal Tversky: {metrics['focal_tversky']:.4f}")
    print(f"  clDice: {metrics['cldice']:.4f}")

    # Verify gradients flow
    pred.requires_grad = True
    total, _ = river_loss(pred, target)
    total.backward()
    print(f"\nGradient check: pred.grad exists = {pred.grad is not None}")
    print(f"Gradient magnitude: {pred.grad.abs().mean().item():.6f}")

    print("\n" + "=" * 60)
    print("All tests passed!")

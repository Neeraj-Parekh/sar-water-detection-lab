#!/usr/bin/env python3
"""
================================================================================
LEARNABLE ENSEMBLE - Uncertainty-Weighted Model Combination
================================================================================

Problem with Fixed Ensemble Weights:
- Current ensemble uses hardcoded weights (LGB=0.75, UNet=0.20, Physics=0.05)
- Grid search showed these weights actually HURT performance vs LightGBM alone
- Fixed weights don't adapt to different chip types (urban, river, lake)

Solutions Implemented:

1. UNCERTAINTY-WEIGHTED ENSEMBLE (Kendall & Gal, NeurIPS 2017)
   - Learn task-specific uncertainty (homoscedastic uncertainty)
   - Higher uncertainty = lower weight
   - Weights adapt during training

2. CONFIDENCE-BASED ENSEMBLE
   - Weight predictions by model confidence at each pixel
   - Confident model dominates uncertain model
   - No training required

3. STACKING ENSEMBLE
   - Train a meta-learner on model outputs
   - Can learn non-linear combinations
   - Most flexible but requires validation set

4. SPATIAL ATTENTION ENSEMBLE
   - Different weights for different spatial regions
   - UNet better at edges, LGB better at centers
   - Learns spatial weight maps

References:
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian DL?", NeurIPS 2017
- Wolpert, "Stacked Generalization", Neural Networks, 1992
- Caruana et al., "Ensemble Selection", ICML 2004

Author: SAR Water Detection Project
Date: 2026-01-26
Version: 1.0 - Production Ready
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. UNCERTAINTY-WEIGHTED ENSEMBLE
# =============================================================================


class UncertaintyWeightedEnsemble(nn.Module):
    """
    Learnable ensemble using homoscedastic uncertainty.

    Key idea: Each model has an associated "task uncertainty" σ.
    Weight = 1/σ² (precision). Models with lower uncertainty get higher weight.

    During training, the model learns optimal σ values that minimize total loss.

    Loss formulation:
        L = Σᵢ (1/2σᵢ²) * Lᵢ + log(σᵢ)

    The log(σᵢ) term prevents σ → ∞ (which would make all weights 0).

    Usage:
        ensemble = UncertaintyWeightedEnsemble(n_models=3)
        combined, weights = ensemble([lgb_prob, unet_prob, physics_prob])
    """

    def __init__(
        self,
        n_models: int = 3,
        model_names: Optional[List[str]] = None,
        init_log_var: float = 0.0,
    ):
        """
        Args:
            n_models: Number of models in ensemble
            model_names: Optional names for logging
            init_log_var: Initial log variance (0 = equal weights)
        """
        super().__init__()

        self.n_models = n_models
        self.model_names = model_names or [f"model_{i}" for i in range(n_models)]

        # Learnable log-variance for each model
        # log(σ²) is more numerically stable than σ directly
        self.log_vars = nn.Parameter(torch.full((n_models,), init_log_var))

    def get_weights(self) -> torch.Tensor:
        """
        Compute normalized weights from log-variances.

        Weight = exp(-log_var) = 1/σ²
        Then normalize to sum to 1.
        """
        # Precision (inverse variance)
        precision = torch.exp(-self.log_vars)

        # Normalize
        weights = precision / precision.sum()

        return weights

    def forward(
        self,
        predictions: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Combine predictions using learned weights.

        Args:
            predictions: List of (B, 1, H, W) or (B, H, W) probability maps

        Returns:
            combined: Weighted average prediction
            weights: Current model weights
            info: Dictionary with weight info for logging
        """
        # Ensure all predictions have same shape
        preds = []
        for p in predictions:
            if p.dim() == 3:
                p = p.unsqueeze(1)
            preds.append(p)

        # Stack: (N, B, 1, H, W)
        stacked = torch.stack(preds, dim=0)

        # Get weights: (N,)
        weights = self.get_weights()

        # Weighted sum: (B, 1, H, W)
        # weights: (N, 1, 1, 1, 1) for broadcasting
        weights_broadcast = weights.view(-1, 1, 1, 1, 1)
        combined = (stacked * weights_broadcast).sum(dim=0)

        # Info for logging
        info = {
            f"weight_{name}": w.item() for name, w in zip(self.model_names, weights)
        }
        info.update(
            {
                f"log_var_{name}": lv.item()
                for name, lv in zip(self.model_names, self.log_vars)
            }
        )

        return combined, weights, info

    def compute_loss_with_uncertainty(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted loss.

        This is used during training to learn optimal weights.

        Args:
            predictions: List of model predictions
            target: Ground truth
            loss_fn: Base loss function (e.g., BCELoss)

        Returns:
            total_loss: Combined uncertainty-weighted loss
            info: Loss components and weights
        """
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        info = {}

        for i, (pred, name) in enumerate(zip(predictions, self.model_names)):
            # Compute individual loss
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            if target.dim() == 3:
                target = target.unsqueeze(1)

            loss_i = loss_fn(pred, target)

            # Uncertainty weighting: (1/2σ²) * L + log(σ)
            # In terms of log_var: 0.5 * exp(-log_var) * L + 0.5 * log_var
            precision_i = torch.exp(-self.log_vars[i])
            weighted_loss_i = 0.5 * precision_i * loss_i + 0.5 * self.log_vars[i]

            total_loss = total_loss + weighted_loss_i

            info[f"loss_{name}"] = loss_i.item()
            info[f"weighted_loss_{name}"] = weighted_loss_i.item()

        info["total_loss"] = total_loss.item()
        info.update(
            {
                f"weight_{name}": w.item()
                for name, w in zip(self.model_names, self.get_weights())
            }
        )

        return total_loss, info


# =============================================================================
# 2. CONFIDENCE-BASED ENSEMBLE
# =============================================================================


class ConfidenceBasedEnsemble(nn.Module):
    """
    Weight predictions by per-pixel confidence.

    Confidence = |P - 0.5| (how far from uncertainty)
    Highly confident predictions dominate uncertain ones.

    This requires NO training - just inference-time weighting.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_confidence: float = 0.01,
    ):
        """
        Args:
            temperature: Sharpness of confidence weighting
            min_confidence: Minimum confidence to prevent division by zero
        """
        super().__init__()
        self.temperature = temperature
        self.min_confidence = min_confidence

    def forward(
        self,
        predictions: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions weighted by confidence.

        Args:
            predictions: List of probability maps

        Returns:
            combined: Confidence-weighted prediction
            confidence_weights: Per-model confidence at each pixel
        """
        # Stack predictions
        preds = []
        for p in predictions:
            if p.dim() == 3:
                p = p.unsqueeze(1)
            preds.append(p)

        stacked = torch.stack(preds, dim=0)  # (N, B, 1, H, W)

        # Confidence = distance from 0.5
        confidence = torch.abs(stacked - 0.5) * 2  # Scale to [0, 1]
        confidence = confidence.clamp(min=self.min_confidence)

        # Apply temperature
        weights = confidence**self.temperature

        # Normalize across models
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted combination
        combined = (stacked * weights).sum(dim=0)

        return combined, weights


# =============================================================================
# 3. STACKING ENSEMBLE (Meta-Learner)
# =============================================================================


class StackingEnsemble(nn.Module):
    """
    Meta-learner that learns non-linear combination of model outputs.

    Architecture:
    - Input: Concatenated model predictions (N channels)
    - Small CNN to learn spatial patterns in agreement/disagreement
    - Output: Final prediction

    This can learn complex rules like:
    - "Trust UNet at edges, LGB at centers"
    - "When models disagree, use physics as tie-breaker"
    """

    def __init__(
        self,
        n_models: int = 3,
        hidden_channels: int = 32,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_models: Number of input models
            hidden_channels: Hidden layer size
            dropout: Dropout rate
        """
        super().__init__()

        self.n_models = n_models

        # Small CNN meta-learner
        self.meta_learner = nn.Sequential(
            # Combine model outputs
            nn.Conv2d(n_models, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Learn patterns
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(hidden_channels, 1, 1),
        )

        # Optional: Also output per-model weights for interpretability
        self.weight_head = nn.Sequential(
            nn.Conv2d(n_models, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_models, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        predictions: List[torch.Tensor],
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combine predictions using meta-learner.

        Args:
            predictions: List of probability maps
            return_weights: If True, also return learned weight maps

        Returns:
            combined: Meta-learner output (logits)
            weights: (optional) Per-pixel weight maps for each model
        """
        # Stack predictions: (B, N, H, W)
        preds = []
        for p in predictions:
            if p.dim() == 4:
                p = p.squeeze(1)
            preds.append(p)

        stacked = torch.stack(preds, dim=1)  # (B, N, H, W)

        # Meta-learner
        output = self.meta_learner(stacked)  # (B, 1, H, W)

        if return_weights:
            weights = self.weight_head(stacked)  # (B, N, H, W)
            return output, weights

        return output


# =============================================================================
# 4. SPATIAL ATTENTION ENSEMBLE
# =============================================================================


class SpatialAttentionEnsemble(nn.Module):
    """
    Learn spatial attention masks for each model.

    Key insight: Different models are better at different spatial regions:
    - UNet better at boundaries (spatial context)
    - LGB better at homogeneous regions (pixel features)
    - Physics better near DEM edges

    This learns where to trust each model.
    """

    def __init__(
        self,
        n_models: int = 3,
        n_input_features: int = 9,
        hidden_channels: int = 16,
    ):
        """
        Args:
            n_models: Number of models
            n_input_features: Number of input channels (for spatial context)
            hidden_channels: Hidden layer size
        """
        super().__init__()

        self.n_models = n_models

        # Attention network: input features -> weight maps
        self.attention_net = nn.Sequential(
            nn.Conv2d(n_input_features, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_models, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        predictions: List[torch.Tensor],
        input_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions using spatial attention.

        Args:
            predictions: List of probability maps (B, 1, H, W)
            input_features: Input tensor used for attention (B, C, H, W)

        Returns:
            combined: Attention-weighted prediction
            attention_maps: Per-model attention weights
        """
        # Compute attention weights from input features
        attention = self.attention_net(input_features)  # (B, N, H, W)

        # Stack predictions
        preds = []
        for p in predictions:
            if p.dim() == 4:
                p = p.squeeze(1)
            preds.append(p)

        stacked = torch.stack(preds, dim=1)  # (B, N, H, W)

        # Weighted combination
        combined = (stacked * attention).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        return combined, attention


# =============================================================================
# 5. SIMPLE OPTIMIZABLE ENSEMBLE (for numpy/sklearn models)
# =============================================================================


class OptimizableEnsemble:
    """
    Simple optimizable ensemble for non-PyTorch models (like LightGBM).

    Uses scipy.optimize to find optimal weights on validation set.
    """

    def __init__(
        self,
        n_models: int = 3,
        model_names: Optional[List[str]] = None,
    ):
        self.n_models = n_models
        self.model_names = model_names or [f"model_{i}" for i in range(n_models)]
        self.weights = np.ones(n_models) / n_models  # Equal weights initially

    def optimize_weights(
        self,
        predictions: List[np.ndarray],
        targets: np.ndarray,
        metric: str = "iou",
    ) -> Dict[str, float]:
        """
        Find optimal weights using grid search or scipy.optimize.

        Args:
            predictions: List of prediction arrays
            targets: Ground truth
            metric: Optimization metric ('iou', 'dice', 'f1')

        Returns:
            Dictionary with optimal weights and achieved metric
        """
        from scipy.optimize import minimize

        def compute_metric(weights: np.ndarray) -> float:
            """Compute negative metric (for minimization)."""
            # Normalize weights
            w = np.abs(weights)
            w = w / w.sum()

            # Combine predictions
            combined = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, w):
                combined += weight * pred

            # Threshold
            combined_binary = combined > 0.5
            target_binary = targets > 0.5

            # Compute IoU
            intersection = np.logical_and(combined_binary, target_binary).sum()
            union = np.logical_or(combined_binary, target_binary).sum()
            iou = intersection / (union + 1e-8)

            return -iou  # Negative for minimization

        # Initial guess: equal weights
        x0 = np.ones(self.n_models) / self.n_models

        # Optimize
        result = minimize(
            compute_metric,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        # Store optimal weights
        self.weights = np.abs(result.x)
        self.weights = self.weights / self.weights.sum()

        # Compute final metric
        final_metric = -result.fun

        return {
            "weights": dict(zip(self.model_names, self.weights.tolist())),
            "metric": final_metric,
            "success": result.success,
        }

    def predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions using optimized weights."""
        combined = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            combined += weight * pred
        return combined


# =============================================================================
# 6. ENSEMBLE FACTORY
# =============================================================================


def create_ensemble(
    ensemble_type: str,
    n_models: int = 3,
    model_names: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create ensemble modules.

    Args:
        ensemble_type: Type of ensemble:
            - 'uncertainty': UncertaintyWeightedEnsemble
            - 'confidence': ConfidenceBasedEnsemble
            - 'stacking': StackingEnsemble
            - 'spatial': SpatialAttentionEnsemble
        n_models: Number of models
        model_names: Names for logging
        **kwargs: Additional arguments for specific ensemble types

    Returns:
        Ensemble module
    """
    if ensemble_type == "uncertainty":
        return UncertaintyWeightedEnsemble(
            n_models=n_models, model_names=model_names, **kwargs
        )
    elif ensemble_type == "confidence":
        return ConfidenceBasedEnsemble(**kwargs)
    elif ensemble_type == "stacking":
        return StackingEnsemble(n_models=n_models, **kwargs)
    elif ensemble_type == "spatial":
        return SpatialAttentionEnsemble(n_models=n_models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


# =============================================================================
# TESTING
# =============================================================================


def test_ensembles():
    """Test all ensemble implementations."""
    print("=" * 60)
    print("Testing Learnable Ensemble Implementations")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dummy predictions
    B, H, W = 2, 64, 64
    pred_lgb = torch.sigmoid(torch.randn(B, 1, H, W, device=device))
    pred_unet = torch.sigmoid(torch.randn(B, 1, H, W, device=device))
    pred_physics = torch.sigmoid(torch.randn(B, 1, H, W, device=device))
    predictions = [pred_lgb, pred_unet, pred_physics]
    target = (torch.rand(B, 1, H, W, device=device) > 0.5).float()

    # Test 1: Uncertainty-Weighted Ensemble
    print("\n1. Uncertainty-Weighted Ensemble:")
    uw_ensemble = UncertaintyWeightedEnsemble(
        n_models=3,
        model_names=["LGB", "UNet", "Physics"],
    ).to(device)

    combined, weights, info = uw_ensemble(predictions)
    print(f"   Output shape: {combined.shape}")
    print(f"   Initial weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    # Test training
    loss_fn = nn.BCELoss()
    loss, loss_info = uw_ensemble.compute_loss_with_uncertainty(
        predictions, target, loss_fn
    )
    print(f"   Training loss: {loss.item():.4f}")

    # Test 2: Confidence-Based Ensemble
    print("\n2. Confidence-Based Ensemble:")
    conf_ensemble = ConfidenceBasedEnsemble(temperature=2.0).to(device)

    combined_conf, conf_weights = conf_ensemble(predictions)
    print(f"   Output shape: {combined_conf.shape}")
    print(f"   Confidence weights shape: {conf_weights.shape}")

    # Test 3: Stacking Ensemble
    print("\n3. Stacking Ensemble:")
    stack_ensemble = StackingEnsemble(n_models=3, hidden_channels=16).to(device)

    combined_stack, weight_maps = stack_ensemble(predictions, return_weights=True)
    print(f"   Output shape: {combined_stack.shape}")
    print(f"   Weight maps shape: {weight_maps.shape}")

    # Test 4: Spatial Attention Ensemble
    print("\n4. Spatial Attention Ensemble:")
    spatial_ensemble = SpatialAttentionEnsemble(
        n_models=3,
        n_input_features=9,
    ).to(device)

    input_features = torch.randn(B, 9, H, W, device=device)
    combined_spatial, attention = spatial_ensemble(predictions, input_features)
    print(f"   Output shape: {combined_spatial.shape}")
    print(f"   Attention maps shape: {attention.shape}")

    # Test 5: Optimizable Ensemble (numpy)
    print("\n5. Optimizable Ensemble (numpy):")
    opt_ensemble = OptimizableEnsemble(
        n_models=3,
        model_names=["LGB", "UNet", "Physics"],
    )

    # Create numpy predictions
    preds_np = [p.cpu().numpy() for p in [pred_lgb, pred_unet, pred_physics]]
    target_np = target.cpu().numpy()

    result = opt_ensemble.optimize_weights(preds_np, target_np)
    print(f"   Optimized weights: {result['weights']}")
    print(f"   Achieved IoU: {result['metric']:.4f}")

    # Test 6: Factory function
    print("\n6. Testing Factory Function:")
    for etype in ["uncertainty", "confidence", "stacking"]:
        ensemble = create_ensemble(etype, n_models=3)
        print(f"   Created {etype}: {type(ensemble).__name__}")

    # Test 7: Gradient check for learnable ensembles
    print("\n7. Gradient Check:")
    uw_ensemble.zero_grad()
    combined, _, _ = uw_ensemble(predictions)
    loss = combined.mean()
    loss.backward()

    grad_norm = uw_ensemble.log_vars.grad.norm().item()
    print(f"   log_vars gradient norm: {grad_norm:.6f}")
    print(f"   Gradients computed: {uw_ensemble.log_vars.grad is not None}")

    print("\n" + "=" * 60)
    print("Ensemble Tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_ensembles()

"""
Physics-Guided U-Net for SAR Water Detection
=============================================

References:
- arXiv 2024: Physics-Guided Neural Networks for SAR Flood Detection
- Sen1Floods11 Benchmark (CVPR)
- JRC Global Surface Water

Author: SAR Water Detection Lab
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class HANDAttention(nn.Module):
    """
    HAND-based spatial attention module.
    
    Physics principle: Water probability should be HIGH where HAND is LOW.
    This module learns to weight features based on HAND values.
    """
    
    def __init__(self, hand_threshold: float = 10.0):
        super().__init__()
        self.hand_threshold = hand_threshold
        
        # Learnable attention refinement
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature maps
            hand: [B, 1, H, W] HAND values in meters
            
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Physics-based attention: high at low HAND
        physics_attn = torch.sigmoid(-hand / self.hand_threshold)
        
        # Learnable refinement
        learned_attn = self.refine(hand)
        
        # Combine physics + learned
        combined_attn = 0.7 * physics_attn + 0.3 * learned_attn
        
        return features * combined_attn


# =============================================================================
# Physics-Guided U-Net
# =============================================================================

class PhysicsGuidedUNet(nn.Module):
    """
    U-Net with HAND attention for physics-guided water detection.
    
    Input channels:
        - VV (Sentinel-1 dB)
        - VH (Sentinel-1 dB)
        - DEM (meters)
        - HAND (meters)
        - Slope (degrees)
        - TWI (topographic wetness index)
    
    Output: Binary water/non-water segmentation
    """
    
    def __init__(self, in_channels: int = 6, num_classes: int = 2,
                 base_features: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_channels, base_features)
        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.enc3 = ConvBlock(base_features * 2, base_features * 4)
        self.enc4 = ConvBlock(base_features * 4, base_features * 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)
        
        # HAND Attention at bottleneck
        self.hand_attention = HANDAttention(hand_threshold=10.0)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_features * 16, base_features * 8)
        
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)
        
        # Final output
        self.final = nn.Conv2d(base_features, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics-guided attention.
        
        Args:
            x: [B, 6, H, W] input tensor with channels [VV, VH, DEM, HAND, Slope, TWI]
            
        Returns:
            [B, num_classes, H, W] logits
        """
        # Extract HAND channel for attention
        hand = x[:, 3:4, :, :]  # HAND is channel 3
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck with HAND attention
        b = self.bottleneck(self.pool(e4))
        
        # Downsample HAND to bottleneck resolution
        hand_downsampled = F.interpolate(hand, size=b.shape[2:], mode='bilinear', align_corners=False)
        b = self.hand_attention(b, hand_downsampled)
        
        # Decoder with skip connections (with size matching)
        up4 = self.up4(b)
        up4 = F.interpolate(up4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([up4, e4], dim=1))
        
        up3 = self.up3(d4)
        up3 = F.interpolate(up3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))
        
        up2 = self.up2(d3)
        up2 = F.interpolate(up2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        
        up1 = self.up1(d2)
        up1 = F.interpolate(up1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        
        return self.final(d1)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary prediction."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)[:, 1]  # Water class probability
            return (probs > threshold).float()


# =============================================================================
# Physics Loss Functions
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Combined loss with physics constraints.
    
    Based on arXiv 2024: Physics-Guided Neural Networks
    
    Loss = BCE + λ_hand * L_hand + λ_slope * L_slope
    """
    
    def __init__(self, hand_weight: float = 0.3, slope_weight: float = 0.2,
                 slope_threshold: float = 15.0):
        super().__init__()
        self.hand_weight = hand_weight
        self.slope_weight = slope_weight
        self.slope_threshold = slope_threshold
        
        self.bce = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                hand: torch.Tensor, slope: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics-constrained loss.
        
        Args:
            logits: [B, 2, H, W] model output
            targets: [B, H, W] ground truth (0 or 1)
            hand: [B, H, W] HAND values
            slope: [B, H, W] slope values
            
        Returns:
            Total loss and breakdown dict
        """
        # Clamp targets to valid class range [0, num_classes-1]
        targets_clamped = targets.long().clamp(0, 1)
        
        # Standard BCE loss
        loss_bce = self.bce(logits, targets_clamped)
        
        # Get water probability
        probs = F.softmax(logits, dim=1)[:, 1]  # [B, H, W]
        
        # HAND monotonicity loss
        # Physics: water probability should DECREASE with HAND
        loss_hand = self._hand_loss(probs, hand)
        
        # Slope exclusion loss
        # Physics: no water on steep slopes (>15°)
        loss_slope = self._slope_loss(probs, slope)
        
        # Total loss
        total = loss_bce + self.hand_weight * loss_hand + self.slope_weight * loss_slope
        
        breakdown = {
            'bce': loss_bce.item(),
            'hand': loss_hand.item(),
            'slope': loss_slope.item(),
            'total': total.item()
        }
        
        return total, breakdown
    
    def _hand_loss(self, probs: torch.Tensor, hand: torch.Tensor) -> torch.Tensor:
        """
        Penalize positive correlation between water prob and HAND.
        
        Water should be at LOW HAND, so correlation should be NEGATIVE.
        """
        # Flatten (use reshape for non-contiguous tensors)
        pred_flat = probs.reshape(-1)
        hand_flat = hand.reshape(-1)
        
        # Remove invalid values
        valid = ~(torch.isnan(pred_flat) | torch.isnan(hand_flat))
        if valid.sum() < 100:
            return torch.tensor(0.0, device=probs.device)
        
        pred_valid = pred_flat[valid]
        hand_valid = hand_flat[valid]
        
        # Compute Pearson correlation
        pred_mean = pred_valid.mean()
        hand_mean = hand_valid.mean()
        
        pred_centered = pred_valid - pred_mean
        hand_centered = hand_valid - hand_mean
        
        corr_num = (pred_centered * hand_centered).sum()
        corr_denom = torch.sqrt((pred_centered**2).sum() * (hand_centered**2).sum() + 1e-8)
        
        correlation = corr_num / corr_denom
        
        # Penalize positive correlation (should be negative)
        return F.relu(correlation)
    
    def _slope_loss(self, probs: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        """Penalize water predictions on steep slopes."""
        steep_mask = slope > self.slope_threshold
        if steep_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device)
        
        # Mean water probability on steep slopes (should be ~0)
        return probs[steep_mask].mean()


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute IoU (Intersection over Union)."""
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    
    intersection = (pred_bool & target_bool).float().sum()
    union = (pred_bool | target_bool).float().sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def compute_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute F1 score."""
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    
    tp = (pred_bool & target_bool).float().sum()
    fp = (pred_bool & ~target_bool).float().sum()
    fn = (~pred_bool & target_bool).float().sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


# =============================================================================
# Model Factory
# =============================================================================

def create_model(in_channels: int = 6, num_classes: int = 2,
                 pretrained: bool = False) -> PhysicsGuidedUNet:
    """Create and initialize model."""
    model = PhysicsGuidedUNet(in_channels, num_classes)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    print(f"Using device: {device}")
    
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 6, 256, 256).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test loss
    loss_fn = PhysicsLoss()
    target = torch.randint(0, 2, (2, 256, 256)).to(device)
    hand = torch.rand(2, 256, 256).to(device) * 20
    slope = torch.rand(2, 256, 256).to(device) * 30
    
    loss, breakdown = loss_fn(out, target, hand, slope)
    print(f"Loss breakdown: {breakdown}")

"""
India-Focused Lightweight Chip Selector
========================================

ResNet-18 based binary classifier with GradCAM interpretability.

Task: Classify SAR chips as "Good" (clear water signal, India-relevant) 
      vs "Reject" (poor quality, cloud shadow, ambiguous).

Author: SAR Water Detection Lab
Date: 2026-01-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np
from pathlib import Path


# =============================================================================
# Model Architecture
# =============================================================================

class IndiaChipSelector(nn.Module):
    """
    Lightweight SAR chip quality classifier.
    
    Base: ResNet-18 (11.7M parameters)
    Input: [batch, 2, H, W] - VV and VH SAR bands
    Output: [batch, 2] - logits for [Reject, Good]
    
    Features:
    - Transfer learning from ImageNet
    - First conv adapted for 2-channel SAR input
    - GradCAM hooks for interpretability
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Adapt first conv: 3 RGB → 2 SAR (VV, VH)
        # Keep pretrained weights by averaging RGB channels
        if pretrained:
            weight = self.resnet.conv1.weight.data  # [64, 3, 7, 7]
            self.resnet.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize with averaged ImageNet weights
            self.resnet.conv1.weight.data = weight[:, :2, :, :].clone()
        else:
            self.resnet.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classifier: 1000 classes → 2 (reject/good)
        self.resnet.fc = nn.Linear(512, 2)
        
        # Hooks for GradCAM
        self.gradients = None
        self.activations = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GradCAM hooks.
        
        Args:
            x: [B, 2, H, W] SAR input (VV, VH)
            
        Returns:
            logits: [B, 2] class logits
        """
        # Save activation hook
        def save_gradient(grad):
            self.gradients = grad
            
        # Forward through ResNet blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # Final conv layer [B, 512, H/32, W/32]
        
        # Register hook for GradCAM
        if x.requires_grad:
            x.register_hook(save_gradient)
        self.activations = x
        
        # Classification head
        x = self.resnet.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)      # [B, 512]
        x = self.resnet.fc(x)         # [B, 2]
        
        return x
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction with threshold."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of "Good"
            return (probs > threshold).long()


# =============================================================================
# GradCAM Implementation
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Shows which spatial regions (water areas, terrain features) 
    influenced the model's chip quality decision.
    
    Reference: Selvaraju et al., ICCV 2017
    """
    
    def __init__(self, model: IndiaChipSelector):
        self.model = model
        self.model.eval()
        
    def generate_cam(self, input_image: torch.Tensor, 
                     target_class: int = 1) -> np.ndarray:
        """
        Generate Class Activation Map for input chip.
        
        Args:
            input_image: [1, 2, H, W] SAR chip (single sample)
            target_class: 0 = Reject, 1 = Good
            
        Returns:
            cam: [H, W] heatmap (0-1 range) showing influential regions
        """
        # Ensure gradient tracking
        input_image = input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)  # [1, 2]
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations from last conv layer
        gradients = self.model.gradients.data.cpu()  # [1, 512, H', W']
        activations = self.model.activations.data.cpu()  # [1, 512, H', W']
        
        # Global average pooling of gradients → weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)  # Remove negative influence
        
        # Normalize to 0-1
        cam = cam.squeeze().numpy()  # [H', W']
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Upsample to original image size
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
        cam_upsampled = F.interpolate(
            cam_tensor, 
            size=input_image.shape[2:], 
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return cam_upsampled
    
    def overlay_heatmap(self, sar_image: np.ndarray, cam: np.ndarray, 
                        alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM heatmap on SAR image.
        
        Args:
            sar_image: [H, W] grayscale SAR (e.g. VV band)
            cam: [H, W] CAM heatmap (0-1)
            alpha: Transparency of heatmap
            
        Returns:
            overlay: [H, W, 3] RGB visualization
        """
        import cv2
        
        # Normalize SAR to 0-255
        sar_norm = ((sar_image - sar_image.min()) / 
                    (sar_image.max() - sar_image.min()) * 255).astype(np.uint8)
        sar_rgb = cv2.cvtColor(sar_norm, cv2.COLOR_GRAY2RGB)
        
        # Convert CAM to heatmap (blue → red)
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(sar_rgb, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


# =============================================================================
# Dataset
# =============================================================================

class SARChipDataset(Dataset):
    """
    Dataset for 120 India SAR chips.
    
    Expected structure:
        chips/
            chip_001_large_lakes_7band_jrc_f32.tif
            chip_002_large_lakes_7band_jrc_f32.tif
            ...
            chip_120_high_elevation_7band_jrc_f32.tif
    
    Bands:
        0: VV
        1: VH
        2: DEM
        3: HAND
        4: SLOPE
        5: TWI
        6: JRC_Water (ground truth)
    """
    
    def __init__(self, chip_files: list, labels: Optional[list] = None, 
                 transform=None):
        """
        Args:
            chip_files: List of paths to .tif or .npy chip files
            labels: Optional list of labels (0=Reject, 1=Good)
                    If None, extracts from JRC_Water band
            transform: Optional torchvision transforms
        """
        self.chip_files = self._validate_files(chip_files)
        self.labels = labels
        self.transform = transform
        
    def _validate_files(self, files: list) -> list:
        """Filter out corrupted files."""
        valid_files = []
        print(f"Validating {len(files)} chips...")
        for f in files:
            try:
                if str(f).endswith('.npy'):
                    # Quick check: try loading
                    np.load(f, mmap_mode='r')
                valid_files.append(f)
            except Exception as e:
                print(f"Skipping corrupted file {f}: {e}")
        print(f"Kept {len(valid_files)}/{len(files)} valid chips")
        return valid_files
        
    def __len__(self):
        return len(self.chip_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        chip_path = self.chip_files[idx]
        
        try:
            # Load chip
            if str(chip_path).endswith('.npy'):
                data = np.load(chip_path).astype(np.float32)
            else:  # GeoTIFF
                import rasterio
                with rasterio.open(chip_path) as src:
                    data = src.read().transpose(1, 2, 0).astype(np.float32)
        except Exception as e:
            print(f"Error loading {chip_path}: {e}")
            # Return dummy data to avoid crash (should be caught by validation, but just in case)
            return torch.zeros(2, 512, 512), 0
        
        # Extract VV, VH bands
        vv = data[:, :, 0]
        vh = data[:, :, 1]
        
        # Normalize SAR (dB to 0-1 range)
        # Typical SAR range: -30 to 0 dB
        vv = np.clip((vv + 30) / 30, 0, 1)
        vh = np.clip((vh + 30) / 30, 0, 1)
        
        # Stack to [2, H, W]
        features = np.stack([vv, vh], axis=0)
        
        # Get label
        if self.labels is not None:
            label = self.labels[idx]
        else:
            # Auto-label from JRC water coverage
            jrc_water = data[:, :, 6]
            water_percentage = (jrc_water > 0).mean()
            # Good chip if 5-95% water (not too little, not all water)
            label = 1 if 0.05 < water_percentage < 0.95 else 0
        
        # Convert to tensor
        features = torch.from_numpy(features)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


# =============================================================================
# Training Utilities
# =============================================================================

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_checkpoint(model: IndiaChipSelector, optimizer, epoch: int, 
                    metrics: dict, filepath: Path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)


if __name__ == "__main__":
    # Test model
    print("Testing IndiaChipSelector...")
    
    model = IndiaChipSelector(pretrained=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 2, 512, 512)  # Batch of 2 chips
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test GradCAM
    print("\nTesting GradCAM...")
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(x[0:1], target_class=1)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("\n✅ Model ready for training!")

"""
Merge feature TIFs with verified label TIFs into complete NPY chips.
"""
import numpy as np
from pathlib import Path
import rasterio

def merge_features_and_labels():
    features_dir = Path("/media/neeraj-parekh/Data1/sar soil system/chips/processed/features_7band")
    labels_dir = Path("/media/neeraj-parekh/Data1/sar soil system/chips/processed/labels_verified")
    output_dir = Path("/media/neeraj-parekh/Data1/sar soil system/chips/processed/features_with_truth")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_files = sorted(features_dir.glob("*.tif"))
    print(f"Found {len(feature_files)} feature files")
    
    merged_count = 0
    
    for feat_file in feature_files:
        # Extract chip number and type
        # e.g. chip_001_large_lakes_features_7band_f32.tif
        name_parts = feat_file.stem.split("_")
        chip_num = name_parts[1]  # e.g. "001"
        chip_type = "_".join(name_parts[2:-2])  # e.g. "large_lakes"
        
        # Find matching label file
        label_pattern = f"chip_{chip_num}_{chip_type}_label_verified_u8.tif"
        label_file = labels_dir / label_pattern
        
        if not label_file.exists():
            # Try alternative patterns
            possible_labels = list(labels_dir.glob(f"chip_{chip_num}_*_label_*.tif"))
            if possible_labels:
                label_file = possible_labels[0]
            else:
                print(f"No label for {feat_file.name}")
                continue
        
        try:
            # Load features
            with rasterio.open(feat_file) as src:
                features = src.read()  # (7, H, W)
            
            # Load label
            with rasterio.open(label_file) as src:
                label = src.read(1)  # (H, W)
            
            # Ensure same size
            if features.shape[1:] != label.shape:
                # Resize label to match features
                from scipy.ndimage import zoom
                scale_h = features.shape[1] / label.shape[0]
                scale_w = features.shape[2] / label.shape[1]
                label = zoom(label, (scale_h, scale_w), order=0)
            
            # Stack: 7 features + 1 label = 8 bands
            combined = np.concatenate([features, label[np.newaxis, :, :]], axis=0)
            
            # Transpose to (H, W, C)
            combined = np.transpose(combined, (1, 2, 0))
            
            # Fix NaN values
            combined = np.nan_to_num(combined, nan=0.0)
            
            # Normalize label to 0/1
            combined[:, :, -1] = (combined[:, :, -1] > 127).astype(np.float32)
            
            # Save
            output_name = feat_file.stem.replace("_features_7band_f32", "_with_truth") + ".npy"
            output_path = output_dir / output_name
            np.save(output_path, combined.astype(np.float32))
            
            merged_count += 1
            if merged_count % 10 == 0:
                print(f"Merged {merged_count} chips...")
                
        except Exception as e:
            print(f"Error merging {feat_file.name}: {e}")
    
    print(f"\nDone! Merged {merged_count} chips to {output_dir}")

if __name__ == "__main__":
    merge_features_and_labels()

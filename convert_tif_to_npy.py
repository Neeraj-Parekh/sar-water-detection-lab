"""
Convert TIF chips to NPY format for GPU pipeline.
"""
import numpy as np
from pathlib import Path
import sys

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    print("GDAL not available, trying rasterio...")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def convert_tif_to_npy_gdal(tif_path: Path, output_dir: Path):
    """Convert TIF to NPY using GDAL."""
    ds = gdal.Open(str(tif_path))
    if ds is None:
        print(f"Failed to open {tif_path}")
        return None
    
    # Read all bands
    bands = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        bands.append(band.ReadAsArray())
    
    # Stack to (H, W, C)
    data = np.stack(bands, axis=-1)
    
    # Save as NPY
    output_path = output_dir / f"{tif_path.stem}.npy"
    np.save(output_path, data)
    
    ds = None
    return output_path

def convert_tif_to_npy_rasterio(tif_path: Path, output_dir: Path):
    """Convert TIF to NPY using rasterio."""
    with rasterio.open(tif_path) as src:
        # Read all bands (C, H, W) -> transpose to (H, W, C)
        data = src.read()
        data = np.transpose(data, (1, 2, 0))
    
    # Save as NPY
    output_path = output_dir / f"{tif_path.stem}.npy"
    np.save(output_path, data)
    
    return output_path

def main():
    input_dir = Path("/media/neeraj-parekh/Data1/sar soil system/chips/processed/features_7band")
    output_dir = Path("/media/neeraj-parekh/Data1/sar soil system/chips/processed/features_npy")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(input_dir.glob("*.tif"))
    print(f"Found {len(tif_files)} TIF files")
    
    for i, tif_path in enumerate(tif_files):
        try:
            if HAS_RASTERIO:
                output = convert_tif_to_npy_rasterio(tif_path, output_dir)
            elif HAS_GDAL:
                output = convert_tif_to_npy_gdal(tif_path, output_dir)
            else:
                print("Neither GDAL nor rasterio available!")
                sys.exit(1)
            
            if output and i % 10 == 0:
                print(f"Converted {i+1}/{len(tif_files)}: {output.name}")
        except Exception as e:
            print(f"Error converting {tif_path.name}: {e}")
    
    print(f"\nDone! Converted {len(tif_files)} files to {output_dir}")

if __name__ == "__main__":
    main()

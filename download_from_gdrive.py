#!/usr/bin/env python3
"""
Download chips and DEM from Google Drive with auto-resume capability.
Uses gdown library for robust Google Drive downloads.
"""
import subprocess
import sys
import os
from pathlib import Path
import time

# Google Drive folder IDs
DEM_FOLDER_URL = "https://drive.google.com/drive/folders/1MTbohNjCp9XXAR8_67XacUqurNaJCm_o"
CHIPS_FOLDER_URL = "https://drive.google.com/drive/folders/1kL8dd803Qx8tz6vqRh1uWrVvy5nHANLq"

def install_gdown():
    """Install gdown if not present."""
    try:
        import gdown
        print(f"gdown version: {gdown.__version__}")
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--upgrade"])
        import gdown
    return True

def download_folder_with_retry(url: str, output_dir: str, max_retries: int = 5):
    """Download a Google Drive folder with automatic retry on failure.
    
    Args:
        url: Google Drive folder URL
        output_dir: Local directory to save files
        max_retries: Maximum number of retry attempts
    """
    import gdown
    
    os.makedirs(output_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}: Downloading to {output_dir}")
            
            # Use gdown to download folder
            gdown.download_folder(
                url=url,
                output=output_dir,
                quiet=False,
                use_cookies=False,
                remaining_ok=True  # Continue even if some files fail
            )
            
            print(f"✓ Download completed successfully to {output_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Exponential backoff: 30s, 60s, 90s...
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return False
    
    return False

def main():
    """Main download function."""
    base_dir = Path.home() / "sar_water_detection"
    chips_dir = base_dir / "chips_gdrive"
    dem_dir = base_dir / "dem"
    
    print("=" * 60)
    print("SAR Water Detection - Google Drive Data Downloader")
    print("=" * 60)
    
    # Install gdown
    if not install_gdown():
        print("Failed to install gdown")
        sys.exit(1)
    
    # Download chips
    print("\n[1/2] Downloading chips folder...")
    chips_success = download_folder_with_retry(CHIPS_FOLDER_URL, str(chips_dir))
    
    # Download DEM
    print("\n[2/2] Downloading DEM folder...")
    dem_success = download_folder_with_retry(DEM_FOLDER_URL, str(dem_dir))
    
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  Chips: {'✓ Success' if chips_success else '✗ Failed'} -> {chips_dir}")
    print(f"  DEM:   {'✓ Success' if dem_success else '✗ Failed'} -> {dem_dir}")
    print("=" * 60)
    
    # List downloaded files
    if chips_dir.exists():
        files = list(chips_dir.rglob("*"))
        print(f"\nChips folder contains {len([f for f in files if f.is_file()])} files")
    
    if dem_dir.exists():
        files = list(dem_dir.rglob("*"))
        print(f"DEM folder contains {len([f for f in files if f.is_file()])} files")

if __name__ == "__main__":
    main()

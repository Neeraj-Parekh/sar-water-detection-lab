#!/usr/bin/env python3
"""
SAR Water Detection Lab - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="sar-water-detection-lab",
    version="1.0.0",
    author="Neeraj Parekh",
    author_email="neeraj@example.com",
    description="Interactive tool for SAR-based water body detection using 47+ algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Neeraj-Parekh/sar-water-detection-lab",
    project_urls={
        "Bug Tracker": "https://github.com/Neeraj-Parekh/sar-water-detection-lab/issues",
        "Documentation": "https://github.com/Neeraj-Parekh/sar-water-detection-lab/wiki",
        "Source Code": "https://github.com/Neeraj-Parekh/sar-water-detection-lab",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Framework :: Streamlit",
    ],
    keywords="SAR, remote sensing, water detection, geospatial, streamlit, image processing",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.31.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "rasterio>=1.3.9",
        "shapely>=2.0.0",
        "pyproj>=3.6.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "PyWavelets>=1.4.1",
    ],
    extras_require={
        "ml": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "scikit-learn>=1.3.0",
            "lightgbm>=4.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sar-lab=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
)

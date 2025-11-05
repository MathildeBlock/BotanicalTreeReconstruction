"""
Botanical Tree Reconstruction Package

A comprehensive toolkit for 3D tree reconstruction from aerial imagery using
deep learning segmentation and COLMAP structure-from-motion.

This package provides:
- Semantic segmentation of aerial tree images
- COLMAP-based 3D reconstruction pipeline
- Point cloud filtering and optimization
- Comprehensive visualization tools
- Command-line interface for easy usage

Usage:
    # CLI usage
    tree-reconstruct pipeline --config config.yaml
    tree-reconstruct segment --input-dir images/ --output-dir masks/
    tree-reconstruct colmap --input-dir images/ --output-dir models/
    
    # Python API usage
    from tree_reconstruction import segmentation, colmap_pipeline, visualization
    
    # Run segmentation
    segmentation.run_segmentation("images/", "masks/")
    
    # Run COLMAP
    colmap_pipeline.run_colmap_pipeline("images/", "models/")
    
    # Create visualizations
    visualization.create_visualizations("models/sparse/0", "plots/")
"""

__version__ = "0.1.0"
__author__ = "Tree Reconstruction Team"

# Import main modules
from . import segmentation
from . import colmap_pipeline
from . import filtering
from . import visualization
from . import config
from . import cli

# Import utilities
from .read_write_model import read_model, write_model

__all__ = [
    "segmentation",
    "colmap_pipeline", 
    "filtering",
    "visualization",
    "config",
    "cli",
    "read_model",
    "write_model"
]
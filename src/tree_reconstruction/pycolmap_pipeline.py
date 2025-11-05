#!/usr/bin/env python3
"""
COLMAP Pipeline Module - PyColmap Version

This module provides a unified interface for running COLMAP structure-from-motion
reconstruction using pycolmap Python bindings, creating proper COLMAP folder structures.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    pycolmap = None

from .read_write_model import read_model, write_model

console = Console()

class PyColmapPipeline:
    """COLMAP pipeline using pycolmap Python bindings"""
    
    def __init__(self):
        if not PYCOLMAP_AVAILABLE:
            raise ImportError("pycolmap is required. Install with: pip install pycolmap")
        
        console.print("[green]✅ PyColmap pipeline initialized[/green]")
        
    def create_colmap_workspace(self, output_dir: Path) -> Dict[str, Path]:
        """Create proper COLMAP workspace structure"""
        
        workspace = {
            'root': output_dir,
            'images': output_dir / "images",
            'sparse': output_dir / "sparse",
            'dense': output_dir / "dense",
            'database': output_dir / "database.db"
        }
        
        # Create directories
        for key, path in workspace.items():
            if key != 'database':
                path.mkdir(parents=True, exist_ok=True)
                
        # Create sparse subdirectories for different models
        (workspace['sparse'] / "0").mkdir(exist_ok=True)
        
        console.print(f"[green]📁 Created COLMAP workspace at {output_dir}[/green]")
        return workspace
    
    def import_images(self, image_dir: Path, workspace: Dict[str, Path]) -> bool:
        """Import images into COLMAP workspace"""
        
        console.print("[blue]📷 Importing images...[/blue]")
        
        try:
            # Copy or symlink images to workspace
            if not workspace['images'].exists():
                workspace['images'].mkdir(parents=True, exist_ok=True)
                
            image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + \
                         list(image_dir.glob("*.png")) + list(image_dir.glob("*.PNG"))
            
            for img_file in image_files:
                dest = workspace['images'] / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
            
            console.print(f"[green]✅ Imported {len(image_files)} images[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Image import failed: {e}[/red]")
            return False
    
    def extract_features(self, workspace: Dict[str, Path], 
                        feature_type: str = "sift",
                        max_features: int = 30000) -> bool:
        """Extract features using pycolmap"""
        
        console.print("[blue]🔍 Extracting features...[/blue]")
        
        try:
            # Set SIFT options
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_num_features = max_features
            sift_options.estimate_affine_shape = True
            sift_options.domain_size_pooling = True
            
            if pycolmap.has_cuda:
                sift_options.use_gpu = True
                console.print("[cyan]🚀 Using GPU for feature extraction[/cyan]")
            
            # Use the correct PyColmap API
            pycolmap.extract_features(
                database_path=str(workspace['database']),
                image_path=str(workspace['images']),
                sift_options=sift_options
            )
            
            console.print(f"[green]✅ Feature extraction completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Feature extraction failed: {e}[/red]")
            return False
    
    def match_features(self, workspace: Dict[str, Path], 
                      matching_type: str = "sequential") -> bool:
        """Match features using pycolmap"""
        
        console.print("[blue]🔗 Matching features...[/blue]")
        
        try:
            if matching_type == "sequential":
                # Sequential matching (good for ordered images like drone flights)
                pycolmap.match_sequential(
                    database_path=str(workspace['database']),
                    sequential_options=pycolmap.SequentialMatchingOptions()
                )
            else:
                # Exhaustive matching (more thorough but slower)
                pycolmap.match_exhaustive(
                    database_path=str(workspace['database']),
                    exhaustive_options=pycolmap.ExhaustiveMatchingOptions()
                )
            
            console.print("[green]✅ Feature matching completed[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Feature matching failed: {e}[/red]")
            return False
    
    def incremental_mapping(self, workspace: Dict[str, Path]) -> bool:
        """Run incremental Structure-from-Motion mapping"""
        
        console.print("[blue]🏗️ Running incremental mapping...[/blue]")
        
        try:
            # Set up incremental mapping options
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = 10  # Minimum number of images for a model
            mapper_options.abs_pose_min_num_inliers = 30
            mapper_options.abs_pose_min_inlier_ratio = 0.25
            mapper_options.ba_refine_focal_length = True
            mapper_options.ba_refine_principal_point = True
            mapper_options.ba_refine_extra_params = True
            
            # Run incremental mapping
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(workspace['database']),
                image_path=str(workspace['images']),
                output_path=str(workspace['sparse']),
                options=mapper_options
            )
            
            if len(reconstructions) == 0:
                console.print("[red]❌ No reconstruction created[/red]")
                return False
            
            # Save the largest reconstruction as model 0
            largest_reconstruction = max(reconstructions, key=lambda x: len(x.images))
            
            sparse_model_dir = workspace['sparse'] / "0"
            sparse_model_dir.mkdir(exist_ok=True)
            
            # Write reconstruction to standard COLMAP format
            largest_reconstruction.write_binary(str(sparse_model_dir))
            
            num_images = len(largest_reconstruction.images)
            num_points = len(largest_reconstruction.points3D)
            
            console.print(f"[green]✅ Incremental mapping completed[/green]")
            console.print(f"[green]   📷 {num_images} registered images[/green]")
            console.print(f"[green]   🎯 {num_points} 3D points[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Incremental mapping failed: {e}[/red]")
            return False
    
    def bundle_adjustment(self, workspace: Dict[str, Path]) -> bool:
        """Run bundle adjustment to refine the reconstruction"""
        
        console.print("[blue]🔧 Running bundle adjustment...[/blue]")
        
        try:
            sparse_model_dir = workspace['sparse'] / "0"
            
            # Read reconstruction
            reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
            
            # Set bundle adjustment options
            ba_options = pycolmap.BundleAdjustmentOptions()
            ba_options.solver_options.function_tolerance = 1e-6
            ba_options.solver_options.gradient_tolerance = 1e-10
            ba_options.solver_options.parameter_tolerance = 1e-8
            ba_options.solver_options.max_num_iterations = 100
            
            # Run bundle adjustment
            success = pycolmap.bundle_adjustment(reconstruction, ba_options)
            
            if success:
                # Save refined reconstruction
                reconstruction.write_binary(str(sparse_model_dir))
                console.print("[green]✅ Bundle adjustment completed[/green]")
                return True
            else:
                console.print("[red]❌ Bundle adjustment failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Bundle adjustment failed: {e}[/red]")
            return False
    
    def create_visualization_data(self, workspace: Dict[str, Path]) -> Dict[str, Any]:
        """Create data for visualization"""
        
        try:
            sparse_model_dir = workspace['sparse'] / "0"
            
            if not (sparse_model_dir / "cameras.bin").exists():
                return {}
            
            # Read the reconstruction
            cameras, images, points3D = read_model(sparse_model_dir, ext=".bin")
            
            stats = {
                'num_cameras': len(cameras),
                'num_images': len(images),
                'num_points': len(points3D),
                'workspace_path': str(workspace['root']),
                'sparse_model_path': str(sparse_model_dir)
            }
            
            return stats
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create visualization data: {e}[/red]")
            return {}

def run_pycolmap_pipeline(input_dir: Path,
                         output_dir: Path,
                         feature_type: str = "sift",
                         max_features: int = 30000,
                         matching_type: str = "sequential",
                         run_bundle_adjustment: bool = True) -> bool:
    """
    Run complete COLMAP pipeline using pycolmap
    
    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for COLMAP workspace
        feature_type: Feature descriptor type (sift, etc.)
        max_features: Maximum number of features per image
        matching_type: Feature matching strategy ('sequential' or 'exhaustive')
        run_bundle_adjustment: Whether to run bundle adjustment
        
    Returns:
        Success status
    """
    
    if not PYCOLMAP_AVAILABLE:
        console.print("[red]❌ pycolmap not available. Install with: pip install pycolmap[/red]")
        return False
    
    console.print("[bold blue]🚀 Starting PyColmap Pipeline[/bold blue]")
    console.print(f"📁 Input: {input_dir}")
    console.print(f"📁 Output: {output_dir}")
    
    pipeline = PyColmapPipeline()
    
    console.print("🔧 Running COLMAP pipeline stages...")
    
    # Create workspace first
    workspace = pipeline.create_colmap_workspace(output_dir)
    
    stages = [
        ("Importing images", lambda: pipeline.import_images(input_dir, workspace)),
        ("Extracting features", lambda: pipeline.extract_features(workspace, feature_type, max_features)),
        ("Matching features", lambda: pipeline.match_features(workspace, matching_type)),
        ("Incremental mapping", lambda: pipeline.incremental_mapping(workspace)),
    ]
    
    if run_bundle_adjustment:
        stages.append(("Bundle adjustment", lambda: pipeline.bundle_adjustment(workspace)))
    
    for i, (stage_name, stage_func) in enumerate(stages):
        console.print(f"[cyan]Step {i+1}/{len(stages)}: {stage_name}[/cyan]")
        
        success = stage_func()
        
        if not success:
            console.print(f"[red]❌ Pipeline failed at: {stage_name}[/red]")
            return False
        
        console.print(f"[green]✅ {stage_name} completed[/green]")
    
    # Create visualization data
    stats = pipeline.create_visualization_data(workspace)
    
    console.print("[bold green]🎉 PyColmap Pipeline Completed![/bold green]")
    if stats:
        console.print(f"[green]📊 Results: {stats['num_images']} images, {stats['num_points']} 3D points[/green]")
        console.print(f"[green]📁 COLMAP workspace: {stats['workspace_path']}[/green]")
        console.print(f"[green]📁 Sparse model: {stats['sparse_model_path']}[/green]")
    
    return True

# Legacy function for backward compatibility
def run_colmap_pipeline(input_dir: Path,
                       output_dir: Path,
                       feature_type: str = "sift",
                       max_features: int = 30000,
                       use_gpu: bool = True,
                       sequential: bool = True,
                       run_dense: bool = False) -> bool:
    """Legacy wrapper for the original function signature"""
    
    matching_type = "sequential" if sequential else "exhaustive"
    
    return run_pycolmap_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_type=feature_type,
        max_features=max_features,
        matching_type=matching_type,
        run_bundle_adjustment=True
    )

if __name__ == "__main__":
    import typer
    typer.run(run_pycolmap_pipeline)
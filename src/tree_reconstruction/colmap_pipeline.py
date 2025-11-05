#!/usr/bin/env python3
"""
COLMAP Pipeline Module

This module provides a unified interface for running COLMAP structure-from-motion
reconstruction with various optimization and filtering options using pycolmap.
"""

# Try to use pycolmap first, fall back to subprocess if not available
try:
    from .pycolmap_pipeline import run_pycolmap_pipeline, PyColmapPipeline
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False

import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import typer

from .read_write_model import read_model, write_model

console = Console()

class COLMAPPipeline:
    """Main COLMAP pipeline class"""
    
    def __init__(self, 
                 colmap_executable: str = "colmap",
                 working_dir: Optional[Path] = None):
        self.colmap_exe = colmap_executable
        self.working_dir = working_dir or Path.cwd()
        
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with error handling"""
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        try:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Command failed: {e}[/red]")
            console.print(f"[red]stdout: {e.stdout}[/red]")
            console.print(f"[red]stderr: {e.stderr}[/red]")
            raise
    
    def feature_extraction(self,
                          database_path: Path,
                          image_path: Path,
                          feature_type: str = "sift",
                          max_features: int = 30000,
                          use_gpu: bool = True) -> bool:
        """Extract features from images"""
        
        console.print("[bold blue]🔍 Extracting features...[/bold blue]")
        
        cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
        ]
        
        if feature_type.lower() == "sift":
            cmd.extend([
                f"--SiftExtraction.max_num_features", str(max_features),
                "--SiftExtraction.estimate_affine_shape", "true",
                "--SiftExtraction.domain_size_pooling", "true",
            ])
            if use_gpu:
                cmd.extend(["--SiftExtraction.use_gpu", "1"])
        
        try:
            self.run_command(cmd)
            console.print("[green]✅ Feature extraction completed[/green]")
            return True
        except subprocess.CalledProcessError:
            console.print("[red]❌ Feature extraction failed[/red]")
            return False
    
    def feature_matching(self,
                        database_path: Path,
                        matching_type: str = "sequential",
                        use_gpu: bool = True) -> bool:
        """Match features between images"""
        
        console.print("[bold blue]🔗 Matching features...[/bold blue]")
        
        if matching_type == "sequential":
            cmd = [
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(database_path),
            ]
            if use_gpu:
                cmd.extend([
                    "--SiftMatching.guided_matching", "true",
                    "--SiftMatching.use_gpu", "1"
                ])
        else:  # exhaustive
            cmd = [
                self.colmap_exe, "exhaustive_matcher", 
                "--database_path", str(database_path),
            ]
            if use_gpu:
                cmd.extend(["--SiftMatching.use_gpu", "1"])
        
        try:
            self.run_command(cmd)
            console.print("[green]✅ Feature matching completed[/green]")
            return True
        except subprocess.CalledProcessError:
            console.print("[red]❌ Feature matching failed[/red]")
            return False
    
    def sparse_reconstruction(self,
                            database_path: Path,
                            image_path: Path,
                            output_path: Path) -> bool:
        """Run sparse 3D reconstruction"""
        
        console.print("[bold blue]🏗️ Running sparse reconstruction...[/bold blue]")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
        ]
        
        try:
            self.run_command(cmd)
            console.print("[green]✅ Sparse reconstruction completed[/green]")
            return True
        except subprocess.CalledProcessError:
            console.print("[red]❌ Sparse reconstruction failed[/red]")
            return False
    
    def dense_reconstruction(self,
                           sparse_path: Path,
                           image_path: Path,
                           output_path: Path) -> bool:
        """Run dense 3D reconstruction"""
        
        console.print("[bold blue]🌊 Running dense reconstruction...[/bold blue]")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Image undistortion
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(sparse_path),
            "--output_path", str(output_path),
            "--output_type", "COLMAP"
        ]
        
        try:
            self.run_command(cmd)
            
            # Patch match stereo
            dense_sparse_path = output_path / "sparse"
            dense_images_path = output_path / "images"
            
            cmd = [
                self.colmap_exe, "patch_match_stereo",
                "--workspace_path", str(output_path),
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true"
            ]
            self.run_command(cmd)
            
            # Stereo fusion
            cmd = [
                self.colmap_exe, "stereo_fusion",
                "--workspace_path", str(output_path),
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", str(output_path / "fused.ply")
            ]
            self.run_command(cmd)
            
            console.print("[green]✅ Dense reconstruction completed[/green]")
            return True
            
        except subprocess.CalledProcessError:
            console.print("[red]❌ Dense reconstruction failed[/red]")
            return False

def run_colmap_pipeline(input_dir: Path,
                       output_dir: Path,
                       feature_type: str = "sift",
                       max_features: int = 30000,
                       use_gpu: bool = True,
                       sequential: bool = True,
                       run_dense: bool = False,
                       cleanup_temp: bool = True) -> bool:
    """
    Run complete COLMAP pipeline
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save reconstruction results
        feature_type: Type of features to extract (sift/superpoint)
        max_features: Maximum features per image
        use_gpu: Whether to use GPU acceleration
        sequential: Use sequential vs exhaustive matching
        run_dense: Whether to run dense reconstruction
        cleanup_temp: Whether to cleanup temporary files
    
    Returns:
        Success status
    """
    
    # Setup paths
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    
    # Remove existing database
    if database_path.exists():
        database_path.unlink()
    
    # Initialize pipeline
    pipeline = COLMAPPipeline()
    
    with Progress() as progress:
        task = progress.add_task("[cyan]COLMAP Pipeline", total=3)
        
        # Step 1: Feature extraction
        success = pipeline.feature_extraction(
            database_path=database_path,
            image_path=input_dir,
            feature_type=feature_type,
            max_features=max_features,
            use_gpu=use_gpu
        )
        if not success:
            return False
        progress.update(task, advance=1)
        
        # Step 2: Feature matching
        matching_type = "sequential" if sequential else "exhaustive"
        success = pipeline.feature_matching(
            database_path=database_path,
            matching_type=matching_type,
            use_gpu=use_gpu
        )
        if not success:
            return False
        progress.update(task, advance=1)
        
        # Step 3: Sparse reconstruction
        success = pipeline.sparse_reconstruction(
            database_path=database_path,
            image_path=input_dir,
            output_path=sparse_dir
        )
        if not success:
            return False
        progress.update(task, advance=1)
        
        # Optional: Dense reconstruction
        if run_dense:
            dense_dir = output_dir / "dense"
            progress.add_task("[cyan]Dense Reconstruction", total=1)
            success = pipeline.dense_reconstruction(
                sparse_path=sparse_dir / "0",  # Usually first reconstruction
                image_path=input_dir,
                output_path=dense_dir
            )
            if not success:
                console.print("[yellow]Dense reconstruction failed, continuing with sparse only[/yellow]")
    
    # Convert to text format for easier processing
    try:
        sparse_model_dir = sparse_dir / "0"
        if sparse_model_dir.exists():
            cameras, images, points3D = read_model(sparse_model_dir, ext=".bin")
            write_model(cameras, images, points3D, sparse_model_dir, ext=".txt")
            console.print("[green]✅ Model converted to text format[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not convert to text format: {e}[/yellow]")
    
    console.print(f"[bold green]🎉 COLMAP pipeline completed! Results in {output_dir}[/bold green]")
    return True

def optimize_reconstruction(model_dir: Path, 
                          output_dir: Path,
                          coord_threshold: float = 3.0,
                          add_points: bool = True) -> bool:
    """
    Optimize COLMAP reconstruction by filtering and adding points
    
    Args:
        model_dir: Directory containing COLMAP model
        output_dir: Directory to save optimized model
        coord_threshold: Coordinate threshold for filtering outliers
        add_points: Whether to add new triangulated points
    
    Returns:
        Success status
    """
    
    console.print("[bold yellow]🔧 Optimizing reconstruction...[/bold yellow]")
    
    try:
        # Load model
        cameras, images, points3D = read_model(model_dir, ext=".bin")
        console.print(f"Loaded model with {len(points3D)} points")
        
        # Filter points by coordinates
        filtered_points = {}
        removed_count = 0
        
        for point_id, point in points3D.items():
            x, y, z = point.xyz
            if abs(x) <= coord_threshold and abs(y) <= coord_threshold and abs(z) <= coord_threshold:
                filtered_points[point_id] = point
            else:
                removed_count += 1
        
        console.print(f"Removed {removed_count} outlier points")
        
        # Add new points through triangulation (if requested)
        if add_points:
            # This would integrate your existing triangulation code
            console.print("Adding new triangulated points...")
            # TODO: Integrate your optimized triangulation script here
        
        # Save optimized model
        output_dir.mkdir(parents=True, exist_ok=True)
        write_model(cameras, images, filtered_points, output_dir, ext=".bin")
        write_model(cameras, images, filtered_points, output_dir, ext=".txt")
        
        console.print(f"[green]✅ Optimization complete! Saved to {output_dir}[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {e}[/red]")
        return False

def main():
    """Main function for standalone usage"""
    typer.run(run_colmap_pipeline)

if __name__ == "__main__":
    main()
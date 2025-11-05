#!/usr/bin/env python3
"""
Botanical Tree Reconstruction CLI

A comprehensive tool for 3D tree reconstruction from aerial imagery using 
segmentation models and COLMAP structure-from-motion.

Usage:
    tree-reconstruct segment --input-dir /path/to/images --output-dir /path/to/masks
    tree-reconstruct colmap --input-dir /path/to/images --output-dir /path/to/sparse
    tree-reconstruct filter --input-model /path/to/sparse --output-model /path/to/filtered
    tree-reconstruct visualize --input-model /path/to/sparse --output-dir /path/to/plots
    tree-reconstruct pipeline --config config.yaml
"""

import typer
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from . import segmentation
from . import colmap_pipeline
from . import filtering
from . import visualization
from . import config

app = typer.Typer(
    name="tree-reconstruct",
    help="🌳 Botanical Tree Reconstruction Pipeline",
    rich_markup_mode="rich"
)
console = Console()

@app.command()
def segment(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory containing input images"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save segmentation masks"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to segmentation model"),
    confidence: float = typer.Option(0.5, "--confidence", "-c", help="Confidence threshold for segmentation"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to run inference on (cuda/cpu)"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size for inference"),
):
    """🎯 Run semantic segmentation on aerial images"""
    console.print(f"[bold green]Running segmentation on images in {input_dir}[/bold green]")
    
    segmentation.run_segmentation(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        confidence=confidence,
        device=device,
        batch_size=batch_size
    )
    
    console.print(f"[bold green]✅ Segmentation complete! Results saved to {output_dir}[/bold green]")

@app.command()
def colmap(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory containing images"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save COLMAP model"),
    feature_type: str = typer.Option("sift", "--features", "-f", help="Feature type (sift/superpoint)"),
    max_features: int = typer.Option(30000, "--max-features", help="Maximum number of features per image"),
    use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU acceleration"),
    sequential: bool = typer.Option(True, "--sequential/--exhaustive", help="Use sequential vs exhaustive matching"),
):
    """📐 Run COLMAP structure-from-motion reconstruction"""
    console.print(f"[bold blue]Running COLMAP reconstruction on {input_dir}[/bold blue]")
    
    colmap_pipeline.run_colmap_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        feature_type=feature_type,
        max_features=max_features,
        use_gpu=use_gpu,
        sequential=sequential
    )
    
    console.print(f"[bold blue]✅ COLMAP reconstruction complete! Model saved to {output_dir}[/bold blue]")

@app.command()
def filter(
    input_model: Path = typer.Option(..., "--input-model", "-i", help="Path to input COLMAP model"),
    output_model: Path = typer.Option(..., "--output-model", "-o", help="Path to save filtered model"),
    mask_dir: Optional[Path] = typer.Option(None, "--masks", "-m", help="Directory containing segmentation masks"),
    filter_type: str = typer.Option("coordinate", "--filter-type", "-t", help="Filter type (coordinate/mask/outlier)"),
    coord_threshold: float = typer.Option(3.0, "--coord-thresh", help="Coordinate threshold for filtering"),
    outlier_threshold: float = typer.Option(2.0, "--outlier-thresh", help="Outlier threshold"),
):
    """🔧 Filter and clean COLMAP point clouds"""
    console.print(f"[bold yellow]Filtering COLMAP model from {input_model}[/bold yellow]")
    
    filtering.filter_point_cloud(
        input_model=input_model,
        output_model=output_model,
        mask_dir=mask_dir,
        filter_type=filter_type,
        coord_threshold=coord_threshold,
        outlier_threshold=outlier_threshold
    )
    
    console.print(f"[bold yellow]✅ Filtering complete! Clean model saved to {output_model}[/bold yellow]")

@app.command()
def visualize(
    input_model: Path = typer.Option(..., "--input-model", "-i", help="Path to COLMAP model"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save visualizations"),
    image_dir: Optional[Path] = typer.Option(None, "--images", help="Directory containing original images"),
    plot_type: str = typer.Option("all", "--plot-type", "-t", help="Plot type (all/3d/projected/comparison)"),
    num_images: int = typer.Option(6, "--num-images", "-n", help="Number of images to visualize"),
):
    """📊 Create visualizations of 3D reconstruction results"""
    console.print(f"[bold magenta]Creating visualizations for {input_model}[/bold magenta]")
    
    visualization.create_visualizations(
        input_model=input_model,
        output_dir=output_dir,
        image_dir=image_dir,
        plot_type=plot_type,
        num_images=num_images
    )
    
    console.print(f"[bold magenta]✅ Visualizations complete! Saved to {output_dir}[/bold magenta]")

@app.command()
def pipeline(
    config_file: Path = typer.Option(..., "--config", "-c", help="Path to pipeline configuration file"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Run specific stage only"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be executed without running"),
):
    """🚀 Run complete reconstruction pipeline from configuration"""
    console.print(f"[bold cyan]Running pipeline from config: {config_file}[/bold cyan]")
    
    # Load configuration
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create pipeline table
    table = Table(title="Pipeline Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")
    
    stages = [
        ("segment", "Semantic Segmentation"),
        ("colmap", "3D Reconstruction"),
        ("filter", "Point Cloud Filtering"), 
        ("visualize", "Result Visualization")
    ]
    
    for stage_name, description in stages:
        if stage is None or stage == stage_name:
            if dry_run:
                table.add_row(stage_name, "DRY RUN", description)
            else:
                table.add_row(stage_name, "RUNNING", description)
    
    console.print(table)
    
    if not dry_run:
        config.run_pipeline(cfg, stage)
    
    console.print("[bold cyan]✅ Pipeline complete![/bold cyan]")

@app.command()
def status(
    project_dir: Path = typer.Option(".", "--project", "-p", help="Project directory to check"),
):
    """📋 Show project status and statistics"""
    console.print(f"[bold white]Project Status: {project_dir}[/bold white]")
    
    # Create status table
    table = Table(title="Project Overview")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    # Check for different components
    components = [
        ("Images", "data/raw", "Original aerial images"),
        ("Masks", "data/processed/masks", "Segmentation masks"),
        ("Sparse Model", "models/sparse", "COLMAP reconstruction"),
        ("Filtered Model", "models/filtered", "Cleaned point cloud"),
        ("Visualizations", "reports/figures", "Analysis plots")
    ]
    
    for name, path, desc in components:
        full_path = project_dir / path
        if full_path.exists():
            count = len(list(full_path.glob("*"))) if full_path.is_dir() else 1
            table.add_row(name, "✅ Found", f"{desc} ({count} items)")
        else:
            table.add_row(name, "❌ Missing", desc)
    
    console.print(table)

@app.command()
def init(
    project_dir: Path = typer.Option(".", "--project", "-p", help="Directory to initialize"),
    template: str = typer.Option("standard", "--template", "-t", help="Project template (standard/advanced)"),
):
    """🎯 Initialize a new tree reconstruction project"""
    console.print(f"[bold green]Initializing new project in {project_dir}[/bold green]")
    
    # Create directory structure
    directories = [
        "data/raw",
        "data/processed/masks", 
        "data/processed/filtered_images",
        "models/sparse",
        "models/filtered",
        "reports/figures",
        "configs"
    ]
    
    for dir_path in directories:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create default config
    default_config = {
        "project_name": "tree_reconstruction",
        "data": {
            "input_images": "data/raw",
            "output_masks": "data/processed/masks",
            "filtered_images": "data/processed/filtered_images"
        },
        "segmentation": {
            "model_path": None,
            "confidence": 0.5,
            "device": "cuda",
            "batch_size": 4
        },
        "colmap": {
            "feature_type": "sift",
            "max_features": 30000,
            "use_gpu": True,
            "sequential": True
        },
        "filtering": {
            "coord_threshold": 3.0,
            "outlier_threshold": 2.0
        },
        "visualization": {
            "plot_types": ["3d", "projected", "comparison"],
            "num_images": 6
        }
    }
    
    config_path = project_dir / "configs" / "default.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    console.print(f"[bold green]✅ Project initialized! Config saved to {config_path}[/bold green]")

def main():
    """Main entry point for the CLI"""
    app()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Configuration and Pipeline Management

This module provides configuration management and pipeline orchestration
for the tree reconstruction workflow.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.progress import Progress, TaskID
import time

from . import segmentation
from . import colmap_pipeline  
from . import filtering
from . import visualization

console = Console()

class PipelineConfig:
    """Configuration manager for reconstruction pipeline"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with configuration file"""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                console.print(f"[green]Loaded configuration from {config_path}[/green]")
                return config
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
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
                "batch_size": 4,
                "min_vegetation_ratio": 0.1
            },
            "colmap": {
                "feature_type": "sift",
                "max_features": 30000,
                "use_gpu": True,
                "sequential": True,
                "run_dense": False
            },
            "filtering": {
                "filter_type": "combined",
                "coord_threshold": 3.0,
                "outlier_threshold": 2.0,
                "min_observations": 3,
                "add_points": False
            },
            "visualization": {
                "plot_types": ["3d", "projected", "stats"],
                "num_images": 6,
                "point_size": 8,
                "max_points": 50000
            },
            "output": {
                "sparse_model": "models/sparse",
                "filtered_model": "models/filtered",
                "visualizations": "reports/figures",
                "logs": "logs"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Path) -> None:
        """Save configuration to file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        console.print(f"[green]Configuration saved to {output_path}[/green]")

class ReconstructionPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        self.config = config
        self.project_root = Path.cwd()
        
    def setup_directories(self) -> None:
        """Setup project directory structure"""
        directories = [
            self.config.get('data.input_images'),
            self.config.get('data.output_masks'),
            self.config.get('data.filtered_images'),
            self.config.get('output.sparse_model'),
            self.config.get('output.filtered_model'),
            self.config.get('output.visualizations'),
            self.config.get('output.logs')
        ]
        
        for dir_path in directories:
            if dir_path:
                full_path = self.project_root / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
    
    def run_segmentation_stage(self) -> bool:
        """Run segmentation stage"""
        console.print("[bold cyan]🎯 Running Segmentation Stage[/bold cyan]")
        
        input_dir = self.project_root / self.config.get('data.input_images')
        output_dir = self.project_root / self.config.get('data.output_masks')
        
        if not input_dir.exists() or not any(input_dir.iterdir()):
            console.print(f"[red]No images found in {input_dir}[/red]")
            return False
        
        try:
            segmentation.run_segmentation(
                input_dir=input_dir,
                output_dir=output_dir,
                model_path=self.config.get('segmentation.model_path'),
                confidence=self.config.get('segmentation.confidence', 0.5),
                device=self.config.get('segmentation.device', 'cuda'),
                batch_size=self.config.get('segmentation.batch_size', 4)
            )
            
            # Filter images based on vegetation content
            filtered_dir = self.project_root / self.config.get('data.filtered_images')
            segmentation.filter_images_by_mask(
                image_dir=input_dir,
                mask_dir=output_dir,
                output_dir=filtered_dir,
                min_vegetation_ratio=self.config.get('segmentation.min_vegetation_ratio', 0.1)
            )
            
            return True
            
        except Exception as e:
            console.print(f"[red]Segmentation stage failed: {e}[/red]")
            return False
    
    def run_colmap_stage(self) -> bool:
        """Run COLMAP reconstruction stage"""
        console.print("[bold blue]📐 Running COLMAP Stage[/bold blue]")
        
        # Use filtered images if available, otherwise original images
        filtered_dir = self.project_root / self.config.get('data.filtered_images')
        input_dir = self.project_root / self.config.get('data.input_images')
        
        if filtered_dir.exists() and any(filtered_dir.iterdir()):
            image_dir = filtered_dir
            console.print("Using filtered images for reconstruction")
        else:
            image_dir = input_dir
            console.print("Using original images for reconstruction")
        
        output_dir = self.project_root / self.config.get('output.sparse_model')
        
        try:
            success = colmap_pipeline.run_colmap_pipeline(
                input_dir=image_dir,
                output_dir=output_dir,
                feature_type=self.config.get('colmap.feature_type', 'sift'),
                max_features=self.config.get('colmap.max_features', 30000),
                use_gpu=self.config.get('colmap.use_gpu', True),
                sequential=self.config.get('colmap.sequential', True),
                run_dense=self.config.get('colmap.run_dense', False)
            )
            return success
            
        except Exception as e:
            console.print(f"[red]COLMAP stage failed: {e}[/red]")
            return False
    
    def run_filtering_stage(self) -> bool:
        """Run filtering stage"""
        console.print("[bold yellow]🔧 Running Filtering Stage[/bold yellow]")
        
        input_model = self.project_root / self.config.get('output.sparse_model') / "sparse" / "0"
        output_model = self.project_root / self.config.get('output.filtered_model')
        mask_dir = self.project_root / self.config.get('data.output_masks')
        
        if not input_model.exists():
            console.print(f"[red]No sparse model found at {input_model}[/red]")
            return False
        
        try:
            filtering.filter_point_cloud(
                input_model=input_model,
                output_model=output_model,
                mask_dir=mask_dir if mask_dir.exists() else None,
                filter_type=self.config.get('filtering.filter_type', 'combined'),
                coord_threshold=self.config.get('filtering.coord_threshold', 3.0),
                outlier_threshold=self.config.get('filtering.outlier_threshold', 2.0),
                min_observations=self.config.get('filtering.min_observations', 3),
                add_points=self.config.get('filtering.add_points', False)
            )
            return True
            
        except Exception as e:
            console.print(f"[red]Filtering stage failed: {e}[/red]")
            return False
    
    def run_visualization_stage(self) -> bool:
        """Run visualization stage"""
        console.print("[bold magenta]📊 Running Visualization Stage[/bold magenta]")
        
        # Visualize filtered model if available, otherwise sparse model
        filtered_model = self.project_root / self.config.get('output.filtered_model')
        sparse_model = self.project_root / self.config.get('output.sparse_model') / "sparse" / "0"
        
        if filtered_model.exists():
            input_model = filtered_model
            console.print("Visualizing filtered model")
        elif sparse_model.exists():
            input_model = sparse_model
            console.print("Visualizing sparse model")
        else:
            console.print("[red]No model found for visualization[/red]")
            return False
        
        output_dir = self.project_root / self.config.get('output.visualizations')
        
        # Get image directory
        filtered_dir = self.project_root / self.config.get('data.filtered_images')
        input_dir = self.project_root / self.config.get('data.input_images')
        
        image_dir = filtered_dir if filtered_dir.exists() else input_dir
        
        try:
            for plot_type in self.config.get('visualization.plot_types', ['3d', 'projected', 'stats']):
                visualization.create_visualizations(
                    input_model=input_model,
                    output_dir=output_dir,
                    image_dir=image_dir,
                    plot_type=plot_type,
                    num_images=self.config.get('visualization.num_images', 6)
                )
            return True
            
        except Exception as e:
            console.print(f"[red]Visualization stage failed: {e}[/red]")
            return False

def run_pipeline(config_dict: Dict[str, Any], stage: Optional[str] = None) -> bool:
    """
    Run the complete reconstruction pipeline
    
    Args:
        config_dict: Configuration dictionary
        stage: Specific stage to run (None for all stages)
        
    Returns:
        Success status
    """
    
    # Create config object
    config = PipelineConfig()
    config.config = config_dict
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(config)
    pipeline.setup_directories()
    
    # Define stages
    stages = [
        ("segment", "Segmentation", pipeline.run_segmentation_stage),
        ("colmap", "COLMAP Reconstruction", pipeline.run_colmap_stage),
        ("filter", "Point Cloud Filtering", pipeline.run_filtering_stage),
        ("visualize", "Visualization", pipeline.run_visualization_stage)
    ]
    
    # Filter stages if specific stage requested
    if stage:
        stages = [(s, n, f) for s, n, f in stages if s == stage]
        if not stages:
            console.print(f"[red]Unknown stage: {stage}[/red]")
            return False
    
    # Run stages
    with Progress() as progress:
        task = progress.add_task("[cyan]Pipeline Progress", total=len(stages))
        
        for stage_name, stage_desc, stage_func in stages:
            console.print(f"\n[bold]Running {stage_desc}...[/bold]")
            
            try:
                success = stage_func()
                if success:
                    console.print(f"[green]✅ {stage_desc} completed successfully[/green]")
                else:
                    console.print(f"[red]❌ {stage_desc} failed[/red]")
                    if stage is None:  # Only fail entire pipeline if running all stages
                        return False
                        
            except Exception as e:
                console.print(f"[red]❌ {stage_desc} failed with error: {e}[/red]")
                if stage is None:
                    return False
                    
            progress.update(task, advance=1)
    
    console.print("\n[bold green]🎉 Pipeline completed![/bold green]")
    return True

def create_example_configs() -> None:
    """Create example configuration files"""
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Basic configuration
    basic_config = PipelineConfig()._get_default_config()
    basic_config_path = configs_dir / "basic.yaml"
    with open(basic_config_path, 'w') as f:
        yaml.dump(basic_config, f, default_flow_style=False)
    
    # Advanced configuration with more options
    advanced_config = basic_config.copy()
    advanced_config.update({
        "colmap": {
            **advanced_config["colmap"],
            "run_dense": True,
            "max_features": 50000
        },
        "filtering": {
            **advanced_config["filtering"],
            "add_points": True,
            "filter_type": "mask"
        },
        "visualization": {
            **advanced_config["visualization"],
            "plot_types": ["3d", "projected", "poses", "stats"],
            "max_points": 100000
        }
    })
    
    advanced_config_path = configs_dir / "advanced.yaml"
    with open(advanced_config_path, 'w') as f:
        yaml.dump(advanced_config, f, default_flow_style=False)
    
    console.print(f"[green]Example configs created in {configs_dir}[/green]")

def main():
    """Main function for standalone usage"""
    create_example_configs()

if __name__ == "__main__":
    main()
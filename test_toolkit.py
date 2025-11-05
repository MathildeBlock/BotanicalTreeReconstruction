#!/usr/bin/env python3
"""
Test and Demo Script for Botanical Tree Reconstruction

This script demonstrates the key functionality of the reconstruction toolkit
and can be used to verify that everything is working correctly.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from rich.console import Console
from rich.table import Table

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from tree_reconstruction import __version__
    from tree_reconstruction.config import PipelineConfig, create_example_configs
    from tree_reconstruction.cli import app
    import tree_reconstruction.segmentation as segmentation
    import tree_reconstruction.colmap_pipeline as colmap_pipeline
    import tree_reconstruction.filtering as filtering
    import tree_reconstruction.visualization as visualization
    PACKAGE_IMPORTED = True
except ImportError as e:
    PACKAGE_IMPORTED = False
    IMPORT_ERROR = str(e)

console = Console()

def test_imports():
    """Test that all modules can be imported"""
    console.print("[bold blue]🔍 Testing Package Imports[/bold blue]")
    
    if PACKAGE_IMPORTED:
        console.print("[green]✅ Package imported successfully[/green]")
        console.print(f"[dim]Version: {__version__}[/dim]")
        return True
    else:
        console.print(f"[red]❌ Import failed: {IMPORT_ERROR}[/red]")
        return False

def test_configuration():
    """Test configuration system"""
    console.print("[bold blue]⚙️ Testing Configuration System[/bold blue]")
    
    try:
        # Test default config creation
        config = PipelineConfig()
        default_config = config._get_default_config()
        
        # Test config access
        project_name = config.get("project_name", "default")
        segmentation_conf = config.get("segmentation.confidence", 0.5)
        
        console.print("[green]✅ Configuration system working[/green]")
        console.print(f"[dim]Project: {project_name}, Confidence: {segmentation_conf}[/dim]")
        
        # Test config file creation
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config.save(config_path)
            
            if config_path.exists():
                console.print("[green]✅ Configuration file creation working[/green]")
            else:
                console.print("[red]❌ Configuration file creation failed[/red]")
                return False
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Configuration test failed: {e}[/red]")
        return False

def test_cli_commands():
    """Test CLI command structure"""
    console.print("[bold blue]🖥️ Testing CLI Commands[/bold blue]")
    
    try:
        # Test that CLI app exists and has commands
        commands = []
        for command_name in dir(app):
            if not command_name.startswith('_') and hasattr(getattr(app, command_name), '__call__'):
                commands.append(command_name)
        
        console.print(f"[green]✅ CLI system loaded with {len(app.commands)} commands[/green]")
        
        # List available commands
        table = Table(title="Available CLI Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")
        
        command_descriptions = {
            "segment": "Run semantic segmentation",
            "colmap": "Run COLMAP reconstruction", 
            "filter": "Filter point clouds",
            "visualize": "Create visualizations",
            "pipeline": "Run complete pipeline",
            "status": "Show project status",
            "init": "Initialize new project"
        }
        
        for cmd_name, cmd_obj in app.commands.items():
            description = command_descriptions.get(cmd_name, "Pipeline command")
            table.add_row(cmd_name, description)
        
        console.print(table)
        return True
        
    except Exception as e:
        console.print(f"[red]❌ CLI test failed: {e}[/red]")
        return False

def test_module_functions():
    """Test key functions from each module"""
    console.print("[bold blue]🔧 Testing Module Functions[/bold blue]")
    
    try:
        # Test segmentation module
        seg_model = segmentation.TreeSegmentationModel(device="cpu")
        console.print("[green]✅ Segmentation model can be initialized[/green]")
        
        # Test COLMAP pipeline
        pipeline = colmap_pipeline.COLMAPPipeline()
        console.print("[green]✅ COLMAP pipeline can be initialized[/green]")
        
        # Test filtering
        # This would require a real model, so just test class existence
        filter_class = filtering.PointCloudFilter
        console.print("[green]✅ Point cloud filter class accessible[/green]")
        
        # Test visualization
        vis_class = visualization.ReconstructionVisualizer
        console.print("[green]✅ Visualization class accessible[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Module function test failed: {e}[/red]")
        return False

def create_demo_project():
    """Create a demonstration project structure"""
    console.print("[bold blue]🏗️ Creating Demo Project[/bold blue]")
    
    try:
        demo_dir = Path("demo_project")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
        
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed/masks",
            "data/processed/filtered_images",
            "models/sparse",
            "models/filtered", 
            "reports/figures",
            "configs",
            "logs"
        ]
        
        for dir_path in directories:
            (demo_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create example configurations
        config = PipelineConfig()
        
        # Basic config
        basic_config_path = demo_dir / "configs" / "basic.yaml"
        config.save(basic_config_path)
        
        # Create README
        readme_content = """# Demo Tree Reconstruction Project

This is a demonstration project created by the test script.

## Quick Start

1. Copy your aerial images to `data/raw/`
2. Run the pipeline: `tree-reconstruct pipeline --config configs/basic.yaml`

## Directory Structure

- `data/raw/`: Input aerial images
- `data/processed/`: Processed images and masks
- `models/`: 3D reconstruction models
- `reports/figures/`: Generated visualizations
- `configs/`: Configuration files
- `logs/`: Processing logs

## Example Commands

```bash
# Run complete pipeline
tree-reconstruct pipeline --config configs/basic.yaml

# Run individual stages
tree-reconstruct segment --input-dir data/raw --output-dir data/processed/masks
tree-reconstruct colmap --input-dir data/raw --output-dir models/sparse
tree-reconstruct filter --input-model models/sparse/sparse/0 --output-model models/filtered
tree-reconstruct visualize --input-model models/filtered --output-dir reports/figures

# Check project status
tree-reconstruct status --project .
```
"""
        
        readme_path = demo_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        console.print(f"[green]✅ Demo project created at {demo_dir}[/green]")
        console.print(f"[dim]Configuration: {basic_config_path}[/dim]")
        console.print(f"[dim]Instructions: {readme_path}[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Demo project creation failed: {e}[/red]")
        return False

def run_installation_check():
    """Check if external dependencies are available"""
    console.print("[bold blue]🔍 Checking External Dependencies[/bold blue]")
    
    dependencies = {
        "torch": "PyTorch",
        "cv2": "OpenCV",
        "numpy": "NumPy", 
        "matplotlib": "Matplotlib",
        "yaml": "PyYAML",
        "rich": "Rich",
        "typer": "Typer",
        "PIL": "Pillow"
    }
    
    results = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            results.append((name, "✅ Available", "green"))
        except ImportError:
            results.append((name, "❌ Missing", "red"))
    
    # Check COLMAP
    import subprocess
    try:
        result = subprocess.run(["colmap", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            results.append(("COLMAP", "✅ Available", "green"))
        else:
            results.append(("COLMAP", "❌ Not working", "red"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        results.append(("COLMAP", "❌ Not found", "red"))
    
    # Create results table
    table = Table(title="Dependency Check")
    table.add_column("Dependency", style="cyan")
    table.add_column("Status", style="white")
    
    for name, status, color in results:
        table.add_row(name, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    
    # Check if all critical dependencies are available
    missing_critical = [name for name, status, color in results 
                       if color == "red" and name in ["PyTorch", "OpenCV", "NumPy"]]
    
    if missing_critical:
        console.print(f"[red]❌ Critical dependencies missing: {', '.join(missing_critical)}[/red]")
        return False
    else:
        console.print("[green]✅ All critical dependencies available[/green]")
        return True

def main():
    """Run all tests"""
    console.print("[bold cyan]🌳 Botanical Tree Reconstruction - System Test[/bold cyan]")
    console.print("=" * 60)
    
    tests = [
        ("Package Import", test_imports),
        ("Dependencies", run_installation_check),
        ("Configuration", test_configuration),
        ("CLI Commands", test_cli_commands),
        ("Module Functions", test_module_functions),
        ("Demo Project", create_demo_project)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print()
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with exception: {e}[/red]")
            results.append((test_name, False))
    
    # Summary
    console.print()
    console.print("[bold cyan]📊 Test Summary[/bold cyan]")
    console.print("=" * 25)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "[green]✅ PASS[/green]" if success else "[red]❌ FAIL[/red]"
        console.print(f"{test_name:<20} {status}")
    
    console.print()
    console.print(f"[bold]Results: {passed}/{total} tests passed[/bold]")
    
    if passed == total:
        console.print("[bold green]🎉 All tests passed! The toolkit is ready to use.[/bold green]")
        console.print()
        console.print("Next steps:")
        console.print("1. Copy aerial images to demo_project/data/raw/")
        console.print("2. Run: cd demo_project")
        console.print("3. Run: tree-reconstruct pipeline --config configs/basic.yaml")
    else:
        console.print("[bold red]⚠️ Some tests failed. Check the issues above.[/bold red]")
        if not results[0][1]:  # Package import failed
            console.print("\n[yellow]Install the package first:[/yellow]")
            console.print("pip install -e .")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
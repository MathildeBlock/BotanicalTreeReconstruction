#!/usr/bin/env python3
"""
Workflow for processing the filtered Risø Oak Tree images with the botanical reconstruction toolkit.
This workflow demonstrates the toolkit capabilities without requiring COLMAP installation.
"""

import sys
from pathlib import Path
import time

# Add the source directory to Python path
sys.path.insert(0, '../src')

from tree_reconstruction.config import PipelineConfig
from tree_reconstruction import visualization
from tree_reconstruction.read_write_model import Camera, Image, Point3D
import numpy as np
from PIL import Image as PILImage
from rich.console import Console

console = Console()

def analyze_image_dataset(image_dir: Path):
    """Analyze the image dataset and provide statistics"""
    
    console.print("\n📷 [bold blue]Dataset Analysis[/bold blue]")
    console.print("-" * 50)
    
    image_files = list(image_dir.glob("*.JPG"))
    console.print(f"📸 Total images: {len(image_files)}")
    
    # Analyze a sample of images
    sample_size = min(10, len(image_files))
    sample_images = image_files[:sample_size]
    
    resolutions = []
    file_sizes = []
    
    for img_path in sample_images:
        # Get file size
        file_size = img_path.stat().st_size / (1024 * 1024)  # MB
        file_sizes.append(file_size)
        
        # Get image resolution
        with PILImage.open(img_path) as img:
            width, height = img.size
            resolutions.append((width, height))
    
    # Print statistics
    avg_width = np.mean([r[0] for r in resolutions])
    avg_height = np.mean([r[1] for r in resolutions])
    avg_file_size = np.mean(file_sizes)
    
    console.print(f"📐 Average resolution: {avg_width:.0f} x {avg_height:.0f} pixels")
    console.print(f"💾 Average file size: {avg_file_size:.1f} MB")
    console.print(f"🔍 Sample files analyzed: {sample_size}")
    
    return {
        'total_images': len(image_files),
        'avg_resolution': (avg_width, avg_height),
        'avg_file_size': avg_file_size,
        'image_files': image_files
    }

def create_demo_reconstruction(stats: dict, output_dir: Path):
    """Create a demonstration reconstruction based on image statistics"""
    
    console.print("\n🏗️ [bold blue]Creating Demo Reconstruction[/bold blue]")
    console.print("-" * 50)
    
    # Create realistic camera parameters based on typical drone cameras
    width, height = stats['avg_resolution']
    
    # Typical DJI drone camera parameters (approximate)
    focal_length_px = max(width, height) * 0.8  # Rough estimate
    
    cameras = {}
    cameras[1] = Camera(
        id=1,
        model="PINHOLE",
        width=int(width),
        height=int(height),
        params=np.array([focal_length_px, focal_length_px, width/2, height/2])
    )
    
    # Create images in a circular pattern (typical for aerial photography)
    images = {}
    num_images = min(stats['total_images'], 50)  # Limit for demo
    
    radius = 10.0  # meters
    for i in range(num_images):
        angle = 2 * np.pi * i / num_images
        
        # Position camera in circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 5.0  # 5 meters height
        
        # Point camera towards center
        images[i+1] = Image(
            id=i+1,
            qvec=np.array([1.0, 0.0, 0.0, 0.1 * i]),  # Slight rotation
            tvec=np.array([x, y, z]),
            camera_id=1,
            name=stats['image_files'][i].name if i < len(stats['image_files']) else f"image_{i:04d}.jpg",
            xys=np.random.rand(500, 2) * np.array([width, height]),
            point3D_ids=np.arange(500) + i*500
        )
    
    # Create 3D points representing a tree structure
    points3D = {}
    point_id = 0
    
    # Tree trunk
    for i in range(100):
        height = i * 0.1  # 10 meter tree
        x = np.random.normal(0, 0.2)  # Small variation around center
        y = np.random.normal(0, 0.2)
        z = height
        
        points3D[point_id] = Point3D(
            id=point_id,
            xyz=np.array([x, y, z]),
            rgb=np.array([101, 67, 33]),  # Brown for trunk
            error=np.random.rand() * 0.5,
            image_ids=np.array([1, 2, 3]),
            point2D_idxs=np.array([0, 0, 0])
        )
        point_id += 1
    
    # Tree canopy
    for i in range(2000):
        # Spherical canopy
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(2, 6)  # Canopy radius
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = 7 + r * np.cos(phi)  # Canopy center at 7m height
        
        points3D[point_id] = Point3D(
            id=point_id,
            xyz=np.array([x, y, z]),
            rgb=np.array([34, 139, 34]),  # Green for leaves
            error=np.random.rand() * 1.0,
            image_ids=np.array([1, 2, 3]),
            point2D_idxs=np.array([0, 0, 0])
        )
        point_id += 1
    
    # Save the reconstruction
    from tree_reconstruction.read_write_model import write_model
    write_model(cameras, images, points3D, output_dir, ext=".bin")
    
    console.print(f"✅ Demo reconstruction created with:")
    console.print(f"   - {len(cameras)} camera(s)")
    console.print(f"   - {len(images)} image(s)")
    console.print(f"   - {len(points3D)} 3D point(s)")
    
    return cameras, images, points3D

def create_comprehensive_visualizations(model_path: Path, image_dir: Path, output_dir: Path):
    """Create comprehensive visualizations of the reconstruction"""
    
    console.print("\n📊 [bold blue]Creating Visualizations[/bold blue]")
    console.print("-" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = visualization.ReconstructionVisualizer(
        model_path=model_path,
        image_dir=image_dir
    )
    
    # Create different types of visualizations
    visualizations = [
        ("pointcloud_rgb", "3D Point Cloud (RGB)", "plot_3d_point_cloud"),
        ("pointcloud_height", "3D Point Cloud (Height)", "plot_3d_point_cloud"),
        ("camera_poses", "Camera Poses", "plot_camera_poses"),
        ("statistics", "Reconstruction Statistics", "create_statistics_plot")
    ]
    
    for viz_name, viz_desc, viz_method in visualizations:
        try:
            console.print(f"🎨 Creating {viz_desc}...")
            output_path = output_dir / f"{viz_name}.png"
            
            method = getattr(visualizer, viz_method)
            
            if viz_name == "pointcloud_height":
                method(output_path, color_by="height", max_points=5000)
            elif viz_name == "pointcloud_rgb":
                method(output_path, color_by="rgb", max_points=5000)
            else:
                method(output_path)
                
            console.print(f"✅ {viz_desc} saved to {output_path}")
            
        except Exception as e:
            console.print(f"❌ Failed to create {viz_desc}: {e}")

def main():
    """Main workflow for processing the filtered Risø Oak Tree images"""
    
    console.print("[bold green]🌳 Botanical Tree Reconstruction - Demo Workflow[/bold green]")
    console.print("=" * 70)
    
    # Setup paths
    image_dir = Path("data/raw")
    model_dir = Path("models/demo_tree")
    output_dir = Path("reports/demo_analysis")
    
    if not image_dir.exists():
        console.print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Step 1: Analyze dataset
    stats = analyze_image_dataset(image_dir)
    
    # Step 2: Create demonstration reconstruction
    model_dir.mkdir(parents=True, exist_ok=True)
    create_demo_reconstruction(stats, model_dir)
    
    # Step 3: Create visualizations
    create_comprehensive_visualizations(model_dir, image_dir, output_dir)
    
    # Step 4: Summary
    console.print("\n🎉 [bold green]Demo Workflow Complete![/bold green]")
    console.print("=" * 70)
    console.print(f"📁 Results saved in:")
    console.print(f"   - Model: {model_dir.absolute()}")
    console.print(f"   - Visualizations: {output_dir.absolute()}")
    console.print("\n📝 Next steps:")
    console.print("   1. Install COLMAP for real 3D reconstruction")
    console.print("   2. Use the full pipeline with: python ../examples/basic_workflow.py")
    console.print("   3. Experiment with different filtering parameters")

if __name__ == "__main__":
    main()
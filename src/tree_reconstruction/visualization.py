#!/usr/bin/env python3
"""
Visualization Module for Tree Reconstruction

This module provides comprehensive visualization tools for COLMAP reconstructions,
including 3D point clouds, projected points, comparisons, and quality assessments.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from rich.console import Console
from rich.progress import track

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    
from .read_write_model import read_model

console = Console()

class ReconstructionVisualizer:
    """Main visualization class for COLMAP reconstructions"""
    
    def __init__(self, model_path: Path, image_dir: Optional[Path] = None):
        """Initialize visualizer with COLMAP model"""
        self.model_path = model_path
        self.image_dir = image_dir
        self.cameras, self.images, self.points3D = self._load_model()
        
    def _load_model(self):
        """Load COLMAP model"""
        try:
            cameras, images, points3D = read_model(self.model_path, ext=".bin")
        except:
            cameras, images, points3D = read_model(self.model_path, ext=".txt")
        
        console.print(f"Loaded model: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")
        return cameras, images, points3D
    
    def plot_3d_point_cloud(self, output_path: Path, 
                           color_by: str = "rgb",
                           point_size: float = 1.0,
                           max_points: int = 50000) -> None:
        """
        Create 3D point cloud visualization
        
        Args:
            output_path: Path to save the plot
            color_by: How to color points ('rgb', 'height', 'error', 'uniform')
            point_size: Size of points in plot
            max_points: Maximum number of points to plot (for performance)
        """
        console.print("[blue]📊 Creating 3D point cloud visualization...[/blue]")
        
        if not self.points3D:
            console.print("[red]No 3D points to visualize[/red]")
            return
        
        # Extract point data
        points = []
        colors = []
        errors = []
        
        for point in list(self.points3D.values())[:max_points]:
            points.append(point.xyz)
            colors.append(point.rgb / 255.0)  # Normalize RGB
            errors.append(point.error)
        
        points = np.array(points)
        colors = np.array(colors)
        errors = np.array(errors)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color points based on selection
        if color_by == "rgb" and colors.size > 0:
            scatter_colors = colors
        elif color_by == "height":
            scatter_colors = points[:, 2]  # Z coordinate
        elif color_by == "error":
            scatter_colors = errors
        else:
            scatter_colors = 'blue'
        
        # Create scatter plot
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=scatter_colors, s=point_size, alpha=0.6)
        
        # Add colorbar if using scalar coloring
        if color_by in ["height", "error"]:
            plt.colorbar(scatter, label=color_by.capitalize())
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'3D Point Cloud ({len(points)} points)\nColored by {color_by}')
        
        # Equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ 3D plot saved to {output_path}[/green]")
    
    def plot_projected_points(self, output_path: Path, 
                             num_images: int = 6,
                             point_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                             point_size: float = 8) -> None:
        """
        Create visualization of projected 3D points on images
        
        Args:
            output_path: Path to save the plot
            num_images: Number of images to show
            point_color: RGB color for points
            point_size: Size of points
        """
        console.print("[blue]📸 Creating projected points visualization...[/blue]")
        
        if not self.image_dir:
            console.print("[red]No image directory provided[/red]")
            return
        
        # Find images with points
        images_with_points = [img for img in self.images.values() 
                             if any(pid != -1 for pid in img.point3D_ids)]
        
        if not images_with_points:
            console.print("[red]No images with 3D points found[/red]")
            return
        
        # Select images to visualize
        selected_images = images_with_points[:num_images]
        
        # Calculate grid layout
        cols = min(3, len(selected_images))
        rows = (len(selected_images) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, image_obj in enumerate(selected_images):
            ax = axes[i]
            
            # Load image
            img_path = self.image_dir / image_obj.name
            if not img_path.exists():
                console.print(f"[yellow]Warning: Image not found: {img_path}[/yellow]")
                continue
            
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get camera info
            camera = self.cameras[image_obj.camera_id]
            cam_w, cam_h = camera.width, camera.height
            orig_h, orig_w = img.shape[:2]
            
            # Extract projected points
            points_2d = []
            for kp, pid in zip(image_obj.xys, image_obj.point3D_ids):
                if pid == -1:
                    continue
                
                # Scale to original image size
                u = kp[0] * (orig_w / cam_w)
                v = kp[1] * (orig_h / cam_h)
                points_2d.append([u, v])
            
            if points_2d:
                points_2d = np.array(points_2d)
                
                # Display image and points
                ax.imshow(img)
                ax.scatter(points_2d[:, 0], points_2d[:, 1], 
                          c=[point_color], s=point_size, alpha=0.8)
                ax.set_title(f"{image_obj.name}\n({len(points_2d)} points)")
            else:
                ax.imshow(img)
                ax.set_title(f"{image_obj.name}\n(0 points)")
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(selected_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Projected points plot saved to {output_path}[/green]")
    
    def plot_camera_poses(self, output_path: Path, scale: float = 0.1) -> None:
        """
        Visualize camera poses in 3D space
        
        Args:
            output_path: Path to save the plot
            scale: Scale factor for camera visualization
        """
        console.print("[blue]📷 Creating camera pose visualization...[/blue]")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract camera positions and orientations
        positions = []
        for image in self.images.values():
            # Convert quaternion and translation to camera position
            qvec = image.qvec
            tvec = image.tvec
            
            # Convert quaternion to rotation matrix
            R = qvec_to_rotmat(qvec)
            
            # Camera position is -R^T * t
            camera_pos = -R.T @ tvec
            positions.append(camera_pos)
        
        positions = np.array(positions)
        
        # Plot camera positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', s=50, alpha=0.8, label='Cameras')
        
        # Plot 3D points
        if self.points3D:
            points = np.array([point.xyz for point in self.points3D.values()])
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='blue', s=1, alpha=0.3, label='3D Points')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Poses and Point Cloud ({len(positions)} cameras)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Camera pose plot saved to {output_path}[/green]")
    
    def create_statistics_plot(self, output_path: Path) -> None:
        """Create comprehensive statistics visualization"""
        console.print("[blue]📈 Creating statistics visualization...[/blue]")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Point count per image
        point_counts = [len([p for p in img.point3D_ids if p != -1]) 
                       for img in self.images.values()]
        axes[0, 0].hist(point_counts, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Points per Image')
        axes[0, 0].set_xlabel('Number of Points')
        axes[0, 0].set_ylabel('Number of Images')
        
        # 2. Reprojection errors
        errors = [point.error for point in self.points3D.values()]
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Reprojection Errors')
        axes[0, 1].set_xlabel('Error (pixels)')
        axes[0, 1].set_ylabel('Number of Points')
        
        # 3. Point observation counts
        obs_counts = [len(point.image_ids) for point in self.points3D.values()]
        axes[0, 2].hist(obs_counts, bins=30, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Observations per Point')
        axes[0, 2].set_xlabel('Number of Observations')
        axes[0, 2].set_ylabel('Number of Points')
        
        # 4. 3D point distribution (XY)
        points = np.array([point.xyz for point in self.points3D.values()])
        if len(points) > 0:
            axes[1, 0].scatter(points[:, 0], points[:, 1], alpha=0.5, s=1)
            axes[1, 0].set_title('Point Distribution (Top View)')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            axes[1, 0].set_aspect('equal')
        
        # 5. Height distribution
        if len(points) > 0:
            axes[1, 1].hist(points[:, 2], bins=30, alpha=0.7, color='gold')
            axes[1, 1].set_title('Height Distribution')
            axes[1, 1].set_xlabel('Z coordinate')
            axes[1, 1].set_ylabel('Number of Points')
        
        # 6. Summary statistics table
        axes[1, 2].axis('off')
        stats_text = f"""
        RECONSTRUCTION STATISTICS
        
        Cameras: {len(self.cameras)}
        Images: {len(self.images)}
        3D Points: {len(self.points3D)}
        
        Avg Points/Image: {np.mean(point_counts):.1f}
        Avg Observations/Point: {np.mean(obs_counts):.1f}
        Avg Reprojection Error: {np.mean(errors):.2f}px
        
        Point Cloud Bounds:
        X: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]
        Y: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]
        Z: [{np.min(points[:, 2]):.2f}, {np.max(points[:, 2]):.2f}]
        """
        axes[1, 2].text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Statistics plot saved to {output_path}[/green]")

def compare_models(model1_path: Path, model2_path: Path, 
                  output_path: Path, 
                  image_dir: Optional[Path] = None,
                  model1_name: str = "Model 1",
                  model2_name: str = "Model 2") -> None:
    """
    Create comparison visualization between two COLMAP models
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        output_path: Path to save comparison plot
        image_dir: Directory containing images
        model1_name: Name for first model
        model2_name: Name for second model
    """
    console.print(f"[blue]⚖️ Comparing models: {model1_name} vs {model2_name}[/blue]")
    
    # Load both models
    vis1 = ReconstructionVisualizer(model1_path, image_dir)
    vis2 = ReconstructionVisualizer(model2_path, image_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Model comparison statistics
    stats1 = {
        'cameras': len(vis1.cameras),
        'images': len(vis1.images),
        'points': len(vis1.points3D),
        'avg_error': np.mean([p.error for p in vis1.points3D.values()]) if vis1.points3D else 0
    }
    
    stats2 = {
        'cameras': len(vis2.cameras),
        'images': len(vis2.images), 
        'points': len(vis2.points3D),
        'avg_error': np.mean([p.error for p in vis2.points3D.values()]) if vis2.points3D else 0
    }
    
    # 1. Point count comparison
    categories = ['Cameras', 'Images', '3D Points']
    model1_values = [stats1['cameras'], stats1['images'], stats1['points']]
    model2_values = [stats2['cameras'], stats2['images'], stats2['points']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, model1_values, width, label=model1_name, alpha=0.8)
    axes[0, 0].bar(x + width/2, model2_values, width, label=model2_name, alpha=0.8)
    axes[0, 0].set_title('Model Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()
    
    # 2. 3D point clouds side by side
    if vis1.points3D:
        points1 = np.array([p.xyz for p in vis1.points3D.values()])
        axes[0, 1].scatter(points1[:, 0], points1[:, 1], alpha=0.5, s=1, label=model1_name)
        
    if vis2.points3D:
        points2 = np.array([p.xyz for p in vis2.points3D.values()])
        axes[0, 1].scatter(points2[:, 0], points2[:, 1], alpha=0.5, s=1, label=model2_name)
    
    axes[0, 1].set_title('Point Clouds (Top View)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')
    
    # 3. Error comparison
    if vis1.points3D and vis2.points3D:
        errors1 = [p.error for p in vis1.points3D.values()]
        errors2 = [p.error for p in vis2.points3D.values()]
        
        axes[0, 2].hist(errors1, bins=30, alpha=0.5, label=model1_name, density=True)
        axes[0, 2].hist(errors2, bins=30, alpha=0.5, label=model2_name, density=True)
        axes[0, 2].set_title('Reprojection Error Distribution')
        axes[0, 2].set_xlabel('Error (pixels)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
    
    # 4-6. Additional comparison plots...
    # Hide remaining plots for now
    for i in range(1, 2):
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✅ Model comparison saved to {output_path}[/green]")

def create_visualizations(input_model: Path,
                         output_dir: Path,
                         image_dir: Optional[Path] = None,
                         plot_type: str = "all",
                         num_images: int = 6) -> None:
    """
    Create comprehensive visualizations
    
    Args:
        input_model: Path to COLMAP model
        output_dir: Directory to save visualizations
        image_dir: Directory containing images
        plot_type: Type of plots to create
        num_images: Number of images for multi-image plots
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = ReconstructionVisualizer(input_model, image_dir)
    
    if plot_type in ["all", "3d"]:
        visualizer.plot_3d_point_cloud(output_dir / "point_cloud_3d.png")
    
    if plot_type in ["all", "projected"] and image_dir:
        visualizer.plot_projected_points(output_dir / "projected_points.png", num_images)
    
    if plot_type in ["all", "poses"]:
        visualizer.plot_camera_poses(output_dir / "camera_poses.png")
    
    if plot_type in ["all", "stats"]:
        visualizer.create_statistics_plot(output_dir / "statistics.png")
    
    console.print(f"[green]✅ All visualizations saved to {output_dir}[/green]")

def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def main():
    """Main function for standalone usage"""
    import typer
    typer.run(create_visualizations)

if __name__ == "__main__":
    main()
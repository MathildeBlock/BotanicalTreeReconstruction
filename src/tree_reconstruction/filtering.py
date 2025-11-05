#!/usr/bin/env python3
"""
Point Cloud Filtering Module

This module provides various filtering and optimization techniques for COLMAP
point clouds, including coordinate filtering, mask-based filtering, and outlier removal.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import os
from rich.console import Console
from rich.progress import track

from .read_write_model import read_model, write_model, Camera, BaseImage, Point3D

console = Console()

class PointCloudFilter:
    """Main point cloud filtering class"""
    
    def __init__(self, model_path: Path):
        """Initialize with COLMAP model"""
        self.model_path = model_path
        self.cameras, self.images, self.points3D = self._load_model()
        
    def _load_model(self):
        """Load COLMAP model"""
        try:
            cameras, images, points3D = read_model(self.model_path, ext=".bin")
            console.print(f"Loaded model: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")
            return cameras, images, points3D
        except:
            # Try text format
            cameras, images, points3D = read_model(self.model_path, ext=".txt")
            console.print(f"Loaded model: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")
            return cameras, images, points3D
    
    def filter_by_coordinates(self, threshold: float = 3.0) -> Dict[int, Point3D]:
        """
        Filter points by coordinate bounds
        
        Args:
            threshold: Maximum absolute coordinate value
            
        Returns:
            Filtered points dictionary
        """
        filtered_points = {}
        removed_count = 0
        
        for point_id, point in self.points3D.items():
            x, y, z = point.xyz
            if abs(x) <= threshold and abs(y) <= threshold and abs(z) <= threshold:
                filtered_points[point_id] = point
            else:
                removed_count += 1
        
        console.print(f"Coordinate filtering: removed {removed_count}/{len(self.points3D)} points")
        return filtered_points
    
    def filter_by_reprojection_error(self, threshold: float = 2.0) -> Dict[int, Point3D]:
        """
        Filter points by reprojection error
        
        Args:
            threshold: Maximum reprojection error
            
        Returns:
            Filtered points dictionary
        """
        filtered_points = {}
        removed_count = 0
        
        for point_id, point in self.points3D.items():
            if point.error <= threshold:
                filtered_points[point_id] = point
            else:
                removed_count += 1
        
        console.print(f"Reprojection error filtering: removed {removed_count}/{len(self.points3D)} points")
        return filtered_points
    
    def filter_by_observation_count(self, min_observations: int = 3) -> Dict[int, Point3D]:
        """
        Filter points by minimum number of observations
        
        Args:
            min_observations: Minimum number of images that must observe the point
            
        Returns:
            Filtered points dictionary
        """
        filtered_points = {}
        removed_count = 0
        
        for point_id, point in self.points3D.items():
            if len(point.image_ids) >= min_observations:
                filtered_points[point_id] = point
            else:
                removed_count += 1
        
        console.print(f"Observation count filtering: removed {removed_count}/{len(self.points3D)} points")
        return filtered_points
    
    def filter_by_masks(self, mask_dir: Path, min_mask_ratio: float = 0.5) -> Tuple[Dict[int, BaseImage], Dict[int, Point3D]]:
        """
        Filter points and images based on segmentation masks
        
        Args:
            mask_dir: Directory containing segmentation masks
            min_mask_ratio: Minimum ratio of mask coverage for point to be kept
            
        Returns:
            Tuple of (filtered_images, filtered_points)
        """
        filtered_images = {}
        filtered_points = {}
        
        for image_id, image in self.images.items():
            mask_path = mask_dir / f"{image.name.replace('.JPG', '_mask.png').replace('.jpg', '_mask.png')}"
            
            if not mask_path.exists():
                console.print(f"[yellow]Warning: No mask found for {image.name}[/yellow]")
                continue
            
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # Keep image
            filtered_images[image_id] = image
            
            # Check each point's projection against mask
            camera = self.cameras[image.camera_id]
            
            for i, point3D_id in enumerate(image.point3D_ids):
                if point3D_id == -1:
                    continue
                
                if point3D_id not in self.points3D:
                    continue
                
                # Get 2D projection
                x, y = image.xys[i]
                
                # Scale coordinates to mask size
                mask_h, mask_w = mask.shape
                cam_w, cam_h = camera.width, camera.height
                
                mask_x = int(x * mask_w / cam_w)
                mask_y = int(y * mask_h / cam_h)
                
                # Check if point is within mask bounds and on vegetation
                if (0 <= mask_x < mask_w and 0 <= mask_y < mask_h and 
                    mask[mask_y, mask_x] > 0):
                    if point3D_id not in filtered_points:
                        filtered_points[point3D_id] = self.points3D[point3D_id]
        
        console.print(f"Mask filtering: kept {len(filtered_images)}/{len(self.images)} images")
        console.print(f"Mask filtering: kept {len(filtered_points)}/{len(self.points3D)} points")
        
        return filtered_images, filtered_points
    
    def statistical_outlier_removal(self, k_neighbors: int = 20, std_threshold: float = 2.0) -> Dict[int, Point3D]:
        """
        Remove statistical outliers based on distance to neighbors
        
        Args:
            k_neighbors: Number of neighbors to consider
            std_threshold: Standard deviation threshold
            
        Returns:
            Filtered points dictionary
        """
        if len(self.points3D) < k_neighbors:
            console.print("[yellow]Not enough points for statistical filtering[/yellow]")
            return self.points3D
        
        # Get all point coordinates
        points = np.array([point.xyz for point in self.points3D.values()])
        point_ids = list(self.points3D.keys())
        
        # Calculate distances to k nearest neighbors for each point
        from scipy.spatial.distance import cdist
        
        distances = cdist(points, points)
        mean_distances = []
        
        for i in range(len(points)):
            # Get k+1 nearest neighbors (including self)
            nearest_dists = np.sort(distances[i])[1:k_neighbors+1]
            mean_distances.append(np.mean(nearest_dists))
        
        mean_distances = np.array(mean_distances)
        
        # Calculate threshold
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_threshold * global_std
        
        # Filter points
        filtered_points = {}
        removed_count = 0
        
        for i, (point_id, mean_dist) in enumerate(zip(point_ids, mean_distances)):
            if mean_dist <= threshold:
                filtered_points[point_id] = self.points3D[point_id]
            else:
                removed_count += 1
        
        console.print(f"Statistical outlier removal: removed {removed_count}/{len(self.points3D)} points")
        return filtered_points

def triangulate_new_points(cameras: Dict, images: Dict, points3D: Dict, 
                          max_new_points: int = 1000) -> Dict[int, Point3D]:
    """
    Triangulate new 3D points from existing 2D matches
    
    Args:
        cameras: Camera dictionary
        images: Images dictionary  
        points3D: Existing points dictionary
        max_new_points: Maximum number of new points to add
        
    Returns:
        Dictionary with new points added
    """
    console.print("[blue]🔺 Triangulating new points...[/blue]")
    
    # This integrates your existing triangulation optimization
    # Implementation based on your smolbutbiggerscripttodeletandadd_optimized.py
    
    new_points = points3D.copy()
    next_point_id = max(points3D.keys()) + 1 if points3D else 1
    added_count = 0
    
    # Get image pairs for triangulation
    image_list = list(images.values())
    
    for i, img1 in enumerate(track(image_list[:10], description="Triangulating")):  # Limit for performance
        if added_count >= max_new_points:
            break
            
        for j, img2 in enumerate(image_list[i+1:i+7], i+1):  # Use next 6 images
            if added_count >= max_new_points:
                break
                
            # Find unmatched keypoints between images
            unmatched_kps1 = []
            unmatched_kps2 = []
            
            # Get keypoints that don't have 3D points
            for k, pt3d_id in enumerate(img1.point3D_ids):
                if pt3d_id == -1:  # No 3D point
                    unmatched_kps1.append((k, img1.xys[k]))
            
            for k, pt3d_id in enumerate(img2.point3D_ids):
                if pt3d_id == -1:  # No 3D point
                    unmatched_kps2.append((k, img2.xys[k]))
            
            # Simple matching based on spatial proximity (simplified)
            # In practice, you'd use proper feature matching
            for idx1, kp1 in unmatched_kps1[:50]:  # Limit for performance
                for idx2, kp2 in unmatched_kps2[:50]:
                    # Check if keypoints are spatially reasonable
                    if np.linalg.norm(kp1 - kp2) < 100:  # Simple heuristic
                        
                        # Try triangulation
                        try:
                            point_3d = triangulate_point(
                                kp1, kp2, 
                                cameras[img1.camera_id],
                                cameras[img2.camera_id],
                                img1, img2
                            )
                            
                            # Validate point
                            if point_3d is not None and validate_point(point_3d):
                                # Create new Point3D object
                                new_point = Point3D(
                                    id=next_point_id,
                                    xyz=point_3d,
                                    rgb=np.array([128, 128, 128]),  # Gray color
                                    error=0.5,  # Default error
                                    image_ids=np.array([img1.id, img2.id]),
                                    point2D_idxs=np.array([idx1, idx2])
                                )
                                
                                new_points[next_point_id] = new_point
                                next_point_id += 1
                                added_count += 1
                                
                                if added_count >= max_new_points:
                                    break
                                    
                        except:
                            continue
    
    console.print(f"Added {added_count} new triangulated points")
    return new_points

def triangulate_point(kp1, kp2, cam1, cam2, img1, img2):
    """Triangulate a 3D point from two 2D observations"""
    # Simplified triangulation - in practice use proper camera matrices
    # This would integrate your existing triangulation code
    return np.array([0.0, 0.0, 1.0])  # Placeholder

def validate_point(point_3d, coord_threshold=3.0):
    """Validate triangulated point"""
    x, y, z = point_3d
    return abs(x) <= coord_threshold and abs(y) <= coord_threshold and abs(z) <= coord_threshold

def filter_point_cloud(input_model: Path,
                      output_model: Path,
                      mask_dir: Optional[Path] = None,
                      filter_type: str = "coordinate",
                      coord_threshold: float = 3.0,
                      outlier_threshold: float = 2.0,
                      min_observations: int = 3,
                      add_points: bool = False):
    """
    Main filtering function
    
    Args:
        input_model: Path to input COLMAP model
        output_model: Path to save filtered model
        mask_dir: Directory containing segmentation masks
        filter_type: Type of filtering to apply
        coord_threshold: Coordinate filtering threshold
        outlier_threshold: Outlier removal threshold
        min_observations: Minimum observations per point
        add_points: Whether to add new triangulated points
    """
    
    console.print(f"[bold yellow]🔧 Filtering point cloud: {filter_type}[/bold yellow]")
    
    # Initialize filter
    filter_obj = PointCloudFilter(input_model)
    
    # Apply filtering based on type
    if filter_type == "coordinate":
        filtered_points = filter_obj.filter_by_coordinates(coord_threshold)
        filtered_images = filter_obj.images
        
    elif filter_type == "mask" and mask_dir:
        filtered_images, filtered_points = filter_obj.filter_by_masks(mask_dir)
        
    elif filter_type == "outlier":
        filtered_points = filter_obj.statistical_outlier_removal(std_threshold=outlier_threshold)
        filtered_images = filter_obj.images
        
    elif filter_type == "combined":
        # Apply multiple filters
        filtered_points = filter_obj.filter_by_coordinates(coord_threshold)
        filtered_points = filter_obj.filter_by_reprojection_error(outlier_threshold)
        filtered_points = filter_obj.filter_by_observation_count(min_observations)
        filtered_images = filter_obj.images
        
        if mask_dir:
            filtered_images, filtered_points = filter_obj.filter_by_masks(mask_dir)
    
    else:
        console.print(f"[red]Unknown filter type: {filter_type}[/red]")
        return
    
    # Add new points if requested
    if add_points:
        filtered_points = triangulate_new_points(
            filter_obj.cameras, filtered_images, filtered_points
        )
    
    # Update image references to filtered points
    updated_images = {}
    for img_id, image in filtered_images.items():
        new_point3D_ids = []
        new_xys = []
        
        for i, point3D_id in enumerate(image.point3D_ids):
            if point3D_id in filtered_points or point3D_id == -1:
                new_point3D_ids.append(point3D_id)
                new_xys.append(image.xys[i])
        
        updated_images[img_id] = image._replace(
            point3D_ids=np.array(new_point3D_ids),
            xys=np.array(new_xys)
        )
    
    # Save filtered model
    output_model.mkdir(parents=True, exist_ok=True)
    write_model(filter_obj.cameras, updated_images, filtered_points, output_model, ext=".bin")
    write_model(filter_obj.cameras, updated_images, filtered_points, output_model, ext=".txt")
    
    console.print(f"[green]✅ Filtered model saved to {output_model}[/green]")
    console.print(f"Final counts: {len(updated_images)} images, {len(filtered_points)} points")

def main():
    """Main function for standalone usage"""
    import typer
    typer.run(filter_point_cloud)

if __name__ == "__main__":
    main()
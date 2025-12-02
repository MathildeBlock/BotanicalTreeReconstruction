#!/usr/bin/env python3
"""
Visualize only the new points added by ray enhancement
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

# Add the parent directory to the path to import read_write_model
sys.path.append(str(Path(__file__).parent))
from read_write_model import read_model, Camera, Image, Point3D

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize only new ray-generated points')
    parser.add_argument('--images', type=str, required=True, help='Path to original images directory')
    parser.add_argument('--masks', type=str, help='Path to segmentation masks directory (optional)')
    parser.add_argument('--filtered_model', type=str, required=True, help='Path to filtered COLMAP model directory')
    parser.add_argument('--ray_model', type=str, required=True, help='Path to ray-enhanced COLMAP model directory')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--n_images', type=int, default=3, help='Number of images to sample for visualization')
    parser.add_argument('--point-size', type=float, default=1.0, help='Size of projected points', dest='point_size')
    parser.add_argument('--mask_type', type=str, choices=['rough', 'fine', 'both'], default='both', 
                       help='Type of masks to use')
    
    return parser.parse_args()

def project_points(points_3d, camera, image):
    """Project 3D points onto image plane"""
    if len(points_3d) == 0:
        return np.array([])
    
    # Get camera parameters
    if camera.model == 'SIMPLE_RADIAL':
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
        k1 = camera.params[3]  # Radial distortion parameter
    elif camera.model == 'SIMPLE_PINHOLE':
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
        k1 = 0.0  # No distortion
    elif camera.model == 'PINHOLE':
        fx, fy = camera.params[0], camera.params[1]
        cx, cy = camera.params[2], camera.params[3]
        k1 = 0.0  # No distortion
    else:
        print(f"Warning: Unsupported camera model {camera.model}")
        fx = fy = 1000  # Default values
        cx, cy = camera.width / 2, camera.height / 2
        k1 = 0.0  # No distortion
    
    # Get rotation and translation
    qvec = image.qvec
    tvec = image.tvec
    
    # Convert quaternion to rotation matrix
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Transform points to camera coordinate system
    points_cam = R @ points_3d.T + tvec.reshape(-1, 1)
    
    # Only keep points in front of camera
    valid_depth = points_cam[2, :] > 0
    points_cam = points_cam[:, valid_depth]
    
    if points_cam.shape[1] == 0:
        return np.array([])
    
    # Project to image plane
    x_proj = points_cam[0, :] / points_cam[2, :]
    y_proj = points_cam[1, :] / points_cam[2, :]
    
    # Apply radial distortion if present
    if abs(k1) > 1e-10:
        r2 = x_proj**2 + y_proj**2
        distortion_factor = 1 + k1 * r2
        x_proj *= distortion_factor
        y_proj *= distortion_factor
    
    # Convert to pixel coordinates
    u = fx * x_proj + cx
    v = fy * y_proj + cy
    
    points_2d = np.column_stack([u, v])
    
    # Filter points within image bounds
    valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.width)
    valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.height)
    valid_points = points_2d[valid_x & valid_y]
    
    return valid_points

def find_mask_file(masks_dir, image_name, mask_type):
    """Find corresponding mask file for an image"""
    if not masks_dir:
        return None
        
    base_name = Path(image_name).stem
    mask_files = []
    
    if mask_type in ['rough', 'both']:
        rough_mask = Path(masks_dir) / f"{base_name}_rough.png"
        if rough_mask.exists():
            mask_files.append(rough_mask)
    
    if mask_type in ['fine', 'both']:
        fine_mask = Path(masks_dir) / f"{base_name}_fine.png"
        if fine_mask.exists():
            mask_files.append(fine_mask)
    
    return mask_files[0] if mask_files else None

def extract_new_ray_points(filtered_model_path, ray_model_path):
    """Extract only the new points added by ray enhancement"""
    print("Loading models to extract new ray points...")
    
    # Load both models
    cameras_filt, images_filt, points_filt = read_model(filtered_model_path)
    cameras_ray, images_ray, points_ray = read_model(ray_model_path)
    
    # Get point IDs that are only in ray model (new points)
    filtered_point_ids = set(points_filt.keys())
    ray_point_ids = set(points_ray.keys())
    new_ray_point_ids = ray_point_ids - filtered_point_ids
    
    # Extract coordinates of new ray points
    new_ray_points = np.array([points_ray[pid].xyz for pid in new_ray_point_ids])
    
    print(f"Filtered model: {len(points_filt)} points")
    print(f"Ray model: {len(points_ray)} points")
    print(f"New ray points: {len(new_ray_points)} points")
    
    return cameras_ray, images_ray, new_ray_points

def create_ray_only_visualization(args):
    """Create visualization showing only new ray-generated points"""
    
    # Extract new ray points
    cameras_ray, images_ray, new_ray_points = extract_new_ray_points(args.filtered_model, args.ray_model)
    
    # Get available images
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Error: Images directory {image_dir} not found")
        return
    
    # Find images that exist in ray model
    available_images = []
    for img_path in image_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img_name = img_path.name
            
            # Check if image exists in ray model
            found_in_ray = any(img_data.name == img_name for img_data in images_ray.values())
            
            if found_in_ray:
                available_images.append(img_name)
    
    if not available_images:
        print("Error: No common images found")
        return
    
    # Sample images for visualization
    sample_images = random.sample(available_images, min(args.n_images, len(available_images)))
    
    # Create visualization with 3 columns: original, mask, ray_points_only
    n_cols = 3
    n_rows = len(sample_images)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['Original Image', 'Segmentation Mask', 'New Ray Points Only']
    
    # Set column titles
    for col, title in enumerate(column_titles):
        if n_rows > 0:
            axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    for row, image_name in enumerate(sample_images):
        # Load original image
        image_path = image_dir / image_name
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Column 1: Original image
        axes[row, 0].imshow(img)
        axes[row, 0].set_xlim(0, img.shape[1])
        axes[row, 0].set_ylim(img.shape[0], 0)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        
        # Column 2: Segmentation mask
        if args.masks:
            mask_file = find_mask_file(args.masks, image_name, args.mask_type)
            if mask_file:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Create colored mask overlay
                    mask_colored = np.zeros_like(img)
                    mask_colored[mask > 3] = [0, 255, 0]  # Green for tree areas
                    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
                    axes[row, 1].imshow(overlay)
                else:
                    axes[row, 1].imshow(img)
            else:
                axes[row, 1].imshow(img)
        else:
            axes[row, 1].imshow(img)
        
        axes[row, 1].set_xlim(0, img.shape[1])
        axes[row, 1].set_ylim(img.shape[0], 0)
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        
        # Column 3: New ray points only
        axes[row, 2].imshow(img)
        
        # Get camera and image data for ray model
        camera_ray = list(cameras_ray.values())[0] if cameras_ray else None
        image_data_ray = None
        
        if cameras_ray and images_ray:
            for img_data in images_ray.values():
                if img_data.name == image_name:
                    image_data_ray = img_data
                    break
        
        if camera_ray and image_data_ray and len(new_ray_points) > 0:
            points_2d = project_points(new_ray_points, camera_ray, image_data_ray)
            if len(points_2d) > 0:
                axes[row, 2].scatter(points_2d[:, 0], points_2d[:, 1], 
                                   c='red', s=args.point_size, alpha=0.8, edgecolors='white', linewidth=0.1)
                axes[row, 2].set_xlabel(f'{len(points_2d)} new ray points projected')
            else:
                axes[row, 2].set_xlabel('No new ray points visible')
        else:
            axes[row, 2].set_xlabel('No ray model data')
        
        axes[row, 2].set_xlim(0, img.shape[1])
        axes[row, 2].set_ylim(img.shape[0], 0)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
    
    plt.tight_layout()
    
    # Save visualization
    print(f"Saving new ray points visualization to {args.output}")
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Ray-only visualization complete!")

def main():
    args = parse_args()
    create_ray_only_visualization(args)

if __name__ == '__main__':
    main()
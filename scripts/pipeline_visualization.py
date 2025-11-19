#!/usr/bin/env python3
"""
pipeline_visualization.py - Comprehensive Pipeline Visualization

This script creates a comprehensive visualization showing the complete botanical tree
reconstruction pipeline: original images, segmentation masks, projected points from
original COLMAP model, projected points from filtered COLMAP model, and ray-added points.
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
    parser = argparse.ArgumentParser(description='Create comprehensive pipeline visualization')
    parser.add_argument('--images', type=str, required=True, help='Path to original images directory')
    parser.add_argument('--masks', type=str, help='Path to segmentation masks directory (optional)')
    parser.add_argument('--original_model', type=str, required=True, help='Path to original COLMAP model directory')
    parser.add_argument('--filtered_model', type=str, required=True, help='Path to filtered COLMAP model directory')
    parser.add_argument('--ray_model', type=str, help='Path to ray-enhanced COLMAP model directory')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--n_images', type=int, default=3, help='Number of images to sample for visualization')
    parser.add_argument('--point_size', type=float, default=1.0, help='Size of projected points')
    parser.add_argument('--mask_type', type=str, choices=['rough', 'fine', 'both'], default='both', 
                       help='Type of masks to use - should match what was used in filtering (rough, fine, or both)')
    
    return parser.parse_args()

def project_points(points_3d, camera, image):
    """Project 3D points onto image plane"""
    if len(points_3d) == 0:
        return np.array([])
    
    # Get camera parameters
    if camera.model == 'SIMPLE_RADIAL':
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
        # k = camera.params[3]  # Radial distortion parameter (not used in simple projection)
    elif camera.model == 'SIMPLE_PINHOLE':
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
    elif camera.model == 'PINHOLE':
        fx, fy = camera.params[0], camera.params[1]
        cx, cy = camera.params[2], camera.params[3]
    else:
        print(f"Warning: Unsupported camera model {camera.model}")
        print("Expected SIMPLE_RADIAL camera model")
        fx = fy = 1000  # Default values
        cx, cy = camera.width / 2, camera.height / 2
    
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
    points_cam = (R @ points_3d.T).T + tvec
    
    # Remove points behind camera
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    
    if len(points_cam) == 0:
        return np.array([])
    
    # Project to image plane
    points_2d = np.zeros((len(points_cam), 2))
    points_2d[:, 0] = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    points_2d[:, 1] = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    
    # Filter points within image bounds
    valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.width)
    valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.height)
    valid_points = points_2d[valid_x & valid_y]
    
    return valid_points

def load_model_data(model_path):
    """Load COLMAP model and extract 3D points"""
    if not model_path or not Path(model_path).exists():
        return None, None, np.array([])
    
    cameras, images, points3D = read_model(model_path)
    
    # Extract 3D point coordinates
    points = np.array([point.xyz for point in points3D.values()])
    
    return cameras, images, points

def find_mask_file(masks_dir, image_name, mask_type):
    """Find corresponding mask file for an image"""
    if not masks_dir:
        return None
        
    masks_path = Path(masks_dir)
    if not masks_path.exists():
        return None
    
    # Remove extension and add mask suffix
    base_name = Path(image_name).stem
    
    # Handle different mask types
    if mask_type == 'both':
        # Try to find and combine both rough and fine masks using the same logic as filtering
        rough_suffix = "_rough.png"
        fine_suffix = "_fine.png"
        rough_name = base_name + rough_suffix
        fine_name = base_name + fine_suffix
        rough_path = masks_path / rough_name
        fine_path = masks_path / fine_name
        
        # Check if both masks exist
        if rough_path.exists() and fine_path.exists():
            # Load both masks
            rough_mask = cv2.imread(str(rough_path), cv2.IMREAD_GRAYSCALE)
            fine_mask = cv2.imread(str(fine_path), cv2.IMREAD_GRAYSCALE)
            
            # Create combined mask using the same logic as filtering: (rough_mask > threshold) | (fine_mask > threshold)
            threshold = 10  # Default threshold used in filtering
            combined_mask = ((rough_mask > threshold) | (fine_mask > threshold)).astype(np.uint8) * 255
            
            # Save temporary combined mask
            combined_path = masks_path / f"{base_name}_combined.png"
            cv2.imwrite(str(combined_path), combined_mask)
            
            return combined_path, "combined (rough | fine)"
        elif rough_path.exists():
            return rough_path, "rough"
        elif fine_path.exists():
            return fine_path, "fine"
        else:
            return None, None
    else:
        # Look for specific mask type
        mask_suffix = f"_{mask_type}.png"
        mask_name = base_name + mask_suffix
        mask_path = masks_path / mask_name
        
        if mask_path.exists():
            return mask_path, mask_type
        
        # Try alternative naming conventions
        for suffix in [f"_{mask_type}.jpg", f"_{mask_type}.jpeg", ".png", ".jpg", ".jpeg"]:
            alt_mask_path = masks_path / (base_name + suffix)
            if alt_mask_path.exists():
                return alt_mask_path, mask_type
    
    return None, None



def create_pipeline_visualization(args):
    """Create comprehensive pipeline visualization"""
    
    # All parameters now provided directly via command line arguments
    
    # Validate required arguments (all are required now)
    if not args.images:
        print("Error: --images is required")
        return
    if not args.original_model:
        print("Error: --original_model is required")
        return
    if not args.filtered_model:
        print("Error: --filtered_model is required")
        return
    
    # Load all models
    print("\nLoading models...")
    original_data = load_model_data(args.original_model)
    filtered_data = load_model_data(args.filtered_model)
    ray_data = load_model_data(args.ray_model)
    
    cameras_orig, images_orig, points_orig = original_data
    cameras_filt, images_filt, points_filt = filtered_data
    cameras_ray, images_ray, points_ray = ray_data
    
    print(f"Original model: {len(points_orig)} points")
    print(f"Filtered model: {len(points_filt)} points") 
    print(f"Ray-enhanced model: {len(points_ray)} points")
    
    # Get available images
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Error: Images directory {image_dir} not found")
        return
    
    # Find common images across all models that exist
    available_images = []
    for img_path in image_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            img_name = img_path.name
            
            # Check if image exists in at least one model
            found_in_model = False
            for images_dict in [images_orig, images_filt, images_ray]:
                if images_dict:
                    for image_data in images_dict.values():
                        if image_data.name == img_name:
                            found_in_model = True
                            break
            
            if found_in_model:
                available_images.append(img_name)
    
    if not available_images:
        print("Error: No common images found")
        return
    
    # Sample images for visualization
    sample_images = random.sample(available_images, min(args.n_images, len(available_images)))
    
    # Create visualization with 5 columns: original, mask, original_points, filtered_points, ray_points
    n_cols = 5
    n_rows = len(sample_images)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['Original Image', 'Segmentation Mask', 'Original COLMAP', 'Filtered COLMAP', 'Ray Enhanced']
    
    # Set column titles
    for col, title in enumerate(column_titles):
        if n_rows > 0:
            axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    for row, image_name in enumerate(sample_images):
        # Load original image
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"Warning: Image {image_path} not found")
            continue
            
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Column 1: Original Image
        axes[row, 0].imshow(img)
        axes[row, 0].set_xlabel(f'{image_name}')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        
        # Column 2: Segmentation Mask
        mask_result = find_mask_file(args.masks, image_name, args.mask_type)
        if mask_result[0]:  # If mask found
            mask_path, actual_mask_type = mask_result
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Create colored mask overlay
            mask_colored = np.zeros_like(img)
            mask_colored[mask > 0] = [0, 255, 0]  # Green for tree regions
            
            # Blend with original image
            alpha = 0.6
            blended = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)
            axes[row, 1].imshow(blended)
            axes[row, 1].set_xlabel(f'{actual_mask_type} mask')
        else:
            axes[row, 1].imshow(img)
            axes[row, 1].text(0.5, 0.5, 'No mask found', transform=axes[row, 1].transAxes,
                             ha='center', va='center', fontsize=12, color='red',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[row, 1].set_xlabel('No mask')
        
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        
        # Get camera and image data for projections
        camera = None
        image_data = None
        
        # Try to find camera and image data from available models
        for cameras_dict, images_dict in [(cameras_orig, images_orig), 
                                         (cameras_filt, images_filt), 
                                         (cameras_ray, images_ray)]:
            if cameras_dict and images_dict:
                camera = list(cameras_dict.values())[0]  # Assume single camera
                for img_data in images_dict.values():
                    if img_data.name == image_name:
                        image_data = img_data
                        break
                if camera and image_data:
                    break
        
        # Columns 3-5: Point projections
        point_data = [
            (points_orig, 'red', 'Original'),
            (points_filt, 'blue', 'Filtered'),
            (points_ray, 'green', 'Ray Enhanced')
        ]
        
        for col_idx, (points, color, label) in enumerate(point_data, start=2):
            axes[row, col_idx].imshow(img)
            
            if camera and image_data and len(points) > 0:
                points_2d = project_points(points, camera, image_data)
                if len(points_2d) > 0:
                    axes[row, col_idx].scatter(points_2d[:, 0], points_2d[:, 1], 
                                             c=color, s=args.point_size, alpha=0.7)
                    axes[row, col_idx].set_xlabel(f'{len(points_2d)} points projected')
                else:
                    axes[row, col_idx].set_xlabel('No points visible')
            else:
                axes[row, col_idx].set_xlabel('No model data')
            
            axes[row, col_idx].set_xlim(0, img.shape[1])
            axes[row, col_idx].set_ylim(img.shape[0], 0)
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])
    
    plt.tight_layout()
    
    # Save visualization
    print(f"Saving comprehensive visualization to {args.output}")
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Pipeline visualization complete!")

def main():
    args = parse_args()
    create_pipeline_visualization(args)

if __name__ == '__main__':
    main()
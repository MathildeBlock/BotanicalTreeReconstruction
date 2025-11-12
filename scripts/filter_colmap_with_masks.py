import os
import numpy as np
import cv2
from read_write_model import read_model, write_model
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import sys

import argparse
import sys

# # Basic usage with OR combination
# python smolscripttodeletpointsandmakecolmap.py \
#   --colmap /work3/s204201/AP_colmap/sparse/0 \
#   --images /work3/s204201/RisøOakTree-filtered \
#   --rough-mask /work3/s203788/Segmented_Risøe/best_deeplabv3_tree_DICE_50_ish_epochs.pth \
#   --fine-mask /work3/s203788/Segmented_Risøe/PATCH_best_deeplabv3_tree_DICE_50_ish_epochs.pth

# # Custom threshold with AND combination
# python smolscripttodeletpointsandmakecolmap.py \
#   --colmap /path/to/sparse \
#   --images /path/to/images \
#   --rough-mask /path/to/rough \
#   --fine-mask /path/to/fine \
#   --threshold 25 \
#   --combine and \
#   --examples 10

# # With custom output directory
# python smolscripttodeletpointsandmakecolmap.py \
#   --colmap /path/to/sparse \
#   --images /path/to/images \
#   --rough-mask /path/to/rough \
#   --fine-mask /path/to/fine \
#   --output /custom/output/path

def create_mask_overlay_visualization(img_path, rough_mask, fine_mask, combined_mask, output_path, mask_thresh, combine_op):
    """Create and save a visualization showing the original image with mask overlays."""
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image for visualization: {img_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Mask Analysis: {os.path.basename(img_path)} (thresh={mask_thresh}, op={combine_op})", fontsize=16)
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Rough mask overlay
    rough_overlay = img_rgb.copy()
    rough_overlay[rough_mask > mask_thresh] = [255, 0, 0]  # Red overlay
    axes[0, 1].imshow(rough_overlay)
    axes[0, 1].set_title("Rough Mask Overlay (Red)")
    axes[0, 1].axis('off')
    
    # Fine mask overlay
    fine_overlay = img_rgb.copy()
    fine_overlay[fine_mask > mask_thresh] = [0, 255, 0]  # Green overlay
    axes[1, 0].imshow(fine_overlay)
    axes[1, 0].set_title("Fine Mask Overlay (Green)")
    axes[1, 0].axis('off')
    
    # Combined mask overlay
    combined_overlay = img_rgb.copy()
    combined_overlay[combined_mask] = [0, 0, 255]  # Blue overlay
    axes[1, 1].imshow(combined_overlay)
    axes[1, 1].set_title(f"Combined Mask ({combine_op}) - Blue")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Filter COLMAP 3D points using segmentation masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OR combination
  python script.py --colmap /path/to/sparse --images /path/to/images --rough-mask /path/to/rough --fine-mask /path/to/fine
  
  # With custom threshold and AND combination
  python script.py --colmap /path/to/sparse --images /path/to/images --rough-mask /path/to/rough --fine-mask /path/to/fine --threshold 15 --combine and
  
  # With more examples and custom output
  python script.py --colmap /path/to/sparse --images /path/to/images --rough-mask /path/to/rough --fine-mask /path/to/fine --output /custom/output --examples 10
        """
    )
    
    # Required arguments
    parser.add_argument('--colmap', required=True,
                       help='Path to COLMAP model directory (e.g., sparse/0)')
    parser.add_argument('--images', required=True,
                       help='Path to directory containing original images')
    parser.add_argument('--rough-mask', required=True,
                       help='Path to directory containing rough mask files')
    parser.add_argument('--fine-mask', required=True,
                       help='Path to directory containing fine mask files')
    
    # Optional arguments
    parser.add_argument('--output', 
                       help='Output directory for filtered model (default: auto-generated in colmap dir)')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Mask threshold value (default: 10)')
    parser.add_argument('--combine', choices=['or', 'and'], default='or',
                       help='How to combine masks: "or" (union) or "and" (intersection) (default: or)')
    parser.add_argument('--examples', type=int, default=5,
                       help='Number of example visualizations to save (default: 5)')
    parser.add_argument('--format', choices=['bin', 'txt'], default='bin',
                       help='COLMAP model format (default: bin)')
    parser.add_argument('--rough-suffix', default='_soft_mask.png',
                       help='Suffix for rough mask files (default: _soft_mask.png)')
    parser.add_argument('--fine-suffix', default='_mask_soft.png',
                       help='Suffix for fine mask files (default: _mask_soft.png)')
    parser.add_argument('--img-ext', default='.JPG',
                       help='Image file extension to replace (default: .JPG)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.colmap):
        print(f"Error: COLMAP model directory '{args.colmap}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        print(f"Error: Images directory '{args.images}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.rough_mask):
        print(f"Error: Rough mask directory '{args.rough_mask}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.fine_mask):
        print(f"Error: Fine mask directory '{args.fine_mask}' does not exist.")
        sys.exit(1)
    
    # Print configuration
    print("🔧 Configuration:")
    print(f"  COLMAP model: {args.colmap}")
    print(f"  Images: {args.images}")
    print(f"  Rough masks: {args.rough_mask}")
    print(f"  Fine masks: {args.fine_mask}")
    print(f"  Mask threshold: {args.threshold}")
    print(f"  Combine operation: {args.combine}")
    print(f"  Examples to save: {args.examples}")
    print(f"  Model format: {args.format}")
    
    # Load model
    print(f"\n📂 Loading COLMAP model...")
    try:
        cameras, images, points3D = read_model(args.colmap, ext=f".{args.format}")
        print(f"✅ Loaded: {len(cameras)} cameras, {len(images)} images, {len(points3D)} points")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Track which 3D points are kept
    kept_point_ids = set()
    
    # Create timestamp and output directories
    timestamp = datetime.now().strftime("%d%m_%H%M")
    
    if args.output:
        output_model_dir = args.output
    else:
        output_model_dir = os.path.join(args.colmap, f"filtered_{args.combine}_{args.threshold}_{timestamp}")
    
    viz_dir = f"{output_model_dir}_visualizations"
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Track processing stats
    viz_count = 0
    processed_images = 0
    skipped_images = 0
    
    print(f"\n🔄 Processing {len(images)} images...")
    
    # Process images
    for img_obj in images.values():
        img_path = os.path.join(args.images, img_obj.name)
        
        # Check if image file exists and can be loaded
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Warning: Could not load image {img_path}, skipping...")
            skipped_images += 1
            continue
        
        orig_h, orig_w = img.shape[:2]

        # Load masks
        rough_mask_path = os.path.join(args.rough_mask, img_obj.name.replace(args.img_ext, args.rough_suffix))
        fine_mask_path = os.path.join(args.fine_mask, img_obj.name.replace(args.img_ext, args.fine_suffix))
        
        rough_mask = cv2.imread(rough_mask_path, cv2.IMREAD_GRAYSCALE)
        fine_mask = cv2.imread(fine_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if masks were loaded successfully
        if rough_mask is None or fine_mask is None:
            print(f"⚠️  Warning: Could not load masks for {img_obj.name}, skipping...")
            skipped_images += 1
            continue
        
        # Combine masks based on operation
        if args.combine == 'or':
            combined_mask = (rough_mask > args.threshold) | (fine_mask > args.threshold)
        else:  # and
            combined_mask = (rough_mask > args.threshold) & (fine_mask > args.threshold)

        # Save example visualizations for first few images
        if viz_count < args.examples:
            viz_filename = f"mask_overlay_{args.combine}_{args.threshold}_{img_obj.name.replace(args.img_ext, '.png')}"
            viz_path = os.path.join(viz_dir, viz_filename)
            create_mask_overlay_visualization(img_path, rough_mask, fine_mask, combined_mask, viz_path, args.threshold, args.combine)
            viz_count += 1

        # Update point3D_ids
        original_valid_points = np.sum(img_obj.point3D_ids != -1)
        points_removed = 0
        
        for i, pid in enumerate(img_obj.point3D_ids):
            if pid == -1:
                continue
            u, v = img_obj.xys[i]
            u_int = int(round(u))
            v_int = int(round(v))
            if (u_int < 0 or u_int >= combined_mask.shape[1] or 
                v_int < 0 or v_int >= combined_mask.shape[0] or 
                not combined_mask[v_int, u_int]):
                img_obj.point3D_ids[i] = -1  # delete from image
                points_removed += 1
            else:
                kept_point_ids.add(pid)
        
        processed_images += 1
        if processed_images % 50 == 0:
            print(f"  Processed {processed_images}/{len(images)} images...")

    # Remove deleted points from points3D
    points3D_filtered = {pid: pt for pid, pt in points3D.items() if pid in kept_point_ids}
    
    # Save updated model
    print(f"\n💾 Saving filtered model...")
    write_model(cameras, images, points3D_filtered, output_model_dir, ext=f".{args.format}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  Original 3D points: {len(points3D):,}")
    print(f"  Filtered 3D points: {len(points3D_filtered):,}")
    print(f"  Points removed: {len(points3D) - len(points3D_filtered):,} ({((len(points3D) - len(points3D_filtered)) / len(points3D) * 100):.1f}%)")
    print(f"  Images processed: {processed_images}")
    print(f"  Images skipped: {skipped_images}")
    print(f"  Example visualizations saved: {viz_count}")
    print(f"  Output model: {output_model_dir}")
    print(f"  Visualizations: {viz_dir}")

if __name__ == "__main__":
    main()

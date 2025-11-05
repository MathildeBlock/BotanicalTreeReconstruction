import os
import numpy as np
import cv2
from read_write_model import read_model, write_model

colmap_model_dir = "/work3/s204201/AP_colmap/sparsefilteredimgs30_000/0"
images_dir = "/work3/s204201/RisøOakTree"
rough_mask_dir = "/work3/s203788/Segmented_Risøe/best_deeplabv3_tree_DICE_50_ish_epochs.pth"
fine_mask_dir = "/work3/s203788/Segmented_Risøe/PATCH_best_deeplabv3_tree_DICE_50_ish_epochs.pth"
mask_thresh = 3

# Load model
cameras, images, points3D = read_model(colmap_model_dir, ext=".bin")

# Track which 3D points are kept
kept_point_ids = set()

# Process images
for img_obj in images.values():
    img_path = os.path.join(images_dir, img_obj.name)
    orig_h, orig_w = cv2.imread(img_path).shape[:2]

    # Load masks
    rough_mask = cv2.imread(os.path.join(rough_mask_dir, img_obj.name.replace(".JPG","_soft_mask.png")), cv2.IMREAD_GRAYSCALE)
    fine_mask = cv2.imread(os.path.join(fine_mask_dir, img_obj.name.replace(".JPG","_mask_soft.png")), cv2.IMREAD_GRAYSCALE)
    combined_mask = (rough_mask > mask_thresh) & (fine_mask > mask_thresh)

    # Update point3D_ids
    for i, pid in enumerate(img_obj.point3D_ids):
        if pid == -1:
            continue
        u, v = img_obj.xys[i]
        u_int = int(round(u))
        v_int = int(round(v))
        if u_int < 0 or u_int >= combined_mask.shape[1] or v_int < 0 or v_int >= combined_mask.shape[0] or not combined_mask[v_int, u_int]:
            img_obj.point3D_ids[i] = -1  # delete from image
        else:
            kept_point_ids.add(pid)

# Remove deleted points from points3D
points3D_filtered = {pid: pt for pid, pt in points3D.items() if pid in kept_point_ids}

# Save updated model
output_model_dir = os.path.join(colmap_model_dir, "segmentation_filtered_simple")
os.makedirs(output_model_dir, exist_ok=True)
write_model(cameras, images, points3D_filtered, output_model_dir, ext=".bin")
print(f"Saved filtered model with {len(points3D_filtered)} points to {output_model_dir}")

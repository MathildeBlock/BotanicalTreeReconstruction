#!/usr/bin/env python3
"""
Voxel-hash ray-density point insertion for COLMAP models (full script),
memory-safe version using spatial chunking (process one coarse chunk of space
at a time so we never hold the whole scene's voxels in memory).

Key idea: split scene bounding box into 3D chunks (--chunk_size meters).
For each chunk:
  - sample rays (workers), but workers only add samples that fall inside the chunk
  - merge per-worker voxel aggregates for that chunk
  - compute centroids for voxels meeting min image support
  - append centroids and free memory for the chunk
"""
import os
import argparse
import math
import numpy as np
import cv2
from read_write_model import read_model, write_model, Image, Point3D
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict
import random
import itertools

# -------------------- Argument parsing --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Voxel-hash ray-density point insertion for COLMAP models.")
    p.add_argument("--colmap_model_dir", type=str, required=True,
                   help="Path to COLMAP model directory (sparse) containing cameras.bin, images.bin, points3D.bin")
    p.add_argument("--images_dir", type=str, default=".",
                   help="Path to original images (optional, only for naming/consistency)")
    p.add_argument("--rough_mask_dir", type=str, required=True,
                   help="Directory with rough masks (filename convention: IMAGE.JPG -> IMAGE_soft_mask.png)")
    p.add_argument("--fine_mask_dir", type=str, required=True,
                   help="Directory with fine masks (filename convention: IMAGE.JPG -> IMAGE_mask_soft.png)")
    p.add_argument("--mask_thresh", type=int, default=3, help="Mask threshold used for combining rough/fine masks")

    p.add_argument("--samples_per_image", type=int, default=2000, help="Sampled mask pixels per image")
    p.add_argument("--depth_samples", type=int, default=500, help="Depth samples per ray")
    p.add_argument("--depth_min", type=float, default=1.0, help="Minimum depth (m)")
    p.add_argument("--depth_max", type=float, default=7.0, help="Maximum depth (m)")

    p.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size in meters")
    p.add_argument("--min_image_support", type=int, default=20, help="Minimum DISTINCT images supporting a voxel")
    p.add_argument("--merge_eps_mult", type=float, default=1.5,
                   help="Multiplier for MERGE_EPS relative to VOXEL_SIZE; MERGE_EPS = voxel_size*merge_eps_mult")

    p.add_argument("--uv_match_tol", type=float, default=1.5, help="Pixels tolerance for 2D matching")
    p.add_argument("--max_samples_per_voxel_for_centroid", type=int, default=2000,
                   help="Max per-voxel samples used to compute centroid (soft cap)")

    p.add_argument("--n_workers", type=int, default=min(8, multiprocessing.cpu_count()),
                   help="Number of worker processes for sampling")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--chunk_size", type=float, default=1.5,
                   help="Spatial chunk size in meters for processing (smaller = less memory, more runtime)")

    p.add_argument("--output_dir", type=str, required=True,
                   help="Base output directory. A folder will be created inside this path containing the model outputs.")
    p.add_argument("--output-folder-name", type=str, default=None,
                   help="Optional explicit subfolder name inside --output-dir. If omitted, a name is generated from params.")

    return p.parse_args()

args = parse_args()

# -------------------- Config (from args) --------------------
colmap_model_dir = args.colmap_model_dir
images_dir = args.images_dir
rough_mask_dir = args.rough_mask_dir
fine_mask_dir = args.fine_mask_dir
mask_thresh = int(args.mask_thresh)

SAMPLES_PER_IMAGE = int(args.samples_per_image)
DEPTH_SAMPLES = int(args.depth_samples)
DEPTH_MIN = float(args.depth_min)
DEPTH_MAX = float(args.depth_max)

VOXEL_SIZE = float(args.voxel_size)
MIN_IMAGE_SUPPORT = int(args.min_image_support)
MERGE_EPS = VOXEL_SIZE * float(args.merge_eps_mult)

UV_MATCH_TOL = float(args.uv_match_tol)
MAX_SAMPLES_PER_VOXEL_FOR_CENTROID = int(args.max_samples_per_voxel_for_centroid)

N_WORKERS = int(args.n_workers)
RNG_SEED = int(args.seed)
CHUNK_SIZE = float(args.chunk_size)

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# prepare output folder
if args.output_folder_name:
    folder_name = args.output_folder_name
else:
    folder_name = (f"GITSH_samples_pr_images_{SAMPLES_PER_IMAGE}_"
                   f"min_image_support{MIN_IMAGE_SUPPORT}_voxsize_{VOXEL_SIZE}_"
                   f"depths_samples_{DEPTH_SAMPLES}_maskthresh_{mask_thresh}_chunk_{CHUNK_SIZE}")
output_model_dir = os.path.join(args.output_dir, folder_name)
os.makedirs(output_model_dir, exist_ok=True)

print("Parameters:")
print(f" COLMAP model dir: {colmap_model_dir}")
print(f" rough mask dir: {rough_mask_dir}")
print(f" fine mask dir: {fine_mask_dir}")
print(f" output dir: {output_model_dir}")
print(f" samples/image: {SAMPLES_PER_IMAGE}, depth samples: {DEPTH_SAMPLES}, depth range: [{DEPTH_MIN},{DEPTH_MAX}]")
print(f" voxel size: {VOXEL_SIZE}, min image support: {MIN_IMAGE_SUPPORT}, merge_eps: {MERGE_EPS}")
print(f" chunk_size (m): {CHUNK_SIZE}, workers: {N_WORKERS}, seed: {RNG_SEED}")

# -------------------- Helpers (from your minimal rays script) --------------------
def qvec_to_R(q):
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy]
    ], dtype=float)

def camera_center(R, t):
    return -R.T @ np.array(t).reshape(3,)

def camera_intrinsics(cam):
    model = getattr(cam, "model", "")
    p = list(getattr(cam, "params", []))
    model_up = model.upper() if isinstance(model, str) else ""

    if model_up in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        if len(p) >= 3:
            f = float(p[0]); cx = float(p[1]); cy = float(p[2])
            return f, f, cx, cy
    elif model_up in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE"):
        if len(p) >= 4:
            return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if len(p) >= 4:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError("Unknown camera params/model in camera_intrinsics")

def choose_world_to_camera(R, C, pts, max_samples=500):
    if pts.shape[0] == 0:
        return R, (0, 0)
    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], max_samples, replace=False)
        samp = pts[idx]
    else:
        samp = pts
    x1 = (R @ (samp.T - C.reshape(3,1))).T
    x2 = (R.T @ (samp.T - C.reshape(3,1))).T
    c1 = int(np.sum(x1[:,2] > 0)); c2 = int(np.sum(x2[:,2] > 0))
    return (R if c1 >= c2 else R.T), (c1, c2)

# -------------------- Load COLMAP model --------------------
print("Loading COLMAP model...")
cameras, images, points3D = read_model(colmap_model_dir, ext=".bin")
print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3D)} 3D points")
next_point3d_id = max(points3D.keys()) + 1 if points3D else 1

existing_pts = np.vstack([p.xyz for p in points3D.values()]) if points3D else np.zeros((0,3), dtype=float)

# -------------------- Prepare lightweight image descriptors for workers --------------------
image_items = []
camera_centers = []
for img_id, img_obj in images.items():
    cam = cameras[img_obj.camera_id]
    try:
        width = int(cam.width)
        height = int(cam.height)
    except Exception:
        width = int(getattr(img_obj, 'width', 0))
        height = int(getattr(img_obj, 'height', 0))
    R = qvec_to_R(np.array(img_obj.qvec, dtype=float))
    C = camera_center(R, img_obj.tvec)
    camera_centers.append(C)
    item = {
        'img_id': int(img_id),
        'name': img_obj.name,
        'camera_id': int(img_obj.camera_id),
        'width': width,
        'height': height,
        'qvec': img_obj.qvec,
        'tvec': img_obj.tvec,
        'C': C.tolist(),  # include for worker convenience
        'R': R.tolist()
    }
    image_items.append(item)
camera_centers = np.vstack(camera_centers) if camera_centers else np.zeros((0,3), dtype=float)

# -------------------- Compute scene bounding box and chunk grid --------------------
# conservative bounding box: camera centers extended by DEPTH_MAX in all directions
if camera_centers.shape[0] == 0:
    print("No camera centers found — exiting.")
    exit(1)

scene_min = np.min(camera_centers, axis=0) - float(DEPTH_MAX)
scene_max = np.max(camera_centers, axis=0) + float(DEPTH_MAX)
scene_extent = scene_max - scene_min
n_chunks = np.maximum(1, np.ceil(scene_extent / CHUNK_SIZE).astype(int))

print("Scene AABB:", scene_min, scene_max)
print("Chunk grid dims (nx,ny,nz):", tuple(n_chunks.tolist()))
# generate chunk bounding boxes (centered, axis aligned)
chunk_bounds = []
for ix in range(n_chunks[0]):
    for iy in range(n_chunks[1]):
        for iz in range(n_chunks[2]):
            cmin = scene_min + np.array([ix,iy,iz], dtype=float) * CHUNK_SIZE
            cmax = cmin + CHUNK_SIZE
            chunk_bounds.append((tuple(cmin.tolist()), tuple(cmax.tolist()), (ix,iy,iz)))
print(f"Total chunks: {len(chunk_bounds)}")

# -------------------- Worker: sample rays but only accumulate samples inside given chunk bounds ----
def sample_for_image_worker_chunk(item, rough_mask_dir, fine_mask_dir, mask_thresh,
                                  samples_per_image, depth_samples, depth_min, depth_max,
                                  voxel_size, rng_seed, chunk_min, chunk_max, R_cam_to_world_arr, C_arr):
    """
    Sample mask pixels for a single image, generate ray points, but only accumulate
    those points that fall inside [chunk_min, chunk_max). Return per-voxel aggregates.
    """
    np.random.seed(int(rng_seed) + int(item['img_id']) % 10000)
    random.seed(int(rng_seed) + int(item['img_id']) % 10000)

    img_id = int(item['img_id'])
    img_name = item['name']
    cam_id = int(item['camera_id'])

    rough_fname = img_name.replace(".JPG", "_rough.png")
    fine_fname = img_name.replace(".JPG", "_fine.png")
    rough_path = os.path.join(rough_mask_dir, rough_fname)
    fine_path = os.path.join(fine_mask_dir, fine_fname)

    rough = cv2.imread(rough_path, cv2.IMREAD_GRAYSCALE)
    fine = cv2.imread(fine_path, cv2.IMREAD_GRAYSCALE)
    if rough is None or fine is None:
        return {'vox_map': {}, 'samples': 0}

    combined = ((rough > mask_thresh) | (fine > mask_thresh)).astype(np.uint8)
    ys, xs = np.where(combined > 0)
    if len(xs) == 0:
        return {'vox_map': {}, 'samples': 0}

    k = min(samples_per_image, len(xs))
    sel = np.random.choice(len(xs), size=k, replace=False)
    xs_sel = xs[sel].astype(np.float32)
    ys_sel = ys[sel].astype(np.float32)
    uv_pixels = np.vstack([xs_sel, ys_sel]).T  # (k,2): u,v

    cam_obj = cameras[cam_id]
    fx, fy, cx, cy = camera_intrinsics(cam_obj)

    u = uv_pixels[:, 0]; v = uv_pixels[:, 1]
    dir_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)  # (k,3)

    R_cam_to_world = np.array(R_cam_to_world_arr, dtype=float).reshape(3,3)
    C = np.array(C_arr, dtype=float).reshape(3,)

    rays_world = (R_cam_to_world @ dir_cam.T).T
    norms = np.linalg.norm(rays_world, axis=1, keepdims=True)
    rays_world = rays_world / (norms + 1e-12)

    depths = np.linspace(depth_min, depth_max, depth_samples, dtype=np.float32)

    inv_vs = 1.0 / float(voxel_size)
    vox_map = {}  # (ix,iy,iz) -> [sum_x,sum_y,sum_z,count,set(img_ids)]
    total_samples = 0

    # bounding checks (avoid computing voxel for points outside chunk)
    cmin = np.array(chunk_min, dtype=float)
    cmax = np.array(chunk_max, dtype=float)

    # iterate rays one by one to avoid allocating huge arrays
    for r_idx in range(rays_world.shape[0]):
        ray = rays_world[r_idx]
        for d in depths:
            p = C + ray * float(d)
            # quickly skip if outside chunk in world coords
            if (p[0] < cmin[0] or p[0] >= cmax[0] or
                p[1] < cmin[1] or p[1] >= cmax[1] or
                p[2] < cmin[2] or p[2] >= cmax[2]):
                continue
            ix = int(math.floor(p[0] * inv_vs))
            iy = int(math.floor(p[1] * inv_vs))
            iz = int(math.floor(p[2] * inv_vs))
            key = (ix, iy, iz)
            if key in vox_map:
                entry = vox_map[key]
                entry[0] += float(p[0]); entry[1] += float(p[1]); entry[2] += float(p[2])
                entry[3] += 1
                # union image id set (it's already present)
            else:
                vox_map[key] = [float(p[0]), float(p[1]), float(p[2]), 1, {int(img_id)}]
            total_samples += 1

    return {'vox_map': vox_map, 'samples': total_samples}

# -------------------- Chunked sampling loop (process one spatial chunk at a time) --------------------
print(f"Processing {len(chunk_bounds)} chunks, chunk_size={CHUNK_SIZE} m ...")
all_centroids = []
all_vox_info = []

# Precompute per-image chosen World->Camera convention, Rcw and C to pass to workers
per_image_conv = {}
for it in image_items:
    R = qvec_to_R(np.array(it['qvec'], dtype=float))
    C = camera_center(R, it['tvec'])
    chosen_w2c, _ = choose_world_to_camera(R, C, existing_pts)
    Rcw = chosen_w2c.T
    per_image_conv[int(it['img_id'])] = (Rcw.astype(np.float32), C.astype(np.float32))

# Process each chunk sequentially
for cidx, (cmin, cmax, cgrid) in enumerate(tqdm(chunk_bounds, desc="Chunks")):
    # Submit sampling tasks for this chunk
    worker_inputs = []
    for it in image_items:
        img_id = int(it['img_id'])
        Rcw, C = per_image_conv[img_id]
        worker_inputs.append((it, rough_mask_dir, fine_mask_dir, mask_thresh,
                              SAMPLES_PER_IMAGE, DEPTH_SAMPLES, DEPTH_MIN, DEPTH_MAX,
                              VOXEL_SIZE, RNG_SEED, cmin, cmax, Rcw, C))
    # Merge per-worker voxel aggregates for this chunk only
    chunk_vox_map = {}  # key -> [sumx,sumy,sumz,count,set(img_ids)]
    chunk_total_samples = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(sample_for_image_worker_chunk, *args): args[0]['img_id'] for args in worker_inputs}
        for fut in as_completed(futures):
            res = fut.result()
            vox_map = res.get('vox_map', {})
            samples = res.get('samples', 0)
            chunk_total_samples += samples
            # merge
            for k, v in vox_map.items():
                sumx, sumy, sumz, cnt, imgset = v[0], v[1], v[2], v[3], v[4]
                if k in chunk_vox_map:
                    g = chunk_vox_map[k]
                    g[0] += sumx; g[1] += sumy; g[2] += sumz; g[3] += cnt
                    g[4].update(imgset)
                else:
                    chunk_vox_map[k] = [sumx, sumy, sumz, cnt, set(imgset)]
    # chunk merge done
    print(f"Chunk {cidx+1}/{len(chunk_bounds)} grid{cgrid}: samples_in_chunk={chunk_total_samples}, distinct_voxels={len(chunk_vox_map)}")

    # Compute centroids for voxels meeting min image support inside this chunk
    chunk_vox_info = []
    for vk, v in chunk_vox_map.items():
        sumx, sumy, sumz, cnt, imgset = v
        if cnt > 0 and len(imgset) >= MIN_IMAGE_SUPPORT:
            centroid = np.array([sumx / cnt, sumy / cnt, sumz / cnt], dtype=float)
            chunk_vox_info.append({
                'voxel_key': vk,
                'sum': np.array([sumx, sumy, sumz], dtype=float),
                'count': int(cnt),
                'image_ids': set(imgset),
                'centroid': centroid
            })
    if chunk_vox_info:
        centroids_chunk = np.vstack([v['centroid'] for v in chunk_vox_info]).astype(float)
        all_centroids.append(centroids_chunk)
        all_vox_info.extend(chunk_vox_info)
    # free chunk memory
    del chunk_vox_map
    # optional: force GC
    import gc
    gc.collect()

if all_centroids:
    centroids = np.vstack(all_centroids).astype(float)
else:
    centroids = np.zeros((0,3), dtype=float)

print(f"Total centroids after chunked accumulation: {len(centroids)}")

# -------------------- Attach centroids -> project & match/append 2D keypoints (same as before) -----
print("Caching combined masks (reading each mask once)...")
combined_masks = {}
for img_id, img_obj in images.items():
    rough_fname = img_obj.name.replace(".JPG", "_soft_mask.png")
    fine_fname = img_obj.name.replace(".JPG", "_mask_soft.png")
    rough_path = os.path.join(rough_mask_dir, rough_fname)
    fine_path = os.path.join(fine_mask_dir, fine_fname)
    rough = cv2.imread(rough_path, cv2.IMREAD_GRAYSCALE)
    fine = cv2.imread(fine_path, cv2.IMREAD_GRAYSCALE)
    if rough is None or fine is None:
        combined_masks[int(img_id)] = None
    else:
        combined_masks[int(img_id)] = ((rough > mask_thresh) | (fine > mask_thresh)).astype(np.uint8)

print("Precomputing camera matrices and building per-image KD-trees (if available)...")
camera_data = {}
try:
    from scipy.spatial import cKDTree
    have_kdt = True
except Exception:
    have_kdt = False

for img_id, img_obj in images.items():
    cam = cameras[img_obj.camera_id]
    fx, fy, cx, cy = camera_intrinsics(cam)
    R = qvec_to_R(img_obj.qvec)
    t = np.array(img_obj.tvec).reshape(3,)
    C = -R.T @ t
    try:
        width = int(cam.width)
        height = int(cam.height)
    except Exception:
        width = int(getattr(img_obj, 'width', 0))
        height = int(getattr(img_obj, 'height', 0))
    xys = np.array(img_obj.xys, dtype=float) if hasattr(img_obj, 'xys') else np.zeros((0, 2), dtype=float)
    pids = np.array(img_obj.point3D_ids, dtype=int) if hasattr(img_obj, 'point3D_ids') else np.array([], dtype=int)

    tree = None
    if have_kdt and xys.shape[0] > 0:
        try:
            tree = cKDTree(xys)
        except Exception:
            tree = None

    camera_data[int(img_id)] = {
        'cam_obj': cam,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'R': R, 't': t, 'C': C,
        'width': width, 'height': height,
        'xys': xys, 'pids': pids, 'tree': tree
    }

def project_point_to_image_precomputed(X, cam_data):
    R = cam_data['R']; t = cam_data['t']
    X_cam = (R @ X.reshape(3,1)).reshape(3,) + t
    z = X_cam[2]
    fx, fy, cx, cy = cam_data['fx'], cam_data['fy'], cam_data['cx'], cam_data['cy']
    u = (fx * X_cam[0] / (z + 1e-12)) + cx
    v = (fy * X_cam[1] / (z + 1e-12)) + cy
    return u, v, z

print("Attaching centroids to images (optimized projection & matching)...")
images_new = dict(images)
created_points = []

use_vox_image_lists = (len(all_vox_info) == centroids.shape[0] and all('image_ids' in v for v in all_vox_info))

total_centroids = centroids.shape[0]
print(f"Will process {total_centroids} centroids.")
for k in tqdm(range(total_centroids), desc="Centroids"):
    centroid = centroids[k]
    if use_vox_image_lists:
        candidate_imgs = list(map(int, all_vox_info[k]['image_ids']))
    else:
        candidate_imgs = list(map(int, images.keys()))

    attached_imgs = []
    attached_2d_idxs = []

    for img_id in candidate_imgs:
        cam_data = camera_data.get(int(img_id))
        if cam_data is None:
            continue

        u, v, z = project_point_to_image_precomputed(centroid, cam_data)
        if z <= 0:
            continue
        if not (0 <= u < cam_data['width'] and 0 <= v < cam_data['height']):
            continue

        mask = combined_masks.get(int(img_id))
        if mask is not None:
            u_i = int(np.clip(np.round(u).astype(int), 0, mask.shape[1] - 1))
            v_i = int(np.clip(np.round(v).astype(int), 0, mask.shape[0] - 1))
            if mask[v_i, u_i] == 0:
                continue

        tree = cam_data['tree']
        if tree is not None:
            d, idx = tree.query([u, v], k=1)
            if d <= UV_MATCH_TOL:
                attached_imgs.append(int(img_id))
                attached_2d_idxs.append(int(idx))
                continue

        xys = cam_data['xys']
        if xys.shape[0] > 0:
            diffs = xys - np.array([u, v], dtype=float)
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            best_idx = int(np.argmin(d2))
            if d2[best_idx] <= (UV_MATCH_TOL ** 2):
                attached_imgs.append(int(img_id))
                attached_2d_idxs.append(best_idx)
                continue

        img_obj = images_new[int(img_id)]
        new_xys = np.vstack([np.array(img_obj.xys, dtype=float), np.array([[u, v]], dtype=float)])
        new_pids = np.concatenate([np.array(img_obj.point3D_ids, dtype=int), np.array([-1], dtype=int)])
        updated_img = Image(id=img_obj.id, qvec=img_obj.qvec, tvec=img_obj.tvec, camera_id=img_obj.camera_id,
                            name=img_obj.name, xys=new_xys, point3D_ids=new_pids)
        images_new[int(img_id)] = updated_img
        attached_imgs.append(int(img_id))
        attached_2d_idxs.append(new_xys.shape[0] - 1)

    if len(set(attached_imgs)) >= MIN_IMAGE_SUPPORT:
        created_points.append({
            'xyz': centroid.astype(float),
            'image_ids': np.array(attached_imgs, dtype=int),
            'point2D_idxs': np.array(attached_2d_idxs, dtype=int)
        })

print(f"Prepared {len(created_points)} new 3D points after projection & 2D matching (>= {MIN_IMAGE_SUPPORT} images).")

# -------------------- Create Point3D objects and update images_new --------------------
points3D_new = dict(points3D)
added_count = 0
pid_counter = next_point3d_id

for cp in tqdm(created_points, desc="Creating Point3D"):
    pid = pid_counter
    pid_counter += 1
    added_count += 1

    xyz = cp['xyz']
    img_ids = cp['image_ids']
    p2d_idxs = cp['point2D_idxs']

    new_pt = Point3D(
        id=int(pid),
        xyz=xyz.astype(float),
        rgb=np.array([0, 255, 0], dtype=np.uint8),
        error=1.0,
        image_ids=np.array(img_ids, dtype=int),
        point2D_idxs=np.array(p2d_idxs, dtype=int)
    )
    points3D_new[int(pid)] = new_pt

    for img_id, idx2d in zip(img_ids, p2d_idxs):
        img_obj = images_new[int(img_id)]
        pids_arr = np.array(img_obj.point3D_ids, dtype=int)
        if idx2d < 0 or idx2d >= pids_arr.shape[0]:
            continue
        pids_arr[int(idx2d)] = int(pid)
        updated_img = Image(id=img_obj.id, qvec=img_obj.qvec, tvec=img_obj.tvec,
                            camera_id=img_obj.camera_id, name=img_obj.name,
                            xys=np.array(img_obj.xys, dtype=float), point3D_ids=pids_arr)
        images_new[int(img_id)] = updated_img

print(f"Added {added_count} new 3D points to the model.")

# -------------------- Save model --------------------
print("Saving updated model...")
write_model(cameras, images_new, points3D_new, output_model_dir, ext=".bin")
write_model(cameras, images_new, points3D_new, output_model_dir, ext=".txt")
print(f"Saved model to {output_model_dir}")
for fname in sorted(os.listdir(output_model_dir)):
    fpath = os.path.join(output_model_dir, fname)
    try:
        size = os.path.getsize(fpath)
        print(f"{fname:30s} {size/1024:.1f} KB")
    except Exception:
        print(f"{fname:30s} <could not stat>")
print("Done.")
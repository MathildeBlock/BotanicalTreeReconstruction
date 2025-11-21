#!/usr/bin/env python3
"""
Segmentation Inference Script

Unified script for running tree segmentation inference with DeepLabV3.
Supports both full-image and patch-based (tiled) inference modes.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import torch.nn as nn
from pathlib import Path


def load_ckpt_state(ckpt_path):
    """Load checkpoint state dict, handling various formats."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state_dict" in ckpt):
        sd = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    else:
        sd = ckpt
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    return new_sd


def sliding_windows(width, height, tile_size, overlap):
    """Generate sliding window coordinates for patch-based inference."""
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    
    # Ensure last tile touches the border
    if not xs or xs[-1] != width - tile_size:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != height - tile_size:
        ys.append(max(0, height - tile_size))
    
    for y in ys:
        for x in xs:
            yield x, y


def load_model(model_path, device):
    """Load and initialize the segmentation model."""
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Initialize model
    model = segmentation.deeplabv3_resnet50(weights=None)
    
    # Adjust heads for binary segmentation (1 output channel)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    
    model = model.to(device)
    
    # Load checkpoint
    sd = load_ckpt_state(model_path)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Loaded checkpoint:")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    
    model.eval()
    return model


def full_image_inference(model, image, device, input_size=512):
    """Run inference on full image (resized)."""
    orig_w, orig_h = image.size
    
    # Transform: resize, normalize
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_dict = model(input_tensor)
        out = out_dict.get("out") if isinstance(out_dict, dict) else out_dict
        
        if out is None:
            raise RuntimeError("Model did not return 'out'")
        
        probs = torch.sigmoid(out)
        soft_mask = probs.squeeze().cpu().numpy()
    
    # Resize mask back to original size
    soft_mask_img = Image.fromarray((soft_mask * 255).astype(np.uint8))
    mask_resized = soft_mask_img.resize((orig_w, orig_h), Image.NEAREST)
    
    return np.array(mask_resized).astype(np.float32) / 255.0


def patch_inference(model, image, device, tile_size=512, overlap=64):
    """Run patch-based inference with sliding windows."""
    W, H = image.size
    print(f"  Image size: {W} x {H}, using patches of {tile_size}x{tile_size} with {overlap}px overlap")
    
    # Transform for patches (no resize, just normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Accumulation arrays
    prob_acc = np.zeros((H, W), dtype=np.float32)
    count_acc = np.zeros((H, W), dtype=np.float32)
    
    # Process tiles
    total_tiles = len(list(sliding_windows(W, H, tile_size, overlap)))
    processed_tiles = 0
    
    for (x0, y0) in sliding_windows(W, H, tile_size, overlap):
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        
        # Crop tile
        tile = image.crop((x0, y0, min(x1, W), min(y1, H)))
        
        # Handle edge cases by resizing
        resized_back = False
        if tile.size != (tile_size, tile_size):
            tile = tile.resize((tile_size, tile_size), Image.BILINEAR)
            resized_back = True
        
        # Run inference
        inp = transform(tile).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_dict = model(inp)
            out = out_dict.get("out") if isinstance(out_dict, dict) else out_dict
            
            if out is None:
                raise RuntimeError("Model did not return 'out'")
            
            probs = torch.sigmoid(out)
            probs_np = probs.squeeze().cpu().numpy()
        
        # Resize back if needed
        if resized_back:
            orig_w = min(x1, W) - x0
            orig_h = min(y1, H) - y0
            probs_np = Image.fromarray((probs_np * 255).astype(np.uint8))
            probs_np = probs_np.resize((orig_w, orig_h), Image.BILINEAR)
            probs_np = np.array(probs_np).astype(np.float32) / 255.0
        
        # Add to accumulators
        if resized_back:
            prob_acc[y0:y0+orig_h, x0:x0+orig_w] += probs_np
            count_acc[y0:y0+orig_h, x0:x0+orig_w] += 1.0
        else:
            prob_acc[y0:y1, x0:x1] += probs_np
            count_acc[y0:y1, x0:x1] += 1.0
        
        processed_tiles += 1
        if processed_tiles % 10 == 0:
            print(f"    Processed {processed_tiles}/{total_tiles} patches")
    
    # Average overlapping predictions
    mask_probs = np.zeros_like(prob_acc)
    valid = count_acc > 0
    mask_probs[valid] = prob_acc[valid] / count_acc[valid]
    
    return mask_probs


def save_single_mask(mask_probs, base_name, output_folder, mask_type):
    """Save a single mask with appropriate suffix based on type."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{mask_type}.png"
    mask_path = output_folder / f"{base_name}{suffix}"
    
    # Save soft mask (0-255)
    soft_mask_img = Image.fromarray((mask_probs * 255).astype(np.uint8))
    soft_mask_img.save(mask_path)
    
    return mask_path


def main():
    parser = argparse.ArgumentParser(
        description='Tree segmentation inference with DeepLabV3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model', required=True,
                       help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--input', required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', required=True,
                       help='Output directory for masks')
    
    # Mask type
    parser.add_argument('--mask-type', choices=['rough', 'fine', 'both'], default='both',
                       help='Type of mask to generate: rough (fast, full-image), fine (detailed, patch-based), or both')
    
    # Inference parameters (used automatically based on mask type)
    parser.add_argument('--input-size', type=int, default=512,
                       help='Input size for rough (full-image) mode')
    parser.add_argument('--tile-size', type=int, default=512,
                       help='Tile size for patch mode')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between patches in patch mode')
    
    # Other options
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for inference')
    parser.add_argument('--image-ext', nargs='+', default=['.jpg', '.jpeg', '.png'],
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist.")
        return 1
    
    # Load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Find images
    input_path = Path(args.input)
    image_files = []
    for ext in args.image_ext:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found with extensions {args.image_ext} in {args.input}")
        return 1
    
    print(f"\nFound {len(image_files)} images")
    print(f"Mask type: {args.mask_type}")
    if args.mask_type == "both":
        print(f"  Rough masks: full-image mode (fast, {args.input_size}x{args.input_size})")
        print(f"  Fine masks: patch mode (detailed, {args.tile_size}x{args.tile_size} patches with {args.overlap}px overlap)")
    elif args.mask_type == "rough":
        print(f"  Mode: full-image ({args.input_size}x{args.input_size})")
    else:
        print(f"  Mode: patch-based ({args.tile_size}x{args.tile_size} patches with {args.overlap}px overlap)")
    print(f"Output directory: {args.output}")
    
    # Process images
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing ({i}/{len(image_files)}): {img_path.name}")
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Generate masks based on type
            if args.mask_type == 'both':
                # Generate both rough (full) and fine (patch) masks
                print(f"  Generating rough mask (full-image mode)...")
                rough_mask = full_image_inference(model, image, device, args.input_size)
                
                print(f"  Generating fine mask (patch mode)...")
                fine_mask = patch_inference(model, image, device, args.tile_size, args.overlap)
                
                # Save both
                base_name = img_path.stem
                rough_path = save_single_mask(rough_mask, base_name, args.output, "rough")
                fine_path = save_single_mask(fine_mask, base_name, args.output, "fine")
                print(f"  Saved: {rough_path} and {fine_path}")
                
            elif args.mask_type == 'rough':
                # Generate rough mask using full-image mode
                print(f"  Generating rough mask (full-image mode)...")
                mask_probs = full_image_inference(model, image, device, args.input_size)
                base_name = img_path.stem
                mask_path = save_single_mask(mask_probs, base_name, args.output, "rough")
                print(f"  Saved: {mask_path}")
                
            else:  # fine
                # Generate fine mask using patch mode
                print(f"  Generating fine mask (patch mode)...")
                mask_probs = patch_inference(model, image, device, args.tile_size, args.overlap)
                base_name = img_path.stem
                mask_path = save_single_mask(mask_probs, base_name, args.output, "fine")
                print(f"  Saved: {mask_path}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    print(f"\nCompleted processing {len(image_files)} images!")
    return 0


if __name__ == "__main__":
    exit(main())
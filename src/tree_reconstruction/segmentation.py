#!/usr/bin/env python3
"""
Segmentation Module for Tree Reconstruction

This module provides functionality for running semantic segmentation inference
on aerial images to identify tree regions.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import segmentation
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import yaml
from PIL import Image
from tqdm import tqdm
import typer
from rich.console import Console

console = Console()

class TreeSegmentationModel:
    """Wrapper for tree segmentation models"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        
    def _load_model(self, model_path: Optional[Path]):
        """Load segmentation model"""
        if model_path and model_path.exists():
            # Load custom trained model
            console.print(f"Loading custom model from {model_path}")
            model = torch.load(model_path, map_location=self.device)
        else:
            # Use pre-trained DeepLabV3 as fallback
            console.print("Using pre-trained DeepLabV3 model")
            model = segmentation.deeplabv3_resnet50(pretrained=True)
            
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def segment_image(self, image: Union[np.ndarray, Path], confidence: float = 0.5) -> np.ndarray:
        """
        Segment a single image
        
        Args:
            image: Input image as numpy array or path to image
            confidence: Confidence threshold for segmentation
            
        Returns:
            Binary mask as numpy array
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        pil_image = Image.fromarray(image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Process output
        if isinstance(output, dict):
            output = output['out']
        
        # Apply softmax and get predictions
        probs = torch.softmax(output, dim=1)
        
        # For now, assume vegetation class is index 1 (adjust based on your model)
        vegetation_prob = probs[0, 1].cpu().numpy()
        
        # Create binary mask
        mask = (vegetation_prob > confidence).astype(np.uint8) * 255
        
        return mask
    
    def segment_batch(self, images: List[np.ndarray], confidence: float = 0.5) -> List[np.ndarray]:
        """Segment a batch of images"""
        masks = []
        for image in images:
            mask = self.segment_image(image, confidence)
            masks.append(mask)
        return masks

def run_segmentation(
    input_dir: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    confidence: float = 0.5,
    device: str = "cuda",
    batch_size: int = 4,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']
):
    """
    Run segmentation on all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save segmentation masks
        model_path: Path to trained segmentation model
        confidence: Confidence threshold for segmentation
        device: Device to run inference on
        batch_size: Batch size for processing
        image_extensions: List of valid image extensions
    """
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
    
    if not image_files:
        console.print(f"[red]No images found in {input_dir}[/red]")
        return
    
    console.print(f"Found {len(image_files)} images to process")
    
    # Initialize model
    model = TreeSegmentationModel(model_path, device)
    
    # Process images
    for img_path in tqdm(image_files, desc="Segmenting images"):
        try:
            # Generate mask
            mask = model.segment_image(img_path, confidence)
            
            # Save mask
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            
        except Exception as e:
            console.print(f"[red]Error processing {img_path}: {e}[/red]")
            continue
    
    console.print(f"[green]Segmentation complete! Masks saved to {output_dir}[/green]")

def create_mask_overlay(image_path: Path, mask_path: Path, output_path: Path, alpha: float = 0.5):
    """Create overlay visualization of image and mask"""
    
    # Load image and mask
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Create colored mask (green for vegetation)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    # Save overlay
    cv2.imwrite(str(output_path), overlay)

def filter_images_by_mask(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    min_vegetation_ratio: float = 0.1
):
    """
    Filter images based on vegetation content in masks
    
    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing segmentation masks
        output_dir: Directory to save filtered images
        min_vegetation_ratio: Minimum ratio of vegetation pixels to keep image
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching image-mask pairs
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
    
    kept_images = 0
    total_images = len(image_files)
    
    for img_path in tqdm(image_files, desc="Filtering images"):
        mask_path = mask_dir / f"{img_path.stem}_mask.png"
        
        if not mask_path.exists():
            console.print(f"[yellow]Warning: No mask found for {img_path.name}[/yellow]")
            continue
        
        # Load mask and calculate vegetation ratio
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        vegetation_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        vegetation_ratio = vegetation_pixels / total_pixels
        
        # Keep image if it has enough vegetation
        if vegetation_ratio >= min_vegetation_ratio:
            output_path = output_dir / img_path.name
            import shutil
            shutil.copy2(img_path, output_path)
            kept_images += 1
    
    console.print(f"[green]Kept {kept_images}/{total_images} images with sufficient vegetation[/green]")

def main():
    """Main function for standalone usage"""
    typer.run(run_segmentation)

if __name__ == "__main__":
    main()
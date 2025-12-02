"""
COLMAP Model Creation Script
"""

import os
import subprocess
import argparse
import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_command(cmd: list[str], description: str) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run as a list of strings
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    logging.info(f"=== {description} ===")
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        logging.info(f"Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        logging.error(f"Command not found: {cmd[0]}")
        logging.error("Make sure COLMAP is installed and in your PATH")
        return False


def create_colmap_model(image_dir: Path, output_dir: Path, max_features: int = 30000) -> bool:
    """
    Create a COLMAP model from an image directory.
    
    Args:
        image_dir: Path to directory containing images
        output_dir: Path to output directory for the model
        max_features: Maximum number of features to extract per image
        
    Returns:
        True if successful, False otherwise
    """
    # Use GPU by default - COLMAP will automatically fall back to CPU if GPU isn't available
    gpu_flag = "1"
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
        logging.info(f"Removed existing database: {database_path}")
    
    # Step 1: Feature extraction
    logging.info(f"Extracting features from images in: {image_dir}")
    logging.info("Using GPU acceleration (with automatic CPU fallback)")
    feature_cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--SiftExtraction.max_num_features", str(max_features),
        "--SiftExtraction.estimate_affine_shape", "true",
        "--SiftExtraction.domain_size_pooling", "true",
        "--SiftExtraction.use_gpu", gpu_flag,
        "--SiftExtraction.max_image_size", "2400",
        "--SiftExtraction.num_threads", "4"  # Limit CPU threads to reduce memory usage
    ]
    
    if not run_command(feature_cmd, f"Feature Extraction ({max_features} features per image)"):
        return False
    
    # Step 2: Feature matching
    logging.info("Matching features between images...")
    match_cmd = [
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.guided_matching", "true",
        "--SiftMatching.use_gpu", gpu_flag
    ]
    
    if not run_command(match_cmd, "Feature Matching"):
        return False
    
    # Step 3: Sparse reconstruction
    logging.info("Creating sparse 3D model...")
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir)
    ]
    
    if not run_command(mapper_cmd, "Sparse Reconstruction"):
        return False
    
    # Check for multiple models and merge if necessary
    model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if len(model_dirs) > 1:
        logging.info(f"Found {len(model_dirs)} separate models - attempting to merge...")
        
        # Sort models by number of images (largest first)
        model_sizes = []
        for model_dir in model_dirs:
            try:
                # Count images in each model by checking images.bin file size as proxy
                images_file = model_dir / "images.bin"
                if images_file.exists():
                    size = images_file.stat().st_size
                    model_sizes.append((model_dir, size))
            except:
                model_sizes.append((model_dir, 0))
        
        model_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if len(model_sizes) >= 2:
            # Try to merge the two largest models
            largest_model = model_sizes[0][0]
            second_model = model_sizes[1][0]
            merged_dir = sparse_dir / "merged"
            
            merge_cmd = [
                "colmap", "model_merger",
                "--input_path1", str(largest_model),
                "--input_path2", str(second_model),
                "--output_path", str(merged_dir)
            ]
            
            if run_command(merge_cmd, f"Merging models {largest_model.name} and {second_model.name}"):
                logging.info(f"Models successfully merged into: {merged_dir}")
            else:
                logging.warning("Model merging failed - keeping separate models")
    
    logging.info(f"COLMAP model created successfully in: {output_dir}")
    return True

def main():
    """Main function to create a COLMAP model from images."""
    parser = argparse.ArgumentParser(
        description="Create COLMAP 3D model from image directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing input images"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for the COLMAP model (default: ../models/colmap_output)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum number of features to extract per image"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input directory
    if not args.image_dir.exists():
        logging.error(f"Image directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    if not args.image_dir.is_dir():
        logging.error(f"Path is not a directory: {args.image_dir}")
        sys.exit(1)
    
    # Set default output directory if not provided
    if args.output is None:
        # Default to models folder in the project directory
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        args.output = project_dir / "models" / "colmap_output"
    
    # Check for images in directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in args.image_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        logging.error(f"No images found in directory: {args.image_dir}")
        logging.error(f"Supported formats: {', '.join(image_extensions)}")
        sys.exit(1)
    
    logging.info(f"Found {len(images)} images in {args.image_dir}")
    
    # Create COLMAP model
    try:
        success = create_colmap_model(
            image_dir=args.image_dir,
            output_dir=args.output,
            max_features=args.max_features
        )
        
        if success:
            logging.info("=== COLMAP model creation completed successfully! ===")
            logging.info(f"Model saved to: {args.output}")
            logging.info(f"Sparse reconstruction: {args.output / 'sparse'}")
            logging.info(f"Database: {args.output / 'database.db'}")
        else:
            logging.error("COLMAP model creation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
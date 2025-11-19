#!/usr/bin/env python3
"""
main_pipeline.py - Complete Botanical Tree Reconstruction Pipeline

This script executes the entire pipeline from raw images to final visualization:
1. Segmentation mask generation (rough & fine)
2. COLMAP 3D reconstruction
3. Point cloud filtering with masks
4. Comprehensive visualization

Usage:
    python main_pipeline.py --images data/raw --model models/model.pth
    python main_pipeline.py --help  # For all options
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

class PipelineRunner:
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%d%m_%H%M")
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = self.base_dir / "scripts"
        
        # Set input images directory
        self.images_dir = Path(args.images)
        
        # Set model path (if provided)
        self.model_path = Path(args.model) if args.model else None
            
        # Set up output directories (use provided paths or defaults)
        self.masks_dir = self.base_dir / "data" / "segmentation_masks"
        self.colmap_output = self.base_dir / "models" / "colmap_reconstruction"
        self.filtered_output = Path(args.filtered_model) if args.filtered_model else self.base_dir / "models" / "colmap_filtered"
        self.ray_output = self.base_dir / "models" / "colmap_ray_enhanced"
        self.viz_output = self.base_dir / "outputs" / f"pipeline_visualization_{self.timestamp}.png"
        
        # Override COLMAP input if provided
        self.colmap_input = None
        if args.colmap_model:
            self.colmap_input = Path(args.colmap_model)
        
        # Ensure output directories exist (only for directories we'll create)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        if not args.colmap_model:
            self.colmap_output.mkdir(parents=True, exist_ok=True)
        if not args.filtered_model:
            self.filtered_output.mkdir(parents=True, exist_ok=True)
        self.ray_output.mkdir(parents=True, exist_ok=True)
        self.viz_output.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting Botanical Tree Reconstruction Pipeline - {self.timestamp}")
        print(f"📁 Input images: {self.images_dir}")
        if args.model:
            print(f"🎯 Model: {args.model}")
        print(f"📊 COLMAP output: {self.colmap_output}")
        print(f"📊 Filtered output: {self.filtered_output}")
        print(f"📊 Ray enhanced: {self.ray_output}")

    def run_command(self, cmd, description, check=True):
        """Run a command with error handling and logging"""
        print(f"\n{'='*60}")
        print(f"🔄 {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=self.scripts_dir)
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {description}:")
            print(f"Exit code: {e.returncode}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            if self.args.continue_on_error:
                print("⚠️  Continuing despite error due to --continue-on-error flag")
                return e
            else:
                raise

    def step1_segmentation(self):
        """Generate segmentation masks"""
        print(f"\n🎯 STEP 1: Generating Segmentation Masks")
        
        cmd = [
            "python", "segmentation_inference.py",
            "--model", str(self.args.model),
            "--input", str(self.images_dir),
            "--output", str(self.masks_dir),
            "--mask-type", self.args.mask_type,
            "--device", self.args.device
        ]
        
        self.run_command(cmd, "Segmentation mask generation")
        
        # Count generated masks
        rough_masks = len(list(self.masks_dir.glob("*_rough.png")))
        fine_masks = len(list(self.masks_dir.glob("*_fine.png")))
        print(f"✅ Generated {rough_masks} rough masks and {fine_masks} fine masks")

    def step2_colmap(self):
        """Create COLMAP 3D reconstruction"""
        print(f"\n📐 STEP 2: COLMAP 3D Reconstruction")
        
        cmd = [
            "python", "colmap_pipeline.py",
            str(self.images_dir),
            "--output", str(self.colmap_output),
            "--max-features", str(self.args.max_features),
            "--log-level", "INFO"
        ]
        
        self.run_command(cmd, "COLMAP 3D reconstruction")
        
        # Check if sparse model was created
        sparse_dir = self.colmap_output / "sparse" / "0"
        if sparse_dir.exists():
            print(f"✅ COLMAP model created at {sparse_dir}")
        else:
            raise FileNotFoundError(f"COLMAP sparse model not found at {sparse_dir}")

    def check_dependencies(self):
        """Check if required dependencies exist for enabled steps"""
        print(f"🔍 Checking pipeline dependencies...")
        
        missing_deps = []
        
        # Check segmentation dependencies
        if not self.args.skip_segmentation:
            if not self.args.model:
                missing_deps.append("Segmentation model path required (--model)")
            elif not self.model_path.exists():
                missing_deps.append(f"Segmentation model not found: {self.model_path}")
        else:
            # If skipping segmentation, check if masks exist in standard location
            if not self.masks_dir.exists():
                missing_deps.append(f"Segmentation masks directory not found: {self.masks_dir}")
                missing_deps.append("  -> Either run segmentation (remove --skip-segmentation) or ensure masks exist in data/segmentation_masks")
        
        # Check COLMAP dependencies  
        if not self.args.skip_filtering:
            if self.args.skip_colmap:
                # If skipping COLMAP but filtering is enabled, need existing model
                if self.args.colmap_model:
                    colmap_model_dir = self.args.colmap_model / "sparse" / "0" if not (self.args.colmap_model / "cameras.bin").exists() else self.args.colmap_model
                else:
                    # Check standard location
                    colmap_model_dir = self.colmap_output / "sparse" / "0"
                
                if not colmap_model_dir.exists():
                    if self.args.colmap_model:
                        missing_deps.append(f"COLMAP model not found: {colmap_model_dir}")
                    else:
                        missing_deps.append(f"COLMAP model not found in standard location: {colmap_model_dir}")
                        missing_deps.append("  -> Either run COLMAP (remove --skip-colmap) or specify --colmap-model")
        
        # Check filtering dependencies for visualization
        if not self.args.skip_visualization:
            if self.args.skip_filtering:
                # If skipping filtering but visualization is enabled, check if filtered model exists
                if not self.filtered_output.exists():
                    missing_deps.append(f"Filtered COLMAP model not found: {self.filtered_output}")
                    missing_deps.append("  -> Either run filtering (remove --skip-filtering) or provide existing filtered model")
            
            # Also need original COLMAP model for visualization
            if self.args.skip_colmap and not self.args.colmap_model:
                colmap_model_dir = self.colmap_output / "sparse" / "0"
                if not colmap_model_dir.exists():
                    missing_deps.append(f"Original COLMAP model needed for visualization: {colmap_model_dir}")
                    missing_deps.append("  -> Either run COLMAP (remove --skip-colmap) or specify --colmap-model")
        
        if missing_deps:
            print(f"\n❌ Missing dependencies:")
            for dep in missing_deps:
                if dep.startswith("  ->"):
                    print(f"    {dep}")
                else:
                    print(f"  • {dep}")
            print(f"\n💡 Suggestions:")
            print(f"  • Run without skip flags to generate missing files")
            print(f"  • Or provide existing files in the expected locations")
            print(f"  • Use --help to see all available options")
            sys.exit(1)
        
        print(f"✅ All dependencies satisfied")

    def step3_filtering(self):
        """Filter point cloud with segmentation masks"""
        print(f"\n🔍 STEP 3: Point Cloud Filtering")
        
        # Determine which COLMAP model to use
        if self.colmap_input:
            # Use provided existing model
            colmap_model_path = self.colmap_input
            if not (colmap_model_path / "cameras.bin").exists():
                # Assume it's a model directory that contains sparse/0
                colmap_model_path = colmap_model_path / "sparse" / "0"
        else:
            # Use the model we just created
            colmap_model_path = self.colmap_output / "sparse" / "0"
        
        cmd = [
            "python", "filter_colmap_with_masks.py",
            "--colmap", str(colmap_model_path),
            "--images", str(self.images_dir),
            "--rough-mask", str(self.masks_dir),
            "--fine-mask", str(self.masks_dir),
            "--output", str(self.filtered_output),
            "--visibility-threshold", str(self.args.visibility_threshold),
            "--combine", self.args.combine_masks,
            "--examples", str(self.args.filter_examples)
        ]
        
        self.run_command(cmd, "Point cloud filtering with masks")
        print(f"✅ Filtered model saved to {self.filtered_output}")

    def step4_ray_enhancement(self):
        """Add points using ray densification"""
        print(f"\n🌟 STEP 4: Ray Enhancement")
        
        cmd = [
            "python", "add_points_with_rays.py",
            "--colmap_model_dir", str(self.filtered_output),
            "--images_dir", str(self.images_dir),
            "--rough_mask_dir", str(self.masks_dir),
            "--fine_mask_dir", str(self.masks_dir),
            "--output_dir", str(self.ray_output.parent),
            "--output-folder-name", self.ray_output.name,
            "--mask_thresh", "10",
            "--samples_per_image", "1000",
            "--depth_samples", "50"
        ]
        
        self.run_command(cmd, "Ray-based point densification")
        print(f"✅ Ray-enhanced model saved to {self.ray_output}")

    def step5_visualization(self):
        """Create comprehensive pipeline visualization"""
        print(f"\n📊 STEP 5: Pipeline Visualization")
        
        # Determine which original COLMAP model to use for visualization
        if self.colmap_input:
            # Use provided existing model
            original_model_path = self.colmap_input
            if not (original_model_path / "cameras.bin").exists():
                # Assume it's a model directory that contains sparse/0
                original_model_path = original_model_path / "sparse" / "0"
        else:
            # Use the model we just created
            original_model_path = self.colmap_output / "sparse" / "0"
        
        cmd = [
            "python", "pipeline_visualization.py",
            "--images", str(self.images_dir),
            "--masks", str(self.masks_dir),
            "--original_model", str(original_model_path),
            "--filtered_model", str(self.filtered_output),
            "--ray_model", str(self.ray_output),
            "--output", str(self.viz_output),
            "--n_images", str(self.args.viz_images),
            "--point_size", str(self.args.point_size),
            "--mask_type", "both"
        ]
        
        self.run_command(cmd, "Pipeline visualization")
        print(f"✅ Visualization saved to {self.viz_output}")

    def cleanup(self):
        """Clean up temporary files"""
        # No temporary files to clean up
        pass

    def save_pipeline_summary(self):
        """Save a summary of the pipeline run"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Convert args to JSON-serializable format
        args_dict = {}
        for key, value in vars(self.args).items():
            if isinstance(value, Path):
                args_dict[key] = str(value)
            else:
                args_dict[key] = value
        
        summary = {
            "pipeline_run": {
                "timestamp": self.timestamp,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "success": True
            },
            "inputs": {
                "images_directory": str(self.images_dir),
                "model_file": str(self.args.model) if self.args.model else None
            },
            "outputs": {
                "masks_directory": str(self.masks_dir),
                "colmap_model": str(self.colmap_output),
                "filtered_model": str(self.filtered_output),
                "ray_enhanced_model": str(self.ray_output),
                "visualization": str(self.viz_output)
            },
            "parameters": args_dict
        }
        
        summary_path = self.base_dir / "outputs" / f"pipeline_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Pipeline Summary:")
        print(f"⏱️  Duration: {duration}")
        print(f"📁 COLMAP Model: {self.colmap_output}")
        print(f"🔍 Filtered Model: {self.filtered_output}")
        print(f"🌟 Ray Enhanced: {self.ray_output}")
        print(f"📊 Visualization: {self.viz_output}")
        print(f"📄 Summary: {summary_path}")

    def run(self):
        """Execute the complete pipeline"""
        try:
            # Check dependencies before starting
            self.check_dependencies()
            
            if not self.args.skip_segmentation:
                self.step1_segmentation()
            
            if not self.args.skip_colmap:
                self.step2_colmap()
            
            if not self.args.skip_filtering:
                self.step3_filtering()
            
            if not self.args.skip_rays:
                self.step4_ray_enhancement()
            
            if not self.args.skip_visualization:
                self.step5_visualization()
            
            self.save_pipeline_summary()
            print(f"\n🎉 Pipeline completed successfully in {datetime.now() - self.start_time}!")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        finally:
            self.cleanup()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete Botanical Tree Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Basic usage (full pipeline with ray enhancement)
    python main_pipeline.py --images data/raw --model models/model.pth
    
    # Skip segmentation (uses existing masks in data/segmentation_masks/)
    python main_pipeline.py --images data/raw --skip-segmentation
    
    # Skip COLMAP (uses existing model in models/colmap_reconstruction/ or specify custom path)
    python main_pipeline.py --images data/raw --model models/model.pth --skip-colmap
    python main_pipeline.py --images data/raw --model models/model.pth --colmap-model path/to/custom/model --skip-colmap
    
    # Skip filtering (uses existing filtered model in models/colmap_filtered/ or specify custom path)
    python main_pipeline.py --images data/raw --skip-filtering
    python main_pipeline.py --images data/raw --filtered-model path/to/custom/filtered --skip-filtering
    
    # Skip ray enhancement (faster, but less dense point cloud)
    python main_pipeline.py --images data/raw --model models/model.pth --skip-rays
    
    # High quality processing
    python main_pipeline.py --images data/raw --model models/model.pth --max-features 50000 --mask-type both
        """
    )
    
    # Input options
    parser.add_argument("--images", type=Path, required=True, help="Directory containing input images")
    parser.add_argument("--model", type=Path, help="Path to segmentation model (.pth file)")
    
    # Existing model paths (for skipping steps)
    parser.add_argument("--colmap-model", type=Path, help="Path to existing COLMAP model directory (when --skip-colmap)")
    parser.add_argument("--filtered-model", type=Path, help="Path to existing filtered model directory (when --skip-filtering)")
    
    # Processing options
    parser.add_argument("--mask-type", choices=["rough", "fine", "both"], default="both",
                       help="Type of masks to generate")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device for segmentation")
    parser.add_argument("--max-features", type=int, default=20000,
                       help="Maximum features for COLMAP")
    parser.add_argument("--visibility-threshold", type=float, default=0.7,
                       help="Visibility threshold for filtering (0.0-1.0)")
    parser.add_argument("--combine-masks", choices=["or", "and"], default="or",
                       help="How to combine rough and fine masks")
    
    # Output options
    parser.add_argument("--viz-images", type=int, default=3,
                       help="Number of images for visualization")
    parser.add_argument("--point-size", type=float, default=1.0,
                       help="Point size for visualization")
    parser.add_argument("--filter-examples", type=int, default=5,
                       help="Number of filtering examples to save")
    
    # Pipeline control
    parser.add_argument("--skip-segmentation", action="store_true",
                       help="Skip segmentation step (use existing masks)")
    parser.add_argument("--skip-colmap", action="store_true",
                       help="Skip COLMAP reconstruction")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip point cloud filtering")
    parser.add_argument("--skip-rays", action="store_true",
                       help="Skip ray enhancement step")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Skip final visualization")
    
    # Error handling
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue pipeline even if a step fails")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate inputs
    if args.model and not args.model.exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not args.images.exists():
        print(f"❌ Images directory not found: {args.images}")
        sys.exit(1)
    
    # Run pipeline
    runner = PipelineRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Basic Workflow Example

This example demonstrates a complete tree reconstruction workflow
using the botanical tree reconstruction toolkit.
"""

from pathlib import Path
import yaml
from tree_reconstruction import segmentation, colmap_pipeline, filtering, visualization
from tree_reconstruction.config import run_pipeline, PipelineConfig

def basic_workflow_example():
    """Run a basic reconstruction workflow"""
    
    print("🌳 Botanical Tree Reconstruction - Basic Workflow Example")
    print("=" * 60)
    
    # Setup paths
    project_dir = Path("example_project")
    image_dir = project_dir / "data" / "raw"
    
    # Create project structure
    project_dir.mkdir(exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created project directory: {project_dir}")
    print(f"📁 Place your aerial images in: {image_dir}")
    print()
    
    # Create configuration
    config = PipelineConfig()
    config_dict = config._get_default_config()
    config_dict["project_name"] = "basic_example"
    
    # Save configuration
    config_path = project_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"⚙️ Configuration saved to: {config_path}")
    print()
    
    # Check if images exist
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
    
    if not image_files:
        print("ℹ️ To run this example:")
        print(f"   1. Copy your aerial images to: {image_dir}")
        print(f"   2. Run: cd {project_dir}")
        print("   3. Run: tree-reconstruct pipeline --config config.yaml")
        print()
        print("Or run individual stages:")
        print("   tree-reconstruct segment --input-dir data/raw --output-dir data/masks")
        print("   tree-reconstruct colmap --input-dir data/raw --output-dir models/sparse")
        print("   tree-reconstruct filter --input-model models/sparse/sparse/0 --output-model models/filtered")
        print("   tree-reconstruct visualize --input-model models/filtered --output-dir reports/figures")
        return
    
    print(f"📸 Found {len(image_files)} images")
    print("🚀 Running reconstruction pipeline...")
    print()
    
    # Change to project directory
    import os
    original_cwd = os.getcwd()
    os.chdir(project_dir)
    
    try:
        # Run the complete pipeline
        success = run_pipeline(config_dict)
        
        if success:
            print()
            print("✅ Reconstruction completed successfully!")
            print()
            print("📊 Results:")
            print(f"   - Segmentation masks: data/processed/masks/")
            print(f"   - Sparse 3D model: models/sparse/")
            print(f"   - Filtered 3D model: models/filtered/")
            print(f"   - Visualizations: reports/figures/")
            
        else:
            print("❌ Reconstruction failed. Check the logs for details.")
            
    finally:
        os.chdir(original_cwd)

def step_by_step_example():
    """Run reconstruction step by step with detailed control"""
    
    print("🔧 Step-by-Step Reconstruction Example")
    print("=" * 45)
    
    # Define paths
    image_dir = Path("data/raw")
    mask_dir = Path("data/masks")
    filtered_dir = Path("data/filtered")
    sparse_dir = Path("models/sparse")
    clean_dir = Path("models/filtered")
    vis_dir = Path("reports/figures")
    
    # Create directories
    for directory in [mask_dir, filtered_dir, sparse_dir, clean_dir, vis_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Segmentation
    print("🎯 Step 1: Running segmentation...")
    segmentation.run_segmentation(
        input_dir=image_dir,
        output_dir=mask_dir,
        confidence=0.5,
        device="cuda"
    )
    
    # Filter images based on vegetation content
    segmentation.filter_images_by_mask(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=filtered_dir,
        min_vegetation_ratio=0.1
    )
    
    # Step 2: COLMAP reconstruction
    print("📐 Step 2: Running COLMAP reconstruction...")
    colmap_pipeline.run_colmap_pipeline(
        input_dir=filtered_dir,
        output_dir=sparse_dir,
        max_features=30000,
        use_gpu=True,
        sequential=True
    )
    
    # Step 3: Point cloud filtering
    print("🔧 Step 3: Filtering point cloud...")
    filtering.filter_point_cloud(
        input_model=sparse_dir / "sparse" / "0",
        output_model=clean_dir,
        mask_dir=mask_dir,
        filter_type="combined",
        coord_threshold=3.0,
        add_points=False
    )
    
    # Step 4: Visualization
    print("📊 Step 4: Creating visualizations...")
    visualization.create_visualizations(
        input_model=clean_dir,
        output_dir=vis_dir,
        image_dir=filtered_dir,
        plot_type="all",
        num_images=6
    )
    
    print("✅ Step-by-step reconstruction completed!")

if __name__ == "__main__":
    print("Select an example to run:")
    print("1. Basic workflow (recommended)")
    print("2. Step-by-step example")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        basic_workflow_example()
    elif choice == "2":
        step_by_step_example()
    else:
        print("Invalid choice. Running basic workflow...")
        basic_workflow_example()
#!/bin/bash
# run_pipeline_configs.sh - Different configuration examples for the pipeline

# Basic configuration
submit_basic() {
    echo "Submitting basic pipeline job..."
    bsub < run_pipeline.sh
}

# High quality configuration (more features, stricter filtering)
submit_high_quality() {
    echo "Submitting high quality pipeline job..."
    
    # Create temporary job script
    cp run_pipeline.sh run_pipeline_hq.sh
    
    # Modify parameters for high quality
    sed -i 's/--max-features 20000/--max-features 50000/' run_pipeline_hq.sh
    sed -i 's/--visibility-threshold 0.7/--visibility-threshold 0.8/' run_pipeline_hq.sh
    sed -i 's/--combine-masks or/--combine-masks and/' run_pipeline_hq.sh
    sed -i 's/#BSUB -W 04:00/#BSUB -W 06:00/' run_pipeline_hq.sh
    sed -i 's/#BSUB -J botanical_pipeline/#BSUB -J botanical_pipeline_hq/' run_pipeline_hq.sh
    
    bsub < run_pipeline_hq.sh
    rm run_pipeline_hq.sh
}

# Fast configuration (skip segmentation if masks exist)
submit_fast() {
    echo "Submitting fast pipeline job (skipping segmentation)..."
    
    # Create temporary job script
    cp run_pipeline.sh run_pipeline_fast.sh
    
    # Add skip segmentation flag
    sed -i 's/--device cuda/--device cuda --skip-segmentation/' run_pipeline_fast.sh
    sed -i 's/#BSUB -W 04:00/#BSUB -W 02:00/' run_pipeline_fast.sh
    sed -i 's/#BSUB -J botanical_pipeline/#BSUB -J botanical_pipeline_fast/' run_pipeline_fast.sh
    
    bsub < run_pipeline_fast.sh
    rm run_pipeline_fast.sh
}

# Show usage
show_usage() {
    echo "Usage: ./run_pipeline_configs.sh [basic|high_quality|fast]"
    echo ""
    echo "Configurations:"
    echo "  basic       - Standard pipeline (20k features, 0.7 threshold, OR masks)"
    echo "  high_quality - High quality (50k features, 0.8 threshold, AND masks)"
    echo "  fast        - Skip segmentation (use existing masks)"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline_configs.sh basic"
    echo "  ./run_pipeline_configs.sh high_quality"
}

# Main script
case "$1" in
    basic)
        submit_basic
        ;;
    high_quality)
        submit_high_quality
        ;;
    fast)
        submit_fast
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
#!/bin/bash
#BSUB -J botanical_pipeline
#BSUB -q gpua100
#BSUB -W 06:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -B
## BSUB -N
## BSUB -u xx@dtu.dk # write email here and uncomment

# Load required modules
module load colmap/3.8-cuda-11.8-avx512

# Activate conda environment if needed
source /work3/s204201/miniconda3/etc/profile.d/conda.sh
conda activate colmap_env

# Project paths
PROJECT_DIR="/work3/s204201/BotanicalTreeReconstruction"
IMAGES_DIR="/work3/s204201/BotanicalTreeReconstruction/data/raw"
MODEL_PATH="/work3/s204201/BotanicalTreeReconstruction/models/model.pth"

echo "=== Starting Botanical Tree Reconstruction Pipeline ==="
echo "Project Directory: $PROJECT_DIR"
echo "Images Directory: $IMAGES_DIR"
echo "Model Path: $MODEL_PATH"
echo "Job ID: $LSB_JOBID"
echo "Date: $(date)"

# Change to project directory
cd $PROJECT_DIR

# Verify inputs exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check COLMAP is available
if ! command -v colmap &> /dev/null; then
    echo "ERROR: COLMAP not found in PATH"
    exit 1
fi

echo "=== Environment Check Complete ==="

# Run the main pipeline
echo "=== Executing Main Pipeline ==="
cd scripts

python run_pipeline.py \
    --images "$IMAGES_DIR" \
    --model "$MODEL_PATH" \
    --mask-type both \
    --max-features 30000 \
    --visibility-threshold 0.7 \
    --combine-masks or \
    --viz-images 3 \
    --filter-examples 5 \
    --device cuda \
    --skip-rays

PIPELINE_EXIT_CODE=$?

echo "=== Pipeline Completed ==="
echo "Exit Code: $PIPELINE_EXIT_CODE"
echo "End Time: $(date)"

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Pipeline completed successfully!"
    
    # List outputs
    echo "=== Generated Outputs ==="
    find ../models -name "*$(date +%d%m)*" -type d 2>/dev/null | head -5
    find ../outputs -name "*$(date +%d%m)*" -type f 2>/dev/null | head -5
    
else
    echo "ERROR: Pipeline failed with exit code $PIPELINE_EXIT_CODE"
    echo "Check the job error file for details: ${LSB_JOBID}.err"
fi

exit $PIPELINE_EXIT_CODE
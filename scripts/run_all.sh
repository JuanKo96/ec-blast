#!/bin/bash
#SBATCH -c 8  # Number of Cores
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs and type
#SBATCH -C vram23  
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o sbatch_output/slurm-%A_%a.out  # Output file (%A = job ID, %a = array index)
#SBATCH -e sbatch_output/slurm-%A_%a.err
#SBATCH --array=0-11  # Job array with indices 0-11 for 12 parameter combinations (2×2×3)

# Activate conda environment
source ~/anaconda3/bin/activate ai4sci

# Define parameter arrays
ALPHA_VALUES=(1.0 0.7)
FINE_TUNING_MODES=(full classification_only)
MODEL_NAMES=(
  "facebook/esm2_t6_8M_UR50D"
  "facebook/esm2_t6_8M_UR50D"
  "facebook/esm2_t12_35M_UR50D"
)

# Calculate which parameters to use based on array index
index=$SLURM_ARRAY_TASK_ID

# Calculate indices for each parameter
alpha_idx=$((index % 2))
ft_mode_idx=$(((index / 2) % 2))
model_idx=$(((index / 4) % 3))

# Get parameter values
alpha=${ALPHA_VALUES[$alpha_idx]}
ft_mode=${FINE_TUNING_MODES[$ft_mode_idx]}
model=${MODEL_NAMES[$model_idx]}

echo "Running experiment $SLURM_ARRAY_TASK_ID with parameters:"
echo "  - training.alpha = $alpha"
echo "  - model.fine_tuning_mode = $ft_mode"
echo "  - model.model_name = $model"

# Run the Python script with the selected parameters
python3 -m ec_blast.main training.alpha=$alpha model.layer_idx=-1 model.fine_tuning_mode=$ft_mode \
    model.model_name=\"$model\" model.architecture=hierarchical training.batch_size=16 training.num_epochs=200

wait
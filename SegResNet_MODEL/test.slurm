#!/bin/sh
#SBATCH --account=ie-idi      
#SBATCH --job-name=train_model_jeff
#SBATCH --time=0-00:45:00         

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=32G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
#SBATCH --mail-user=katan@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2023.09-0
conda activate /cluster/home/katan/.conda/envs/tdt4265_project

# Run the generated Python script
python /cluster/home/katan/Main.py



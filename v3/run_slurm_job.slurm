#!/bin/bash
#SBATCH --job-name=imdb_skyline_15k_v3
#SBATCH --output=/specific/scratches/parallel/galzeevi-2023-01-06/slurm-logs/imdb_skyline_15k_v3.log
#SBATCH --partition=cpu-killable
#SBATCH --gpus=0
#SBATCH --cpus-per-task=36
#SBATCH --exclude=rack-iscb-16
#SBATCH --mem-per-cpu=10GB

export HOME="/specific/scratches/parallel/galzeevi-2023-01-06"
export PATH="$HOME/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
source $HOME/anaconda3/etc/profile.d/conda.sh

conda init bash
source activate saqp

echo "STARTING JOB $SLURM_JOB_NAME"
# python k_means_sampler.py --k 15000 --lib "pyclustering" --pool_size 250 --checkpoint 3
python3 skyline_sampler.py --k 15000 --pool_size 250 --checkpoint 3
# python greedy_sampler.py --k 1000 --queries "train" --pool_size 600 --diversity_coeff 0.5 --checkpoint 7
# python brute_force_sampler2.py --k 1000 --checkpoint 10 --pool 1000 --horizon_time 48
# python random_sampler.py
# python3 top_queried_sampler.py

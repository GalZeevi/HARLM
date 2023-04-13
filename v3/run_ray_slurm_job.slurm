#!/bin/bash

#SBATCH --partition=killable
#SBATCH --job-name=saqp_ppo
#SBATCH --output=/specific/scratches/parallel/galzeevi-2023-01-06/slurm-logs/saqp_ppo.log

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1
# n-502,n-503,n-601,rack-gww-dgx1,rack-bgw-dgx1 are bad servers
# Also, exclude any servers that already have Ray clusters running on them
#SBATCH --exclude=n-502,n-503,n-601,rack-gww-dgx1,rack-bgw-dgx1,n-351

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}
export HOME="/specific/scratches/parallel/galzeevi-2023-01-06"
export PATH="$HOME/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
source $HOME/anaconda3/etc/profile.d/conda.sh

conda init bash
source activate saqp

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address

if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 \
  ray start --head --node-ip-address=$ip --port=$port \
   --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i \
      ray start --address $ip_head \
       --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
  sleep 5
done

##############################################################################################

#### call your code below
echo "STARTING PYTHON SCRIPT"
python3 ray_sampler.py --remote True --alg PPO --cpus "${SLURM_CPUS_PER_TASK}" --gpus "${SLURM_GPUS_PER_TASK}" --rollouts 8
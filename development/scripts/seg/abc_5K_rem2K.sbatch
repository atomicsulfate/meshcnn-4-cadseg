#!/bin/bash

#SBATCH -o %x_%j_%N.out   	# Output-File
#SBATCH -e %x_%j_%N.out		# stderr
#SBATCH -D /home/users/m/mandadoalmajano/dev	        # Working Directory
#SBATCH -J MeshCNNABC5KRem2K	# Job Name
#SBATCH --nodes=1
#SBATCH --ntasks=1 		# Anzahl Prozesse P (CPU-Cores) 
#SBATCH --cpus-per-task=40	# Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:2		# 2 GPUs anfordern
#SBATCH --mem=16GB              # 16GiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=7-00:00:00 # Erwartete Laufzeit

#Auf GPU-Knoten rechnen:
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mandadoalmajano@campus.tu-berlin.de

# benötigte SW / Bibliotheken laden (CUDA, etc.)

#module purge

#module load nvidia/cuda/10.1
#module load python/3.7.1 

echo $PWD
echo "Entering working directory"
echo $PWD

cd /home/users/m/mandadoalmajano/dev

echo "Activating virtual environment"
source /home/users/m/mandadoalmajano/.venvs/meshcnn/bin/activate 
type python

echo "running training"
python train.py --dataroot datasets/abc_5K_rem2K --name abc_5K_rem2K --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 512 --ninput_edges 2000 --pool_res 1600 1280 1024 850 --resblocks 3 --lr 0.001 --batch_size 32 --num_aug 1 --gpu_ids 0,1 
exitCode=$?
echo "done training. Exit code was $exitCode"

deactivate

exit $exitCode

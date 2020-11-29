#!/bin/bash

#SBATCH -o %x_%j_%N.out         # Output-File
#SBATCH -e %x_%j_%N.out         # stderr
#SBATCH -D /home/users/m/mandadoalmajano/dev            # Working Directory
#SBATCH -J MeshCNNABCSmall              # Job Name
#SBATCH --nodes=1
#SBATCH --ntasks=1              # Anzahl Prozesse P (CPU-Cores)
#SBATCH --cpus-per-task=1       # Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:1            # 2 GPUs anfordern
#SBATCH --mem=16GB              # 16GiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=00:05:00 # Erwartete Laufzeit

#Auf GPU-Knoten rechnen:
#SBATCH --partition=gpu_short

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

echo "running training"
python train.py --dataroot datasets/abc_small --name abc_small --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 --ninput_edges 5500 --pool_res 4000 3000 1800 --resblocks 3 --lr 0.001 --batch_size 12 --num_aug 20 --slide_verts 0.2
exitCode=$?
echo "done training. Exit code was $exitCode"

deactivate

exit $exitCode

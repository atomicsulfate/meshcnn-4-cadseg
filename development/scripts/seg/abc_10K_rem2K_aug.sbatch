#!/bin/bash

#SBATCH -o %x_%j_%N.out   	# Output-File
#SBATCH -e %x_%j_%N.out		# stderr
#SBATCH -D /home/users/m/mandadoalmajano/dev	        # Working Directory
#SBATCH -J MeshCNNABC10KRem2KAug	# Job Name
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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


#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"
echo "Master: $MASTER"
echo "Slaves: $SLAVES"

cd /home/users/m/mandadoalmajano/dev
source /home/users/m/mandadoalmajano/.venvs/meshcnn/bin/activate
type python

MPORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');
echo "Master port: $MPORT"

COMMAND="train.py --dataroot datasets/abc_10K_rem2K_aug --name abc_10K_rem2K_aug --arch meshunet --dataset_mode segmentation --ncf 32 64 128 256 512 --ninput_edges 2000 --pool_res 1600 1280 1024 850 --resblocks 3 --lr 0.002 --batch_size 16 --num_aug 1 --gpu_ids 0,1 --continue_train"

#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves
RANK=0
set -x
for node in $HOSTLIST; do
        srun -N 1 -n 1 --nodelist=$node \
                python -m torch.distributed.launch \
                --nproc_per_node 2 \
                --nnodes $SLURM_JOB_NUM_NODES \
                --node_rank $RANK \
                --master_addr "$MASTER" --master_port "$MPORT" --use_env \
                $COMMAND &
        RANK=$((RANK+1))
done
wait
exitCode=$?
echo "done training. Exit code was $exitCode"

deactivate

exit $exitCode
